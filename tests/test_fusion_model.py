from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from yp_video.actor import training_labels
from yp_video.contracts.action import RECIPES
from yp_video.core.jsonl import write_jsonl
from yp_video.web import spot_training
from yp_video.web.label_sources import PreparedLabels, check_task_supervision
from yp_video.web.routers import fusion_model
from yp_video.web.train_requests import FusionTrainRequest


class FusionModelStatusTests(unittest.TestCase):
    def test_status_serves_the_registry_and_per_recipe_init_options(self) -> None:
        with (
            patch.object(
                fusion_model.training,
                "annotation_stats",
                return_value={
                    "videos": 2,
                    "events": 20,
                    "per_video": [
                        {"video": "joint", "events": 10},
                        {"video": "action_only", "events": 10},
                    ],
                },
            ),
            patch.object(
                fusion_model, "checkpoint_package_options",
                side_effect=lambda _dir, tasks: [{"label": ",".join(tasks), "value": "x"}],
            ),
            patch.object(fusion_model.association_labels, "labeled_stems", return_value=["joint"]),
            patch.object(fusion_model.rally_spot, "select_training_items", return_value=([], [])),
            patch.object(fusion_model.rally_spot, "rally_stats", return_value={"videos": 0}),
            patch.object(fusion_model.rally_spot, "frame_cache_stats", return_value=[]),
        ):
            payload = fusion_model.status()

        recipes = {row["id"]: row for row in payload["recipes"]}
        self.assertEqual(set(recipes), set(RECIPES))
        self.assertEqual(recipes["rally_winner"]["tasks"], ["rally", "winner"])
        self.assertEqual(recipes["rally_winner"]["fields"], ["extract_fps", "video_limit"])
        self.assertEqual(recipes["association_action"]["serveable_tasks"], ["action", "actor"])
        self.assertEqual(payload["init_checkpoints"]["rally"], [{"label": "rally", "value": "x"}])
        self.assertEqual(payload["task_labels"]["winner"], "Winner")
        self.assertEqual(
            payload["supervision"],
            {"action_videos": 2, "joint_videos": 1, "action_only_videos": 1},
        )


class BuildCommandTests(unittest.TestCase):
    def _prepared(self, dataset: str, extra=()) -> PreparedLabels:
        return PreparedLabels(
            label_dir=Path("/run/labels/x"),
            label_subdirs=("x",),
            frame_dir=Path("/frames"),
            dataset=dataset,
            extra_args=list(extra),
        )

    def test_rally_winner_command(self) -> None:
        req = FusionTrainRequest(recipe="rally_winner", extract_fps=5, audio_backend="logmel", acc_grad_iter=4, batch_size=8)
        cmd = spot_training.build_command(
            req, RECIPES["rally_winner"], self._prepared("yp_rally"),
            save_dir=Path("/run"), init_checkpoint=None, audio_dir=None,
        )
        joined = " ".join(cmd)
        self.assertIn(" yp_rally /frames ", joined)
        self.assertIn("--tasks rally,winner", joined)
        self.assertIn("--sample_fps 5", joined)
        # Rally is visual-only and never accumulates, whatever the form sent.
        self.assertIn("--audio_backend none", joined)
        self.assertIn("--acc_grad_iter 1", joined)
        self.assertIn("--label_dir /run/labels/x --val_ratio 0.2 --split_seed 42", joined)
        self.assertNotIn("--predict", joined)

    def test_association_action_manual_validation_command(self) -> None:
        req = FusionTrainRequest(
            recipe="association_action", validation="manual", validation_videos=["a"],
            sample_fps=30, acc_grad_iter=2, batch_size=8, audio_backend="logmel",
        )
        cmd = spot_training.build_command(
            req, RECIPES["association_action"],
            self._prepared("yp_actions", ["--actor_dir", "/run/labels/actor-candidates"]),
            save_dir=Path("/run"), init_checkpoint=None, audio_dir=Path("/audio"),
        )
        joined = " ".join(cmd)
        self.assertIn("--tasks action,location,actor", joined)
        self.assertIn("--actor_dir /run/labels/actor-candidates", joined)
        self.assertIn("--audio_backend logmel --actor_dir", joined)
        self.assertIn("--audio_dir /audio", joined)
        self.assertIn("--train_labels /run/labels/train --val_labels /run/labels/val", joined)

    def test_run_name_token_per_recipe(self) -> None:
        self.assertEqual(spot_training.recipe_token(RECIPES["rally_winner"]), "ral_win")
        self.assertEqual(spot_training.recipe_token(RECIPES["association_action"]), "ass_act")
        self.assertEqual(spot_training.recipe_token(RECIPES["action"]), "act")

    def test_bad_run_name_is_refused(self) -> None:
        with self.assertRaises(HTTPException) as caught:
            spot_training.resolve_run_name(
                FusionTrainRequest(run_name="../escape"), RECIPES["action"]
            )
        self.assertEqual(caught.exception.status_code, 400)


class SupervisionGateTests(unittest.TestCase):
    def _prepared(self, summary: dict) -> PreparedLabels:
        return PreparedLabels(Path("/x"), ("x",), Path("/f"), "yp_rally", summary=summary)

    def test_winner_head_needs_winner_labels(self) -> None:
        with self.assertRaises(RuntimeError) as caught:
            check_task_supervision(RECIPES["rally_winner"], self._prepared({"rallies_with_winner": 0}))
        self.assertIn("winner", str(caught.exception))
        check_task_supervision(RECIPES["rally_winner"], self._prepared({"rallies_with_winner": 3}))
        check_task_supervision(RECIPES["rally"], self._prepared({"rallies_with_winner": 0}))

    def test_actor_head_needs_actor_targets(self) -> None:
        with self.assertRaises(RuntimeError):
            check_task_supervision(
                RECIPES["association_action"], self._prepared({"actor_targets": {"track": 0}})
            )
        check_task_supervision(RECIPES["action"], self._prepared({"actor_targets": {"track": 0}}))


class FusionLabelScopeTests(unittest.TestCase):
    def test_joint_only_snapshot_fails_when_a_video_has_no_actor_targets(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            label = root / "match_actions.jsonl"
            video = root / "match.mp4"
            video.touch()
            write_jsonl(
                label,
                {"video": "match", "num_frames": 100, "fps": 30},
                [{"id": "event", "frame": 10, "label": "spike"}],
            )
            with (
                patch.object(training_labels, "inspect_action_frame_cache", return_value={"frame_count": 100}),
                patch.object(training_labels, "cut_kind_of", return_value="sideline"),
                patch.object(training_labels.candidates, "build", return_value=([], {})),
            ):
                with self.assertRaises(RuntimeError) as caught:
                    training_labels.prepare_action_training_labels(
                        items=[(label, video)],
                        frame_dir=root / "frames",
                        save_dir=root / "run",
                        tasks=("action", "location", "actor"),
                        require_actor_targets=True,
                    )

        self.assertIn("produced no usable actor targets", str(caught.exception))

    def test_action_only_recipe_writes_no_actor_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            label = root / "match_actions.jsonl"
            video = root / "match.mp4"
            video.touch()
            write_jsonl(
                label,
                {"video": "match", "num_frames": 100, "fps": 30},
                [{"id": "event", "frame": 10, "label": "spike"}],
            )
            with (
                patch.object(training_labels, "inspect_action_frame_cache", return_value={"frame_count": 100}),
                patch.object(training_labels, "cut_kind_of", return_value="sideline"),
                patch.object(training_labels.candidates, "build") as build,
            ):
                summary = training_labels.prepare_action_training_labels(
                    items=[(label, video)],
                    frame_dir=root / "frames",
                    save_dir=root / "run",
                    tasks=("action", "location"),
                )
            build.assert_not_called()
            self.assertFalse((root / "run" / "labels" / "actor-candidates").exists())
            self.assertEqual(summary["videos"], 1)


if __name__ == "__main__":
    unittest.main()
