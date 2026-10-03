from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from yp_video.action import training_labels
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
            patch.object(fusion_model.rally_spot, "select_training_items", return_value=([], [])),
            patch.object(fusion_model.rally_spot, "rally_stats", return_value={"videos": 0}),
        ):
            payload = fusion_model.status()

        recipes = {row["id"]: row for row in payload["recipes"]}
        self.assertEqual(set(recipes), set(RECIPES))
        self.assertEqual(recipes["rally_winner"]["tasks"], ["rally", "winner"])
        self.assertEqual(recipes["rally_winner"]["fields"], ["sample_fps", "acc_grad_iter", "video_limit"])
        self.assertEqual(recipes["rally"]["defaults"]["sample_fps"], 5.0)
        self.assertEqual(recipes["rally_winner"]["defaults"]["sample_fps"], 5.0)
        self.assertEqual(
            recipes["action_rally_winner"]["tasks"],
            ["action", "location", "rally", "winner", "person"],
        )
        self.assertEqual(
            recipes["action_rally_winner"]["defaults"]["action_sample_fps"],
            30.0,
        )
        self.assertEqual(
            recipes["action_rally_winner"]["defaults"]["rally_sample_fps"],
            5.0,
        )
        self.assertEqual(payload["init_checkpoints"]["rally"], [{"label": "rally", "value": "x"}])
        self.assertEqual(payload["task_labels"]["winner"], "Winner")


class BuildCommandTests(unittest.TestCase):
    def _prepared(self, dataset: str, extra=()) -> PreparedLabels:
        task = "rally" if dataset == "yp_rally" else "action"
        return PreparedLabels(
            label_dirs={task: Path("/run/labels/x")},
            label_subdirs=("x",),
            frame_dir=Path("/frames"),
            dataset=dataset,
            extra_args=list(extra),
        )

    def test_rally_winner_command(self) -> None:
        req = FusionTrainRequest(recipe="rally_winner", validation="ratio", sample_fps=5, audio_backend="logmel", acc_grad_iter=4, batch_size=8)
        cmd = spot_training.build_command(
            req, RECIPES["rally_winner"], self._prepared("yp_rally"),
            save_dir=Path("/run"), init_checkpoint=None, audio_dir=None,
        )
        joined = " ".join(cmd)
        self.assertIn(" yp_rally /frames ", joined)
        self.assertIn("--tasks rally,winner", joined)
        self.assertIn("--sample_fps 5", joined)
        # Rally is visual-only whatever the form sent; accumulation passes through.
        self.assertIn("--audio_backend none", joined)
        self.assertIn("--acc_grad_iter 4", joined)
        self.assertIn("--label_dir /run/labels/x --val_ratio 0.1 --split_seed 42", joined)
        self.assertNotIn("--predict", joined)

    def test_multi_fps_command_has_independent_streams(self) -> None:
        req = FusionTrainRequest(
            recipe="action_rally_winner",
            validation="ratio",
            action_learning_rate=3e-4,
            rally_learning_rate=3e-5,
            winner_learning_rate=3e-5,
            action_fg_upsample=0.5,
            action_dilate_len=1,
        )
        self.assertEqual(req.val_ratio, 0.1)
        self.assertEqual(req.start_val_epoch, 0)
        prepared = PreparedLabels(
            label_dirs={
                "action": Path("/run/labels/action-annotations"),
                "rally": Path("/run/labels/rally-annotations"),
            },
            label_subdirs=("action-annotations", "rally-annotations"),
            frame_dir=Path("/frames"),
            dataset="yp_action_rally",
        )
        cmd = spot_training.build_command(
            req,
            RECIPES["action_rally_winner"],
            prepared,
            save_dir=Path("/run"),
            init_checkpoint=None,
            audio_dir=None,
        )
        joined = " ".join(cmd)
        self.assertIn("--tasks action,location,rally,winner,person", joined)
        self.assertIn("--task_sample_fps action=30.0", joined)
        self.assertIn("--task_sample_fps rally=5.0", joined)
        self.assertIn("--task_sample_fps winner=5.0", joined)
        self.assertIn("--task_learning_rate action=0.0003", joined)
        self.assertIn("--task_learning_rate rally=3e-05", joined)
        self.assertIn("--task_learning_rate winner=3e-05", joined)
        self.assertIn("--task_audio_backend action=logmel", joined)
        self.assertIn("--task_audio_backend rally=none", joined)
        self.assertIn("--task_fg_upsample action=0.5", joined)
        self.assertIn("--task_dilate_len action=1", joined)
        self.assertNotIn("--dilate_len", joined)
        self.assertIn(
            "--task_label_dir action=/run/labels/action-annotations", joined
        )
        self.assertIn(
            "--task_label_dir rally=/run/labels/rally-annotations", joined
        )
        self.assertNotIn("--sample_fps ", joined)
        self.assertIn("--audio_backend logmel", joined)

    def test_run_name_token_per_recipe(self) -> None:
        self.assertEqual(spot_training.recipe_token(RECIPES["rally_winner"]), "ral_win")
        self.assertEqual(spot_training.recipe_token(RECIPES["action"]), "act")
        self.assertEqual(
            spot_training.recipe_token(RECIPES["action_rally_winner"]),
            "act_ral_win",
        )

    def test_bad_run_name_is_refused(self) -> None:
        with self.assertRaises(HTTPException) as caught:
            spot_training.resolve_run_name(
                FusionTrainRequest(run_name="../escape", validation="ratio"), RECIPES["action"]
            )
        self.assertEqual(caught.exception.status_code, 400)


class SupervisionGateTests(unittest.TestCase):
    def _prepared(self, summary: dict) -> PreparedLabels:
        return PreparedLabels(
            {"rally": Path("/x")},
            ("x",),
            Path("/f"),
            "yp_rally",
            summary=summary,
        )

    def test_winner_head_needs_winner_labels(self) -> None:
        with self.assertRaises(RuntimeError) as caught:
            check_task_supervision(RECIPES["rally_winner"], self._prepared({"rallies_with_winner": 0}))
        self.assertIn("winner", str(caught.exception))
        check_task_supervision(RECIPES["rally_winner"], self._prepared({"rallies_with_winner": 3}))
        check_task_supervision(RECIPES["rally"], self._prepared({"rallies_with_winner": 0}))

class FusionLabelScopeTests(unittest.TestCase):
    def test_snapshot_carries_rally_spans_for_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            label = root / "match_actions.jsonl"
            video = root / "match.mp4"
            video.touch()
            write_jsonl(
                label,
                {"video": "match", "num_frames": 300, "fps": 30},
                [{"id": "event", "frame": 40, "label": "spike"}],
            )
            rallies = [{"rally_id": 1, "start": 1.0, "end": 2.5}, {"rally_id": 2, "start": 4.0, "end": 6.0}]
            with (
                patch.object(training_labels, "inspect_action_frame_cache", return_value={"frame_count": 300}),
                patch.object(training_labels, "cut_kind_of", return_value="sideline"),
                patch.object(training_labels, "load_rallies", return_value=rallies),
                patch.object(training_labels, "rally_match_span", return_value=(0, 240)),
            ):
                training_labels.prepare_action_training_labels(
                    items=[(label, video)],
                    frame_dir=root / "frames",
                    save_dir=root / "run",
                )
            written = next((root / "run" / "labels").rglob("match_actions.jsonl"))
            meta = json.loads(written.read_text().splitlines()[0])
            self.assertEqual(meta["rally_spans"], [[1.0, 2.5], [4.0, 6.0]])


if __name__ == "__main__":
    unittest.main()
