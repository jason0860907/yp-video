from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from yp_video.core.jsonl import write_jsonl
from yp_video.web.routers import fusion_model


class FusionModelStatusTests(unittest.TestCase):
    def test_registry_exposes_current_and_future_recipes_honestly(self) -> None:
        with (
            patch.object(
                fusion_model.action_train,
                "status",
                return_value={
                    "spot_available": True,
                    "init_checkpoints": [],
                    "resumable_runs": [],
                    "action_annotations": {
                        "videos": 2,
                        "events": 20,
                        "per_video": [
                            {"video": "joint", "events": 10},
                            {"video": "action_only", "events": 10},
                        ],
                    },
                },
            ),
            patch.object(
                fusion_model.association_labels,
                "labeled_stems",
                return_value=["joint"],
            ),
            patch.object(
                fusion_model.spot_associate,
                "list_association_checkpoints",
                return_value=[
                    {"name": "joint", "family": "legacy-actor-head"},
                    {"name": "independent", "family": "yp-association-v1"},
                ],
            ),
        ):
            payload = fusion_model.status()

        recipes = {row["id"]: row for row in payload["recipes"]}
        self.assertTrue(recipes["association_action"]["available"])
        self.assertEqual(
            recipes["association_action"]["predict_outputs"],
            ["association", "action"],
        )
        self.assertFalse(recipes["rally_action"]["available"])
        self.assertIn("sampling rates", recipes["rally_action"]["blocked_on"])
        self.assertFalse(recipes["association_action_rally"]["available"])
        self.assertEqual(
            [row["name"] for row in payload["checkpoints"]],
            ["joint"],
        )
        self.assertEqual(
            payload["supervision"],
            {
                "action_videos": 2,
                "joint_videos": 1,
                "action_only_videos": 1,
            },
        )

    def test_performance_uses_the_shared_task_metrics_reader(self) -> None:
        with patch.object(
            fusion_model,
            "performance_payload",
            return_value={"run": "yp_fusion_one", "entries": []},
        ) as read:
            payload = fusion_model.performance("yp_fusion_one")

        self.assertEqual(payload["run"], "yp_fusion_one")
        read.assert_called_once_with(
            fusion_model.ACTION_CHECKPOINTS_DIR,
            "yp_fusion_one",
            run_prefixes=("yp_fusion_", "yp_actor_only"),
        )


class FusionModelTrainTests(unittest.IsolatedAsyncioTestCase):
    async def test_association_action_maps_to_the_joint_actor_trainer(self) -> None:
        start = AsyncMock(return_value={"id": "job"})
        request = fusion_model.FusionTrainRequest(
            run_name="joint_run",
            audio_backend="none",
            num_epochs=7,
            batch_size=3,
        )
        label_item = (
            Path("/labels/joint_actions.jsonl"),
            Path("/videos/joint.mp4"),
        )
        with (
            patch.object(
                fusion_model.action_train,
                "start_training_job",
                start,
            ),
            patch.object(
                fusion_model.action_train,
                "_action_label_items",
                return_value=[label_item],
            ),
            patch.object(
                fusion_model.association_labels,
                "labeled_stems",
                return_value=["joint"],
            ),
        ):
            result = await fusion_model.train(request)

        self.assertEqual(result, {"id": "job"})
        action_request = start.await_args.args[0]
        flavor = start.await_args.kwargs["flavor"]
        self.assertTrue(action_request.predict_location)
        self.assertTrue(action_request.predict_actor)
        self.assertEqual(action_request.audio_backend, "none")
        self.assertEqual(action_request.num_epochs, 7)
        self.assertEqual(action_request.batch_size, 3)
        self.assertTrue(action_request.save_dir.endswith("/exp/joint_run"))
        self.assertTrue(action_request.checkpoint_dir.endswith("/joint_run"))
        self.assertEqual(flavor.job_type, "fusion_model_train")
        self.assertEqual(flavor.package_type, "actor-association-spot")
        self.assertFalse(start.await_args.kwargs["reuse_existing_labels"])
        self.assertEqual(start.await_args.kwargs["label_items"], [label_item])
        self.assertTrue(start.await_args.kwargs["require_actor_targets"])

    async def test_resume_preserves_the_joint_run_contract_and_label_snapshot(
        self,
    ) -> None:
        start = AsyncMock(return_value={"id": "resume-job"})
        with tempfile.TemporaryDirectory() as raw_dir:
            spot_root = Path(raw_dir) / "yp-spot"
            run = spot_root / "exp" / "yp_actor_only"
            run.mkdir(parents=True)
            (run / "config.json").write_text(
                json.dumps(
                    {
                        "dataset": "yp_actions",
                        "predict_actor": True,
                        "audio_backend": "logmel",
                        "feature_arch": "rny008_gsm",
                        "temporal_arch": "gru",
                        "clip_len": 64,
                        "sample_fps": 30,
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch.object(fusion_model, "SPOT_DIR", spot_root),
                patch.object(
                    fusion_model,
                    "ACTION_CHECKPOINTS_DIR",
                    Path(raw_dir) / "checkpoints",
                ),
                patch.object(
                    fusion_model.action_train,
                    "start_training_job",
                    start,
                ),
            ):
                result = await fusion_model.train(
                    fusion_model.FusionTrainRequest(
                        resume_run=str(run),
                        feature_arch="rny002",
                        audio_backend="none",
                        num_epochs=60,
                    )
                )

        self.assertEqual(result, {"id": "resume-job"})
        action_request = start.await_args.args[0]
        self.assertTrue(action_request.resume)
        self.assertEqual(action_request.save_dir, str(run))
        self.assertIsNone(action_request.init_checkpoint)
        self.assertEqual(action_request.feature_arch, "rny008_gsm")
        self.assertEqual(action_request.audio_backend, "logmel")
        self.assertEqual(action_request.num_epochs, 60)
        self.assertTrue(start.await_args.kwargs["reuse_existing_labels"])

    async def test_manual_validation_videos_map_to_the_holdout_contract(self) -> None:
        start = AsyncMock(return_value={"id": "holdout-job"})
        with patch.object(
            fusion_model.action_train,
            "start_training_job",
            start,
        ):
            await fusion_model.train(
                fusion_model.FusionTrainRequest(
                    run_name="manual_holdout",
                    dataset_scope="partial_labels",
                    validation_mode="manual",
                    validation_videos=[
                        "match_a_actions.jsonl",
                        "match_b_actions.jsonl",
                    ],
                )
            )

        action_request = start.await_args.args[0]
        self.assertEqual(action_request.training_mode, "holdout")
        self.assertEqual(
            action_request.holdout_videos,
            ["match_a_actions.jsonl", "match_b_actions.jsonl"],
        )
        self.assertIsNone(start.await_args.kwargs["label_items"])
        self.assertFalse(start.await_args.kwargs["require_actor_targets"])

    async def test_manual_validation_refuses_an_empty_selection(self) -> None:
        with self.assertRaises(HTTPException) as caught:
            await fusion_model.train(
                fusion_model.FusionTrainRequest(
                    validation_mode="manual",
                    validation_videos=[],
                )
            )

        self.assertEqual(caught.exception.status_code, 400)
        self.assertIn("at least one validation video", str(caught.exception.detail))

    async def test_unimplemented_recipe_is_refused_before_starting_a_job(self) -> None:
        with self.assertRaises(HTTPException) as caught:
            await fusion_model.train(
                fusion_model.FusionTrainRequest(recipe="rally_action")
            )

        self.assertEqual(caught.exception.status_code, 409)
        self.assertIn("multi-task", str(caught.exception.detail))


class FusionLabelScopeTests(unittest.TestCase):
    def test_joint_only_snapshot_fails_when_a_video_has_no_actor_targets(
        self,
    ) -> None:
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
                patch.object(
                    fusion_model.action_train,
                    "inspect_action_frame_cache",
                    return_value={"frame_count": 100},
                ),
                patch.object(
                    fusion_model.action_train,
                    "cut_kind_of",
                    return_value="sideline",
                ),
                patch.object(
                    fusion_model.action_train.actor_labels,
                    "build",
                    return_value=([], {}),
                ),
            ):
                with self.assertRaises(RuntimeError) as caught:
                    fusion_model.action_train._prepare_action_training_labels(
                        items=[(label, video)],
                        frame_dir=root / "frames",
                        save_dir=root / "run",
                        require_actor_targets=True,
                    )

        self.assertIn("produced no usable actor targets", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
