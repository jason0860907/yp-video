from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import TypeAdapter, ValidationError

from yp_video.reid import actor_fixes
from yp_video.reid.resolution import ActorResolution, actor_resolution
from yp_video.web.jobs import MAX_LOG_LINES, Job, JobManager, JobStatus
from yp_video.web.routers.action_train import (
    ActionTrainRequest,
    AnnotationActionTrainRequest,
    VnlActionTrainRequest,
)
from yp_video.web.routers.reid import (
    ActorFixRequest,
    AutoActorRequest,
    OccludedActorRequest,
    PickActorRequest,
)


class DiscriminatedRequestTests(unittest.TestCase):
    def test_action_train_source_owns_its_fields(self) -> None:
        adapter = TypeAdapter(ActionTrainRequest)

        vnl = adapter.validate_python({"source": "vnl_1_5"})
        self.assertIsInstance(vnl, VnlActionTrainRequest)
        annotation = adapter.validate_python(
            {
                "source": "action_annotations",
                "training_mode": "holdout",
                "holdout_videos": ["match.mp4"],
            }
        )
        self.assertIsInstance(annotation, AnnotationActionTrainRequest)

        with self.assertRaises(ValidationError):
            adapter.validate_python(
                {"source": "vnl_1_5", "training_mode": "holdout"}
            )
        with self.assertRaises(ValidationError):
            adapter.validate_python({"source": "vnl_1_5", "resume": True})
        with self.assertRaises(ValidationError):
            adapter.validate_python(
                {
                    "source": "vnl_1_5",
                    "resume": True,
                    "save_dir": "run",
                    "init_checkpoint": "weights",
                }
            )

    def test_actor_fix_mode_owns_its_fields(self) -> None:
        adapter = TypeAdapter(ActorFixRequest)

        pick = adapter.validate_python(
            {"mode": "pick", "event_id": "e1", "box": [1, 2, 3, 4]}
        )
        self.assertIsInstance(pick, PickActorRequest)
        self.assertIsInstance(
            adapter.validate_python({"mode": "occluded", "event_id": "e1"}),
            OccludedActorRequest,
        )
        self.assertIsInstance(
            adapter.validate_python({"mode": "auto", "event_id": "e1"}),
            AutoActorRequest,
        )

        with self.assertRaises(ValidationError):
            adapter.validate_python(
                {
                    "mode": "occluded",
                    "event_id": "e1",
                    "box": [1, 2, 3, 4],
                }
            )
        with self.assertRaises(ValidationError):
            adapter.validate_python({"mode": "pick", "event_id": "e1"})
        with self.assertRaises(ValidationError):
            adapter.validate_python(
                {
                    "mode": "pick",
                    "event_id": "e1",
                    "box": [1, 2, 3, 4],
                    "frame": -1,
                }
            )


class ActorResolutionTests(unittest.TestCase):
    def test_legacy_records_normalize_to_explicit_domain_state(self) -> None:
        self.assertEqual(
            actor_resolution({"crop": "auto.jpg"}), ActorResolution.AUTO
        )
        self.assertEqual(
            actor_resolution({"box_source": "manual", "crop": "fix.jpg"}),
            ActorResolution.MANUAL,
        )
        self.assertEqual(
            actor_resolution({"box_source": "manual", "crop": None}),
            ActorResolution.OCCLUDED,
        )
        self.assertEqual(actor_resolution({}), ActorResolution.UNRESOLVED)

    def test_explicit_state_wins_over_legacy_shape(self) -> None:
        self.assertEqual(
            actor_resolution(
                {
                    "resolution": "occluded",
                    "box_source": "manual",
                    "crop": "stale.jpg",
                }
            ),
            ActorResolution.OCCLUDED,
        )


class JobPayloadTests(unittest.TestCase):
    def test_summary_and_sse_do_not_contain_log_bodies(self) -> None:
        manager = JobManager()
        job = manager.create_job("test")
        job.logs.extend(f"line {index}" for index in range(MAX_LOG_LINES + 2))

        summary = job.to_dict()
        self.assertNotIn("logs", summary)
        self.assertEqual(summary["log_count"], MAX_LOG_LINES)
        self.assertEqual(len(manager.job_logs(job.id) or []), MAX_LOG_LINES)

        event = manager.subscribe(job.id).get_nowait()  # type: ignore[union-attr]
        self.assertNotIn("logs", event)
        self.assertEqual(event["log_count"], MAX_LOG_LINES)

    def test_pruning_keeps_running_jobs(self) -> None:
        manager = JobManager()
        running = Job(id="running", type="test")
        manager.jobs[running.id] = running
        for index in range(205):
            job = Job(id=f"done-{index}", type="test")
            job.status = JobStatus.COMPLETED
            job.created_at = float(index)
            manager.jobs[job.id] = job

        manager._prune_terminal_jobs()

        self.assertIn(running.id, manager.jobs)
        self.assertEqual(len(manager.jobs), 200)


class ActorFixTransactionTests(unittest.TestCase):
    def test_derived_and_annotation_files_roll_back_together(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            reid_file = root / "match_reid.jsonl"
            players_file = root / "match_players.json"
            embedding_file = root / "match.model.npy"
            crop_dir = root / "crops"
            masked_dir = root / "masked"
            for path, content in (
                (reid_file, b"reid-before"),
                (players_file, b"players-before"),
                (embedding_file, b"embedding-before"),
            ):
                path.write_bytes(content)

            def mutate_derived(*_args, **_kwargs):
                reid_file.write_bytes(b"reid-after")
                embedding_file.write_bytes(b"embedding-after")
                crop_dir.mkdir()
                (crop_dir / "new.jpg").write_bytes(b"new crop")
                return {"id": "event-1"}

            def fail_annotation(*_args, **_kwargs):
                players_file.write_bytes(b"players-after")
                raise RuntimeError("annotation write failed")

            with (
                patch.object(actor_fixes.store, "reid_path", return_value=reid_file),
                patch.object(
                    actor_fixes.store, "players_path", return_value=players_file
                ),
                patch.object(
                    actor_fixes.store, "embedded_models", return_value=["model"]
                ),
                patch.object(
                    actor_fixes.store,
                    "embedding_path",
                    return_value=embedding_file,
                ),
                patch.object(actor_fixes.store, "crop_dir", return_value=crop_dir),
                patch.object(
                    actor_fixes.store, "masked_crop_dir", return_value=masked_dir
                ),
                patch.object(
                    actor_fixes.pipeline,
                    "apply_actor_fix",
                    side_effect=mutate_derived,
                ),
                patch.object(
                    actor_fixes.identity,
                    "apply_actor_fix_annotation",
                    side_effect=fail_annotation,
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "annotation write failed"
                ):
                    actor_fixes.apply(
                        root / "match.mp4",
                        actor_fixes.PickActor(
                            mode="pick",
                            event_id="event-1",
                            box=(1, 2, 3, 4),
                        ),
                    )

            self.assertEqual(reid_file.read_bytes(), b"reid-before")
            self.assertEqual(players_file.read_bytes(), b"players-before")
            self.assertEqual(embedding_file.read_bytes(), b"embedding-before")
            self.assertFalse((crop_dir / "new.jpg").exists())


if __name__ == "__main__":
    unittest.main()
