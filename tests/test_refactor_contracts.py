from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pydantic import TypeAdapter, ValidationError

from yp_video.actor.labels import ActorLabel, ActorVerdict
from yp_video.actor.resolution import ActorResolution, actor_resolution
from yp_video.contracts.reid import (
    CHECKPOINT_MANIFEST_NAME,
    CHECKPOINT_TYPE,
    REID_CONTRACT_VERSION,
)
from yp_video.extraction import actor_fix, pipeline
from yp_video.reid import checkpoints, store
from yp_video.web.jobs import MAX_LOG_LINES, Job, JobManager, JobStatus
from yp_video.web.routers.action_train import (
    ActionTrainRequest,
    AnnotationActionTrainRequest,
    VnlActionTrainRequest,
)
from yp_video.web.routers.actor_association import (
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
            {
                "mode": "pick",
                "event_id": "e1",
                "box": [1, 2, 3, 4],
            }
        )
        self.assertIsInstance(pick, PickActorRequest)
        self.assertIsInstance(
            adapter.validate_python(
                {
                    "mode": "occluded",
                    "event_id": "e1",
                    }
            ),
            OccludedActorRequest,
        )
        self.assertIsInstance(
            adapter.validate_python(
                {
                    "mode": "auto",
                    "event_id": "e1",
                    }
            ),
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
            adapter.validate_python(
                {
                    "mode": "pick",
                    "event_id": "e1",
                    }
            )
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
    def test_state_is_read_never_inferred(self) -> None:
        self.assertEqual(
            actor_resolution({"resolution": "occluded", "crop": "stale.jpg"}),
            ActorResolution.OCCLUDED,
        )
        self.assertEqual(
            actor_resolution({"resolution": "auto", "crop": None}),
            ActorResolution.AUTO,
        )

    def test_a_record_without_state_fails_loudly(self) -> None:
        """A crop is not evidence of how the actor was chosen — guessing here
        is what let 'manual' and 'occluded' drift apart before."""
        for record in ({"id": "e1", "crop": "auto.jpg"}, {"id": "e1"}):
            with self.assertRaisesRegex(ValueError, "re-run extraction"):
                actor_resolution(record)


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
    def test_active_weight_family_is_synchronous_and_others_are_deferred(
        self,
    ) -> None:
        models = [
            "clip-reid",
            "clip-reid-masked",
            "clip-reident",
            "clip-reident-masked",
        ]
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            reid_file = root / "match_reid.jsonl"
            reid_file.write_bytes(b"reid")
            model_files = {
                model: root / f"match.{model}.npy" for model in models
            }
            for path in model_files.values():
                path.write_bytes(b"embedding")

            with (
                patch.object(
                    actor_fix.extraction_store,
                    "records_path",
                    return_value=reid_file,
                ),
                patch.object(
                    actor_fix.actor_labels,
                    "actors_path",
                    return_value=root / "actors.json",
                ),
                patch.object(
                    actor_fix.store,
                    "players_path",
                    return_value=root / "players.json",
                ),
                patch.object(
                    actor_fix.store,
                    "embedding_refresh_path",
                    return_value=root / "embedding-refresh.json",
                ),
                patch.object(
                    actor_fix.store,
                    "embedded_models",
                    return_value=models,
                ),
                patch.object(
                    actor_fix.store,
                    "embedding_path",
                    side_effect=lambda _stem, model: model_files[model],
                ),
                patch.object(
                    actor_fix.extraction_store,
                    "crop_dir",
                    return_value=root / "crops",
                ),
                patch.object(
                    actor_fix.extraction_store,
                    "masked_crop_dir",
                    return_value=root / "masked",
                ),
                patch.object(
                    actor_fix.pipeline,
                    "apply_actor_fix",
                    return_value={"id": "event-1", "actor_revision": 7},
                ) as apply_actor_fix,
                patch.object(actor_fix.actor_labels, "save"),
                patch.object(actor_fix.identity, "drop_assignment"),
            ):
                result = actor_fix.apply(
                    root / "match.mp4",
                    actor_fix.MarkOccluded(
                        mode="occluded", event_id="event-1"
                    ),
                    active_model="clip-reident-masked",
                )

        self.assertEqual(
            apply_actor_fix.call_args.kwargs["models"],
            ["clip-reident", "clip-reident-masked"],
        )
        self.assertEqual(
            result.refreshing_models,
            ("clip-reid", "clip-reid-masked"),
        )
        self.assertEqual(result.actor_revision, 7)

    def test_request_mode_resolves_to_one_command_without_branching(
        self,
    ) -> None:
        """Transport → command → label, with no re-branching on the way."""
        adapter = TypeAdapter(ActorFixRequest)
        cases = {
            ("pick", ActorVerdict.MANUAL): {
                "mode": "pick",
                "event_id": "e1",
                "box": [1, 2, 3, 4],
                "frame": 9,
                "snap": False,
            },
            ("occluded", ActorVerdict.OCCLUDED): {
                "mode": "occluded",
                "event_id": "e1",
            },
            ("auto", None): {"mode": "auto", "event_id": "e1"},
        }
        for (mode, verdict), payload in cases.items():
            command = adapter.validate_python(payload).command
            self.assertEqual(command.mode, mode)
            self.assertEqual(command.event_id, "e1")
            self.assertEqual(
                command.label.verdict if command.label else None, verdict
            )

    def test_each_command_carries_the_label_it_stands_for(self) -> None:
        """One uniform write per fix — the mode never re-branches downstream."""
        self.assertEqual(
            actor_fix.PickActor(
                mode="pick",
                event_id="e1",
                box=(1, 2, 3, 4),
                frame=9,
                snap=False,
            ).label,
            ActorLabel(
                ActorVerdict.MANUAL, box=(1, 2, 3, 4), frame=9, snap=False
            ),
        )
        self.assertEqual(
            actor_fix.MarkOccluded(mode="occluded", event_id="e1").label,
            ActorLabel(ActorVerdict.OCCLUDED),
        )
        self.assertIsNone(
            actor_fix.RevertActor(mode="auto", event_id="e1").label
        )

    def test_derived_and_annotation_files_roll_back_together(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            record_file = root / "match.jsonl"
            actors_file = root / "match_actors.json"
            players_file = root / "match_players.json"
            embedding_file = root / "match.model.npy"
            crop_dir = root / "crops"
            masked_dir = root / "masked"
            for path, content in (
                (record_file, b"records-before"),
                (actors_file, b"actors-before"),
                (players_file, b"players-before"),
                (embedding_file, b"embedding-before"),
            ):
                path.write_bytes(content)

            def mutate_derived(*_args, **_kwargs):
                record_file.write_bytes(b"records-after")
                embedding_file.write_bytes(b"embedding-after")
                actor_fix.store.mark_actor_embedding_stale(
                    "match", ["model"], "event-1"
                )
                crop_dir.mkdir()
                (crop_dir / "new.jpg").write_bytes(b"new crop")
                return {"id": "event-1"}

            def fail_label(*_args, **_kwargs):
                actors_file.write_bytes(b"actors-after")
                raise RuntimeError("label write failed")

            with (
                patch.object(
                    actor_fix.extraction_store,
                    "records_path",
                    return_value=record_file,
                ),
                patch.object(
                    actor_fix.actor_labels,
                    "actors_path",
                    return_value=actors_file,
                ),
                patch.object(
                    actor_fix.store, "players_path", return_value=players_file
                ),
                patch.object(
                    actor_fix.store,
                    "embedding_refresh_path",
                    return_value=root / "embedding-refresh.json",
                ),
                patch.object(
                    actor_fix.store, "embedded_models", return_value=["model"]
                ),
                patch.object(
                    actor_fix.store,
                    "embedding_path",
                    return_value=embedding_file,
                ),
                patch.object(
                    actor_fix.extraction_store, "crop_dir", return_value=crop_dir
                ),
                patch.object(
                    actor_fix.extraction_store,
                    "masked_crop_dir",
                    return_value=masked_dir,
                ),
                patch.object(
                    actor_fix.pipeline,
                    "apply_actor_fix",
                    side_effect=mutate_derived,
                ),
                patch.object(
                    actor_fix.actor_labels, "save", side_effect=fail_label
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "label write failed"):
                    actor_fix.apply(
                        root / "match.mp4",
                        actor_fix.PickActor(
                            mode="pick",
                            event_id="event-1",
                            box=(1, 2, 3, 4),
                        ),
                        active_model="model",
                    )

            self.assertEqual(record_file.read_bytes(), b"records-before")
            self.assertEqual(actors_file.read_bytes(), b"actors-before")
            self.assertEqual(players_file.read_bytes(), b"players-before")
            self.assertEqual(embedding_file.read_bytes(), b"embedding-before")
            self.assertFalse((root / "embedding-refresh.json").exists())
            self.assertFalse((crop_dir / "new.jpg").exists())


class ActorEmbeddingRefreshTests(unittest.TestCase):
    def test_variants_with_shared_weights_use_one_batch(self) -> None:
        class FakeEmbedder:
            def __init__(self) -> None:
                self.calls: list[list[Path]] = []

            def embed_paths(self, paths):
                self.calls.append(list(paths))
                return np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        embedder = FakeEmbedder()
        saved: dict[str, np.ndarray] = {}
        record = {"id": "event-1", "crop": "event-1.jpg"}
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            with (
                patch.object(
                    pipeline,
                    "embedded_models",
                    return_value=["clip-reident", "clip-reident-masked"],
                ),
                patch.object(
                    pipeline,
                    "_record_revision_is_current",
                    return_value=True,
                ),
                patch.object(
                    pipeline,
                    "build_embedders",
                    return_value={"clip-reident": embedder},
                ),
                patch.object(pipeline, "crop_dir", return_value=root / "crops"),
                patch(
                    "yp_video.extraction.store.masked_crop_dir",
                    return_value=root / "masked",
                ),
                patch.object(pipeline, "_masked_record_crop"),
                patch.object(
                    pipeline,
                    "load_embedding_matrix",
                    side_effect=lambda _stem, _model: np.zeros(
                        (2, 2), dtype=np.float32
                    ),
                ),
                patch.object(
                    pipeline,
                    "save_embedding_matrix",
                    side_effect=lambda _stem, model, matrix: saved.__setitem__(
                        model, matrix.copy()
                    ),
                ),
                patch.object(pipeline, "mark_actor_embedding_refreshed"),
            ):
                updated = pipeline._patch_embedding_row(
                    "match",
                    record,
                    0,
                    object(),
                    models=["clip-reident", "clip-reident-masked"],
                    expected_revision=1,
                )

        self.assertEqual(len(embedder.calls), 1)
        self.assertEqual(len(embedder.calls[0]), 2)
        self.assertEqual(
            updated, ["clip-reident", "clip-reident-masked"]
        )
        np.testing.assert_array_equal(
            saved["clip-reident"][0], np.array([1.0, 2.0])
        )
        np.testing.assert_array_equal(
            saved["clip-reident-masked"][0], np.array([3.0, 4.0])
        )

    def test_matrix_mtime_is_the_durable_stale_marker(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            reid_file = root / "match_reid.jsonl"
            fresh = root / "match.fresh.npy"
            stale = root / "match.stale.npy"
            for path in (reid_file, fresh, stale):
                path.write_bytes(b"x")
            os.utime(stale, ns=(1, 1))
            os.utime(reid_file, ns=(2, 2))
            os.utime(fresh, ns=(3, 3))

            paths = {"fresh": fresh, "stale": stale}
            with (
                patch.object(store, "records_path", return_value=reid_file),
                patch.object(
                    store,
                    "embedded_models",
                    return_value=["fresh", "stale"],
                ),
                patch.object(
                    store,
                    "embedding_path",
                    side_effect=lambda _stem, model: paths[model],
                ),
            ):
                self.assertEqual(
                    store.stale_embedding_models("match"), ["stale"]
                )
                self.assertTrue(store.embedding_is_fresh("match", "fresh"))
                self.assertFalse(store.embedding_is_fresh("match", "stale"))

    def test_pending_events_keep_a_new_matrix_stale_until_all_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            reid_file = root / "match_reid.jsonl"
            matrix = root / "match.model.npy"
            refreshes = root / "match_embedding-refresh.json"
            reid_file.write_bytes(b"reid")
            matrix.write_bytes(b"matrix")
            os.utime(reid_file, ns=(1, 1))
            os.utime(matrix, ns=(2, 2))

            with (
                patch.object(store, "records_path", return_value=reid_file),
                patch.object(
                    store, "embedding_path", return_value=matrix
                ),
                patch.object(
                    store,
                    "embedding_refresh_path",
                    return_value=refreshes,
                ),
                patch.object(
                    store, "embedded_models", return_value=["model"]
                ),
            ):
                store.mark_actor_embedding_stale(
                    "match", ["model"], "event-1"
                )
                store.mark_actor_embedding_stale(
                    "match", ["model"], "event-2"
                )
                store.mark_actor_embedding_refreshed(
                    "match", "model", "event-1"
                )
                self.assertFalse(
                    store.embedding_is_fresh("match", "model")
                )

                store.mark_actor_embedding_refreshed(
                    "match", "model", "event-2"
                )
                self.assertTrue(store.embedding_is_fresh("match", "model"))
                self.assertFalse(refreshes.exists())


class ReidCheckpointPolicyTests(unittest.TestCase):
    @staticmethod
    def _write_package(
        root: Path,
        name: str,
        *,
        best_metric: str | None,
        best_value: float | None,
    ) -> Path:
        package = root / name
        package.mkdir()
        (package / "checkpoint.pt").write_bytes(b"weights")
        manifest = {
            "type": CHECKPOINT_TYPE,
            "contract_version": REID_CONTRACT_VERSION,
            "run_name": name,
            "checkpoint": "checkpoint.pt",
            "best": (
                {"metric": best_metric, "value": best_value}
                if best_metric is not None
                else None
            ),
        }
        (package / CHECKPOINT_MANIFEST_NAME).write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return package

    def test_official_package_stays_active_above_new_training_runs(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            official = self._write_package(
                root,
                checkpoints.DEFAULT_CHECKPOINT_PACKAGE,
                best_metric=None,
                best_value=None,
            )
            self._write_package(
                root,
                "new-candidate",
                best_metric="train_loss",
                best_value=999.0,
            )

            with patch.object(
                checkpoints, "REID_CHECKPOINTS_DIR", root
            ):
                rows = checkpoints.list_checkpoints()
                default = checkpoints.default_checkpoint()

        self.assertEqual(default, official.resolve())
        self.assertEqual(rows[0]["run_name"], "clip-reident-paper")
        self.assertTrue(rows[0]["active"])
        self.assertFalse(rows[1]["active"])

    def test_candidate_is_not_an_implicit_fallback_without_official(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            self._write_package(
                root,
                "candidate-only",
                best_metric="m_ap",
                best_value=1.0,
            )

            with patch.object(
                checkpoints, "REID_CHECKPOINTS_DIR", root
            ):
                self.assertIsNone(checkpoints.default_checkpoint())
                self.assertFalse(checkpoints.list_checkpoints()[0]["active"])


if __name__ == "__main__":
    unittest.main()
