from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pydantic import TypeAdapter, ValidationError

from yp_video.actor import labels as actor_labels
from yp_video.actor.labels import ActorLabel, ActorVerdict
from yp_video.actor.resolution import ActorResolution, actor_resolution
from yp_video.contracts.reid import (
    CHECKPOINT_MANIFEST_NAME,
    CHECKPOINT_TYPE,
    REID_CONTRACT_VERSION,
)
from yp_video.extraction import actor_fix, cropping, pipeline
from yp_video.reid import checkpoints, store
from yp_video.web.jobs import MAX_LOG_LINES, Job, JobManager, JobStatus, JobType
from yp_video.web.routers import extraction as extraction_router
from yp_video.web.routers.actor_association import (
    ActorFixRequest,
    AutoActorRequest,
    OccludedActorRequest,
    PickActorRequest,
)


class DiscriminatedRequestTests(unittest.TestCase):
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
        job = manager.create_job(JobType.DOWNLOAD)
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
                patch.object(actor_fix.store, "drop_assignment"),
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


class CroppingTests(unittest.TestCase):
    """The rules the four crop callers used to each keep a copy of.

    Extraction's automatic pick, a replayed label, the fix endpoint and
    reassociation all cut pixels the same way; the copies had drifted on
    exactly the two questions below.
    """

    DETECTION = {"box": [100, 100, 140, 200], "score": 1.6}

    def _record(self, **extra):
        return {"id": "e1", "frame": 500, "detections": [self.DETECTION], **extra}

    def _cut(self, record, target, contact):
        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as raw_dir:
            return cropping.cut(
                record,
                frame,
                cropping.person_for(record, target),
                source_frame=target.frame,
                contact=contact,
                frame_size=(400, 400),
                out_dir=Path(raw_dir),
            )

    def test_a_same_frame_crop_is_anchored_on_the_contact_point(self) -> None:
        record = self._record()
        target = cropping.CropTarget((100, 100, 140, 200), 500, snap=True)
        self.assertIsNotNone(self._cut(record, target, (300.0, 150.0)))
        # The display box unions the ball, so it reaches out to it.
        self.assertGreaterEqual(record["box"][2], 300)
        self.assertEqual(record["crop_schema"], cropping.CROP_SCHEMA_VERSION)
        self.assertNotIn("crop_frame", record)

    def test_a_cross_frame_crop_ignores_the_contact_point(self) -> None:
        """The point belongs to the event frame, where the player is not —
        unioning it there would drag the crop across the court."""
        record = self._record()
        target = cropping.CropTarget((100, 100, 140, 200), 812, snap=True)
        self.assertIsNotNone(self._cut(record, target, (300.0, 150.0)))
        self.assertLess(record["box"][2], 300)
        self.assertEqual(record["crop_frame"], 812)

    def test_crop_frame_is_cleared_when_the_pick_comes_home(self) -> None:
        """A stale crop_frame sends the tracklet link and the next re-crop to
        a frame the actor is no longer cropped from."""
        record = self._record(crop_frame=812)
        target = cropping.CropTarget((100, 100, 140, 200), 500, snap=True)
        self.assertIsNotNone(self._cut(record, target, (110.0, 150.0)))
        self.assertNotIn("crop_frame", record)

    def test_snapping_is_vetoed_across_frames(self) -> None:
        """Stored detections belong to the event frame; on another one the
        nearest is somebody else standing there."""
        record = self._record()
        same = cropping.person_for(
            record, cropping.CropTarget((102, 102, 138, 198), 500, snap=True)
        )
        across = cropping.person_for(
            record, cropping.CropTarget((102, 102, 138, 198), 812, snap=True)
        )
        self.assertEqual(list(same.xyxy), self.DETECTION["box"])
        self.assertEqual(same.score, self.DETECTION["score"])
        self.assertEqual(across.xyxy, (102, 102, 138, 198))
        self.assertEqual(across.score, 0.0)

    def test_a_vetoed_snap_embeds_the_box_as_drawn(self) -> None:
        """snap=False means no stored detection IS this player, so anything
        close enough to snap to would be the occluder in front of them."""
        record = self._record()
        person = cropping.person_for(
            record, cropping.CropTarget((102, 102, 138, 198), 500, snap=False)
        )
        self.assertEqual(person.xyxy, (102, 102, 138, 198))


class StagesStopWhereTheyShouldTests(unittest.TestCase):
    """Detection, association and embedding are three jobs, in that order.

    Detection is perception and decides nothing; association picks and crops;
    an embedding answers "who is this person" about a crop, so it can only be
    asked once somebody has agreed the right person was cropped. Folding any
    pair together is what made the first association pass a different code
    path from every later one, and made every actor fix re-embed a crop it was
    about to replace.
    """

    def test_detection_neither_associates_nor_embeds(self) -> None:
        """The signature is the contract: it takes no policy and no weights."""
        import inspect

        params = inspect.signature(pipeline.detect_video).parameters
        self.assertEqual(
            sorted(params), ["on_progress", "video_path"]
        )
        source = inspect.getsource(pipeline.detect_video)
        for forbidden in ("cut(", "ActorAssociationService", "embed_video"):
            self.assertNotIn(forbidden, source, f"detection must not {forbidden}")

    def test_retired_detector_output_is_queued_for_migration(self) -> None:
        """Old records are not silently treated as current segmentation data."""
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            old = root / "old.jsonl"
            current = root / "current.jsonl"
            old.write_text(
                json.dumps({"source": {"detector": "retired-detector"}}) + "\n"
            )
            current.write_text(
                json.dumps(
                    {"source": {"detector": extraction_router.DETECTOR_NAME}}
                )
                + "\n"
            )
            self.assertFalse(extraction_router._has_current_detections(old))
            self.assertTrue(extraction_router._has_current_detections(current))

    def test_a_re_detect_keeps_the_association_already_made(self) -> None:
        """Refreshing the candidate list is not an opinion about the answer —
        and one of those answers may be a human verdict."""
        self.assertLessEqual(
            {"status", "resolution", "crop", "actor_box", "track", "actor_revision"},
            pipeline._ASSOCIATION_FIELDS,
        )

    def test_a_fix_before_any_embedding_is_allowed(self) -> None:
        """Actor review runs BEFORE embedding, so the fix endpoint must not
        require the stage that depends on it."""
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            (root / "match_reid.jsonl").write_bytes(b"reid")

            with (
                patch.object(
                    actor_fix.extraction_store,
                    "records_path",
                    return_value=root / "match_reid.jsonl",
                ),
                patch.object(
                    actor_fix.actor_labels,
                    "actors_path",
                    return_value=root / "actors.json",
                ),
                patch.object(
                    actor_fix.store, "players_path", return_value=root / "players.json"
                ),
                patch.object(
                    actor_fix.store,
                    "embedding_refresh_path",
                    return_value=root / "refresh.json",
                ),
                # Nothing embedded yet — the ordinary state during review.
                patch.object(actor_fix.store, "embedded_models", return_value=[]),
                patch.object(
                    actor_fix.extraction_store, "crop_dir", return_value=root / "crops"
                ),
                patch.object(
                    actor_fix.extraction_store,
                    "masked_crop_dir",
                    return_value=root / "masked",
                ),
                patch.object(
                    actor_fix.pipeline,
                    "apply_actor_fix",
                    return_value={"id": "e1", "actor_revision": 1},
                ) as applied,
                patch.object(actor_fix.actor_labels, "save"),
                patch.object(actor_fix.store, "drop_assignment"),
            ):
                result = actor_fix.apply(
                    root / "match.mp4",
                    actor_fix.MarkOccluded(mode="occluded", event_id="e1"),
                    active_model=None,
                )

        self.assertEqual(applied.call_args.kwargs["models"], [])
        self.assertEqual(result.refreshing_models, ())

    def test_a_named_model_must_still_actually_exist(self) -> None:
        """None means "nothing is embedded"; a NAME that is not there is a
        caller bug, and silently embedding nothing would hide it."""
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            (root / "match_reid.jsonl").write_bytes(b"reid")
            with (
                patch.object(
                    actor_fix.extraction_store,
                    "records_path",
                    return_value=root / "match_reid.jsonl",
                ),
                patch.object(
                    actor_fix.store, "embedded_models", return_value=["clip-reid"]
                ),
                self.assertRaises(FileNotFoundError),
            ):
                actor_fix.apply(
                    root / "match.mp4",
                    actor_fix.MarkOccluded(mode="occluded", event_id="e1"),
                    active_model="clip-reident",
                )


class ConfirmableAnswerTests(unittest.TestCase):
    """What a human is allowed to endorse, and what endorsing it records."""

    def test_a_pick_becomes_confirmed_auto(self) -> None:
        out = actor_labels.confirmations_for([
            {"id": "e1", "frame": 10, "resolution": "auto", "actor_box": [1, 2, 3, 4]},
        ])
        self.assertEqual(out["e1"].verdict, ActorVerdict.CONFIRMED_AUTO)
        self.assertEqual(out["e1"].box, (1.0, 2.0, 3.0, 4.0))

    def test_an_explicit_occlusion_becomes_the_occluded_verdict(self) -> None:
        """The model said nobody is visible; agreeing with that IS a verdict,
        and it is the training truth the NONE head is scored on."""
        out = actor_labels.confirmations_for([
            {
                "id": "e1", "frame": 10, "resolution": "unresolved",
                "association": {"decision": "abstained", "kind": "occluded"},
            },
        ])
        self.assertEqual(out["e1"].verdict, ActorVerdict.OCCLUDED)
        self.assertIsNone(out["e1"].box)

    def test_untracked_is_not_endorsable(self) -> None:
        """It says somebody DID act and tracking lost them — re-running
        tracking may fix it, and a verdict would bury it."""
        self.assertEqual(
            actor_labels.confirmations_for([
                {
                    "id": "e1", "frame": 10, "resolution": "unresolved",
                    "association": {"decision": "abstained", "kind": "untracked"},
                },
            ]),
            {},
        )

    def test_a_bare_abstention_is_not_endorsable(self) -> None:
        """No `kind` at all — the geometry simply found nobody, which is not
        the same claim as "nobody is visible"."""
        self.assertEqual(
            actor_labels.confirmations_for([
                {"id": "e1", "frame": 10, "resolution": "unresolved"},
            ]),
            {},
        )

    def test_a_human_verdict_is_never_re_endorsed(self) -> None:
        for resolution in ("manual", "occluded"):
            self.assertEqual(
                actor_labels.confirmations_for([
                    {
                        "id": "e1", "frame": 10, "resolution": resolution,
                        "actor_box": [1, 2, 3, 4],
                        "association": {"kind": "occluded"},
                    },
                ]),
                {},
                resolution,
            )


class MaskedCropReuseTests(unittest.TestCase):
    """Masking is a segmentation pass over every crop in the video. Paying it
    again to reproduce files that are already on disk is what made registering
    a second masked embedder cost a full re-mask."""

    RECORDS = [{"id": "e1", "crop": "e1.jpg"}]

    def _run(self, root: Path):
        crops, masked = root / "crops", root / "masked"
        with (
            patch("yp_video.person.seg.crop_masker"),
            patch("cv2.imread", return_value=object()),
            patch("yp_video.extraction.store.masked_crop_dir", return_value=masked),
            patch.object(pipeline, "_masked_record_crop") as mask_one,
        ):
            out = pipeline._mask_crops(
                "match", [crops / "e1.jpg"], [0], self.RECORDS, None
            )
        return out, mask_one.call_count

    def test_an_up_to_date_masked_crop_is_not_recut(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            (root / "crops").mkdir()
            (root / "masked").mkdir()
            (root / "crops" / "e1.jpg").write_bytes(b"source")
            (root / "masked" / "e1.jpg").write_bytes(b"masked")

            out, calls = self._run(root)

        self.assertEqual(calls, 0)
        self.assertEqual(out, [root / "masked" / "e1.jpg"])

    def test_a_masked_crop_older_than_its_source_is_recut(self) -> None:
        """An automatic pick keeps its crop FILENAME across a re-extraction,
        so existence alone would serve a mask of the previous pick's pixels."""
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            (root / "crops").mkdir()
            (root / "masked").mkdir()
            stale = root / "masked" / "e1.jpg"
            stale.write_bytes(b"masked")
            os.utime(stale, ns=(0, 0))
            (root / "crops" / "e1.jpg").write_bytes(b"re-extracted")

            _out, calls = self._run(root)

        self.assertEqual(calls, 1)

    def test_a_missing_masked_crop_is_cut(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            (root / "crops").mkdir()
            (root / "crops" / "e1.jpg").write_bytes(b"source")

            out, calls = self._run(root)

        self.assertEqual(calls, 1)
        # Aligned with the input either way — the embedder turns a path that
        # does not exist into a NaN row.
        self.assertEqual(out, [root / "masked" / "e1.jpg"])


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
