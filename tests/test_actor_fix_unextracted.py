"""An action added after detection is listed and fixable like any other."""

from __future__ import annotations

import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np

from yp_video.actor.labels import ActorLabel, ActorVerdict
from yp_video.core.cache import StatCache
from yp_video.core.jsonl import read_jsonl, write_jsonl
from yp_video.extraction import actor_fix, pipeline, store
from yp_video.reid import store as reid_store


class UnextractedEventTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.records = self.root / "records.jsonl"
        self.actions = self.root / "actions.jsonl"
        self.matrix = self.root / "v.m.npy"
        # Detection ran on e1; e2 (and a score, which nobody performs) came later.
        write_jsonl(self.records, {"frame_size": [1920, 1080]},
                    [{"id": "e1", "frame": 10, "label": "serve", "resolution": "auto"}])
        write_jsonl(self.actions, {}, [
            {"id": "e1", "frame": 10, "label": "serve"},
            {"id": "e2", "frame": 40, "label": "receive", "visible": False},
            {"id": "e3", "frame": 70, "label": "score"},
        ])
        np.save(self.matrix, np.ones((1, 2), dtype=np.float32))
        stack = ExitStack()
        self.addCleanup(stack.close)
        for target, name, value in (
            (pipeline, "records_path", self.records),
            (store, "records_path", self.records),
            (pipeline, "action_annotation_path", self.actions),
            (store, "action_annotation_path", self.actions),
        ):
            stack.enter_context(patch.object(target, name, return_value=value))
        stack.enter_context(patch.object(store, "load_rallies", return_value=[]))
        stack.enter_context(patch.object(reid_store, "embedding_path", return_value=self.matrix))

    def test_only_events_without_a_record_get_one(self) -> None:
        rows = pipeline.unextracted_records([{"id": "e1"}], "v")
        self.assertEqual([r["id"] for r in rows], ["e2"])
        self.assertEqual(rows[0]["resolution"], "unresolved")

    def test_a_fix_creates_the_record_and_keeps_matrices_aligned(self) -> None:
        record = pipeline._apply_actor_fix(
            "v", "unused.mp4", "e2", ActorLabel(ActorVerdict.OCCLUDED), ["m"]
        )
        self.assertEqual(record["id"], "e2")
        self.assertEqual([r["id"] for r in read_jsonl(self.records)[1]], ["e1", "e2"])
        matrix = np.load(self.matrix)
        self.assertEqual(matrix.shape, (2, 2))
        self.assertTrue(np.isnan(matrix[1]).all())

    def test_a_failed_fix_shrinks_the_matrix_back(self) -> None:
        grown: list[tuple[int, int]] = []

        def fail_save(*_args):
            grown.append(np.load(self.matrix).shape)
            raise OSError("disk full")

        with (
            patch.object(reid_store, "embedded_models", return_value=["m"]),
            patch.object(pipeline, "embedded_models", return_value=["m"]),
            patch.object(pipeline, "mark_actor_embedding_stale"),
            patch.object(actor_fix.actor_labels, "actors_path", return_value=self.root / "actors.json"),
            patch.object(reid_store, "players_path", return_value=self.root / "players.json"),
            patch.object(reid_store, "embedding_refresh_path", return_value=self.root / "refresh.json"),
            patch.object(store, "crop_dir", return_value=self.root / "crops"),
            patch.object(store, "masked_crop_dir", return_value=self.root / "masked"),
            patch.object(actor_fix.actor_labels, "save", side_effect=fail_save),
            self.assertRaises(OSError),
        ):
            actor_fix.apply("v", "unused.mp4", actor_fix.MarkOccluded(mode="occluded", event_id="e2"))
        self.assertEqual(grown, [(2, 2)])
        self.assertEqual(np.load(self.matrix).shape, (1, 2))
        self.assertEqual([r["id"] for r in read_jsonl(self.records)[1]], ["e1"])



class NoRecordsFixTests(unittest.TestCase):
    """A video Player Detection never ran on: the label is the whole fix."""

    def test_the_label_lands_and_nothing_derived_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            records, actions = root / "records.jsonl", root / "actions.jsonl"
            write_jsonl(actions, {}, [{"id": "e1", "frame": 10, "label": "serve"}])
            with (
                patch.object(store, "records_path", return_value=records),
                patch.object(pipeline, "action_annotation_path", return_value=actions),
                patch.object(actor_fix.actor_labels, "actors_path", return_value=root / "actors.json"),
                patch.object(actor_fix.actor_labels._store, "_cache", StatCache()),
                patch.object(pipeline, "apply_actor_fix") as derived,
            ):
                result = actor_fix.apply(
                    "v", "unused.mp4",
                    actor_fix.PickActor(mode="pick", event_id="e1", box=(1, 2, 3, 4)),
                )
                labels = actor_fix.actor_labels.load("v")

            derived.assert_not_called()
            self.assertFalse(records.exists())
            self.assertEqual(labels["e1"], ActorLabel(ActorVerdict.MANUAL, 10, (1, 2, 3, 4)))
            self.assertEqual((result.refreshing_models, result.actor_revision), ((), None))


if __name__ == "__main__":
    unittest.main()
