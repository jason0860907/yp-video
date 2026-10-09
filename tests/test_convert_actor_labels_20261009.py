"""The one-off actor label conversion (scripts/convert_actor_labels_20261009.py).

Deleted together with the script once the real run has landed."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from yp_video.actor import box_style
from yp_video.actor.labels import ActorLabel, ActorVerdict
from yp_video.core.cache import StatCache
from yp_video.core.jsonl import write_jsonl

_spec = importlib.util.spec_from_file_location(
    "convert_actor_labels", Path(__file__).parents[1] / "scripts" / "convert_actor_labels_20261009.py"
)
convert = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(convert)


def write_dense(path: Path, *, frames, counts, boxes, scores) -> None:
    np.savez_compressed(
        path,
        frames=np.asarray(frames, np.int32),
        counts=np.asarray(counts, np.int32),
        boxes=np.asarray(boxes, np.float16).reshape(-1, 4),
        scores=np.asarray(scores, np.float16),
        meta=np.array(json.dumps({"fps": 30.0, "stride": 1, "frame_size": [100, 100]})),
    )


class ConvertActorLabelsTests(unittest.TestCase):
    def setUp(self) -> None:
        raw = tempfile.TemporaryDirectory()
        self.addCleanup(raw.cleanup)
        root = Path(raw.name)
        self.action, self.records, dense = root / "game_actions.jsonl", root / "game.jsonl", root / "game_dense.npz"
        write_jsonl(self.action, {}, [
            {"id": "a", "frame": 10, "label": "attack"},
            {"id": "b", "frame": 11, "label": "attack"},
            {"id": "c", "frame": 10, "label": "set"},
            {"id": "s", "frame": 12, "label": "score"},
        ])
        # Frames 10 and 11 hold the same two people, the right one moving.
        write_dense(dense, frames=[10, 11], counts=[2, 2],
                    boxes=[[.10, .10, .20, .50], [.60, .10, .70, .50],
                           [.12, .10, .22, .50], [.60, .10, .70, .50]],
                    scores=[.9, .9, .9, .9])
        for item in (
            patch.object(convert, "action_annotation_path", return_value=self.action),
            patch.object(convert, "records_path", return_value=self.records),
            patch.object(box_style, "dense_path", return_value=dense),
            patch.object(box_style, "_dense_cache", StatCache()),
        ):
            item.start()
            self.addCleanup(item.stop)

    def _convert(self, actors: dict, records: list[dict]):
        write_jsonl(self.records, {"frame_size": [100, 100]}, records)
        return convert.convert_video("game", {"version": 2, "actors": actors})

    def test_each_label_becomes_its_record_pick_on_the_event_frame(self) -> None:
        payload, counts, changes = self._convert(
            {
                # The record cut a Medium box on the event frame: snapped.
                "a": {"verdict": "manual", "track": "1:2", "box": [60, 10, 70, 50]},
                # Cut on frame 10, the event since moved to 11: followed.
                "b": {"verdict": "confirmed_auto", "box": [10, 10, 20, 50], "frame": 10},
                "c": {"verdict": "occluded"},
                "s": {"verdict": "manual", "box": [1, 1, 5, 5], "frame": 12},
                "gone": {"verdict": "occluded"},
            },
            [
                {"id": "a", "frame": 10, "resolution": "manual", "actor_box": [11, 11, 21, 52]},
                {"id": "b", "frame": 10, "resolution": "auto", "actor_box": [10, 10, 20, 50]},
            ],
        )
        self.assertEqual(payload["version"], 3)
        self.assertEqual(payload["actors"], {
            # The record's pick, not the old label's tracklet anchor.
            "a": {"verdict": "manual", "frame": 10, "box": [10.0, 10.0, 20.0, 50.0]},
            "b": {"verdict": "confirmed_auto", "frame": 11, "box": [12.0, 10.0, 22.0, 50.0]},
            "c": {"verdict": "occluded"},
        })
        self.assertEqual(counts, {
            "manual:record:event_frame:snapped": 1,
            "confirmed_auto:record:followed:snapped": 1,
            "occluded": 1,
            "dropped_skip_label": 1,
            "dropped_orphan": 1,
        })
        self.assertEqual(changes, [])

    def test_without_a_usable_record_the_old_box_is_kept_where_it_cannot_follow(self) -> None:
        payload, counts, changes = self._convert(
            # Nobody near this box on frame 10: it stays on its own frame.
            {"b": {"verdict": "manual", "track": "1:2", "box": [30, 10, 40, 50], "frame": 10}},
            [{"id": "b", "frame": 11, "resolution": "auto", "actor_box": [60, 10, 70, 50]}],
        )
        label = ActorLabel.from_payload(payload["actors"]["b"])
        self.assertEqual(label, ActorLabel(ActorVerdict.MANUAL, 10, (30.0, 10.0, 40.0, 50.0)))
        self.assertEqual(counts, {"manual:label:kept_own_frame:unmatched": 1})
        self.assertEqual(changes, [])

    def test_the_person_check_is_mutual_centre_containment(self) -> None:
        self.assertTrue(convert._inside([5, 5, 6, 6], [0, 0, 10, 10]))
        self.assertFalse(convert._inside([50, 5, 60, 6], [0, 0, 10, 10]))

    def test_only_version_2_files_convert(self) -> None:
        with self.assertRaises(ValueError):
            convert.convert_video("game", {"version": 3, "actors": {}})


if __name__ == "__main__":
    unittest.main()
