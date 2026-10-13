"""The actor head's per-event training rows Fusion Train snapshots."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from yp_video.actor import labels as actor_labels
from yp_video.actor import training_labels
from yp_video.core.cache import StatCache
from yp_video.core.jsonl import write_jsonl


def _write_dense(path: Path) -> None:
    """Frames 10-30 covered: an actor-like box, a weak far box, and on frame
    20 two people competing for the same spot."""
    np.savez_compressed(
        path,
        frames=np.asarray([10, 11, 20, 30], np.int32),
        counts=np.asarray([3, 2, 2, 2], np.int32),
        boxes=np.asarray([[.10, .10, .20, .50], [.60, .10, .70, .50], [.11, .10, .21, .50],
                          [.12, .10, .22, .50], [.60, .10, .70, .50],
                          [.10, .10, .20, .40], [.10, .20, .20, .50],
                          [.30, .10, .40, .50], [.60, .10, .70, .50]], np.float16),
        scores=np.asarray([.9, .3, .9, .9, .9, .8, .8, .9, .9], np.float16),
        meta=np.array(json.dumps({"fps": 30.0, "stride": 1, "frame_size": [100, 100]})),
    )


class ActorTrainingLabelTests(unittest.TestCase):
    def _write(self, events: list[dict], actors: dict | None) -> tuple[dict, list[dict]]:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            action_dir, out_dir = root / "action", root / "actor"
            action_dir.mkdir()
            write_jsonl(action_dir / "game_actions.jsonl", {"video": "game", "fps": 30.0}, events)
            actors_file, dense = root / "game_actors.json", root / "game_dense.npz"
            if actors is not None:
                actors_file.write_text(json.dumps({"version": 4, "actors": actors}))
            _write_dense(dense)
            (out_dir).mkdir()
            (out_dir / "stale_actor_events.jsonl").write_text("{}\n")
            with (
                patch.object(actor_labels, "actors_path", return_value=actors_file),
                patch.object(actor_labels._store, "_cache", StatCache()),
                patch.object(training_labels, "dense_path", return_value=dense),
            ):
                summary = training_labels.write_actor_labels(action_dir, label_dir=out_dir)
            files = sorted(out_dir.iterdir())
            rows = [json.loads(line) for path in files for line in path.read_text().splitlines()]
            self.assertEqual([path.name for path in files], ["game_actor_events.jsonl"] if rows else [])
            return summary, rows

    def test_rows_put_the_actor_first_among_the_event_frame_proposals(self) -> None:
        summary, rows = self._write(
            [{"id": "a", "frame": 10, "label": "spike"},
             {"id": "s", "frame": 12, "label": "score"},
             {"id": "b", "frame": 30, "label": "set"}],
            {"a": {"verdict": "manual", "frame": 10, "box": [10, 10, 20, 50]},
             "b": {"verdict": "confirmed", "frame": 30, "box": [80, 10, 90, 50]}},
        )
        first, second = rows
        self.assertEqual(first["id"], "a")
        self.assertTrue(first["snapped"])
        self.assertEqual(first["box_source"], "event_frame")
        # The actor (snapped onto its dense box), then the other people at
        # or above the proposal cut; the duplicate re-detection and the weak
        # box are not candidates.
        np.testing.assert_allclose(first["boxes"], [[.10, .10, .20, .50]], atol=1e-3)
        self.assertEqual(set(first) - {"boxes"}, {"video", "id", "frame", "label", "box_source",
                                                   "snapped", "verdict"})
        # No dense box matches: the drawn box is kept, flagged unsnapped.
        self.assertFalse(second["snapped"])
        np.testing.assert_allclose(second["boxes"][0], [.8, .1, .9, .5])
        self.assertEqual(len(second["boxes"]), 3)
        self.assertEqual(summary["events"], 2)
        self.assertEqual(summary["score_events"], 1)
        self.assertEqual(summary["unsnapped"], 1)
        self.assertEqual(summary["classes"], {"spike": 1, "set": 1})

    def test_a_moved_event_follows_its_actor_and_the_rest_is_counted(self) -> None:
        summary, rows = self._write(
            [{"id": "moved", "frame": 11, "label": "receive"},
             {"id": "hidden", "frame": 30, "label": "set"},
             {"id": "nobody", "frame": 10, "label": "serve"},
             {"id": "x", "frame": 20, "label": "block"},
             {"id": "y", "frame": 20, "label": "spike"}],
            {"moved": {"verdict": "manual", "frame": 10, "box": [10, 10, 20, 50]},
             "hidden": {"verdict": "occluded"},
             "x": {"verdict": "manual", "frame": 20, "box": [10, 10, 20, 50]},
             "gone": {"verdict": "occluded"}},
        )
        (row,) = rows
        self.assertEqual(row["box_source"], "followed")
        np.testing.assert_allclose(row["boxes"][0], [.12, .10, .22, .50], atol=1e-3)
        self.assertEqual((summary["occluded"], summary["no_human_actor"], summary["shared_frame"],
                          summary["orphan_actor_labels"]), (1, 1, 2, 1))

    def test_an_agent_proposal_is_not_training_truth(self) -> None:
        summary, rows = self._write(
            [{"id": "agent", "frame": 10, "label": "spike"},
             {"id": "agent-hidden", "frame": 30, "label": "set"},
             {"id": "human", "frame": 11, "label": "receive"}],
            {"agent": {"verdict": "agent", "frame": 10, "box": [10, 10, 20, 50]},
             "agent-hidden": {"verdict": "agent_occluded"},
             "human": {"verdict": "confirmed", "frame": 11, "box": [11, 10, 21, 50]}},
        )
        self.assertEqual([row["id"] for row in rows], ["human"])
        self.assertEqual(summary["agent_proposals"], 2)

    def test_a_video_without_actor_labels_writes_nothing(self) -> None:
        summary, rows = self._write([{"id": "a", "frame": 10, "label": "spike"}], None)
        self.assertEqual(rows, [])
        self.assertEqual(summary["videos_without_actor_labels"], 1)

    def test_duplicate_event_ids_fail(self) -> None:
        with self.assertRaises(ValueError):
            self._write([{"id": "a", "frame": 10, "label": "spike"},
                         {"id": "a", "frame": 30, "label": "set"}],
                        {"a": {"verdict": "occluded"}})


if __name__ == "__main__":
    unittest.main()
