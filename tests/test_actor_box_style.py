"""The 2XLarge box-style rule (one rule for the actor snapshot and the
Association box check) and the per-video check built on it."""

from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np

from yp_video.actor import box_check as box_check_module
from yp_video.actor import box_style
from yp_video.actor import labels as actor_labels
from yp_video.actor.box_check import box_check, pending_count
from yp_video.actor.labels import ActorLabel, ActorVerdict
from yp_video.actor.person_labels import DensePass
from yp_video.core.cache import StatCache
from yp_video.core.jsonl import write_jsonl
from yp_video.tracklets.geometry import TrackRef
from yp_video.web.routers import actor_association as router


class BoxStyleRuleTests(unittest.TestCase):
    def test_same_frame_human_box_survives_stale_track_id(self) -> None:
        label = ActorLabel(ActorVerdict.CONFIRMED_AUTO, TrackRef(1, 999), (10., 20., 40., 80.), 10)
        result, reason = box_style.resolve_target(label, 10, {}, 100, 100)
        self.assertEqual(result, [.1, .2, .4, .8])
        self.assertEqual(reason, 'human_box_exact_frame')

    def test_cross_frame_follows_the_dense_boxes_to_the_event_frame(self) -> None:
        label = ActorLabel(ActorVerdict.MANUAL, None, (10., 20., 40., 80.), 9)
        people = {9: [[.1, .2, .4, .8], [.6, .2, .9, .8]], 10: [[.6, .2, .9, .8], [.12, .2, .42, .8]]}
        result, reason = box_style.resolve_target(label, 10, people, 100, 100)
        self.assertEqual(result, [.12, .2, .42, .8])
        self.assertEqual(reason, 'cross_frame_resolved')
        # Frame 11 has nobody where the person was: unresolved, not guessed.
        people[11] = [[.6, .2, .9, .8]]
        self.assertEqual(box_style.resolve_target(label, 11, people, 100, 100),
                         (None, box_style.CROSS_FRAME_UNRESOLVED))

    def test_occluded_is_not_background_and_invalid_boxes_fail(self) -> None:
        label = ActorLabel(ActorVerdict.OCCLUDED)
        self.assertEqual(box_style.resolve_target(label, 10, {}, 100, 100), (None, 'occluded'))
        with self.assertRaises(ValueError):
            box_style.normalize([2, 3, 1, 4], 100, 100)
        with self.assertRaises(ValueError):
            box_style.normalize([0, 0, float('nan'), 5], 100, 100)

    def test_snap_takes_the_clear_dense_box_and_refuses_a_contested_one(self) -> None:
        human = [.10, .10, .20, .50]
        mine, duplicate, far = [.11, .10, .21, .52], [.11, .11, .21, .51], [.60, .10, .70, .50]
        self.assertEqual(box_style.match(human, [far, mine, duplicate]), (mine, box_style.SNAPPED))
        self.assertEqual(box_style.match(human, [far]), (None, box_style.UNMATCHED))
        # Two people overlapping the human box about equally: no guess.
        upper, lower = [.10, .10, .20, .40], [.10, .20, .20, .50]
        self.assertEqual(box_style.match(human, [upper, lower]), (None, box_style.CONTESTED))
        self.assertEqual(box_style.settle(human, None), (None, box_style.NOT_COVERED))


def write_dense(path: Path, *, frames, counts, boxes, scores, stride=1) -> None:
    np.savez_compressed(
        path,
        frames=np.asarray(frames, np.int32),
        counts=np.asarray(counts, np.int32),
        boxes=np.asarray(boxes, np.float16).reshape(-1, 4),
        scores=np.asarray(scores, np.float16),
        meta=np.array(json.dumps({"fps": 30.0, "stride": stride, "frame_size": [100, 100]})),
    )


class DensePassTests(unittest.TestCase):
    def test_a_frame_reads_alone_with_the_stride_fill(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "game_dense.npz"
            write_dense(path, frames=[10, 12, 40], counts=[2, 0, 1],
                        boxes=[[.1, .1, .2, .2], [.3, .3, .4, .4], [.5, .5, .6, .6]],
                        scores=[.9, .2, .9], stride=2)
            dense = DensePass(path)
            boxes, scores = dense.at(11, .1)
            np.testing.assert_allclose(boxes, [[.1, .1, .2, .2], [.3, .3, .4, .4]], atol=1e-3)
            np.testing.assert_allclose(scores, [.9, .2], atol=1e-3)
            self.assertEqual(dense.at(11, .4)[0], dense.at(10, .4)[0])
            self.assertEqual(dense.at(12, .1), ([], []))
            # After a span's last frame and before the first: not covered.
            self.assertIsNone(dense.at(13, .1))
            self.assertIsNone(dense.at(9, .1))
            self.assertIsNotNone(dense.at(40, .1))


class BoxCheckTests(unittest.TestCase):
    @contextmanager
    def _video(self, actors: dict):
        """A scratch video: actions on frames 10/20/30/40, a dense pass that
        covers 10–30, and the given actor labels."""
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            action, actors_file, dense = (root / "game_actions.jsonl",
                                          root / "game_actors.json", root / "game_dense.npz")
            write_jsonl(action, {"fps": 30.0}, [
                {"id": f"e{frame}", "frame": frame, "label": "attack"} for frame in (10, 20, 30, 40)
            ] + [{"id": "s", "frame": 50, "label": "score"}])
            actors_file.write_text(json.dumps({"version": 2, "actors": actors}))
            write_dense(dense, frames=[10, 20, 30], counts=[2, 2, 1],
                        boxes=[[.10, .10, .20, .50], [.60, .10, .70, .50],
                               [.10, .10, .20, .40], [.10, .20, .20, .50],
                               [.60, .10, .70, .50]],
                        scores=[.9, .3, .8, .8, .9])
            with (
                patch.object(box_check_module, "action_annotation_path", return_value=action),
                patch.object(box_check_module, "dense_path", return_value=dense),
                patch.object(actor_labels, "actors_path", return_value=actors_file),
                patch.object(actor_labels._store, "_cache", StatCache()),
                patch.object(box_check_module, "_cache", StatCache()),
            ):
                yield

    def test_each_boxed_label_gets_the_snapshot_status_and_its_frame_boxes(self) -> None:
        box = [10, 10, 20, 50]
        with self._video({
            "e10": {"verdict": "manual", "box": box},
            "e20": {"verdict": "confirmed_auto", "box": box, "frame": 20},
            "e30": {"verdict": "manual", "box": box},
            "e40": {"verdict": "manual", "box": box},
            "s": {"verdict": "manual", "box": box},
        }):
            entries = {e["id"]: e for e in box_check("game")}
            self.assertEqual({k: e["status"] for k, e in entries.items()}, {
                "e10": "snapped", "e20": "contested", "e30": "unmatched", "e40": "not_covered",
            })
            self.assertEqual(entries["e10"]["boxes"], [
                {"box": [10.0, 10.0, 20.0, 50.0], "score": 0.9},
                {"box": [60.0, 10.0, 70.0, 50.0], "score": 0.3},
            ])
            self.assertEqual(entries["e40"]["boxes"], [])
            self.assertEqual(entries["e20"]["label_box"], [10.0, 10.0, 20.0, 50.0])
            self.assertEqual(pending_count("game"), 3)
            self.assertEqual(router.get_box_check("game.mp4"), box_check("game"))

    def test_occluded_and_boxless_labels_are_skipped_and_cross_frame_is_followed(self) -> None:
        with self._video({
            "e10": {"verdict": "occluded"},
            "e20": {"verdict": "manual", "track": "1:2"},
            # Drawn on 10, followed to 30 through 11–29, which the pass skipped.
            "e30": {"verdict": "manual", "box": [60, 10, 70, 50], "frame": 10},
        }):
            self.assertEqual([(e["id"], e["status"], e["label_frame"]) for e in box_check("game")],
                             [("e30", "cross_frame_unresolved", 10)])

    def test_a_new_label_file_recomputes_the_check(self) -> None:
        with self._video({"e30": {"verdict": "manual", "box": [10, 10, 20, 50]}}):
            self.assertEqual(pending_count("game"), 1)
            actor_labels.save("game", "e30", ActorLabel(ActorVerdict.MANUAL, box=(60., 10., 70., 50.), snap=False))
            self.assertEqual(pending_count("game"), 0)


if __name__ == "__main__":
    unittest.main()
