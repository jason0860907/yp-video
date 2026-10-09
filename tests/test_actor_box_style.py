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
from yp_video.core.cache import DiskStatCache, StatCache
from yp_video.core.jsonl import write_jsonl
from yp_video.extraction import done
from yp_video.extraction import store as extraction_store
from yp_video.web.routers import actor_association as router


class BoxStyleRuleTests(unittest.TestCase):
    def test_a_label_on_its_event_frame_is_taken_as_is(self) -> None:
        label = ActorLabel(ActorVerdict.CONFIRMED_AUTO, 10, (10., 20., 40., 80.))
        self.assertEqual(box_style.resolve_target(label, 10, {}, 100, 100),
                         ([.1, .2, .4, .8], box_style.EVENT_FRAME))

    def test_a_moved_event_follows_the_dense_boxes_to_its_frame(self) -> None:
        label = ActorLabel(ActorVerdict.MANUAL, 9, (10., 20., 40., 80.))
        people = {9: [[.1, .2, .4, .8], [.6, .2, .9, .8]], 10: [[.6, .2, .9, .8], [.12, .2, .42, .8]]}
        self.assertEqual(box_style.resolve_target(label, 10, people, 100, 100),
                         ([.12, .2, .42, .8], box_style.FOLLOWED))
        # Frame 11 has nobody where the person was: unresolved, not guessed.
        people[11] = [[.6, .2, .9, .8]]
        self.assertEqual(box_style.resolve_target(label, 11, people, 100, 100),
                         (None, box_style.UNRESOLVED))

    def test_occluded_is_not_background_and_invalid_boxes_fail(self) -> None:
        label = ActorLabel(ActorVerdict.OCCLUDED)
        self.assertEqual(box_style.resolve_target(label, 10, {}, 100, 100), (None, box_style.OCCLUDED))
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


@contextmanager
def scratch_video(actors: dict | None = None):
    """A scratch video: actions on frames 10/20/30/40 (plus a score), a dense
    pass covering 10–30, and the given actor labels (none = no label file)."""
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        action, actors_file, dense = (root / "game_actions.jsonl",
                                      root / "game_actors.json", root / "game_dense.npz")
        write_jsonl(action, {"fps": 30.0}, [
            {"id": f"e{frame}", "frame": frame, "label": "attack"} for frame in (10, 20, 30, 40)
        ] + [{"id": "s", "frame": 50, "label": "score"}])
        if actors is not None:
            actors_file.write_text(json.dumps({"version": 3, "actors": actors}))
        write_dense(dense, frames=[10, 20, 30], counts=[2, 2, 1],
                    boxes=[[.10, .10, .20, .50], [.60, .10, .70, .50],
                           [.10, .10, .20, .40], [.10, .20, .20, .50],
                           [.60, .10, .70, .50]],
                    scores=[.9, .3, .8, .8, .9])
        with (
            patch.object(extraction_store, "action_annotation_path", return_value=action),
            patch.object(extraction_store, "rally_annotation_path", return_value=None),
            patch.object(extraction_store, "load_rallies", return_value=[]),
            patch.object(box_check_module, "dense_path", return_value=dense),
            patch.object(box_style, "dense_path", return_value=dense),
            patch.object(box_style, "_dense_cache", StatCache()),
            patch.object(actor_labels, "actors_path", return_value=actors_file),
            patch.object(actor_labels._store, "_cache", StatCache()),
            patch.object(box_check_module, "_cache", StatCache()),
            patch.object(box_check_module, "_pending", DiskStatCache(root / "pending.json")),
        ):
            yield


class EventBoxTests(unittest.TestCase):
    def test_the_event_frame_box_of_a_label(self) -> None:
        with scratch_video():
            on_event = ActorLabel(ActorVerdict.MANUAL, 20, (1., 2., 3., 4.))
            self.assertEqual(box_style.event_box("game", on_event, 20), (1., 2., 3., 4.))
            self.assertIsNone(box_style.event_box("game", ActorLabel(ActorVerdict.OCCLUDED), 20))
            # Moved 10 -> 20 across frames the pass did not cover: lost.
            moved = ActorLabel(ActorVerdict.MANUAL, 10, (60., 10., 70., 50.))
            self.assertIsNone(box_style.event_box("game", moved, 20))

    def test_a_moved_event_is_followed_frame_by_frame(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            dense = Path(raw) / "game_dense.npz"
            # Frame 12 is strode over, so it reads frame 11's boxes.
            write_dense(dense, frames=[10, 11, 13], counts=[2, 1, 1],
                        boxes=[[.10, .10, .20, .50], [.60, .10, .70, .50],
                               [.12, .10, .22, .50], [.14, .10, .24, .50]],
                        scores=[.9, .9, .9, .9], stride=2)
            with (
                patch.object(box_style, "dense_path", return_value=dense),
                patch.object(box_style, "_dense_cache", StatCache()),
            ):
                moved = ActorLabel(ActorVerdict.MANUAL, 10, (10., 10., 20., 50.))
                self.assertEqual(box_style.event_box("game", moved, 13), (14., 10., 24., 50.))
                other = ActorLabel(ActorVerdict.MANUAL, 10, (60., 10., 70., 50.))
                self.assertIsNone(box_style.event_box("game", other, 13))

    def test_settle_box_snaps_only_a_clear_match(self) -> None:
        with scratch_video():
            self.assertEqual(box_style.settle_box("game", (11., 11., 21., 52.), 10), (10., 10., 20., 50.))
            # Two people overlap equally on frame 20; frame 40 is not covered.
            self.assertEqual(box_style.settle_box("game", (10., 10., 20., 50.), 20), (10., 10., 20., 50.))
            self.assertEqual(box_style.settle_box("game", (11., 11., 21., 52.), 40), (11., 11., 21., 52.))


class ConfirmationTests(unittest.TestCase):
    """A confirmed automatic pick stores its event-frame box in 2XLarge style."""

    def test_the_policy_box_snaps_to_its_clear_dense_match(self) -> None:
        def auto(event_id, frame, box, **extra):
            return {"id": event_id, "frame": frame, "resolution": "auto", "actor_box": box, **extra}

        records = [
            auto("clear", 10, [11, 11, 21, 52]),
            auto("contested", 20, [10, 10, 20, 50]),
            # Cut on 10 although the event is on 20: no frame between them
            # to follow it through, so there is nothing to endorse.
            auto("elsewhere", 20, [60, 10, 70, 50], crop_frame=10),
            {"id": "manual", "frame": 10, "resolution": "manual", "actor_box": [1, 1, 5, 5]},
        ]
        with scratch_video():
            labels = done.confirmations_for("game", records)
        self.assertEqual(labels, {
            "clear": ActorLabel(ActorVerdict.CONFIRMED_AUTO, 10, (10., 10., 20., 50.)),
            "contested": ActorLabel(ActorVerdict.CONFIRMED_AUTO, 20, (10., 10., 20., 50.)),
        })


class BoxCheckTests(unittest.TestCase):
    def test_every_event_gets_its_frame_boxes_and_a_label_its_status(self) -> None:
        box = [10, 10, 20, 50]
        with scratch_video({
            "e10": {"verdict": "manual", "frame": 10, "box": box},
            "e20": {"verdict": "confirmed_auto", "frame": 20, "box": box},
            "e30": {"verdict": "manual", "frame": 30, "box": box},
            "e40": {"verdict": "manual", "frame": 40, "box": box},
        }):
            entries = {e["id"]: e for e in box_check("game")}
            self.assertEqual({k: e["status"] for k, e in entries.items()}, {
                "e10": "snapped", "e20": "contested", "e30": "unmatched", "e40": "not_covered",
            })
            self.assertEqual(entries["e10"]["boxes"], [
                {"box": (10.0, 10.0, 20.0, 50.0), "score": 0.9},
                {"box": (60.0, 10.0, 70.0, 50.0), "score": 0.3},
            ])
            self.assertEqual(entries["e40"]["boxes"], [])
            self.assertEqual(entries["e20"]["label_box"], (10.0, 10.0, 20.0, 50.0))
            self.assertEqual(pending_count("game"), 3)
            self.assertEqual(router.get_box_check("game.mp4"), box_check("game"))

    def test_unlabeled_and_occluded_events_carry_boxes_but_no_status(self) -> None:
        with scratch_video({
            "e10": {"verdict": "occluded"},
            # Drawn on 10, lost on the way to 30.
            "e30": {"verdict": "manual", "frame": 10, "box": [10, 10, 20, 50]},
        }):
            entries = {e["id"]: e for e in box_check("game")}
            self.assertEqual(
                {k: (e["status"], e["label_box"], len(e["boxes"])) for k, e in entries.items()},
                {"e10": (None, None, 2), "e20": (None, None, 2),
                 "e30": ("unresolved", None, 1), "e40": (None, None, 0)},
            )

    def test_a_new_label_file_recomputes_the_check(self) -> None:
        with scratch_video({"e30": {"verdict": "manual", "frame": 30, "box": [10, 10, 20, 50]}}):
            self.assertEqual(pending_count("game"), 1)
            actor_labels.save("game", "e30", ActorLabel(ActorVerdict.MANUAL, 30, (60., 10., 70., 50.)))
            self.assertEqual(pending_count("game"), 0)

    def test_an_old_shape_label_file_is_an_error(self) -> None:
        with scratch_video({}):
            actor_labels.actors_path("game").write_text(json.dumps(
                {"version": 2, "actors": {"e10": {"verdict": "manual", "track": "1:2", "box": [1, 2, 3, 4]}}}
            ))
            with self.assertRaises(ValueError):
                actor_labels.load("game")
            actor_labels.actors_path("game").write_text(json.dumps(
                {"version": 3, "actors": {"e10": {"verdict": "manual", "track": "1:2", "frame": 1, "box": [1, 2, 3, 4]}}}
            ))
            with self.assertRaises(ValueError):
                actor_labels.load("game")


if __name__ == "__main__":
    unittest.main()
