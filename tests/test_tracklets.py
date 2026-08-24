"""Resolving a box back to the tracklet it belongs to.

The rules here look arbitrary until you hit the case each one exists for, so
they are pinned: rank on the TIGHT box (a padded display box contains both of
two overlapping players and would hand the win to whoever is bigger), gate on
CONTAINMENT of the display box (IoU against a superset punishes it for being
big), and take the NEAREST detected frame rather than pooling a window (one
real frame's boxes are the truth; a window lists the same player twice).
"""

from __future__ import annotations

import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np

from yp_video.actor.labels import ActorLabel, ActorVerdict
from yp_video.core.cache import StatCache
from yp_video.core.jsonl import write_jsonl
from yp_video.extraction import links
from yp_video.tracklets.geometry import (
    LINK_MIN_CONTAINMENT,
    BoxQuery,
    TrackletIndex,
    TrackRef,
    containment,
    link_boxes,
)


def _pack(mask) -> np.ndarray:
    """Rows of packed bits, the shape store.load_track_masks unpacks."""
    return np.packbits(mask.reshape(mask.shape[0], -1), axis=1)


def _tracklet(rally: int, track: int, frames: list[int], box: list[float]) -> dict:
    return {
        "rally_id": rally,
        "track_id": track,
        "frames": frames,
        "boxes": [box for _ in frames],
        "scores": [0.9 for _ in frames],
    }


class TrackRefTests(unittest.TestCase):
    def test_key_round_trips(self) -> None:
        self.assertEqual(TrackRef(12, 3).key, "12:3")
        self.assertEqual(TrackRef.parse("12:3"), TrackRef(12, 3))
        self.assertEqual(TrackRef(12, 3).payload(), {"rally_id": 12, "track_id": 3})

    def test_identity_is_the_pair(self) -> None:
        """track_id restarts every rally, so it alone identifies nothing."""
        self.assertNotEqual(TrackRef(1, 3), TrackRef(2, 3))
        self.assertNotEqual(TrackRef(1, 3).key, TrackRef(2, 3).key)


class ContainmentTests(unittest.TestCase):
    def test_measures_the_track_box_not_the_overlap(self) -> None:
        # A small track box fully inside a big display box is contained 1.0,
        # while its IoU would be tiny — which is the whole reason for this.
        self.assertEqual(containment([10, 10, 20, 20], [0, 0, 100, 100]), 1.0)
        self.assertEqual(containment([0, 0, 10, 10], [5, 0, 15, 10]), 0.5)
        self.assertEqual(containment([0, 0, 10, 10], [50, 50, 60, 60]), 0.0)


class TrackletIndexTests(unittest.TestCase):
    INDEX = TrackletIndex(
        [_tracklet(1, 1, [10], [0, 0, 10, 10]), _tracklet(1, 2, [12], [50, 0, 60, 10])]
    )

    def test_nearest_detected_frame_wins_and_is_not_pooled(self) -> None:
        # frame 11 is undetected; -1 is searched before +1.
        near = self.INDEX.nearest(11, window=3)
        self.assertEqual([ref for ref, _ in near], [TrackRef(1, 1)])
        self.assertEqual(self.INDEX.nearest(10, window=1), self.INDEX.at(10))
        self.assertEqual(self.INDEX.nearest(99, window=3), [])

    def test_near_pools_the_window_and_keeps_tracklet_order(self) -> None:
        """The candidate-set question, as against nearest()'s snapshot one."""
        found = self.INDEX.near(11, window=3)
        self.assertEqual([w.ref for w in found], [TrackRef(1, 1), TrackRef(1, 2)])
        self.assertEqual([w.rows for w in found], [[0], [0]])
        self.assertEqual(self.INDEX.near(99, window=3), [])

    def test_identity_lookup_is_absent_not_an_error(self) -> None:
        """Re-tracking renumbers track_id, so a stale label names nobody."""
        self.assertEqual(self.INDEX.tracklet(TrackRef(1, 1))["frames"], [10])
        self.assertIsNone(self.INDEX.tracklet(TrackRef(9, 9)))
        self.assertEqual(len(self.INDEX), 2)


class LinkBoxesTests(unittest.TestCase):
    def test_ranks_on_the_tight_box_when_the_display_box_holds_both(self) -> None:
        """The failure this prevents: display box contains two players, and
        containment alone would pick whichever track box is bigger."""
        actor = _tracklet(1, 1, [100], [100, 100, 140, 200])
        bystander = _tracklet(1, 2, [100], [130, 100, 210, 200])  # bigger
        display = [90, 90, 220, 210]  # a padded union containing both

        resolved = link_boxes(
            TrackletIndex([actor, bystander]),
            [BoxQuery(key="e1", frame=100, anchor=[100, 100, 140, 200], gate=display)],
        )
        self.assertEqual(resolved["e1"], TrackRef(1, 1))

    def test_a_candidate_outside_the_gate_resolves_to_nothing(self) -> None:
        """max() always names a winner; the gate is what can say "none"."""
        far = _tracklet(1, 1, [100], [900, 900, 940, 1000])
        resolved = link_boxes(
            TrackletIndex([far]),
            [BoxQuery(key="e1", frame=100, anchor=[0, 0, 40, 100], gate=[0, 0, 50, 110])],
        )
        self.assertNotIn("e1", resolved)

    def test_the_gate_threshold_is_the_documented_one(self) -> None:
        box = [0, 0, 100, 100]
        half = _tracklet(1, 1, [5], box)
        # Exactly half the track box inside the gate — the boundary case.
        gate = [50, 0, 150, 100]
        self.assertEqual(containment(box, gate), LINK_MIN_CONTAINMENT)
        resolved = link_boxes(
            TrackletIndex([half]), [BoxQuery(key="e1", frame=5, anchor=box, gate=gate)]
        )
        self.assertIn("e1", resolved)

    def test_stride_widens_the_frame_search(self) -> None:
        track = _tracklet(1, 1, [10], [0, 0, 10, 10])
        query = BoxQuery(key="e1", frame=12, anchor=[0, 0, 10, 10], gate=[0, 0, 10, 10])
        self.assertNotIn("e1", link_boxes(TrackletIndex([track]), [query], stride=1))
        self.assertIn("e1", link_boxes(TrackletIndex([track]), [query], stride=3))


class ResolveTrackTests(unittest.TestCase):
    """Turning a picked tracklet back into a croppable detection.

    This used to run in the browser. It matters that it does not: the box it
    chooses becomes a crop, the crop becomes an embedding, and an embedding
    has to be reproducible from the saved label long after the click.
    """

    RECORD = {
        "id": "e1",
        "frame": 100,
        "box": [90, 90, 220, 210],
        "detections": [
            {"box": [100, 100, 140, 200], "score": 1.6},   # the player
            {"box": [104, 104, 136, 196], "score": 0.12},  # a tight duplicate
            {"box": [300, 100, 340, 200], "score": 1.4},   # someone else
        ],
    }

    @contextmanager
    def _video(self, *, mask=None, frames=(100,)):
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            tracks = root / "match_tracks.jsonl"
            write_jsonl(
                tracks,
                {"video": "match", "stride": 1},
                [_tracklet(1, 1, list(frames), [98, 98, 142, 202])],
            )
            masks_path = root / "match_masks.npz"
            if mask is not None:
                np.savez_compressed(masks_path, _shape=np.array(mask.shape[1:]), **{"1:1": _pack(mask)})
            index = TrackletIndex(
                [_tracklet(1, 1, list(frames), [98, 98, 142, 202])]
            )
            with (
                patch.object(links, "tracks_path", return_value=tracks),
                patch.object(links, "tracklet_index", return_value=index),
                patch.object(links, "tracks_masks_path", return_value=masks_path),
                patch.object(links, "load_track_masks", return_value=mask),
            ):
                yield

    def test_prefers_the_confident_detection_over_a_tight_duplicate(self) -> None:
        """The browser took the SMALLEST covering box, which over 239 real
        events meant a detection scoring 0.14 where the automatic pick scored
        1.54 — manual crops cut tighter and worse than automatic ones."""
        mask = np.ones((1, 8, 4), dtype=bool)  # the whole track box is player
        with self._video(mask=mask):
            pick = links.resolve_track("match", self.RECORD, TrackRef(1, 1))

        self.assertEqual(list(pick.box), [100, 100, 140, 200])
        self.assertTrue(pick.snap)
        self.assertEqual(pick.frame, 100)

    def test_no_detection_covering_the_mask_vetoes_snapping(self) -> None:
        """Nobody covers the silhouette ⇒ no stored detection IS this player,
        so the raw track box goes through and must not re-snap to a neighbour."""
        mask = np.zeros((1, 8, 4), dtype=bool)
        mask[0, :, 3] = True  # on-pixels only at the far right of the box
        record = {**self.RECORD, "detections": [{"box": [100, 100, 110, 200], "score": 1.6}]}
        with self._video(mask=mask):
            pick = links.resolve_track("match", record, TrackRef(1, 1))

        self.assertEqual(list(pick.box), [98, 98, 142, 202])
        self.assertFalse(pick.snap)

    def test_without_masks_box_iou_decides(self) -> None:
        with self._video(mask=None):
            pick = links.resolve_track("match", self.RECORD, TrackRef(1, 1))
        self.assertEqual(list(pick.box), [100, 100, 140, 200])

    def test_a_track_that_never_reaches_the_event_crops_where_it_does(self) -> None:
        """The client needed a hand-clicked frame for this; the tracklet
        already knows one."""
        with self._video(mask=None, frames=(400,)):
            pick = links.resolve_track("match", self.RECORD, TrackRef(1, 1))

        self.assertEqual(pick.frame, 400)
        self.assertFalse(pick.snap)

    def test_an_unknown_tracklet_resolves_to_nothing(self) -> None:
        """Re-tracking renumbers every id — that must be absent, not wrong."""
        with self._video(mask=None):
            self.assertIsNone(links.resolve_track("match", self.RECORD, TrackRef(9, 9)))

    def test_an_injected_archive_is_used_instead_of_reopening_the_file(self) -> None:
        """Re-deciding a video resolves ~300 tracklet picks; opening the 12 MB
        silhouette archive once per pick was most of what that cost."""
        mask = np.ones((1, 8, 4), dtype=bool)
        with self._video(mask=mask):
            with patch.object(links, "load_track_masks") as reopen:
                pick = links.resolve_track(
                    "match", self.RECORD, TrackRef(1, 1), masks={"1:1": mask}
                )

        reopen.assert_not_called()
        self.assertEqual(list(pick.box), [100, 100, 140, 200])
        self.assertTrue(pick.snap)


class EventTrackPrecedenceTests(unittest.TestCase):
    """A named tracklet outranks the one the box happens to sit on.

    Geometry is the fallback for a policy that answered with a BOX. Running it
    over an answer that already named a tracklet is what made a deliberate
    pick look like it did nothing: two overlapping players each resolve to a
    box matching the other's tracklet, so the board kept showing the one you
    had just clicked away from. Measured at 6.7% of picks on real data.
    """

    #: Two players standing on top of each other. The record's box matches
    #: ACTOR geometrically; the human named the BYSTANDER.
    ACTOR = _tracklet(1, 1, [100], [100, 100, 140, 200])
    BYSTANDER = _tracklet(1, 2, [100], [104, 100, 148, 200])
    RECORD = {
        "id": "e1",
        "frame": 100,
        "box": [90, 90, 160, 210],
        "actor_box": [100, 100, 140, 200],
    }

    @contextmanager
    def _video(self, labels, tracklets=None):
        """``tracklets`` stands in for a re-tracking run that renumbered."""
        tracklets = [self.ACTOR, self.BYSTANDER] if tracklets is None else tracklets
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            tracks, records = root / "t.jsonl", root / "r.jsonl"
            write_jsonl(tracks, {"stride": 1}, tracklets)
            write_jsonl(records, {"video": "match"}, [self.RECORD])
            with (
                patch.object(links, "tracks_path", return_value=tracks),
                patch.object(links, "records_path", return_value=records),
                patch.object(
                    links, "tracklet_index", return_value=TrackletIndex(tracklets)
                ),
                patch.object(links.actor_labels, "load", return_value=labels),
            ):
                yield

    def test_geometry_decides_when_nobody_named_a_tracklet(self) -> None:
        with self._video({}):
            self.assertEqual(links._event_tracks("match")["e1"], TrackRef(1, 1))

    def test_a_human_pick_beats_the_box_it_resolved_to(self) -> None:
        label = ActorLabel(ActorVerdict.MANUAL, track=TrackRef(1, 2), box=(104, 100, 148, 200))
        with self._video({"e1": label}):
            self.assertEqual(links._event_tracks("match")["e1"], TrackRef(1, 2))

    def test_a_policy_pick_beats_geometry_too(self) -> None:
        record = {**self.RECORD, "track": "1:2"}
        with self._video({}), patch.object(links, "read_jsonl_cached") as read:
            read.side_effect = lambda p: (
                ({"stride": 1}, [self.ACTOR, self.BYSTANDER])
                if p.name == "t.jsonl"
                else ({}, [record])
            )
            self.assertEqual(links._event_tracks("match")["e1"], TrackRef(1, 2))

    def test_a_named_tracklet_that_no_longer_exists_falls_back(self) -> None:
        """Re-tracking renumbers every id — honouring a stale name would point
        at whoever inherited the number.

        It falls back to the HUMAN's anchor, not the record's box: they
        pointed at the bystander, so re-deriving hands them the bystander even
        though the policy's box names the actor.
        """
        label = ActorLabel(ActorVerdict.MANUAL, track=TrackRef(9, 9), box=(104, 100, 148, 200))
        with self._video({"e1": label}):
            self.assertEqual(links._event_tracks("match")["e1"], TrackRef(1, 2))

    def test_a_pick_survives_a_re_track_that_swapped_the_ids(self) -> None:
        """The whole point: re-running tracking must not move a human's pick.

        ByteTrack numbers per rally and reuses the numbers, so a re-run can
        hand 1:2 to the other player. The stored pair still EXISTS — which is
        why existence was never proof — but the anchor says it is somebody
        else now, and the same anchor re-derives the person they picked under
        whatever id that person wears today.
        """
        swapped = [_tracklet(1, 2, [100], [100, 100, 140, 200]),   # was the actor's id
                   _tracklet(1, 1, [100], [104, 100, 148, 200])]   # bystander, renumbered
        label = ActorLabel(ActorVerdict.MANUAL, track=TrackRef(1, 2), box=(104, 100, 148, 200))
        with self._video({"e1": label}, tracklets=swapped):
            self.assertEqual(links._event_tracks("match")["e1"], TrackRef(1, 1))

    def test_a_re_track_that_kept_the_id_keeps_the_pick(self) -> None:
        """The other half: an id that survived is honoured, untouched."""
        label = ActorLabel(ActorVerdict.MANUAL, track=TrackRef(1, 2), box=(104, 100, 148, 200))
        with self._video({"e1": label}):
            self.assertEqual(links._event_tracks("match")["e1"], TrackRef(1, 2))

    def test_a_re_derived_pick_refuses_a_near_tie(self) -> None:
        """Two players a hair apart: re-deriving must not guess between them.

        No answer sends the event to the re-pick worklist. A wrong one would
        silently reassign a deliberate pick — the 6.7% this module exists to
        stop.
        """
        twins = [_tracklet(1, 3, [100], [100, 100, 140, 200]),
                 _tracklet(1, 4, [100], [101, 100, 141, 200])]
        # 1:4 exists, so only the anchor can say it is the wrong player now.
        label = ActorLabel(ActorVerdict.MANUAL, track=TrackRef(1, 4), box=(100, 100, 140, 200))
        with self._video({"e1": label}, tracklets=twins):
            self.assertNotIn("e1", links._event_tracks("match"))

    def test_silence_is_not_grounds_to_overturn_a_pick(self) -> None:
        """A tracklet not detected at the anchor frame proves nothing.

        Absence of evidence would otherwise drop a working label: 1.6% of real
        picks name a tracklet the anchor frame has no detection for.
        """
        elsewhere = [_tracklet(1, 2, [400], [104, 100, 148, 200]),
                     _tracklet(1, 1, [400], [100, 100, 140, 200])]
        label = ActorLabel(ActorVerdict.MANUAL, track=TrackRef(1, 2), box=(104, 100, 148, 200))
        with self._video({"e1": label}, tracklets=elsewhere):
            self.assertEqual(links._event_tracks("match")["e1"], TrackRef(1, 2))

    def test_a_policy_box_still_takes_the_best_candidate(self) -> None:
        """The margin is for human picks only — a policy answer wants an
        answer, and nobody deliberated over it."""
        twins = [_tracklet(1, 3, [100], [100, 100, 140, 200]),
                 _tracklet(1, 4, [100], [101, 100, 141, 200])]
        with self._video({}, tracklets=twins):
            self.assertEqual(links._event_tracks("match")["e1"], TrackRef(1, 3))


class LabelBorneOutTests(unittest.TestCase):
    """The single rule the board, the crop and the candidate list share.

    Testing it here rather than three times over: extraction/links,
    extraction/cropping and actor/candidates all answer "who does this label
    name TODAY" through ActorLabel.borne_out_by, so a second opinion about it
    cannot exist to drift.
    """

    ACTOR = _tracklet(1, 1, [100], [100, 100, 140, 200])
    BYSTANDER = _tracklet(1, 2, [100], [104, 100, 148, 200])
    INDEX = TrackletIndex([ACTOR, BYSTANDER])

    def _pick(self, track, box=(104, 100, 148, 200), frame=None):
        return ActorLabel(ActorVerdict.MANUAL, track=track, box=box, frame=frame)

    def test_an_anchor_that_still_fits_keeps_the_pick(self) -> None:
        self.assertEqual(
            self._pick(TrackRef(1, 2)).borne_out_by(self.INDEX, 100), TrackRef(1, 2)
        )

    def test_a_renumbered_id_loses_the_pick(self) -> None:
        """The re-track: 1:2 still exists, wearing the other player."""
        swapped = TrackletIndex(
            [_tracklet(1, 2, [100], [100, 100, 140, 200]),
             _tracklet(1, 1, [100], [104, 100, 148, 200])]
        )
        self.assertIsNone(self._pick(TrackRef(1, 2)).borne_out_by(swapped, 100))

    def test_a_pair_that_is_gone_loses_the_pick(self) -> None:
        self.assertIsNone(self._pick(TrackRef(9, 9)).borne_out_by(self.INDEX, 100))

    def test_a_pick_with_no_box_has_nothing_to_check(self) -> None:
        """Silence, not contradiction — the pick stands."""
        self.assertEqual(
            self._pick(TrackRef(1, 2), box=None).borne_out_by(self.INDEX, 100),
            TrackRef(1, 2),
        )

    def test_a_frame_with_no_detections_is_silence_too(self) -> None:
        self.assertEqual(
            self._pick(TrackRef(1, 2)).borne_out_by(self.INDEX, 400), TrackRef(1, 2)
        )

    def test_a_cross_frame_pick_is_judged_on_its_own_frame(self) -> None:
        """label.frame is where they clicked; the event frame would say 1:1."""
        elsewhere = TrackletIndex(
            [_tracklet(1, 1, [100, 400], [0, 0, 10, 10]),
             _tracklet(1, 2, [400], [104, 100, 148, 200])]
        )
        pick = self._pick(TrackRef(1, 2), frame=400)
        self.assertEqual(pick.borne_out_by(elsewhere, 100), TrackRef(1, 2))

    def test_a_box_only_verdict_names_no_tracklet(self) -> None:
        bare = ActorLabel(ActorVerdict.MANUAL, box=(104, 100, 148, 200))
        self.assertIsNone(bare.borne_out_by(self.INDEX, 100))


class CropFallsBackToTheClickTests(unittest.TestCase):
    """A re-track must not send re-extraction to crop a stranger.

    crop_target resolves a tracklet whenever it is given one, so the guard has
    to be that label_target stops handing it a pick the anchor disowns — then
    the fallback it already carries, the box the person actually clicked,
    takes over.
    """

    def test_a_disowned_pick_crops_the_box_the_human_clicked(self) -> None:
        from yp_video.extraction import cropping

        swapped = TrackletIndex(
            [_tracklet(1, 2, [100], [100, 100, 140, 200]),
             _tracklet(1, 1, [100], [104, 100, 148, 200])]
        )
        record = {"id": "e1", "frame": 100, "box": [90, 90, 160, 210]}
        label = ActorLabel(
            ActorVerdict.MANUAL, track=TrackRef(1, 2), box=(104, 100, 148, 200)
        )
        with tempfile.TemporaryDirectory() as raw:
            tracks = Path(raw) / "t.jsonl"
            write_jsonl(tracks, {"stride": 1}, [])
            with (
                patch.object(links, "tracks_path", return_value=tracks),
                patch.object(links, "tracklet_index", return_value=swapped),
                patch.object(links, "tracks_stride", return_value=1),
                patch.object(cropping, "resolve_track") as resolve,
            ):
                target = cropping.label_target("match", record, label)
        resolve.assert_not_called()
        self.assertEqual(list(target.box), [104, 100, 148, 200])


class UnresolvedLabelsTests(unittest.TestCase):
    """The re-pick worklist: a labeled event resolvable to no tracklet.

    Membership is about what resolves TODAY, not what the label stored — a
    confirm snapshot (box, no track key) that sits on a tracked player is
    fine, and only a label the geometry can do nothing with is work.
    """

    ACTOR = _tracklet(1, 1, [100], [100, 100, 140, 200])
    ON_TRACK = {
        "id": "e1",
        "frame": 100,
        "box": [90, 90, 160, 210],
        "actor_box": [100, 100, 140, 200],
    }
    ON_NOBODY = {
        "id": "e1",
        "frame": 100,
        "box": [400, 90, 470, 210],
        "actor_box": [410, 100, 450, 200],
    }

    @contextmanager
    def _video(self, labels, record, tracked=True):
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            tracks, records = root / "t.jsonl", root / "r.jsonl"
            if tracked:
                write_jsonl(tracks, {"stride": 1}, [self.ACTOR])
            write_jsonl(records, {"video": "match"}, [record])
            with (
                patch.object(links, "tracks_path", return_value=tracks),
                patch.object(links, "records_path", return_value=records),
                patch.object(
                    links,
                    "tracklet_index",
                    return_value=TrackletIndex([self.ACTOR]),
                ),
                patch.object(links.actor_labels, "load", return_value=labels),
                patch.object(links, "_links_cache", StatCache()),
            ):
                yield

    def test_a_confirm_snapshot_that_resolves_is_not_work(self) -> None:
        label = ActorLabel(
            ActorVerdict.CONFIRMED_AUTO, box=(100, 100, 140, 200)
        )
        with self._video({"e1": label}, self.ON_TRACK):
            self.assertEqual(links.unresolved_labels("match"), set())

    def test_a_label_resolving_to_nothing_is_the_worklist(self) -> None:
        label = ActorLabel(ActorVerdict.MANUAL, box=(410, 100, 450, 200))
        with self._video({"e1": label}, self.ON_NOBODY):
            self.assertEqual(links.unresolved_labels("match"), {"e1"})

    def test_occluded_is_a_full_answer_not_work(self) -> None:
        with self._video({"e1": ActorLabel(ActorVerdict.OCCLUDED)}, self.ON_NOBODY):
            self.assertEqual(links.unresolved_labels("match"), set())

    def test_an_untracked_video_has_no_re_pick_work(self) -> None:
        """Nothing resolves before tracking exists. The remedy is running
        tracking, not re-picking players — that gap is the pipeline's."""
        label = ActorLabel(ActorVerdict.MANUAL, box=(410, 100, 450, 200))
        with self._video({"e1": label}, self.ON_NOBODY, tracked=False):
            self.assertEqual(links.unresolved_labels("match"), set())


if __name__ == "__main__":
    unittest.main()
