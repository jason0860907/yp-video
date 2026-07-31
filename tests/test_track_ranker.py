"""The tracklet feature contract the track dataset is built on."""

from __future__ import annotations

import unittest

import numpy as np

from yp_video.actor.track_features import (
    TRACK_CANDIDATE_FEATURE_NAMES,
    TRACK_CONTEXT_FEATURE_NAMES,
    candidates_near,
    extract_track_features,
)
from yp_video.tracklets.geometry import TrackletIndex, TrackRef


def _tracklet(rally, track, frames, box, score=0.9):
    return {
        "rally_id": rally,
        "track_id": track,
        "frames": list(frames),
        "boxes": [list(box) for _ in frames],
        "scores": [score for _ in frames],
    }


def _half_mask(*, fill: str, shape=(96, 48)) -> np.ndarray:
    """A silhouette occupying one half of its box, in mask-grid space."""
    mask = np.zeros(shape, dtype=bool)
    if fill == "left":
        mask[:, : shape[1] // 2] = True
    else:
        mask[:, shape[1] // 2 :] = True
    return mask


class CandidateSetTests(unittest.TestCase):
    def test_only_tracklets_alive_near_the_event_are_candidates(self) -> None:
        tracklets = [
            _tracklet(1, 1, range(95, 106), [0, 0, 40, 100]),
            _tracklet(1, 2, range(500, 511), [0, 0, 40, 100]),  # another rally moment
        ]
        near = candidates_near(TrackletIndex(tracklets), 100)
        self.assertEqual([c.ref for c in near], [TrackRef(1, 1)])

    def test_no_tracklet_alive_is_stated_not_implied(self) -> None:
        """The ~7% of events with nothing tracked. An empty candidate list
        alone would leave the NONE head inferring it from absence."""
        features = extract_track_features([], 10.0, 10.0, 100)
        flag = TRACK_CONTEXT_FEATURE_NAMES.index("no_track_alive")
        self.assertEqual(features.candidates.shape, (0, len(TRACK_CANDIDATE_FEATURE_NAMES)))
        self.assertEqual(features.context[flag], 1.0)


class FeatureTests(unittest.TestCase):
    def _row(self, tracklets, x, y, frame=100, **kw):
        f = extract_track_features(candidates_near(TrackletIndex(tracklets), frame), x, y, frame, **kw)
        return dict(zip(TRACK_CANDIDATE_FEATURE_NAMES, f.candidates[0]))

    def test_presence_at_the_event_frame_is_distinguishable(self) -> None:
        at = self._row([_tracklet(1, 1, [100], [0, 0, 40, 100])], 20, 50)
        near = self._row([_tracklet(1, 1, [97], [0, 0, 40, 100])], 20, 50)
        self.assertEqual(at["present_at_event"], 1.0)
        self.assertEqual(at["frame_gap"], 0.0)
        self.assertEqual(near["present_at_event"], 0.0)
        self.assertEqual(near["frame_gap"], 3.0)

    def test_approach_speed_separates_a_spiker_from_a_bystander(self) -> None:
        """Only a tracklet can say this; it is why the unit changed."""
        closing = {
            "rally_id": 1, "track_id": 1, "frames": [96, 100],
            "boxes": [[300, 0, 340, 100], [0, 0, 40, 100]], "scores": [0.9, 0.9],
        }
        still = _tracklet(1, 2, [96, 100], [0, 0, 40, 100])
        f = extract_track_features(candidates_near(TrackletIndex([closing, still]), 100), 20, 50, 100)
        rows = dict(zip([c.key for c in f.refs], f.candidates))
        speed = TRACK_CANDIDATE_FEATURE_NAMES.index("approach_speed")
        self.assertGreater(rows["1:1"][speed], 0.0)
        self.assertEqual(rows["1:2"][speed], 0.0)

    def test_detection_confidence_is_clamped_in_both_contracts(self) -> None:
        """RF-DETR's score is not a probability — measured max 3.79."""
        row = self._row([_tracklet(1, 1, [100], [0, 0, 40, 100], score=3.5)], 20, 50)
        self.assertEqual(row["score_at_event"], 1.0)
        self.assertEqual(row["score_median"], 1.0)

    def test_the_silhouette_separates_two_players_one_box_contains(self) -> None:
        """The box test says yes to both when they overlap; the outline knows
        which of the two bodies the pixel under the ball belongs to."""
        left = _tracklet(1, 1, [100], [0, 0, 40, 100])
        right = _tracklet(1, 2, [100], [20, 0, 60, 100])
        # Left fills its left half, right fills its right half, so the point
        # at x=30 is inside BOTH boxes but only on the right player.
        masks = {
            "1:1": np.tile(_half_mask(fill="left"), (1, 1, 1)),
            "1:2": np.tile(_half_mask(fill="right"), (1, 1, 1)),
        }
        features = extract_track_features(
            candidates_near(TrackletIndex([left, right]), 100, masks=masks), 45.0, 50.0, 100
        )
        rows = dict(zip([c.key for c in features.refs], features.candidates))
        in_box = TRACK_CANDIDATE_FEATURE_NAMES.index("contact_in_box")
        mask_d = TRACK_CANDIDATE_FEATURE_NAMES.index("mask_distance_height")
        self.assertEqual(rows["1:1"][in_box], rows["1:2"][in_box])
        self.assertEqual(rows["1:2"][mask_d], 0.0)
        self.assertGreater(rows["1:1"][mask_d], 0.0)

    def test_a_video_without_masks_falls_back_to_the_box(self) -> None:
        """Tracked before masks existed. The column keeps measuring the same
        thing, crudely — a sentinel would need a has_mask companion that is
        constant on every corpus tracked since, which this contract refuses."""
        row = self._row([_tracklet(1, 1, [100], [0, 0, 40, 100])], 200.0, 50.0)
        self.assertEqual(row["mask_distance_height"], row["center_distance_height"])

    def test_abstention_sees_the_best_candidate_s_own_distances(self) -> None:
        """The NONE head sees the nearest candidate's silhouette distance."""
        near = extract_track_features(
            candidates_near(TrackletIndex([_tracklet(1, 1, [100], [0, 0, 40, 100])]), 100), 20.0, 50.0, 100
        )
        far = extract_track_features(
            candidates_near(TrackletIndex([_tracklet(1, 1, [100], [0, 0, 40, 100])]), 100), 300.0, 50.0, 100
        )
        mask_d = TRACK_CONTEXT_FEATURE_NAMES.index("top_mask_distance")
        self.assertLess(near.context[mask_d], far.context[mask_d])

    def test_no_feature_is_constant_by_construction(self) -> None:
        """No presence sentinel should be born constant across new data."""
        self.assertNotIn("has_mask", TRACK_CANDIDATE_FEATURE_NAMES)
        names = TRACK_CANDIDATE_FEATURE_NAMES + TRACK_CONTEXT_FEATURE_NAMES
        self.assertFalse(any("wrist" in name for name in names))


if __name__ == "__main__":
    unittest.main()
