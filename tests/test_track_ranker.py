"""The tracklet feature contract, and what it deliberately does NOT inherit.

A tracklet is a different domain object from a box, so it gets its own
contract rather than an overloaded one — and the checkpoint now records
WHICH contract it was trained against, because the validation is by feature
name and a v2 checkpoint made no such statement.
"""

from __future__ import annotations

import unittest

import numpy as np

from yp_video.actor.features import CANDIDATE_FEATURE_NAMES
from yp_video.actor.model import (
    FEATURE_SET_BOX,
    FEATURE_SET_TRACK,
    MODEL_SCHEMA_VERSION,
    AssociationModel,
)
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


def _model(feature_set: str, n_candidate: int, n_context: int) -> AssociationModel:
    return AssociationModel(
        name="m",
        candidate_mean=np.zeros(n_candidate),
        candidate_scale=np.ones(n_candidate),
        context_mean=np.zeros(n_context),
        context_scale=np.ones(n_context),
        candidate_weights=np.zeros(n_candidate),
        none_weights=np.zeros(n_context),
        threshold=0.5,
        none_threshold=0.5,
        feature_set=feature_set,
    )


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
        """Occluded events are decided by how far the nearest player's HANDS
        are from the ball; a centre distance cannot say that, and the NONE
        head used to see nothing else."""
        near = extract_track_features(
            candidates_near(TrackletIndex([_tracklet(1, 1, [100], [0, 0, 40, 100])]), 100), 20.0, 50.0, 100
        )
        far = extract_track_features(
            candidates_near(TrackletIndex([_tracklet(1, 1, [100], [0, 0, 40, 100])]), 100), 300.0, 50.0, 100
        )
        mask_d = TRACK_CONTEXT_FEATURE_NAMES.index("top_mask_distance")
        wrist = TRACK_CONTEXT_FEATURE_NAMES.index("top_wrist_distance")
        self.assertLess(near.context[mask_d], far.context[mask_d])
        self.assertIn("top_wrist_distance", TRACK_CONTEXT_FEATURE_NAMES)
        self.assertEqual(near.context[wrist], 4.0)  # no detections given

    def test_no_feature_is_constant_by_construction(self) -> None:
        """`has_wrist` in the box contract had std 0 and a weight that could
        never move; nothing here may be born that way."""
        self.assertNotIn("has_wrist", TRACK_CANDIDATE_FEATURE_NAMES)


class ContractTests(unittest.TestCase):
    def test_a_checkpoint_declares_which_contract_it_learned(self) -> None:
        model = _model(
            FEATURE_SET_TRACK,
            len(TRACK_CANDIDATE_FEATURE_NAMES),
            len(TRACK_CONTEXT_FEATURE_NAMES),
        )
        payload = model.payload()
        self.assertEqual(payload["feature_set"], FEATURE_SET_TRACK)
        self.assertEqual(payload["schema_version"], MODEL_SCHEMA_VERSION)
        restored = AssociationModel.from_payload(payload)
        self.assertEqual(restored.feature_set, FEATURE_SET_TRACK)

    def test_a_track_model_cannot_be_loaded_as_a_box_model(self) -> None:
        """Both contracts are validated by NAME; without the declaration the
        two are indistinguishable and the weights would be applied to the
        wrong features."""
        payload = _model(
            FEATURE_SET_TRACK,
            len(TRACK_CANDIDATE_FEATURE_NAMES),
            len(TRACK_CONTEXT_FEATURE_NAMES),
        ).payload()
        payload["feature_set"] = FEATURE_SET_BOX
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            AssociationModel.from_payload(payload)

    def test_a_retired_contract_name_is_rejected_not_reinterpreted(self) -> None:
        """A checkpoint naming a contract that no longer exists must say so.
        The old ``track if ... else box`` fallback answered for any string, so
        a track-v1 model was validated against the BOX names and reported as a
        box mismatch — on a model that had never seen a box."""
        payload = _model(
            FEATURE_SET_TRACK,
            len(TRACK_CANDIDATE_FEATURE_NAMES),
            len(TRACK_CONTEXT_FEATURE_NAMES),
        ).payload()
        payload["feature_set"] = "track-v1"
        with self.assertRaisesRegex(ValueError, "Unknown association feature set"):
            AssociationModel.from_payload(payload)

    def test_an_old_checkpoint_fails_loudly(self) -> None:
        payload = _model(
            FEATURE_SET_BOX, len(CANDIDATE_FEATURE_NAMES), 9
        ).payload()
        payload["schema_version"] = 2
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            AssociationModel.from_payload(payload)


if __name__ == "__main__":
    unittest.main()
