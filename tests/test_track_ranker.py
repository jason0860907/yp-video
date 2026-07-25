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
from yp_video.tracklets.geometry import TrackRef


def _tracklet(rally, track, frames, box, score=0.9):
    return {
        "rally_id": rally,
        "track_id": track,
        "frames": list(frames),
        "boxes": [list(box) for _ in frames],
        "scores": [score for _ in frames],
    }


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
        near = candidates_near(tracklets, 100)
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
        f = extract_track_features(candidates_near(tracklets, frame), x, y, frame, **kw)
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
        f = extract_track_features(candidates_near([closing, still], 100), 20, 50, 100)
        rows = dict(zip([c.key for c in f.refs], f.candidates))
        speed = TRACK_CANDIDATE_FEATURE_NAMES.index("approach_speed")
        self.assertGreater(rows["1:1"][speed], 0.0)
        self.assertEqual(rows["1:2"][speed], 0.0)

    def test_detection_confidence_is_clamped_in_both_contracts(self) -> None:
        """RF-DETR's score is not a probability — measured max 3.79."""
        row = self._row([_tracklet(1, 1, [100], [0, 0, 40, 100], score=3.5)], 20, 50)
        self.assertEqual(row["score_at_event"], 1.0)
        self.assertEqual(row["score_median"], 1.0)

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

    def test_an_old_checkpoint_fails_loudly(self) -> None:
        payload = _model(
            FEATURE_SET_BOX, len(CANDIDATE_FEATURE_NAMES), 9
        ).payload()
        payload["schema_version"] = 2
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            AssociationModel.from_payload(payload)


if __name__ == "__main__":
    unittest.main()
