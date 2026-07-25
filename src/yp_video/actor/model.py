"""Serializable linear listwise model for actor candidate + NONE ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from yp_video.actor.ranking import (
    AssociationDecision,
    DecisionReason,
    RankedActor,
)
from yp_video.actor.features import (
    CANDIDATE_FEATURE_NAMES,
    CONTEXT_FEATURE_NAMES,
    AssociationFeatures,
)
from yp_video.actor.track_features import (
    TRACK_CANDIDATE_FEATURE_NAMES,
    TRACK_CONTEXT_FEATURE_NAMES,
)

#: Anything carrying the two numeric blocks. The contract a model was trained
#: against is recorded in ``feature_set``, not enforced by this type.
class FeatureVectors(Protocol):
    # Read-only: the feature dataclasses are frozen, and a mutable protocol
    # attribute would not match them.
    @property
    def candidates(self) -> np.ndarray: ...
    @property
    def context(self) -> np.ndarray: ...


MODEL_SCHEMA_VERSION = 3

#: Which feature contract a checkpoint was trained against. The contract is
#: validated by NAME, and a v2 checkpoint carried no statement of WHICH list
#: those names came from — so a loader could not tell a box model from a
#: tracklet one. Now it must say.
FEATURE_SET_BOX = "box-v2"
FEATURE_SET_TRACK = "track-v1"


def _candidate_names(feature_set: str) -> tuple[str, ...]:
    return (
        TRACK_CANDIDATE_FEATURE_NAMES
        if feature_set == FEATURE_SET_TRACK
        else CANDIDATE_FEATURE_NAMES
    )


def _context_names(feature_set: str) -> tuple[str, ...]:
    return (
        TRACK_CONTEXT_FEATURE_NAMES
        if feature_set == FEATURE_SET_TRACK
        else CONTEXT_FEATURE_NAMES
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    if not len(logits):
        return np.empty(0, dtype=np.float64)
    shifted = logits - np.max(logits)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum()


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + float(np.exp(-value)))
    exponential = float(np.exp(value))
    return exponential / (1.0 + exponential)


@dataclass(frozen=True)
class AssociationModel:
    """One listwise scorer. Training candidates never activate implicitly."""

    name: str
    candidate_mean: np.ndarray
    candidate_scale: np.ndarray
    context_mean: np.ndarray
    context_scale: np.ndarray
    candidate_weights: np.ndarray
    none_weights: np.ndarray
    threshold: float
    none_threshold: float
    feature_set: str = FEATURE_SET_BOX

    def probabilities(self, features: FeatureVectors) -> tuple[np.ndarray, float]:
        """Scores from the two blocks. Either contract fits — this reads only
        the matrices, which is what lets one model class serve both."""
        candidate_matrix = (
            features.candidates - self.candidate_mean
        ) / self.candidate_scale
        context = (features.context - self.context_mean) / self.context_scale
        candidate_logits = candidate_matrix @ self.candidate_weights
        none_logit = float(context @ self.none_weights)
        return _softmax(candidate_logits), _sigmoid(none_logit)

    def decision(self, features: AssociationFeatures) -> AssociationDecision:
        candidate_probabilities, none_probability = self.probabilities(
            features
        )
        if not features.ranked:
            return AssociationDecision(
                version=f"learned:{self.name}",
                ranked=(),
                selected=None,
                reason=DecisionReason.NO_CANDIDATE,
                margin=None,
                confidence=None,
                none_probability=none_probability,
            )

        order = np.argsort(-candidate_probabilities)
        ranked = tuple(
            RankedActor(
                person=features.ranked[int(index)].person,
                source=features.ranked[int(index)].source,
                geometry_cost=1.0
                - float(candidate_probabilities[int(index)]),
                detection_penalty=0.0,
            )
            for index in order
        )
        top_index = int(order[0])
        confidence = float(candidate_probabilities[top_index])
        competitor = max(
            none_probability,
            (
                float(candidate_probabilities[int(order[1])])
                if len(order) > 1
                else 0.0
            ),
        )
        selected = (
            features.ranked[top_index].person
            if confidence >= self.threshold
            and none_probability < self.none_threshold
            else None
        )
        return AssociationDecision(
            version=f"learned:{self.name}",
            ranked=ranked,
            selected=selected,
            reason=(
                DecisionReason.SELECTED
                if selected is not None
                else DecisionReason.AMBIGUOUS
            ),
            margin=confidence - competitor,
            confidence=confidence,
            none_probability=none_probability,
        )

    def payload(self) -> dict:
        return {
            "schema_version": MODEL_SCHEMA_VERSION,
            "type": "actor-association-linear-softmax",
            "name": self.name,
            "feature_set": self.feature_set,
            "feature_contract": {
                "candidate": list(_candidate_names(self.feature_set)),
                "context": list(_context_names(self.feature_set)),
            },
            "candidate_mean": self.candidate_mean.tolist(),
            "candidate_scale": self.candidate_scale.tolist(),
            "context_mean": self.context_mean.tolist(),
            "context_scale": self.context_scale.tolist(),
            "candidate_weights": self.candidate_weights.tolist(),
            "none_weights": self.none_weights.tolist(),
            "threshold": self.threshold,
            "none_threshold": self.none_threshold,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "AssociationModel":
        if payload.get("schema_version") != MODEL_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported association model schema: "
                f"{payload.get('schema_version')!r}"
            )
        feature_set = str(payload.get("feature_set") or FEATURE_SET_BOX)
        expected_candidate = _candidate_names(feature_set)
        expected_context = _context_names(feature_set)
        contract = payload.get("feature_contract") or {}
        if tuple(contract.get("candidate") or ()) != expected_candidate:
            raise ValueError(
                f"Association candidate feature contract mismatch for {feature_set}"
            )
        if tuple(contract.get("context") or ()) != expected_context:
            raise ValueError(
                f"Association context feature contract mismatch for {feature_set}"
            )
        model = cls(
            name=str(payload["name"]),
            candidate_mean=np.asarray(
                payload["candidate_mean"], dtype=np.float64
            ),
            candidate_scale=np.asarray(
                payload["candidate_scale"], dtype=np.float64
            ),
            context_mean=np.asarray(
                payload["context_mean"], dtype=np.float64
            ),
            context_scale=np.asarray(
                payload["context_scale"], dtype=np.float64
            ),
            candidate_weights=np.asarray(
                payload["candidate_weights"], dtype=np.float64
            ),
            none_weights=np.asarray(
                payload["none_weights"], dtype=np.float64
            ),
            threshold=float(payload["threshold"]),
            none_threshold=float(payload["none_threshold"]),
            feature_set=feature_set,
        )
        if model.candidate_weights.shape != (len(expected_candidate),):
            raise ValueError("Invalid association candidate weight shape")
        if model.none_weights.shape != (len(expected_context),):
            raise ValueError("Invalid association NONE weight shape")
        cshape, xshape = (len(expected_candidate),), (len(expected_context),)
        if (
            model.candidate_mean.shape != cshape
            or model.candidate_scale.shape != cshape
            or model.context_mean.shape != xshape
            or model.context_scale.shape != xshape
        ):
            raise ValueError("Invalid association scaler shape")
        arrays = (
            model.candidate_mean,
            model.candidate_scale,
            model.context_mean,
            model.context_scale,
            model.candidate_weights,
            model.none_weights,
        )
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("Association model contains non-finite values")
        if (model.candidate_scale <= 0).any() or (
            model.context_scale <= 0
        ).any():
            raise ValueError("Association model scale must be positive")
        if not 0 <= model.threshold <= 1 or not (
            0 <= model.none_threshold <= 1
        ):
            raise ValueError("Association thresholds must be probabilities")
        return model
