"""Serializable linear listwise model for actor candidate + NONE ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

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


MODEL_SCHEMA_VERSION = 4

#: Which feature contract a checkpoint was trained against. Stated in the
#: payload rather than inferred: contracts get retired, and a checkpoint left
#: on disk from a retired one must fail to LOAD rather than be silently
#: validated against whichever contract happens to be current.
FEATURE_SET_TRACK = "track-v3"

#: The only contracts that exist, and the column names each one means. A
#: lookup and not a fallback: a fallback answers for any string at all, so a
#: checkpoint naming a retired contract would be validated against the live
#: names and the mismatch reported as a column problem on a model that never
#: had those columns.
FEATURE_SETS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    FEATURE_SET_TRACK: (
        TRACK_CANDIDATE_FEATURE_NAMES,
        TRACK_CONTEXT_FEATURE_NAMES,
    ),
}


def _names(feature_set: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    try:
        return FEATURE_SETS[feature_set]
    except KeyError:
        raise ValueError(
            f"Unknown association feature set {feature_set!r} "
            f"(have: {', '.join(sorted(FEATURE_SETS))})"
        ) from None


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
    feature_set: str = FEATURE_SET_TRACK

    def probabilities(self, features: FeatureVectors) -> tuple[np.ndarray, float]:
        """Scores from the two blocks, and nothing else.

        Deliberately numeric all the way through: it reads the two matrices
        and never the candidates they describe, so a new feature contract is
        a new column list rather than a new model class. What the winning
        INDEX refers to is the caller's business — see policy.TrackletPolicy.
        """
        candidate_matrix = (
            features.candidates - self.candidate_mean
        ) / self.candidate_scale
        context = (features.context - self.context_mean) / self.context_scale
        candidate_logits = candidate_matrix @ self.candidate_weights
        none_logit = float(context @ self.none_weights)
        return _softmax(candidate_logits), _sigmoid(none_logit)

    def payload(self) -> dict:
        return {
            "schema_version": MODEL_SCHEMA_VERSION,
            "type": "actor-association-linear-softmax",
            "name": self.name,
            "feature_set": self.feature_set,
            "feature_contract": {
                "candidate": list(_names(self.feature_set)[0]),
                "context": list(_names(self.feature_set)[1]),
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
        feature_set = str(payload.get("feature_set") or FEATURE_SET_TRACK)
        expected_candidate, expected_context = _names(feature_set)
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
