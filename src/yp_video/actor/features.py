"""Stable feature contract for learned actor association.

The detector and rule policy own raw geometry.  This module is the only
adapter that turns that domain state into numeric model input, so training
and inference cannot silently drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from yp_video.actor.ranking import (
    CandidateSource,
    RankedActor,
    rank_candidates,
)
from yp_video.person.detector import PersonBox

CANDIDATE_FEATURE_NAMES = (
    "detection_score",
    "geometry_cost",
    "contact_dx_width",
    "contact_dy_height",
    "center_distance_height",
    "source_box",
    "source_other",
    "legacy_eligible",
    "rank_reciprocal",
)

CONTEXT_FEATURE_NAMES = (
    "bias",
    "log_candidate_count",
    "top_geometry_cost",
    "top_detection_score",
    "top_two_margin",
    "box_candidate_fraction",
    "legacy_candidate_fraction",
)


@dataclass(frozen=True)
class AssociationFeatures:
    """One variable-length candidate set and one fixed-size NONE context."""

    ranked: tuple[RankedActor, ...]
    candidates: np.ndarray
    context: np.ndarray


def _candidate_row(
    candidate: RankedActor,
    x: float,
    y: float,
    rank: int,
) -> list[float]:
    person = candidate.person
    x0, y0, x1, y1 = person.xyxy
    width = max(float(x1 - x0), 1.0)
    height = max(float(y1 - y0), 1.0)
    center_x = (x0 + x1) / 2
    center_y = (y0 + y1) / 2
    return [
        # Clamped: RF-DETR's confidence is not a probability (max 3.79 on the
        # corpus, 13% above 1.0) and this was the only feature with no bound.
        min(max(float(person.score), 0.0), 1.0),
        min(float(candidate.geometry_cost), 8.0),
        min(abs(float(x - center_x)) / width, 4.0),
        min(float(y - y0) / height, 4.0),
        min(float(np.hypot(x - center_x, y - center_y)) / height, 6.0),
        float(candidate.source is CandidateSource.BOX),
        float(candidate.source is CandidateSource.OTHER),
        float(
            person.score >= 0.5
            and candidate.source is not CandidateSource.OTHER
        ),
        1.0 / (rank + 1.0),
    ]


def extract_features(
    people: list[PersonBox],
    x: float,
    y: float,
) -> AssociationFeatures:
    """Extract the versioned numeric contract shared by train and predict."""
    ranked = rank_candidates(people, x, y)
    candidate_rows = [
        _candidate_row(candidate, x, y, rank)
        for rank, candidate in enumerate(ranked)
    ]
    candidates = np.asarray(candidate_rows, dtype=np.float64).reshape(
        len(candidate_rows),
        len(CANDIDATE_FEATURE_NAMES),
    )
    count = len(ranked)
    if count:
        top = ranked[0]
        top_cost = min(float(top.geometry_cost), 8.0)
        top_score = float(top.person.score)
        margin = (
            min(float(ranked[1].cost - top.cost), 8.0)
            if count > 1
            else 8.0
        )
        box_fraction = (
            sum(
                candidate.source is CandidateSource.BOX
                for candidate in ranked
            )
            / count
        )
        legacy_fraction = (
            sum(
                candidate.person.score >= 0.5
                and candidate.source is not CandidateSource.OTHER
                for candidate in ranked
            )
            / count
        )
    else:
        top_cost = 8.0
        top_score = 0.0
        margin = 0.0
        box_fraction = 0.0
        legacy_fraction = 0.0
    context = np.asarray(
        [
            1.0,
            float(np.log1p(count)),
            top_cost,
            top_score,
            margin,
            box_fraction,
            legacy_fraction,
        ],
        dtype=np.float64,
    )
    return AssociationFeatures(ranked, candidates, context)
