"""Candidate ranking, the rule policy, and explainable diagnostics.

Detection answers "which people are visible"; this module answers "which
person plausibly performed the action at the annotated contact point" from
geometry alone. The human's answer to the same question is a label
(see actor/labels.py); this is the machine's.

Two separate things live here, and conflating them was a mistake worth not
repeating:

- ``rule_decision`` is the POLICY production runs on. It picks the best
  confident, geometrically compatible candidate.
- ``rank_candidates`` is the CANDIDATE SET a learned ranker chooses from. It
  keeps everyone the detector supports so the truth is always in the list.

They were once two competing "rules" (V1 and a V2 that also abstained), which
made the candidate generator look like a policy nobody had adopted.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from yp_video.person.detector import (
    PERSON_SCORE_THRESHOLD,
    WRIST_IDXS,
    PersonBox,
)

RULE_BASED = "rule-based"

# ── Geometry policy ───────────────────────────────────────────────
# What counts as "close enough to have made this contact". These live here,
# not on the detector: they describe how an action relates to a body, which
# is this module's question. The detector's own floor
# (PERSON_SCORE_THRESHOLD) stays the detector's.

# The rule only trusts confident detections — the 0.1–0.5 band exists solely
# to give the human picker more boxes to click.
AUTO_PICK_MIN_SCORE = 0.5

# A wrist match counts when the contact point is within this fraction of the
# person's box height from the wrist — roughly ball-diameter reach at contact.
WRIST_REACH_FRAC = 0.6

# Box-geometry fallback for people whose wrists weren't found: the contact
# point may sit up to 35% of box height above the top (ball above the raised
# hand) and 20% of box width outside the horizontal span. Validated on
# annotated sideline footage.
X_PAD_FRAC = 0.20
Y_ABOVE_FRAC = 0.35
# Fallback candidates always rank below any wrist match.
FALLBACK_PENALTY = 10.0

# Detection confidence influences the CANDIDATE ordering without excluding a
# plausible actor. Geometry remains dominant: a wrist-compatible candidate
# always outranks a box fallback, matching the physical meaning of a contact.
DETECTION_PENALTY_WEIGHT = 0.25


class CandidateSource(str, Enum):
    WRIST = "wrist"
    BOX = "box"
    OTHER = "other"


class DecisionReason(str, Enum):
    SELECTED = "selected"
    #: Candidates existed and none was confident enough to pick.
    AMBIGUOUS = "ambiguous"
    NO_CANDIDATE = "no_candidate"
    #: The policy looked and answered "nobody". Distinct from AMBIGUOUS: a
    #: model that abstains on purpose is not the same event as one that could
    #: not separate two players, and the diagnostics are read to tell them
    #: apart.
    ABSTAINED = "abstained"


@dataclass(frozen=True)
class RankedActor:
    person: PersonBox
    source: CandidateSource
    geometry_cost: float
    detection_penalty: float

    @property
    def cost(self) -> float:
        """Lower is more plausible."""
        return self.geometry_cost + self.detection_penalty


@dataclass(frozen=True)
class AssociationDecision:
    version: str
    ranked: tuple[RankedActor, ...]
    selected: PersonBox | None
    reason: DecisionReason
    margin: float | None
    confidence: float | None = None
    none_probability: float | None = None

    def diagnostic(self) -> dict:
        top = self.ranked[0] if self.ranked else None
        return {
            "version": self.version,
            "decision": self.reason.value,
            "candidate_count": len(self.ranked),
            "margin": round(self.margin, 4) if self.margin is not None else None,
            "confidence": (
                round(self.confidence, 4)
                if self.confidence is not None
                else None
            ),
            "none_probability": (
                round(self.none_probability, 4)
                if self.none_probability is not None
                else None
            ),
            "top": (
                {
                    "box": [round(float(v), 1) for v in top.person.xyxy],
                    "cost": round(top.cost, 4),
                    "source": top.source.value,
                    "detection_score": round(float(top.person.score), 3),
                }
                if top is not None
                else None
            ),
        }


def _rank(
    boxes: list[PersonBox],
    x: float,
    y: float,
    *,
    min_detection_score: float,
    detection_penalty_weight: float,
    keep_incompatible: bool,
) -> tuple[RankedActor, ...]:
    ranked: list[RankedActor] = []
    for box in boxes:
        if box.score < min_detection_score:
            continue
        x0, y0, x1, y1 = box.xyxy
        width = max(x1 - x0, 1.0)
        height = max(y1 - y0, 1.0)

        wrist_distance = None
        if box.keypoints is not None:
            wrist_distance = min(
                float(
                    np.hypot(
                        box.keypoints[index][0] - x,
                        box.keypoints[index][1] - y,
                    )
                )
                for index in WRIST_IDXS
            )

        if (
            wrist_distance is not None
            and wrist_distance <= WRIST_REACH_FRAC * height
        ):
            source = CandidateSource.WRIST
            geometry_cost = wrist_distance / height
        else:
            in_x = x0 - X_PAD_FRAC * width <= x <= x1 + X_PAD_FRAC * width
            in_y = y0 - Y_ABOVE_FRAC * height <= y <= y1
            if in_x and in_y:
                source = CandidateSource.BOX
                geometry_cost = (
                    float(np.hypot(x - (x0 + x1) / 2, y - y0))
                    / height
                    + FALLBACK_PENALTY
                )
            elif keep_incompatible:
                # Still a training/ranking candidate. Geometry becomes a
                # strong negative feature, never a hard exclusion.
                source = CandidateSource.OTHER
                geometry_cost = (
                    float(
                        np.hypot(
                            x - (x0 + x1) / 2,
                            y - (y0 + y1) / 2,
                        )
                    )
                    / height
                    + 2 * FALLBACK_PENALTY
                )
            else:
                continue

        ranked.append(
            RankedActor(
                person=box,
                source=source,
                geometry_cost=geometry_cost,
                detection_penalty=detection_penalty_weight
                * (1.0 - min(max(float(box.score), 0.0), 1.0)),
            )
        )
    return tuple(sorted(ranked, key=lambda candidate: candidate.cost))


def rule_decision(
    boxes: list[PersonBox], x: float, y: float
) -> AssociationDecision:
    """The geometric policy production extraction runs on.

    Takes the best confident, geometrically compatible candidate and never
    abstains — its errors are visible as a wrong crop, which is what the
    labeling pages exist to correct.
    """
    ranked = _rank(
        boxes,
        x,
        y,
        min_detection_score=AUTO_PICK_MIN_SCORE,
        detection_penalty_weight=0.0,
        keep_incompatible=False,
    )
    return AssociationDecision(
        version=RULE_BASED,
        ranked=ranked,
        selected=ranked[0].person if ranked else None,
        reason=(
            DecisionReason.SELECTED
            if ranked
            else DecisionReason.NO_CANDIDATE
        ),
        margin=(
            ranked[1].cost - ranked[0].cost if len(ranked) > 1 else None
        ),
    )


def rank_candidates(
    boxes: list[PersonBox], x: float, y: float
) -> tuple[RankedActor, ...]:
    """Every plausible actor, ranked — the learned ranker's candidate set.

    Not a policy: it decides nothing and rejects nobody above the detector's
    own floor. Geometry becomes a strong negative feature instead of a hard
    gate, so a labeled truth is always somewhere in the list and candidate
    recall stays 1.0. What to DO with the ranking is the model's business.
    """
    return _rank(
        boxes,
        x,
        y,
        min_detection_score=PERSON_SCORE_THRESHOLD,
        detection_penalty_weight=DETECTION_PENALTY_WEIGHT,
        keep_incompatible=True,
    )
