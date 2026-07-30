"""The geometric rule policy and its explainable diagnostic.

Detection answers "which people are visible"; this module answers "which
person plausibly performed the action at the annotated contact point" from
geometry alone. The human's answer to the same question is a label
(see actor/labels.py); this is the machine's.

This is the BASELINE, and only that. The learned path ranks TRACKLETS and
lives in actor/track_features.py — it shares no code with this file on
purpose. The wide box candidate set that used to sit here fed a learned box
ranker; both were retired together, because a candidate generator that keeps
everyone the detector supports looks exactly like a policy and decides
nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from yp_video.person.detector import PersonBox

RULE_BASED = "rule-based"

# ── Geometry policy ───────────────────────────────────────────────
# What counts as "close enough to have made this contact". These live here,
# not on the detector: they describe how an action relates to a body, which
# is this module's question. The detector's own floor
# (PERSON_SCORE_THRESHOLD) stays the detector's.

# The rule only trusts confident detections — the 0.1–0.5 band exists solely
# to give the human picker more boxes to click.
AUTO_PICK_MIN_SCORE = 0.5

# The contact point may sit above or beside the segmentation box: the ball is
# often just beyond an extended hand. These pads were validated on annotated
# sideline footage.
X_PAD_FRAC = 0.20
Y_ABOVE_FRAC = 0.35


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
    #: Distance from the contact point to the top-centre of the box, in body
    #: heights. Lower is more plausible.
    geometry_cost: float


@dataclass(frozen=True)
class AssociationDecision:
    """What the rule decided, in the shape the extraction records store.

    Rule-shaped by construction: it names a BOX. A learned policy answers with
    a tracklet and writes its own diagnostic (see policy.TrackletPolicy);
    letting one type carry both is what previously allowed a box-shaped answer
    to pass as an answer to the tracklet question.
    """

    version: str
    ranked: tuple[RankedActor, ...]
    selected: PersonBox | None
    reason: DecisionReason
    margin: float | None

    def diagnostic(self) -> dict:
        top = self.ranked[0] if self.ranked else None
        return {
            "version": self.version,
            "decision": self.reason.value,
            "candidate_count": len(self.ranked),
            "margin": round(self.margin, 4) if self.margin is not None else None,
            "top": (
                {
                    "box": [round(float(v), 1) for v in top.person.xyxy],
                    "cost": round(top.geometry_cost, 4),
                    "detection_score": round(float(top.person.score), 3),
                }
                if top is not None
                else None
            ),
        }


def rule_decision(
    boxes: list[PersonBox], x: float, y: float
) -> AssociationDecision:
    """The geometric policy production extraction runs on.

    Takes the best confident, geometrically compatible candidate and never
    abstains — its errors are visible as a wrong crop, which is what the
    labeling pages exist to correct.
    """
    ranked: list[RankedActor] = []
    for box in boxes:
        if box.score < AUTO_PICK_MIN_SCORE:
            continue
        x0, y0, x1, y1 = box.xyxy
        width = max(x1 - x0, 1.0)
        height = max(y1 - y0, 1.0)
        in_x = x0 - X_PAD_FRAC * width <= x <= x1 + X_PAD_FRAC * width
        in_y = y0 - Y_ABOVE_FRAC * height <= y <= y1
        if not (in_x and in_y):
            continue
        ranked.append(
            RankedActor(
                person=box,
                geometry_cost=float(np.hypot(x - (x0 + x1) / 2, y - y0))
                / height,
            )
        )
    ordered = tuple(
        sorted(ranked, key=lambda candidate: candidate.geometry_cost)
    )
    return AssociationDecision(
        version=RULE_BASED,
        ranked=ordered,
        selected=ordered[0].person if ordered else None,
        reason=(
            DecisionReason.SELECTED if ordered else DecisionReason.NO_CANDIDATE
        ),
        margin=(
            ordered[1].geometry_cost - ordered[0].geometry_cost
            if len(ordered) > 1
            else None
        ),
    )
