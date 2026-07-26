"""Evaluate rule and learned actor-association policies on one label source."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

from yp_video.actor import checkpoints
from yp_video.actor.dataset import (
    AssociationDataset,
    AssociationExample,
    build_dataset,
)
from yp_video.actor.labels import ActorVerdict
from yp_video.actor.model import AssociationModel
from yp_video.actor.ranking import AssociationDecision
from yp_video.actor.review import iter_reviewed
from yp_video.tracklets.geometry import TrackRef


class _Metrics:
    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()

    def add(
        self,
        decision: AssociationDecision,
        example: AssociationExample,
    ) -> None:
        count = self.counts
        count["reviewed"] += 1
        selected_index = _person_index(decision.selected, example)
        top_index = (
            _person_index(decision.ranked[0].person, example)
            if decision.ranked
            else None
        )
        if example.target is None:
            count["occluded"] += 1
            if selected_index is None:
                count["occluded_rejected"] += 1
                count["correct"] += 1
            else:
                count["selected_occluded"] += 1
            return

        count["positive"] += 1
        count["candidate_hit"] += int(
            any(
                _person_index(candidate.person, example)
                == example.target
                for candidate in decision.ranked
            )
        )
        if top_index == example.target:
            count["top1"] += 1
        if selected_index is not None:
            count["selected"] += 1
            if selected_index == example.target:
                count["selected_correct"] += 1
                count["correct"] += 1

    def payload(self) -> dict:
        count = self.counts

        def ratio(numerator: str, denominator: str) -> float | None:
            total = count[denominator]
            return count[numerator] / total if total else None

        selected_total = count["selected"] + count["selected_occluded"]
        return {
            "reviewed": count["reviewed"],
            "positive": count["positive"],
            "occluded": count["occluded"],
            "candidate_recall": ratio("candidate_hit", "positive"),
            "top1_accuracy": ratio("top1", "positive"),
            "auto_coverage": ratio("selected", "positive"),
            "selective_accuracy": ratio(
                "selected_correct", "selected"
            ),
            "operational_precision": (
                count["selected_correct"] / selected_total
                if selected_total
                else None
            ),
            "occluded_rejection_rate": ratio(
                "occluded_rejected", "occluded"
            ),
            "overall_accuracy": ratio("correct", "reviewed"),
        }


def _person_index(person, example: AssociationExample) -> int | None:
    if person is None:
        return None
    return next(
        (
            index
            for index, candidate in enumerate(example.features.ranked)
            if np.allclose(
                np.asarray(candidate.person.xyxy),
                np.asarray(person.xyxy),
                atol=0.1,
                rtol=0.0,
            )
        ),
        None,
    )


def evaluate_dataset(
    dataset: AssociationDataset,
    learned_model: AssociationModel | None = None,
) -> dict:
    metrics: dict[str, _Metrics] = {
        "rule-based": _Metrics(),
    }
    if learned_model is not None:
        metrics[f"learned:{learned_model.name}"] = _Metrics()

    for example in dataset.examples:
        metrics["rule-based"].add(example.production, example)
        if learned_model is not None:
            metrics[f"learned:{learned_model.name}"].add(
                learned_model.decision(example.features),
                example,
            )

    return {
        "models": {
            name: metric.payload() for name, metric in metrics.items()
        },
        "labels": dataset.labels,
        "skipped": dataset.skipped,
        "dataset": dataset.payload(),
    }


def evaluate_association(
    stems: Sequence[str] | None = None,
    *,
    checkpoint: str | None = None,
) -> dict:
    dataset = build_dataset(stems)
    model = (
        checkpoints.load(checkpoint)
        if checkpoint is not None
        else checkpoints.load_active_shadow()
    )
    return evaluate_dataset(dataset, model)


# ── Policy-level evaluation ───────────────────────────────────────
# The metrics above score a DECISION against a box dataset, which only the
# rule and the retired box ranker speak. A policy answers with a tracklet, so
# it is scored on the question it was actually asked: of the players tracking
# found, did you name the one the human named.


class _PolicyScore:
    """Counts for one policy on one slice of the reviewed events."""

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()

    def add(self, event, track) -> None:
        count = self.counts
        count["reviewed"] += 1
        decided = track is not None
        if event.is_occluded:
            count["occluded"] += 1
            count["occluded_rejected"] += int(not decided)
            return
        if event.truth is None:
            # A verdict naming no tracklet — the legacy box labels. They are
            # not answerable in these terms and must not dilute the rate.
            count["unscorable"] += 1
            return
        count["positive"] += 1
        if decided:
            count["decided"] += 1
            count["correct"] += int(track == event.truth)

    def payload(self) -> dict:
        count = self.counts

        def ratio(numerator: str, denominator: str) -> float | None:
            total = count[denominator]
            return count[numerator] / total if total else None

        return {
            "reviewed": count["reviewed"],
            "positive": count["positive"],
            "occluded": count["occluded"],
            "unscorable": count["unscorable"],
            # Of the events with a knowable answer, how many did it get right
            # — abstentions included, because an abstention is a wrong answer
            # when somebody visibly acted.
            "top1_accuracy": ratio("correct", "positive"),
            "auto_coverage": ratio("decided", "positive"),
            "selective_accuracy": ratio("correct", "decided"),
            "occluded_rejection_rate": ratio("occluded_rejected", "occluded"),
        }


#: How much a predicted box must overlap a tracklet to count as naming it.
#: Matches extraction/links.PICK_IOU_MIN — the same question, asked of the
#: same data, should not have two answers.
TRACK_MATCH_IOU = 0.3


def as_track(pick, event):
    """The tracklet a pick names, resolving a BOX answer if that is what it is.

    The rule answers with a box and the pipeline resolves it to a tracklet
    downstream, so scoring the box as "no answer" would score the plumbing
    rather than the policy. Resolution is by overlap at the event frame, which
    is what the answer has to survive in production too.
    """
    from yp_video.person.detector import iou

    if pick.track is not None or pick.box is None:
        return pick.track
    frame = event.context.frame
    best, best_overlap = None, TRACK_MATCH_IOU
    for tracklet in event.context.tracklets:
        for index, at in enumerate(tracklet["frames"]):
            if at != frame:
                continue
            overlap = iou(list(pick.box), list(tracklet["boxes"][index]))
            if overlap >= best_overlap:
                best = TrackRef(tracklet["rally_id"], tracklet["track_id"])
                best_overlap = overlap
            break
    return best


def _contains_contact(box, contact) -> bool:
    from yp_video.actor.ranking import X_PAD_FRAC, Y_ABOVE_FRAC

    x, y = contact
    x0, y0, x1, y1 = (float(v) for v in box)
    width, height = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    return (
        x0 - X_PAD_FRAC * width <= x <= x1 + X_PAD_FRAC * width
        and y0 - Y_ABOVE_FRAC * height <= y <= y1
    )


def is_hard(event) -> bool:
    """More than one tracklet box contains the contact point.

    Pinned here so every arm is scored on the same definition. Note this
    counts TRACKLETS, one box per person; the raw detection list counts the
    same player two or three times and would give a quite different number
    for the same idea.
    """
    if event.context.contact is None:
        return False
    frame = event.context.frame
    hits = 0
    for tracklet in event.context.tracklets:
        for index, at in enumerate(tracklet["frames"]):
            if at != frame:
                continue
            if _contains_contact(tracklet["boxes"][index], event.context.contact):
                hits += 1
            break
    return hits > 1


#: The slices a result has to be read on. The aggregate is dominated by events
#: the rule already got right, so a policy can move it without touching a
#: single case the rule fails — which is the only thing worth moving.
SLICES = {
    "all": lambda event: True,
    "hard": is_hard,
    "manual": lambda event: event.label.verdict is ActorVerdict.MANUAL,
}


def evaluate_policies(
    policies: dict,
    stems: Sequence[str] | None = None,
) -> dict:
    """Score each named policy on the reviewed events, sliced three ways."""
    scores = {
        name: {slice_name: _PolicyScore() for slice_name in SLICES}
        for name in policies
    }
    for event in iter_reviewed(stems):
        member = {name: test(event) for name, test in SLICES.items()}
        for name, policy in policies.items():
            track = as_track(policy.decide(event.context), event)
            for slice_name, inside in member.items():
                if inside:
                    scores[name][slice_name].add(event, track)
    return {
        name: {
            slice_name: score.payload() for slice_name, score in by_slice.items()
        }
        for name, by_slice in scores.items()
    }
