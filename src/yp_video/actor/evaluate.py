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
from yp_video.actor.model import AssociationModel
from yp_video.actor.ranking import AssociationDecision


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
