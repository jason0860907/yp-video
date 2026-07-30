"""The association quality rates, defined once.

The trainer's thresholded scorer (train.py) and the policy evaluator
(evaluate.py) each used to carry their own spelling of these — including two
mathematically-equal but oppositely-phrased ``occluded_rejection_rate``s,
exactly the drift a shared definition exists to prevent.

Both scorers tally the same vocabulary:

    positive           events where a human named a tracklet
    occluded           events the human ruled occluded
    decided            positives the scorer committed an answer on
    correct            decided answers that matched the human's
    occluded_rejected  occluded events the scorer abstained on
"""

from __future__ import annotations

from collections.abc import Mapping


def ratio(count: Mapping[str, int], numerator: str, denominator: str) -> float | None:
    total = count[denominator]
    return count[numerator] / total if total else None


def association_rates(count: Mapping[str, int]) -> dict:
    return {
        "auto_coverage": ratio(count, "decided", "positive"),
        "selective_accuracy": ratio(count, "correct", "decided"),
        "occluded_rejection_rate": ratio(count, "occluded_rejected", "occluded"),
    }
