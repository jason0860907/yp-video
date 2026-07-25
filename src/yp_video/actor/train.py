"""Grouped out-of-fold training for the learned association shadow model."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import minimize

from yp_video.actor import checkpoints
from yp_video.actor.dataset import (
    AssociationDataset,
    AssociationExample,
    TrackDataset,
    TrackExample,
)
from yp_video.actor.model import FEATURE_SET_BOX, AssociationModel


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    folds: int = 5
    l2: float = 0.05
    max_iterations: int = 300
    target_precision: float = 0.9
    min_occluded_rejection: float = 0.5


@dataclass(frozen=True)
class _Prediction:
    example: AssociationExample | TrackExample
    candidate_probabilities: np.ndarray
    none_probability: float


def _scaler(
    examples: tuple[AssociationExample | TrackExample, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    candidate_rows = [
        row
        for example in examples
        for row in example.features.candidates
    ]
    if not candidate_rows:
        raise ValueError("Association training needs candidate detections")
    candidates = np.vstack(candidate_rows)
    contexts = np.vstack(
        [example.features.context for example in examples]
    )
    candidate_mean = candidates.mean(axis=0)
    candidate_scale = candidates.std(axis=0)
    context_mean = contexts.mean(axis=0)
    context_scale = contexts.std(axis=0)
    candidate_scale[candidate_scale < 1e-6] = 1.0
    context_scale[context_scale < 1e-6] = 1.0
    # Preserve the NONE classifier's explicit intercept column.
    context_mean[0], context_scale[0] = 0.0, 1.0
    return candidate_mean, candidate_scale, context_mean, context_scale


def _fit(
    name: str,
    examples: tuple[AssociationExample | TrackExample, ...],
    config: TrainingConfig,
    feature_set: str = FEATURE_SET_BOX,
) -> AssociationModel:
    (
        candidate_mean,
        candidate_scale,
        context_mean,
        context_scale,
    ) = _scaler(examples)
    candidate_size = len(candidate_mean)
    context_size = len(context_mean)

    transformed = [
        (
            (example.features.candidates - candidate_mean)
            / candidate_scale,
            (example.features.context - context_mean) / context_scale,
            example.target,
        )
        for example in examples
    ]

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        candidate_weights = parameters[:candidate_size]
        none_weights = parameters[candidate_size:]
        candidate_loss = 0.0
        none_loss = 0.0
        positive_count = 0
        candidate_gradient = np.zeros_like(candidate_weights)
        none_gradient = np.zeros_like(none_weights)
        for candidates, context, target in transformed:
            if target is not None:
                positive_count += 1
                logits = candidates @ candidate_weights
                logits -= np.max(logits)
                probabilities = np.exp(logits)
                probabilities /= probabilities.sum()
                candidate_loss -= float(
                    np.log(max(probabilities[target], 1e-12))
                )
                probabilities[target] -= 1.0
                candidate_gradient += candidates.T @ probabilities

            none_logit = float(context @ none_weights)
            none_target = float(target is None)
            none_loss += float(np.logaddexp(0.0, none_logit)) - (
                none_target * none_logit
            )
            none_probability = (
                1.0 / (1.0 + float(np.exp(-none_logit)))
                if none_logit >= 0
                else float(np.exp(none_logit))
                / (1.0 + float(np.exp(none_logit)))
            )
            none_gradient += context * (
                none_probability - none_target
            )

        count = max(len(transformed), 1)
        positive_count = max(positive_count, 1)
        regularized = parameters.copy()
        regularized[candidate_size] = 0.0
        loss = (
            candidate_loss / positive_count
            + none_loss / count
            + 0.5 * config.l2 * float(
                regularized @ regularized
            )
        )
        gradient = np.concatenate(
            (
                candidate_gradient / positive_count,
                none_gradient / count,
            )
        ) + config.l2 * regularized
        return loss, gradient

    result = minimize(
        objective,
        np.zeros(candidate_size + context_size, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": config.max_iterations},
    )
    if not result.success:
        raise RuntimeError(
            f"Association optimizer failed: {result.message}"
        )
    return AssociationModel(
        name=name,
        candidate_mean=candidate_mean,
        candidate_scale=candidate_scale,
        context_mean=context_mean,
        context_scale=context_scale,
        candidate_weights=result.x[:candidate_size],
        none_weights=result.x[candidate_size:],
        threshold=0.5,
        none_threshold=0.5,
        feature_set=feature_set,
    )


def _predict(
    model: AssociationModel,
    examples: tuple[AssociationExample | TrackExample, ...],
) -> list[_Prediction]:
    predictions: list[_Prediction] = []
    for example in examples:
        candidate_probabilities, none_probability = model.probabilities(
            example.features
        )
        predictions.append(
            _Prediction(
                example=example,
                candidate_probabilities=candidate_probabilities,
                none_probability=none_probability,
            )
        )
    return predictions


def _metrics(
    predictions: list[_Prediction],
    threshold: float,
    none_threshold: float,
) -> dict:
    counts = {
        "reviewed": len(predictions),
        "positive": 0,
        "occluded": 0,
        "top1": 0,
        "selected_positive": 0,
        "selected_correct": 0,
        "selected_occluded": 0,
        "correct": 0,
    }
    for prediction in predictions:
        example = prediction.example
        probabilities = prediction.candidate_probabilities
        top = int(np.argmax(probabilities)) if len(probabilities) else None
        confidence = (
            float(probabilities[top]) if top is not None else 0.0
        )
        selected = (
            top
            if top is not None
            and confidence >= threshold
            and prediction.none_probability < none_threshold
            else None
        )
        if example.target is None:
            counts["occluded"] += 1
            if selected is None:
                counts["correct"] += 1
            else:
                counts["selected_occluded"] += 1
            continue

        counts["positive"] += 1
        if top == example.target:
            counts["top1"] += 1
        if selected is not None:
            counts["selected_positive"] += 1
            if selected == example.target:
                counts["selected_correct"] += 1
                counts["correct"] += 1

    def ratio(numerator: str, denominator: str) -> float | None:
        value = counts[denominator]
        return counts[numerator] / value if value else None

    selected_total = (
        counts["selected_positive"] + counts["selected_occluded"]
    )
    return {
        "reviewed": counts["reviewed"],
        "positive": counts["positive"],
        "occluded": counts["occluded"],
        "candidate_recall": 1.0 if counts["positive"] else None,
        "top1_accuracy": ratio("top1", "positive"),
        "auto_coverage": ratio("selected_positive", "positive"),
        "selective_accuracy": ratio(
            "selected_correct", "selected_positive"
        ),
        "operational_precision": (
            counts["selected_correct"] / selected_total
            if selected_total
            else None
        ),
        "occluded_rejection_rate": (
            1.0
            - counts["selected_occluded"] / counts["occluded"]
            if counts["occluded"]
            else None
        ),
        "overall_accuracy": ratio("correct", "reviewed"),
        "threshold": threshold,
        "none_threshold": none_threshold,
    }


def _calibrate(
    predictions: list[_Prediction],
    config: TrainingConfig,
) -> tuple[float, float, dict]:
    confidences = sorted(
        {
            float(np.max(prediction.candidate_probabilities))
            for prediction in predictions
            if len(prediction.candidate_probabilities)
        }
    )
    thresholds = sorted(
        set(np.linspace(0.05, 0.99, 95).tolist() + confidences)
    )
    none_thresholds = np.linspace(0.05, 0.95, 19).tolist()
    scored = [
        (
            threshold,
            none_threshold,
            _metrics(predictions, threshold, none_threshold),
        )
        for threshold in thresholds
        for none_threshold in none_thresholds
    ]
    eligible = [
        item
        for item in scored
        if (item[2]["selective_accuracy"] or 0.0)
        >= config.target_precision
        and (item[2]["occluded_rejection_rate"] or 0.0)
        >= config.min_occluded_rejection
        and (item[2]["auto_coverage"] or 0.0) > 0
    ]
    pool = eligible or [
        item
        for item in scored
        if (item[2]["auto_coverage"] or 0.0) > 0
    ]
    if not pool:
        return 1.0, 0.0, _metrics(predictions, 1.0, 0.0)
    threshold, none_threshold, metrics = max(
        pool,
        key=lambda item: (
            item[2]["auto_coverage"] or 0.0,
            item[2]["operational_precision"] or 0.0,
        ),
    )
    return float(threshold), float(none_threshold), metrics


def _folds(
    examples: tuple[AssociationExample | TrackExample, ...],
    config: TrainingConfig,
) -> list[tuple[AssociationExample | TrackExample, ...]]:
    stems = sorted({example.stem for example in examples})
    if len(stems) < 2:
        raise ValueError(
            "Association training needs reviews from at least two videos "
            "for grouped validation"
        )
    random.Random(config.seed).shuffle(stems)
    count = min(config.folds, len(stems))
    fold_stems = [set(stems[index::count]) for index in range(count)]
    return [
        tuple(
            example
            for example in examples
            if example.stem in validation_stems
        )
        for validation_stems in fold_stems
    ]


def train_candidate(
    dataset: AssociationDataset | TrackDataset,
    name: str,
    *,
    config: TrainingConfig | None = None,
    feature_set: str = FEATURE_SET_BOX,
) -> dict:
    """Train with grouped OOF calibration, then fit the saved model on all."""
    resolved_config = config or TrainingConfig()
    examples = dataset.examples
    if len(examples) < 20:
        raise ValueError("Association training needs at least 20 reviews")
    validation_folds = _folds(examples, resolved_config)
    predictions: list[_Prediction] = []
    predictions_by_fold: list[list[_Prediction]] = []
    fold_payload: list[dict] = []
    for index, validation in enumerate(validation_folds):
        validation_ids = {id(example) for example in validation}
        training = tuple(
            example
            for example in examples
            if id(example) not in validation_ids
        )
        fold_model = _fit(
            f"{name}-fold-{index}",
            training,
            resolved_config,
            feature_set,
        )
        fold_predictions = _predict(fold_model, validation)
        predictions.extend(fold_predictions)
        predictions_by_fold.append(fold_predictions)
        fold_payload.append(
            {
                "fold": index,
                "train_examples": len(training),
                "validation_examples": len(validation),
                "validation_stems": sorted(
                    {example.stem for example in validation}
                ),
            }
        )

    threshold, none_threshold, metrics = _calibrate(
        predictions, resolved_config
    )
    fold_payload = [
        {
            **payload,
            "metrics": _metrics(
                fold_predictions,
                threshold,
                none_threshold,
            ),
        }
        for payload, fold_predictions in zip(
            fold_payload, predictions_by_fold, strict=True
        )
    ]
    final_model = replace(
        _fit(name, examples, resolved_config, feature_set),
        threshold=threshold,
        none_threshold=none_threshold,
    )
    manifest = {
        "type": "actor-association-checkpoint",
        "created_at": time.time(),
        "name": name,
        "feature_set": feature_set,
        "training": {
            "examples": len(examples),
            "stems": list(dataset.stems),
            "labels": dataset.labels,
            "skipped": dataset.skipped,
            "config": {
                "seed": resolved_config.seed,
                "folds": len(validation_folds),
                "l2": resolved_config.l2,
                "max_iterations": resolved_config.max_iterations,
                "target_precision": resolved_config.target_precision,
                "min_occluded_rejection": (
                    resolved_config.min_occluded_rejection
                ),
            },
            "folds": fold_payload,
        },
        "metrics": {"grouped_oof": metrics},
        "threshold": threshold,
        "none_threshold": none_threshold,
        "activation": "candidate",
    }
    root = checkpoints.save_candidate(
        final_model,
        manifest,
    )
    return {**manifest, "path": str(root)}
