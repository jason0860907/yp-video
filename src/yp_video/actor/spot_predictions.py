"""What the yp-spot actor head decided, per event.

yp-spot lives in another repo and another venv, reached across a subprocess
boundary, so its answers arrive as a file rather than a function call. This
module is the only reader of that file, and it hands ``SpotActorPolicy`` a
plain lookup so the policy stays a decision and not an IO layer.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from yp_video.config import ASSOCIATION_DIR
from yp_video.core.cache import StatCache
from yp_video.tracklets.geometry import TrackRef

ACTOR_PREDICTIONS_DIR = ASSOCIATION_DIR / "spot"

#: What to call answers written before the file recorded its author. They are
#: readable and scorable; only their provenance is lost, and saying so beats
#: inventing a checkpoint name they might not have come from.
UNRECORDED_RUN = "(unrecorded)"

_cache: StatCache = StatCache()


def predictions_path(stem: str) -> Path:
    return ACTOR_PREDICTIONS_DIR / f"{stem}_actor_predictions.json"


@dataclass(frozen=True)
class SpotPredictions:
    """One video's answers, and which head gave them."""

    run: str
    answers: dict[str, "SpotAnswer"]


@dataclass(frozen=True)
class SpotAnswer:
    """One event's answer, in the head's own terms."""

    track: TrackRef | None
    confidence: float
    #: Which of the three the softmax preferred: a candidate, `occluded`, or
    #: `untracked`. Kept because they abstain identically but mean different
    #: things — `untracked` says go fix tracking, not go relabel.
    kind: str


def _parse(payload: dict) -> SpotPredictions:
    answers: dict[str, SpotAnswer] = {}
    for row in payload.get("events", []):
        event_id = str(row.get("id"))
        track = row.get("track")
        answers[event_id] = SpotAnswer(
            track=TrackRef.parse(track) if track else None,
            confidence=float(row.get("confidence") or 0.0),
            kind=str(row.get("kind") or "untracked"),
        )
    return SpotPredictions(
        run=str(payload.get("checkpoint") or UNRECORDED_RUN),
        answers=answers,
    )


def read(stem: str) -> SpotPredictions | None:
    """This video's answers and their author, or None when it has none."""
    path = predictions_path(stem)
    if not path.exists():
        return None
    return _cache.get(
        stem,
        [path],
        lambda: _parse(json.loads(path.read_text(encoding="utf-8"))),
    )


def load(stem: str) -> dict[str, SpotAnswer]:
    """This video's answers, or an empty mapping when it has none."""
    predictions = read(stem)
    return predictions.answers if predictions is not None else {}


def available_runs(stems: Sequence[str]) -> set[str]:
    """Which heads have answers on disk across these videos."""
    return {
        predictions.run
        for stem in stems
        if (predictions := read(stem)) is not None
    }


def policy_for(stem: str, run: str):
    """That run's policy for this video, or None when it did not answer it.

    None and an empty policy are different claims: a video this head never saw
    must not be scored as a video it abstained on for every event.
    """
    from yp_video.actor.policy import SpotActorPolicy  # noqa: PLC0415

    predictions = read(stem)
    if predictions is None or predictions.run != run:
        return None
    return SpotActorPolicy(predictions.answers, name=predictions.run)
