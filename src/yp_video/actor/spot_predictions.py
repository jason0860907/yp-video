"""What the yp-spot actor head decided, per event.

yp-spot lives in another repo and another venv, reached across a subprocess
boundary, so its answers arrive as a file rather than a function call. This
module is the only reader of that file, and it hands ``SpotActorPolicy`` a
plain lookup so the policy stays a decision and not an IO layer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from yp_video.config import REID_DIR
from yp_video.core.cache import StatCache
from yp_video.tracklets.geometry import TrackRef

ACTOR_PREDICTIONS_DIR = REID_DIR / "association" / "spot"
_cache: StatCache = StatCache()


def predictions_path(stem: str) -> Path:
    return ACTOR_PREDICTIONS_DIR / f"{stem}_actor_predictions.json"


@dataclass(frozen=True)
class SpotAnswer:
    """One event's answer, in the head's own terms."""

    track: TrackRef | None
    confidence: float
    #: Which of the three the softmax preferred: a candidate, `occluded`, or
    #: `untracked`. Kept because they abstain identically but mean different
    #: things — `untracked` says go fix tracking, not go relabel.
    kind: str


def _parse(payload: dict) -> dict[str, SpotAnswer]:
    answers: dict[str, SpotAnswer] = {}
    for row in payload.get("events", []):
        event_id = str(row.get("id"))
        track = row.get("track")
        answers[event_id] = SpotAnswer(
            track=TrackRef.parse(track) if track else None,
            confidence=float(row.get("confidence") or 0.0),
            kind=str(row.get("kind") or "untracked"),
        )
    return answers


def load(stem: str) -> dict[str, SpotAnswer]:
    """This video's answers, or an empty mapping when it has none."""
    path = predictions_path(stem)
    if not path.exists():
        return {}
    return _cache.get(
        stem,
        [path],
        lambda: _parse(json.loads(path.read_text(encoding="utf-8"))),
    )
