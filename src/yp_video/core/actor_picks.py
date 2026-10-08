"""The SPOT pass's actor picks: who touched the ball at each action event.

The fusion model's actor head scores the person head's own boxes at every
action event as it spots it (yp_spot E2EModel.predict), so standard identify
reads the answer instead of re-running the backbone. The picks travel as a
sidecar beside the person boxes, never inside the action events: a saved
action annotation keeps every field it loaded, and a model's actor guess is
association data, not an action label.

One JSON object: ``checkpoint`` (core/person_boxes.checkpoint_identity) and
``picks`` — one ``{frame, label, box, candidates}`` per action event, frame
native, box normalized xyxy or None when the person head proposed nobody.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from yp_video.config import TRACKS_DIR
from yp_video.core.jsonl import atomic_binary
from yp_video.core.person_boxes import checkpoint_identity

Box = tuple[float, float, float, float]


def actor_picks_path(stem: str) -> Path:
    return TRACKS_DIR / f"{stem}_actor_picks.json"


@dataclass(frozen=True)
class SpotActorPick:
    box: Box | None
    candidates: int


def save_actor_picks(action_records: list[dict], target: Path, checkpoint: Path) -> int:
    """Retain the ``actor`` of every spotted action event (native frames).
    Returns how many events carried one."""
    picks = []
    for record in action_records:
        for event in record.get("events") or []:
            actor = event.get("actor")
            if actor is None:
                continue
            picks.append({
                "frame": int(event["frame"]), "label": str(event["label"]),
                "box": actor["box"], "candidates": int(actor["candidates"]),
            })
    payload = {"checkpoint": checkpoint_identity(checkpoint), "picks": picks}
    _validate(payload)
    with atomic_binary(target) as out:
        out.write(json.dumps(payload, allow_nan=False).encode())
    return len(picks)


def load_actor_picks(path: Path) -> dict[tuple[int, str], SpotActorPick]:
    """``(frame, label)`` → the pick the SPOT pass made for that event."""
    payload = json.loads(path.read_text())
    _validate(payload)
    return {
        (p["frame"], p["label"]): SpotActorPick(
            box=tuple(p["box"]) if p["box"] is not None else None, candidates=p["candidates"],
        )
        for p in payload["picks"]
    }


def _validate(payload: dict) -> None:
    if not isinstance(payload.get("checkpoint"), str) or not isinstance(payload.get("picks"), list):
        raise ValueError("Actor picks need a checkpoint identity and a pick list")
    seen = set()
    for pick in payload["picks"]:
        key = (pick["frame"], pick["label"])
        if key in seen:
            raise ValueError(f"Duplicate actor pick for event {key}")
        seen.add(key)
        box, count = pick["box"], pick["candidates"]
        if type(count) is not int or count < 0:
            raise ValueError(f"Invalid candidate count at {key}")
        if box is None:
            if count:
                raise ValueError(f"Actor pick at {key} names no box among {count} candidates")
            continue
        if (count < 1 or len(box) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in box)
                or box[0] >= box[2] or box[1] >= box[3]):
            raise ValueError(f"Invalid actor box at {key}: {box}")
