"""Every labelable event's 2XLarge boxes, and the box check on its label.

Association Label picks an actor by clicking one of the dense pass's 2XLarge
boxes on the event frame (person/dense.py), so the picker needs every
event's boxes — this is where it gets them. For an event whose label has a
box, the snapshot's rule (actor/box_style.py) also says whether that box,
on the event frame, clearly is one of them. Anything but SNAPPED is the
page's box-check queue: a confirmed automatic box no dense box matches, or a
label an action edit moved off its frame and the dense boxes lost on the way.

Cached per video on the files it reads, so the work list counts it on every
load without re-reading a dense pass that has not changed.
"""

from __future__ import annotations

from yp_video.actor import labels as actor_labels
from yp_video.actor.box_style import (
    SNAPPED,
    UNRESOLVED,
    dense_pass,
    pixels,
    resolve_target,
    settle,
)
from yp_video.core.cache import StatCache
from yp_video.extraction.store import action_source_paths, labelable_actions
from yp_video.person.dense import DENSE_SCORE_FLOOR, dense_path

_cache = StatCache()


def box_check(stem: str) -> list[dict]:
    """One entry per labelable event, in frame order; empty when the video
    lacks an action annotation or a dense pass.

    Each entry: event ``id``, ``frame``, ``label``, ``boxes`` — the event
    frame's dense boxes at or above DENSE_SCORE_FLOOR as ``{box, score}`` in
    pixels (empty when the frame is not covered) — and, for an event whose
    actor label has a box, ``status`` (box_style's SNAPPED or why not) and
    ``label_box``, the label's box on the event frame (None when UNRESOLVED);
    both None otherwise. Shared — read-only.
    """
    sources = action_source_paths(stem)
    if not sources or not dense_path(stem).exists():
        return []
    sources.append(dense_path(stem))
    labels = actor_labels.actors_path(stem)
    if labels.exists():
        sources.append(labels)
    try:
        return _cache.get(stem, sources, lambda: _check(stem))
    except FileNotFoundError:
        return []  # deleted between exists() and stat


def pending_count(stem: str) -> int:
    """Events whose label box needs a look (a status other than SNAPPED)."""
    return sum(entry["status"] not in (None, SNAPPED) for entry in box_check(stem))


def _check(stem: str) -> list[dict]:
    dense = dense_pass(stem)
    if dense is None:
        return []
    width, height = dense.meta["frame_size"]
    labels = actor_labels.load(stem)
    out = []
    for event in sorted(labelable_actions(stem, dense.fps), key=lambda e: e["frame"]):
        frame = int(event["frame"])
        label = labels.get(str(event["id"]))
        hit = dense.at(frame, DENSE_SCORE_FLOOR)
        status = label_box = None
        if label is not None and label.box is not None:
            # Only the frames the rule reads: the event's, and every frame a
            # moved label is followed through on its way there.
            lo, hi = sorted((label.frame, frame))
            target, _ = resolve_target(label, frame, dense.people(range(lo, hi + 1)), width, height)
            if target is None:
                status = UNRESOLVED
            else:
                status = settle(target, None if hit is None else hit[0])[1]
                label_box = pixels(target, width, height)
        boxes, scores = hit or ([], [])
        out.append(
            {
                "id": str(event["id"]),
                "frame": frame,
                "label": event.get("label"),
                "status": status,
                "label_box": label_box,
                "boxes": [
                    {"box": pixels(box, width, height), "score": round(score, 3)}
                    for box, score in zip(boxes, scores)
                ],
            }
        )
    return out
