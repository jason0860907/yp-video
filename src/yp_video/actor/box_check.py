"""The 2XLarge box check: which actor labels the person head cannot use as drawn.

For every action event whose actor label carries a box, the snapshot's rule
(actor/box_style.py) decides whether that box clearly is one of the dense
pass's 2XLarge boxes on the event frame. Anything but SNAPPED is the
Association Label page's box-check queue: the reviewer sees the frame's
dense boxes next to the label and clicks the right one, which stores that
exact box and so snaps on the next snapshot.

Cached per video on the files it reads, so the work list counts it on every
load without re-reading a dense pass that has not changed.
"""

from __future__ import annotations

from yp_video.actor import labels as actor_labels
from yp_video.actor.box_style import SNAPPED, resolve_target, settle
from yp_video.actor.labels import ActorVerdict
from yp_video.actor.person_labels import DensePass
from yp_video.core.cache import StatCache
from yp_video.core.jsonl import read_jsonl_cached
from yp_video.extraction.store import SKIP_LABELS, action_annotation_path
from yp_video.person.dense import DENSE_SCORE_FLOOR, dense_path

_cache = StatCache()


def box_check(stem: str) -> list[dict]:
    """One entry per boxed actor label, in frame order; empty when the video
    lacks an action annotation, actor labels or a dense pass.

    Each entry: event ``id``, ``frame``, ``label``, ``status`` (box_style's
    SNAPPED or why not), the human ``label_box`` (pixels) and the frame it
    was drawn on (``label_frame``, None = the event's), and ``boxes`` — the
    event frame's dense boxes at or above DENSE_SCORE_FLOOR as ``{box, score}``
    in pixels (empty when the frame is not covered). Shared — read-only.
    """
    paths = [action_annotation_path(stem), actor_labels.actors_path(stem), dense_path(stem)]
    if paths[0] is None or not all(path.exists() for path in paths):
        return []
    try:
        return _cache.get(stem, paths, lambda: _check(stem))
    except FileNotFoundError:
        return []  # deleted between exists() and stat


def pending_count(stem: str) -> int:
    """Events whose label box needs a look (status other than SNAPPED)."""
    return sum(entry["status"] != SNAPPED for entry in box_check(stem))


def _check(stem: str) -> list[dict]:
    labels = actor_labels.load(stem)
    _meta, rows = read_jsonl_cached(action_annotation_path(stem))
    events = []
    for event in rows:
        label = labels.get(str(event.get("id")))
        if (
            event.get("frame") is not None
            and event.get("label") not in SKIP_LABELS
            and label is not None
            and label.box is not None
            and label.verdict is not ActorVerdict.OCCLUDED
        ):
            events.append((event, label))
    events.sort(key=lambda pair: pair[0]["frame"])
    if not events:
        return []
    dense = DensePass(dense_path(stem))
    width, height = dense.meta["frame_size"]

    # Only the frames the rule reads: each event frame, and every frame a
    # cross-frame label is followed through on its way there.
    needed: set[int] = set()
    for event, label in events:
        frame = int(event["frame"])
        needed.add(frame)
        if label.frame is not None:
            needed.update(range(min(frame, label.frame), max(frame, label.frame) + 1))
    on_frame = {frame: dense.at(frame, DENSE_SCORE_FLOOR) for frame in sorted(needed)}
    people = {frame: hit[0] for frame, hit in on_frame.items() if hit is not None}

    out = []
    for event, label in events:
        frame = int(event["frame"])
        target, reason = resolve_target(label, frame, people, width, height)
        status = reason if target is None else settle(target, people.get(frame))[1]
        boxes, scores = on_frame[frame] or ([], [])
        out.append(
            {
                "id": str(event["id"]),
                "frame": frame,
                "label": event.get("label"),
                "status": status,
                "label_box": list(label.box),
                "label_frame": label.frame,
                "boxes": [
                    {"box": _pixels(box, width, height), "score": round(score, 3)}
                    for box, score in zip(boxes, scores)
                ],
            }
        )
    return out


def _pixels(box: list[float], width: float, height: float) -> list[float]:
    return [round(box[0] * width, 1), round(box[1] * height, 1),
            round(box[2] * width, 1), round(box[3] * height, 1)]
