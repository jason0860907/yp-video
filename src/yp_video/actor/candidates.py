"""The tracklets near an action event, as candidate boxes for the joint
person/action head (``person_action``) and the frame size they normalize by.

Boxes leave here normalized against the source frame size, matching the ``xy``
contact point in the action labels.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from yp_video.core.jsonl import read_jsonl_cached
from yp_video.extraction.store import records_path
from yp_video.tracklets.geometry import box_near
from yp_video.tracklets.store import load_tracklets, tracks_path


def frame_size(stem: str) -> tuple[int, int] | None:
    """The source frame size detection recorded, or None before detection."""
    path = records_path(stem)
    if not path.exists():
        return None
    meta, _records = read_jsonl_cached(path)
    size = meta.get("frame_size")
    if not size or len(size) != 2 or not all(size):
        return None
    return int(size[0]), int(size[1])


def _normalized(
    box: Sequence[float] | None, width: int, height: int
) -> list[float] | None:
    """A box in [0, 1], or None where tracking has none for that frame."""
    if box is None:
        return None
    x0, y0, x1, y1 = (
        round(min(max(float(v) / size, 0.0), 1.0), 5)
        for v, size in zip(box, (width, height, width, height))
    )
    # After clamping: a box hanging off the frame keeps its visible part, and
    # one entirely outside it is no box at all.
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def track_paths(stem: str) -> dict[str, dict[int, Sequence[float]]]:
    """tracklet key → {frame: box}, the whole video."""
    path = tracks_path(stem)
    if not path.exists():
        return {}
    tracklets = load_tracklets(path).records
    return {
        f"{tracklet['rally_id']}:{tracklet['track_id']}": {
            int(frame): box
            for frame, box in zip(tracklet["frames"], tracklet["boxes"])
        }
        for tracklet in tracklets
    }


def candidates_on(
    paths: Mapping[str, Mapping[int, Sequence[float]]], frame: int
) -> list[str]:
    """The tracklets with a box within EVENT_TRACK_MAX_DELTA of this frame,
    in a stable order.

    Membership is decided around the event frame alone, never across the
    whole actor window: a wider window would let a tracklet that had already
    vanished before the contact re-enter as a candidate, and the model would
    be asked to rule out someone who was not there. ±3 frames (0.1 s) is the
    smallest reach that survives stride-2 tracking, where an exact-frame
    rule left 47% of events with no candidate at all (10-03).
    """
    return sorted(key for key, boxes in paths.items() if box_near(boxes, frame) is not None)


def _box_near(boxes: Mapping[int, Sequence[float]], frame: int) -> Sequence[float] | None:
    found = box_near(boxes, frame)
    return found[0] if found is not None else None


def boxes_on(
    paths: Mapping[str, Mapping[int, Sequence[float]]], frame: int, width: int, height: int
) -> list[tuple[str, list[float]]]:
    """``candidates_on`` with each tracklet's nearest box, normalized.

    The candidate set the person/action head scores in advanced identify —
    one box per tracklet, on the event frame, at the same reach.
    """
    out = []
    for key in sorted(paths):
        box = _normalized(_box_near(paths[key], frame), width, height)
        if box is not None:
            out.append((key, box))
    return out
