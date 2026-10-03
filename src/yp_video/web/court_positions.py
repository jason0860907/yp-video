"""Court positions for the labeling page: gathers one local video's inputs —
its calibration, actors' feet (extraction/feet.py) and action annotation —
and places them with the one implementation, court/positions.py.
"""

from __future__ import annotations

from yp_video.contracts.action import event_id
from yp_video.core.jsonl import read_jsonl_cached
from yp_video.core.rallies import load_rallies
from yp_video.court import annotations, positions
from yp_video.extraction import feet as extraction_feet
from yp_video.extraction import store as extraction_store

MAX_FLIGHT_S = positions.MAX_FLIGHT_S


class NotReady(LookupError):
    pass


def compute(stem: str) -> dict:
    calibration = annotations.load(stem)
    if calibration is None:
        raise NotReady("No court calibration for this video")
    path = extraction_store.records_path(stem)
    if not path.exists():
        raise NotReady("No extraction records — run Player Detection and Association first")
    fps = float(read_jsonl_cached(path)[0].get("fps") or 0)
    if not fps:
        raise NotReady("Extraction records carry no fps")
    # Every annotated action is listed, placed or not, so what is missing —
    # and why — shows next to what is not.
    source = extraction_store.action_annotation_path(stem)
    rows = [r for r in (read_jsonl_cached(source)[1] if source is not None else []) if r.get("frame") is not None]
    try:
        feet = extraction_feet.actor_feet(stem, rows)
    except extraction_feet.NotReady as exc:
        raise NotReady(str(exc)) from exc
    spans = load_rallies(stem)
    events = []
    for row in rows:
        time = int(row["frame"]) / fps
        xy = row.get("xy")
        events.append({
            "id": event_id(row),
            "frame": int(row["frame"]),
            "time": time,
            "label": row.get("label"),
            "ball": (float(xy[0]), float(xy[1])) if xy and row.get("visible", True) else None,
            "rally_id": next((s["rally_id"] for s in spans if s["start"] <= time <= s["end"]), None),
        })
    return {"video": stem, **positions.compute(calibration, events, feet)}
