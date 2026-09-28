"""Court positions of one video's play: the actor's feet at each contact,
the ball's 3D point there, where it landed, and the arcs in between.

Joins the court calibration to the extraction records (actor boxes) and the
action annotation (ball points, score landings) — a web-layer join, so the
court package stays free of the pipeline stages it reads.
"""

from __future__ import annotations

import numpy as np

from yp_video.core.jsonl import read_jsonl_cached
from yp_video.core.rallies import load_rallies
from yp_video.court import annotations, camera, geometry
from yp_video.extraction import store as extraction_store

#: Actor answers that name nobody — no feet to place.
_NO_ACTOR = ("unresolved", "occluded")
#: Longest gap between two touches still drawn as one flight. Past this the
#: ball has surely been touched in between by something unlabeled.
MAX_FLIGHT_S = 2.5


class NotReady(LookupError):
    pass


def _rally_of(time: float, spans: list[dict]) -> int | None:
    for span in spans:
        if span["start"] <= time <= span["end"]:
            return span["rally_id"]
    return None


def _visible_xy(event: dict) -> tuple[float, float] | None:
    xy = event.get("xy")
    return (float(xy[0]), float(xy[1])) if xy and event.get("visible", True) else None


def compute(stem: str) -> dict:
    calibration = annotations.load(stem)
    if calibration is None:
        raise NotReady("No court calibration for this video")
    fit = geometry.fit(calibration.points)
    to_court = np.array(fit.image_to_court)
    try:
        cam, camera_error = camera.solve(calibration), None
    except camera.CameraError as exc:
        cam, camera_error = None, str(exc)

    path = extraction_store.records_path(stem)
    if not path.exists():
        raise NotReady("No extraction records — run Player Detection and Association first")
    meta, records = read_jsonl_cached(path)
    width, height = meta["frame_size"]
    fps = float(meta.get("fps") or 0)
    if not fps:
        raise NotReady("Extraction records carry no fps")
    spans = load_rallies(stem)

    events = []
    for r in extraction_store.labelable(records, stem, fps):
        box = r.get("box")
        if not box or r.get("resolution") in _NO_ACTOR:
            continue
        foot = ((box[0] + box[2]) / 2 / width, box[3] / height)
        court_xy = geometry.project(to_court, np.array([foot]))[0]
        ball = _visible_xy(r)
        events.append({
            "id": r["id"],
            "frame": r["frame"],
            "label": r.get("label"),
            "foot_image": foot,
            "ball_image": ball,
            "court_xy": court_xy,
            # The actor's feet anchor the ball's depth along its image ray.
            "ball_3d": camera.lift(cam, ball, tuple(court_xy)) if cam and ball else None,
        })

    # A score marks where the ball came down — on the floor, so the floor
    # homography places it without any actor.
    source = extraction_store.action_annotation_path(stem)
    if source is not None:
        _ann_meta, rows = read_jsonl_cached(source)
        for e in rows:
            ball = _visible_xy(e)
            if e.get("label") != "score" or ball is None or e.get("frame") is None:
                continue
            landing = geometry.project(to_court, np.array([ball]))[0]
            events.append({
                "id": str(e.get("id") or f"f{e['frame']}"),
                "frame": int(e["frame"]),
                "label": "score",
                "foot_image": None,
                "ball_image": ball,
                "court_xy": landing,
                "ball_3d": np.array([landing[0], landing[1], 0.0]),
            })

    events.sort(key=lambda e: e["frame"])
    for e in events:
        e["time"] = e["frame"] / fps
        e["rally_id"] = _rally_of(e["time"], spans)

    arcs = []
    for a, b in zip(events, events[1:]):
        if a["rally_id"] is None or a["rally_id"] != b["rally_id"]:
            continue
        if a["ball_3d"] is None or b["ball_3d"] is None:
            continue
        duration = b["time"] - a["time"]
        if not 0 < duration <= MAX_FLIGHT_S:
            continue
        arcs.append({
            "rally_id": a["rally_id"],
            "from": a["id"],
            "to": b["id"],
            "start": round(a["time"], 3),
            "end": round(b["time"], 3),
            "points": [[round(float(v), 2) for v in p] for p in camera.arc(a["ball_3d"], b["ball_3d"], duration)],
        })

    length, width_m = geometry.COURT_LENGTH, geometry.COURT_WIDTH

    def out(e: dict) -> dict:
        x, y = (float(v) for v in e["court_xy"])
        return {
            "id": e["id"],
            "frame": e["frame"],
            "time": round(e["time"], 3),
            "rally_id": e["rally_id"],
            "label": e["label"],
            "foot_image": [round(float(v), 4) for v in e["foot_image"]] if e["foot_image"] else None,
            "ball_image": [round(float(v), 4) for v in e["ball_image"]] if e["ball_image"] else None,
            "court_xy": [round(x, 2), round(y, 2)],
            "in_court": bool(0 <= x <= length and 0 <= y <= width_m),
            "ball_3d": [round(float(v), 2) for v in e["ball_3d"]] if e["ball_3d"] is not None else None,
        }

    return {
        "video": stem,
        "units": "m",
        "court": {"length": length, "width": width_m},
        "fit_rmse_m": fit.rmse_m,
        "camera": cam.model_dump() if cam else None,
        "camera_error": camera_error,
        "events": [out(e) for e in events],
        "arcs": arcs,
    }
