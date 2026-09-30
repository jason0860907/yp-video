"""Court positions of one video's play: the actor's feet at each contact,
the ball's 3D point there, where it landed, and the arcs in between.

Joins the court calibration to the extraction records (actor boxes), the
actors' tracklets (where a jumper stood) and the action annotation (ball
points, score landings) — a web-layer join, so the court package stays free
of the pipeline stages it reads.

A position that cannot be real is left out rather than drawn: feet past the
free zone, or a contact higher than anyone reaches, is a failed projection —
and one bad point turns every arc it touches into nonsense.
"""

from __future__ import annotations

import numpy as np

from yp_video.core.jsonl import read_jsonl_cached
from yp_video.core.rallies import load_rallies
from yp_video.court import annotations, camera, geometry
from yp_video.extraction import links
from yp_video.extraction import store as extraction_store
from yp_video.tracklets.store import tracklet_index, tracks_path

#: Actor answers that name nobody — no feet to place.
_NO_ACTOR = ("unresolved", "occluded")
#: Contacts made in the air. At the contact the feet are off the floor, and
#: the floor homography puts raised feet far behind where the player stands —
#: twice as far for a 0.4 m jump under a camera 0.8 m up. Their feet come
#: from the takeoff or the landing instead.
_AIRBORNE = ("spike", "block", "serve")
#: How far either side of an airborne contact grounded feet are looked for.
#: Both sides: tracking often loses a jumper mid-air, and a server's track
#: only starts once the rally does, just after the toss.
GROUNDED_WINDOW_S = 0.7
#: How far past the court lines a player can still stand (the free zone).
FREE_ZONE_M = 3.0
#: The highest a ball is plausibly touched.
MAX_CONTACT_Z_M = 4.5
#: Longest gap between two touches still drawn as one flight. Past this the
#: ball has surely been touched in between by something unlabeled.
MAX_FLIGHT_S = 2.0


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


def _grounded_box(tracklet: dict, frame: int, window: int) -> list[float] | None:
    """The actor's box with feet on the floor around an airborne contact.

    Feet lowest in the frame over the window: a jump only raises them, so the
    lowest is the takeoff or the landing (the camera stands above the floor,
    so lower in the frame is lower in the world). A jumper lands about where
    they took off, so either one places them.
    """
    boxes = [
        box
        for f, box in zip(tracklet["frames"], tracklet["boxes"])
        if abs(f - frame) <= window
    ]
    return max(boxes, key=lambda box: box[3]) if boxes else None


def _in_play_area(court_xy: np.ndarray) -> bool:
    x, y = court_xy
    return (
        -FREE_ZONE_M <= x <= geometry.COURT_LENGTH + FREE_ZONE_M
        and -FREE_ZONE_M <= y <= geometry.COURT_WIDTH + FREE_ZONE_M
    )


def _lift(cam: camera.Camera | None, ball: tuple[float, float] | None, court_xy: np.ndarray) -> np.ndarray | None:
    if cam is None or ball is None:
        return None
    point = camera.lift(cam, ball, tuple(court_xy))
    return point if point is not None and 0 <= point[2] <= MAX_CONTACT_Z_M else None


def compute(stem: str) -> dict:
    calibration = annotations.load(stem)
    if calibration is None:
        raise NotReady("No court calibration for this video")
    fit = geometry.fit(calibration.points)
    to_court = np.array(fit.image_to_court)
    try:
        cam = camera.solve(calibration)
    except camera.CameraError:
        cam = None  # no camera, no heights: the floor positions still stand

    path = extraction_store.records_path(stem)
    if not path.exists():
        raise NotReady("No extraction records — run Player Detection and Association first")
    meta, records = read_jsonl_cached(path)
    width, height = meta["frame_size"]
    fps = float(meta.get("fps") or 0)
    if not fps:
        raise NotReady("Extraction records carry no fps")
    if not tracks_path(stem).exists():
        raise NotReady("No tracks — run Player Tracking first")
    spans = load_rallies(stem)
    actors = links.event_tracks(stem)
    index = tracklet_index(stem)
    window = round(GROUNDED_WINDOW_S * fps)

    events = []
    for r in extraction_store.labelable(records, stem, fps):
        box = r.get("box")
        if not box or r.get("resolution") in _NO_ACTOR:
            continue
        if r.get("label") in _AIRBORNE:
            ref = actors.get(r["id"])
            tracklet = index.tracklet(ref) if ref else None
            box = _grounded_box(tracklet, r["frame"], window) if tracklet else None
            if box is None:
                continue
        foot = ((box[0] + box[2]) / 2 / width, box[3] / height)
        court_xy = geometry.project(to_court, np.array([foot]))[0]
        if not _in_play_area(court_xy):
            continue
        ball = _visible_xy(r)
        events.append({
            "id": r["id"],
            "frame": r["frame"],
            "label": r.get("label"),
            "foot_image": foot,
            "ball_image": ball,
            "court_xy": court_xy,
            # The actor's feet anchor the ball's depth along its image ray.
            "ball_3d": _lift(cam, ball, court_xy),
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
            if not _in_play_area(landing):
                continue
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
        "events": [out(e) for e in events],
        "arcs": arcs,
    }
