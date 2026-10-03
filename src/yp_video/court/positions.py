"""Court positions of one video's play: the actor's feet at each contact, the
ball's 3D point there, where it landed, and the arcs in between.

The one implementation: the labeling web page and the production worker's
court job both call `compute`, each gathering the inputs its own way (the web
from local records and annotations, the worker from the analysis and
identify results). Pure — no file is read here.

A position that cannot be real is left out rather than drawn: feet past the
free zone, or a contact higher than anyone reaches, is a failed projection —
and one bad point turns every arc it touches into nonsense.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from yp_video.court import camera, geometry
from yp_video.court.annotations import Calibration

#: How far past the court lines a player can still stand (the free zone).
FREE_ZONE_M = 3.0
#: The highest a ball is plausibly touched.
MAX_CONTACT_Z_M = 4.5
#: Longest gap between two touches still drawn as one flight. Past this the
#: ball has surely been touched in between by something unlabeled.
MAX_FLIGHT_S = 2.0
#: A score marks where the ball came down, not a touch anyone made.
LANDING_LABEL = "score"

#: An event's feet: normalized image (x, y), or why there are none.
Feet = Mapping[str, Sequence[float] | str]


def _in_play_area(court_xy: np.ndarray) -> bool:
    x, y = court_xy
    return (
        -FREE_ZONE_M <= x <= geometry.COURT_LENGTH + FREE_ZONE_M
        and -FREE_ZONE_M <= y <= geometry.COURT_WIDTH + FREE_ZONE_M
    )


def _lift(cam: camera.Camera | None, ball: Sequence[float] | None, court_xy: np.ndarray) -> np.ndarray | None:
    if cam is None or ball is None:
        return None
    point = camera.lift(cam, tuple(ball), tuple(court_xy))
    return point if point is not None and 0 <= point[2] <= MAX_CONTACT_Z_M else None


def _flights(events: list[dict]) -> list[dict]:
    """Ballistic arcs between consecutive placed touches of a rally.

    `events` (time order) holds every touch, placed or not, and only
    neighbours are joined — so an arc never jumps over an unplaced one, which
    would draw a single flight where the ball was really played twice.
    """
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
            # The touch that sent the ball — what the flight is coloured by.
            "label": a["label"],
            "from": a["id"],
            "to": b["id"],
            "start": round(a["time"], 3),
            "end": round(b["time"], 3),
            "points": [[round(float(v), 2) for v in p] for p in camera.arc(a["ball_3d"], b["ball_3d"], duration)],
        })
    return arcs


def compute(calibration: Calibration, events: Sequence[Mapping], feet: Feet) -> dict:
    """Place every event on the court.

    `events`: `{id, frame, time, label, ball, rally_id}` — `ball` the visible
    ball's normalized image point or None, `rally_id` None outside every
    rally. `feet`: event id → the actor's feet in the image, or the reason
    there are none (absent: no detection). Raises geometry.FitError when the
    marks cannot fit the floor.
    """
    to_court = np.array(geometry.fit(calibration.points).image_to_court)
    try:
        cam = camera.solve(calibration)
    except camera.CameraError:
        cam = None  # no camera, no heights: the floor positions still stand

    placed = []
    for event in sorted(events, key=lambda e: e["time"]):
        ball = event.get("ball")
        foot = None
        if event["label"] == LANDING_LABEL:
            # On the floor, so the floor homography places it on its own.
            court_xy = geometry.project(to_court, np.array([ball]))[0] if ball else None
            reason = "ball hidden" if ball is None else None if _in_play_area(court_xy) else "off court"
            if reason:
                court_xy = None
            ball_3d = np.array([court_xy[0], court_xy[1], 0.0]) if court_xy is not None else None
        else:
            found = feet.get(event["id"], "no detection")
            if isinstance(found, str):
                court_xy, reason = None, found
            else:
                foot = (float(found[0]), float(found[1]))
                court_xy = geometry.project(to_court, np.array([foot]))[0]
                reason = None if _in_play_area(court_xy) else "off court"
                if reason:
                    court_xy = None
            # The actor's feet anchor the ball's depth along its image ray.
            ball_3d = _lift(cam, ball, court_xy) if court_xy is not None else None
        placed.append({**event, "foot": foot, "court_xy": court_xy, "ball_3d": ball_3d, "reason": reason})

    length, width = geometry.COURT_LENGTH, geometry.COURT_WIDTH

    def out(e: dict) -> dict:
        xy = e["court_xy"]
        return {
            "id": e["id"],
            "frame": e["frame"],
            "time": round(e["time"], 3),
            "rally_id": e["rally_id"],
            "label": e["label"],
            "foot_image": [round(v, 4) for v in e["foot"]] if e["foot"] and xy is not None else None,
            "ball_image": [round(float(v), 4) for v in e["ball"]] if e.get("ball") else None,
            "court_xy": [round(float(v), 2) for v in xy] if xy is not None else None,
            "in_court": xy is not None and bool(0 <= xy[0] <= length and 0 <= xy[1] <= width),
            "ball_3d": [round(float(v), 2) for v in e["ball_3d"]] if e["ball_3d"] is not None else None,
            # Why court_xy is null: occluded, no association, no detection,
            # outside rally, no takeoff, off court or ball hidden.
            "reason": e["reason"],
        }

    return {
        "units": "m",
        "court": {"length": length, "width": width},
        "events": [out(e) for e in placed],
        "arcs": _flights(placed),
    }
