"""Court positions of one video's play: the actor's feet at each contact, the
ball's 3D point there, where it landed, and the arcs in between — fitted to
the detected ball where it was seen (court/trajectory.py), else the
ballistic path joining the two touches.

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

from yp_video.court import camera, geometry, trajectory
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

#: Points sampled along each arc.
ARC_SAMPLES = 16

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


def _observations(track: Mapping, a: dict, b: dict) -> tuple[np.ndarray, np.ndarray]:
    """The flight's ball sightings: every detected box between the two
    touches, plus each touch's own ball point — (seconds from a, image xy)."""
    fps = float(track["fps"])
    t, xy = [], []
    for e in (a, b):
        if e.get("ball"):
            t.append(e["time"] - a["time"])
            xy.append(e["ball"])
    for frame in range(a["frame"], b["frame"] + 1):
        for x, y, _score in track["frames"].get(str(frame), ()):
            t.append(frame / fps - a["time"])
            xy.append((x, y))
    return np.array(t, dtype=np.float64), np.array(xy, dtype=np.float64).reshape(-1, 2)


def _flights(events: list[dict], cam: camera.Camera | None, aspect: float, track: Mapping | None) -> list[dict]:
    """The ball's arc between each pair of consecutive touches of a rally.

    `events` (time order) holds every touch, placed or not, and only
    neighbours are joined — so an arc never jumps over an unplaced one, which
    would draw a single flight where the ball was really played twice. An arc
    is fitted to the detected ball (source "ball") where enough of the flight
    was seen; otherwise, when both touches are placed in 3D, it is the
    ballistic path joining them (source "touches").
    """
    rng = np.random.default_rng(0)
    arcs = []
    for a, b in zip(events, events[1:]):
        if a["rally_id"] is None or a["rally_id"] != b["rally_id"]:
            continue
        duration = b["time"] - a["time"]
        if not 0 < duration <= MAX_FLIGHT_S:
            continue
        fitted = None
        if cam is not None and track is not None:
            t, xy = _observations(track, a, b)
            anchors = [(e["time"] - a["time"], e["ball_3d"]) for e in (a, b) if e["ball_3d"] is not None]
            fitted = trajectory.fit(cam, aspect, t, xy, duration, rng, anchors)
        if fitted is not None:
            points = fitted.at(np.linspace(0.0, duration, ARC_SAMPLES))
            evidence = {"source": "ball", "frames": fitted.frames, "rmse": round(fitted.rmse, 4)}
        elif a["ball_3d"] is not None and b["ball_3d"] is not None:
            points = camera.arc(a["ball_3d"], b["ball_3d"], duration, ARC_SAMPLES)
            evidence = {"source": "touches"}
        else:
            continue
        arcs.append({
            "rally_id": a["rally_id"],
            # The touch that sent the ball — what the flight is coloured by.
            "label": a["label"],
            "from": a["id"],
            "to": b["id"],
            "start": round(a["time"], 3),
            "end": round(b["time"], 3),
            "points": [[round(float(v), 2) for v in p] for p in points],
            **evidence,
        })
    return arcs


def calibration_summary(calibration: Calibration) -> dict:
    """How well the marks fit, in a form a client can draw without any
    geometry of its own: the floor fit's error, the court's painted lines and
    the net projected back into the frame (normalized image points), and
    whether a camera — hence heights — could be solved.
    """
    fit = geometry.fit(calibration.points)
    to_image = np.array(fit.court_to_image)
    lines = [
        geometry.project(to_image, np.array([[x0, y0], [x1, y1]])).round(4).tolist()
        for x0, y0, x1, y1 in geometry.LINES
    ]
    try:
        cam = camera.solve(calibration)
    except camera.CameraError:
        cam = None
    net = None
    if cam is not None:
        half = geometry.COURT_LENGTH / 2
        net = camera.project(cam, np.array([
            [half, 0.0, calibration.net_height_m], [half, geometry.COURT_WIDTH, calibration.net_height_m],
        ])).round(4).tolist()
    return {
        "floor_rmse_m": round(fit.rmse_m, 3),
        "lines": lines,
        "net": net,
        "camera": {"rmse": round(cam.rmse, 4), "center": [round(v, 2) for v in cam.center]} if cam else None,
    }


def compute(calibration: Calibration, events: Sequence[Mapping], feet: Feet, track: Mapping | None) -> dict:
    """Place every event on the court.

    `events`: `{id, frame, time, label, ball, rally_id}` — `ball` the visible
    ball's normalized image point or None, `rally_id` None outside every
    rally. `feet`: event id → the actor's feet in the image, or the reason
    there are none (absent: no detection). `track`: the video's ball track
    (ball/track.py), None when the ball was not detected. Raises
    geometry.FitError when the marks cannot fit the floor.
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
        "arcs": _flights(placed, cam, calibration.frame_size[0] / calibration.frame_size[1], track),
    }
