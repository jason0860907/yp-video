"""Court positions of one video's play: the actor's feet at each contact, the
ball's 3D point there, where it landed, and the arcs in between — fitted to
the detected ball where it was seen (court/trajectory.py), else the
ballistic path joining the two touches.

The one implementation: the labeling web page and the production worker's
court job both call `compute`, each gathering the inputs its own way (the web
from local records and annotations, the worker from the analysis and
identify results — through `compute_segments`, as an App match is
calibrated again wherever its camera moved). Pure — no file is read here.

A position that cannot be real is left out rather than drawn: feet past the
free zone, or a contact higher than anyone reaches, is a failed projection —
and one bad point turns every arc it touches into nonsense.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator

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


class Segment(BaseModel):
    """One camera position of a video: the calibration that holds from
    `start` until the next segment's start (a camera moved mid-match is
    calibrated again)."""

    model_config = ConfigDict(extra="forbid")
    #: Video seconds the calibration holds from.
    start: float = Field(ge=0, allow_inf_nan=False)
    calibration: Calibration


class Segments(RootModel[list[Segment]]):
    """A video's segments, covering it whole: the first from 0, each later
    one starting after the one before."""

    @model_validator(mode="after")
    def _cover(self) -> "Segments":
        starts = [s.start for s in self.root]
        if not starts or starts[0] != 0:
            raise ValueError("the first segment starts at 0")
        if any(b <= a for a, b in zip(starts, starts[1:])):
            raise ValueError("each segment starts after the one before")
        return self


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


def _observations(track: Mapping, a: dict, last: int, touches: Sequence[dict]) -> tuple[np.ndarray, np.ndarray]:
    """The flight's ball sightings: every detected box from touch a's frame
    to frame `last`, plus the ball point of each of `touches` — (seconds
    from a, image xy)."""
    fps = float(track["fps"])
    t, xy = [], []
    for e in touches:
        if e.get("ball"):
            t.append(e["time"] - a["time"])
            xy.append(e["ball"])
    for frame in range(a["frame"], last + 1):
        for x, y, _score in track["frames"].get(str(frame), ()):
            t.append(frame / fps - a["time"])
            xy.append((x, y))
    return np.array(t, dtype=np.float64), np.array(xy, dtype=np.float64).reshape(-1, 2)


def _runs(events: list[dict]) -> list[list[dict]]:
    """Runs of consecutive touches of one rally, each no more than
    MAX_FLIGHT_S after the one before: the chains the ball is fitted along."""
    runs: list[list[dict]] = []
    for a, b in zip(events, events[1:]):
        if a["rally_id"] is None or a["rally_id"] != b["rally_id"] or not 0 < b["time"] - a["time"] <= MAX_FLIGHT_S:
            continue
        if runs and runs[-1][-1] is a:
            runs[-1].append(b)
        else:
            runs.append([a, b])
    return runs


def _flights(events: list[dict], cam: camera.Camera | None, aspect: float, track: Mapping | None) -> list[dict]:
    """The ball's arc between each pair of consecutive touches of a rally.

    `events` (time order) holds every touch, placed or not, and only
    neighbours are joined — so an arc never jumps over an unplaced one, which
    would draw a single flight where the ball was really played twice.

    Each run of touches is fitted as one chain (trajectory.fit_chain): the
    detected ball over all its flights, with the touches already placed in
    3D (from the actor's feet, or a landing on the floor) as anchors. The
    fitted touch positions replace those events' ball_3d, so dots sit on the
    arcs. An arc is "ball" when enough of its flight was seen on it, else
    "touches" — the gravity path between its two placed touches.

    A placed touch no arc leaves — its next touch unplaced, or the two not
    joinable, or none following within MAX_FLIGHT_S — still gets the flight
    seen after it (`_open_flight`).
    """
    rng = np.random.default_rng(0)
    arcs = []
    for run in _runs(events):
        anchors = {k: e["ball_3d"] for k, e in enumerate(run) if e["ball_3d"] is not None}
        touches: list[np.ndarray | None] = [e["ball_3d"] for e in run]
        shifts = [0.0] * len(run)
        seen = [(0, float("nan"))] * (len(run) - 1)
        if cam is not None:
            empty = (np.zeros(0), np.zeros((0, 2)))
            chain = trajectory.fit_chain(cam, aspect, [
                trajectory.Flight(
                    *(_observations(track, a, b["frame"], (a, b)) if track is not None else empty), b["time"] - a["time"],
                )
                for a, b in zip(run, run[1:])
            ], anchors, rng)
            touches, shifts, seen = chain.touches, chain.shifts, chain.seen
            for e, point in zip(run, touches):
                if point is not None and -trajectory.FLOOR_SLACK_M <= point[2] <= MAX_CONTACT_Z_M:
                    e["ball_3d"] = point
        for k, (a, b) in enumerate(zip(run, run[1:])):
            start, end = a["time"] + shifts[k], b["time"] + shifts[k + 1]
            pa, pb, duration = touches[k], touches[k + 1], end - start
            if pa is None or pb is None or duration <= 0 or not trajectory.plausible(pa, pb, duration):
                continue
            frames, rmse = seen[k]
            evidence = (
                {"source": "ball", "frames": frames, "rmse": round(rmse, 4)}
                if frames >= trajectory.MIN_FRAMES else {"source": "touches"}
            )
            arcs.append({
                "rally_id": a["rally_id"],
                # The touch that sent the ball — what the flight is coloured by.
                "label": a["label"],
                "from": a["id"],
                "to": b["id"],
                # When the ball was really touched: the spotted time, moved
                # by the fit (at most trajectory.TOUCH_SLACK_S).
                "start": round(start, 3),
                "end": round(end, 3),
                "points": [[round(float(v), 2) for v in p] for p in camera.arc(pa, pb, duration, ARC_SAMPLES)],
                **evidence,
            })
    if cam is not None and track is not None:
        sent = {arc["from"] for arc in arcs}
        for a, b in zip(events, [*events[1:], None]):
            # A landing sends nothing: what follows it is a bounce.
            if a["id"] in sent or a["ball_3d"] is None or a["rally_id"] is None or a["label"] == LANDING_LABEL:
                continue
            if b is not None and (b["rally_id"] != a["rally_id"] or not 0 < b["time"] - a["time"] <= MAX_FLIGHT_S):
                b = None
            arc = _open_flight(cam, aspect, track, a, b, rng)
            if arc is not None:
                arcs.append(arc)
        arcs.sort(key=lambda arc: arc["start"])
    return arcs


def _open_flight(
    cam: camera.Camera, aspect: float, track: Mapping, a: dict, b: dict | None, rng: np.random.Generator,
) -> dict | None:
    """The flight from a placed touch no arc leaves — a shank out past the
    free zone, a ball that left the shot, a touch nothing followed: fitted
    from the touch and the boxes seen after it alone, then followed down to
    the floor, or to the next touch `b` if that comes first. Past
    `seen_until` the path is extrapolated. None when too little of it was
    seen.
    """
    window = b["time"] - a["time"] if b is not None else MAX_FLIGHT_S
    last = b["frame"] if b is not None else a["frame"] + int(MAX_FLIGHT_S * float(track["fps"]))
    t, xy = _observations(track, a, last, (a,))
    fitted = trajectory.fit_open(cam, aspect, t, xy, a["ball_3d"], rng)
    if fitted is None:
        return None
    end = min(fitted.landing(), window)
    pa, pb = fitted.at(np.array([0.0, end]))
    if end <= 0 or not trajectory.plausible(pa, pb, end):
        return None
    return {
        "rally_id": a["rally_id"],
        "label": a["label"],
        "from": a["id"],
        # None when nothing followed within MAX_FLIGHT_S.
        "to": b["id"] if b is not None else None,
        "start": round(a["time"], 3),
        "end": round(a["time"] + end, 3),
        "points": [[round(float(v), 2) for v in p] for p in camera.arc(pa, pb, end, ARC_SAMPLES)],
        "source": "ball",
        "frames": fitted.frames,
        "rmse": round(fitted.rmse, 4),
        "seen_until": round(a["time"] + min(fitted.last, end), 3),
    }


def calibration_summary(calibration: Calibration) -> dict:
    """How well the marks fit, in a form a client can draw without any
    geometry of its own: the floor fit's error, the court's painted lines and
    the net projected back into the frame (normalized image points), and
    the camera — hence heights — when one could be solved.
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
        "camera": {
            "rmse": round(cam.rmse, 4),
            "center": [round(v, 2) for v in cam.center],
            # Court metres to normalized image points, so a client can lay
            # any court point — a flight's every sample — onto the video.
            "projection": [[round(v, 6) for v in row] for row in cam.projection],
        } if cam else None,
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

    # First: fitting the arcs refines the touches' ball_3d.
    arcs = _flights(placed, cam, calibration.frame_size[0] / calibration.frame_size[1], track)
    return {
        "units": "m",
        "court": {"length": length, "width": width},
        "events": [out(e) for e in placed],
        "arcs": arcs,
    }


def compute_segments(segments: Segments, events: Sequence[Mapping], feet: Feet, track: Mapping | None) -> dict:
    """`compute` for a video calibrated once per camera position: each
    segment places the events from its start until the next one's, so no
    arc joins touches seen through two cameras. Raises geometry.FitError
    when any segment's marks cannot fit the floor."""
    parts = segments.root
    ends = [s.start for s in parts[1:]] + [float("inf")]
    placed = [
        compute(part.calibration, [e for e in events if part.start <= e["time"] < end], feet, track)
        for part, end in zip(parts, ends)
    ]
    return {
        "units": "m",
        "court": {"length": geometry.COURT_LENGTH, "width": geometry.COURT_WIDTH},
        "events": [e for p in placed for e in p["events"]],
        "arcs": [a for p in placed for a in p["arcs"]],
    }
