"""Court calibration labeling: mark court landmarks, get the floor homography."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TypedDict
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException

from yp_video.core import label_done
from yp_video.court import annotations, camera, geometry, paint
from yp_video.court.camera import Camera
from yp_video.court.geometry import Fit
from yp_video.court.positions import CourtSize
from yp_video.extraction.court_prefill import court_prefill
from yp_video.tracklets.store import tracks_path
from yp_video.web import audit, court_positions, worklists
from yp_video.web.r2_client import cut_frame_source, resolve_cut, sync_to_r2
from yp_video.web.schemas import DoneRequest

router = APIRouter()


def _video(name: str) -> Path:
    video = resolve_cut(Path(unquote(name)).name)
    if video is None:
        raise HTTPException(404, "Video not found")
    return video


def _stem(name: str) -> str:
    return _video(name).stem


class CourtState(TypedDict):
    """A calibration and everything solved from it."""

    #: Landmark → where it sits in the frame (normalized; may lie
    #: OUTSIDE_FRAME past the edge).
    points: dict[geometry.Landmark, paint.Point]
    #: Guide lines drawn along the paint, saved with the marks.
    guides: list[paint.Segment]
    net_height_m: float
    #: The floor homography; None when the marks cannot fit (``fit_error``).
    fit: Fit | None
    fit_error: str | None
    #: The solved pinhole camera; None when it cannot be (``camera_error``).
    camera: Camera | None
    camera_error: str | None
    #: Floor landmark → its court position, metres.
    landmarks: dict[geometry.FloorLandmark, tuple[float, float]]
    #: Net-top marks: their floor position; they stand net_height_m above it.
    net_landmarks: dict[geometry.NetLandmark, tuple[float, float]]
    #: The painted lines, as court-space (x0, y0, x1, y1) segments.
    lines: list[tuple[float, float, float, float]]
    court: CourtSize
    #: How far past the frame edge a mark may sit (fraction of the frame).
    outside_frame: float
    #: Longest gap between touches still drawn as one flight, seconds.
    max_flight_s: float


class CourtPrefill(TypedDict):
    """The court found in the paint: guides, and the floor marks when the
    fit is trusted (court/paint.Found)."""

    guides: list[paint.Segment]
    points: dict[geometry.FloorLandmark, paint.Point]


def _state(calibration: annotations.Calibration | None) -> CourtState:
    points = calibration.points if calibration else {}
    try:
        fit, fit_error = geometry.fit(points), None
    except geometry.FitError as exc:
        fit, fit_error = None, str(exc)
    if calibration is None:
        cam, camera_error = None, fit_error
    else:
        try:
            cam, camera_error = camera.solve(calibration), None
        except camera.CameraError as exc:
            cam, camera_error = None, str(exc)
    return {
        "points": points,
        "guides": calibration.guides if calibration else [],
        "net_height_m": calibration.net_height_m if calibration else annotations.DEFAULT_NET_HEIGHT,
        "fit": fit,
        "fit_error": fit_error,
        "camera": cam,
        "camera_error": camera_error,
        "landmarks": geometry.LANDMARKS,
        "net_landmarks": geometry.NET_LANDMARKS,
        "lines": geometry.LINES,
        "court": {"length": geometry.COURT_LENGTH, "width": geometry.COURT_WIDTH},
        "outside_frame": annotations.OUTSIDE_FRAME,
        "max_flight_s": court_positions.MAX_FLIGHT_S,
    }


def _rows(calibration: annotations.Calibration | None) -> list[dict]:
    """The audit's item view: one row per mark, plus the net height and
    the guide lines."""
    if calibration is None:
        return []
    return [{"key": name, "value": list(xy)} for name, xy in calibration.points.items()] + [
        {"key": "net_height_m", "value": calibration.net_height_m},
        {"key": "guides", "value": [[list(a), list(b)] for a, b in calibration.guides]},
    ]


@router.get("/videos")
def videos() -> list[worklists.CourtVideo]:
    return worklists.court_videos()


@router.get("/video/{name}")
def get_calibration(name: str) -> CourtState:
    stem = _stem(name)
    return _state(annotations.load(stem))


@router.put("/video/{name}")
async def save_calibration(name: str, calibration: annotations.Calibration) -> CourtState:
    stem = await asyncio.to_thread(_stem, name)
    before = await asyncio.to_thread(annotations.load, stem)
    await asyncio.to_thread(annotations.save, stem, calibration)
    sync_to_r2(annotations.annotation_path(stem), "court/annotations")
    audit.record_diff(
        target=stem,
        before=_rows(before),
        after=_rows(calibration),
        key=lambda row: row["key"],
        points=len(calibration.points),
    )
    return _state(calibration)


@router.get("/prefill/{name}")
async def detect_prefill(name: str) -> CourtPrefill:
    """The court found in the paint (court/paint.py): guides, and the floor
    marks when the fit is trusted — drawn before the marking starts, saved
    only with the user's first change."""
    found = await asyncio.to_thread(_detect_prefill, name)
    return {"guides": found.guides, "points": found.points} if found else {"guides": [], "points": {}}


def _detect_prefill(name: str) -> paint.Found | None:
    video = _video(name)
    if not tracks_path(video.stem).exists():
        raise HTTPException(409, "No tracks yet: the court is found where the players stand")
    return court_prefill(video.stem, cut_frame_source(video))


@router.get("/positions/{name}")
def positions(name: str) -> court_positions.CourtPositions:
    """Where the play happened, in court metres (web/court_positions.py)."""
    try:
        return court_positions.compute(_stem(name))
    except (geometry.FitError, court_positions.NotReady) as exc:
        raise HTTPException(409, str(exc)) from exc


@router.put("/done/{name}")
def set_done(name: str, req: DoneRequest) -> label_done.DoneState:
    flags = label_done.set_done(_stem(name), "court", req.done)
    return {"done": flags.get("court", False)}
