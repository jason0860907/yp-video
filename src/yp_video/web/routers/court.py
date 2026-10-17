"""Court calibration labeling: mark court landmarks, get the floor homography."""

from __future__ import annotations

import asyncio
from pathlib import Path
from urllib.parse import unquote

import numpy as np
from fastapi import APIRouter, HTTPException

from yp_video.core import label_done
from yp_video.core.frames import read_frames
from yp_video.court import annotations, camera, geometry, paint
from yp_video.person.dense import dense_path
from yp_video.web import audit, court_positions, worklists
from yp_video.web.r2_client import cut_frame_source, resolve_cut, sync_to_r2
from yp_video.web.schemas import StrictModel

router = APIRouter()

#: Rally frames whose median the paint is found in: the players move, the
#: paint stays.
GUIDE_FRAMES = 12


def _video(name: str) -> Path:
    video = resolve_cut(Path(unquote(name)).name)
    if video is None:
        raise HTTPException(404, "Video not found")
    return video


def _stem(name: str) -> str:
    return _video(name).stem


def _state(calibration: annotations.Calibration | None) -> dict:
    points = calibration.points if calibration else {}
    try:
        fit, fit_error = geometry.fit(points).model_dump(), None
    except geometry.FitError as exc:
        fit, fit_error = None, str(exc)
    if calibration is None:
        cam, camera_error = None, fit_error
    else:
        try:
            cam, camera_error = camera.solve(calibration).model_dump(), None
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
def videos() -> list[dict]:
    return worklists.court_videos()


@router.get("/video/{name}")
def get_calibration(name: str) -> dict:
    stem = _stem(name)
    return _state(annotations.load(stem))


@router.put("/video/{name}")
async def save_calibration(name: str, calibration: annotations.Calibration) -> dict:
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


@router.get("/guides/{name}")
async def detect_guides(name: str) -> dict:
    """Guide lines along the court paint (court/paint.py): drawn before the
    marking starts, saved only with the marks."""
    return {"guides": await asyncio.to_thread(_detect_guides, name)}


def _detect_guides(name: str) -> list[paint.Segment]:
    video = _video(name)
    dense = dense_path(video.stem)
    if not dense.exists():
        raise HTTPException(409, "No person detections yet: the floor is found from the players' feet")
    with np.load(dense) as d:
        rally_frames, boxes, scores = d["frames"], d["boxes"], d["scores"]
    picks = rally_frames[np.linspace(0, len(rally_frames) - 1, GUIDE_FRAMES).astype(int)]
    frames = [image for _, image in read_frames(cut_frame_source(video), picks) if image is not None]
    if not frames:
        raise HTTPException(409, "Could not read the video's frames")
    return paint.guides(frames, paint.floor_top(boxes.astype(np.float32), scores.astype(np.float32)))


@router.get("/positions/{name}")
def positions(name: str) -> dict:
    """Where the play happened, in court metres (web/court_positions.py)."""
    try:
        return court_positions.compute(_stem(name))
    except (geometry.FitError, court_positions.NotReady) as exc:
        raise HTTPException(409, str(exc)) from exc


class DoneRequest(StrictModel):
    done: bool = True


@router.put("/done/{name}")
def set_done(name: str, req: DoneRequest) -> dict:
    flags = label_done.set_done(_stem(name), "court", req.done)
    return {"done": flags.get("court", False)}
