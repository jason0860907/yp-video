"""Person detection labeling: exact-frame review, distinct from predictions."""
from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from yp_video.extraction.pipeline import load_events
from yp_video.person import annotations
from yp_video.web import audit, detection_media, detection_predictions, worklists
from yp_video.web.r2_client import resolve_cut, sync_to_r2

router = APIRouter()


def _video(name: str) -> Path:
    if Path(name).name != name:
        raise HTTPException(400, "Expected a cut filename")
    video = resolve_cut(name)
    if video is None:
        raise HTTPException(404, "Video not found")
    return video


def _meta(video: Path, frame: int | None = None) -> dict:
    meta = detection_media.metadata(video)
    if frame is not None and not 0 <= frame < meta["num_frames"]:
        raise HTTPException(422, "Frame outside video")
    return meta


@router.get("/videos")
def videos() -> list[dict]:
    return worklists.detection_videos()


@router.get("/video/{name}")
def video_info(name: str) -> dict:
    video = _video(name)
    meta = _meta(video)
    data = annotations.load(video.stem)
    if data and data.num_frames != meta["num_frames"]:
        raise HTTPException(409, "Video frame count differs from saved annotations")
    return {
        **meta,
        "prediction_sources": detection_predictions.sources(video.stem, meta["num_frames"], meta["fps"]),
        "frames": {f: {"state": a.state, "count": len(a.boxes)} for f, a in data.frames.items()} if data else {},
        "action_frames": sorted({int(e["frame"]) for e in load_events(video.stem)
                                 if 0 <= int(e["frame"]) < meta["num_frames"]}),
    }


@router.get("/image/{name}")
def image(name: str, frame: int = Query(ge=0), original: bool = False) -> Response:
    video = _video(name)
    _meta(video, frame)
    return Response(detection_media.frame_image(video, frame, original=original), media_type="image/jpeg",
                    headers={"Cache-Control": "private, max-age=60"})


@router.get("/frame/{name}")
def get_frame(name: str, frame: int = Query(ge=0), source: detection_predictions.Source | None = None) -> dict:
    video = _video(name)
    meta = _meta(video, frame)
    data = annotations.load(video.stem)
    if data and data.num_frames != meta["num_frames"]:
        raise HTTPException(409, "Video frame count differs from saved annotations")
    annotation = data.frames.get(frame) if data else None
    prediction = detection_predictions.prediction(video.stem, frame, meta["num_frames"], meta["fps"], source)
    return {"annotation": annotation.model_dump() if annotation else None, "prediction": prediction}


@router.put("/frame/{name}")
async def save_frame(name: str, annotation: annotations.FrameAnnotation, frame: int = Query(ge=0)) -> dict:
    video = await asyncio.to_thread(_video, name)
    meta = await asyncio.to_thread(_meta, video, frame)
    before = await asyncio.to_thread(annotations.load, video.stem)
    previous = before.frames.get(frame) if before else None
    try:
        saved = await asyncio.to_thread(annotations.save, video.stem, meta["num_frames"], frame, annotation)
    except annotations.RevisionConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    sync_to_r2(annotations.annotation_path(video.stem), "person/annotations")
    audit.record_diff(
        target=video.stem,
        before=[{"frame": frame, **previous.model_dump(exclude={"revision"})}] if previous else [],
        after=[{"frame": frame, **saved.model_dump(exclude={"revision"})}],
        key=lambda row: row["frame"],
        frame=frame, state=saved.state, boxes=len(saved.boxes),
    )
    return saved.model_dump()
