"""Video cutter router."""

import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from yp_video.config import (
    CUTS_BROADCAST_DIR,
    CUTS_SIDELINE_DIR,
    RAW_VIDEOS_DIR,
)
from yp_video.core.ffmpeg import FFmpegError, export_segment
from yp_video.web.r2_client import serve_video_or_r2_redirect, sync_to_r2

router = APIRouter()

# Global cap on concurrent FFmpeg export operations across all users.
# Each FFmpeg process is CPU-heavy; letting more than 2 run at once
# on a single VM will just thrash and slow everyone down.
_EXPORT_SEMAPHORE = asyncio.Semaphore(2)


class Segment(BaseModel):
    name: str
    start: float
    end: float


class ExportRequest(BaseModel):
    source: str
    segments: list[Segment]
    # "broadcast" = TV footage (default), "sideline" = practice / side-court
    kind: str = "broadcast"


class ExportResult(BaseModel):
    success: list[str]
    failed: list[str]


@router.get("/videos")
def list_videos() -> list[str]:
    if not RAW_VIDEOS_DIR.exists():
        return []
    return sorted(f.name for f in RAW_VIDEOS_DIR.glob("*.mp4"))


@router.get("/video/{name}")
def stream_video(name: str):
    response = serve_video_or_r2_redirect(RAW_VIDEOS_DIR / name, ("videos",))
    if response:
        return response
    raise HTTPException(404, "Video not found")


@router.delete("/video/{name}")
def delete_video(name: str) -> dict:
    """Delete a raw source video from RAW_VIDEOS_DIR. Local-only, does not touch R2."""
    path = RAW_VIDEOS_DIR / name
    if not path.exists():
        raise HTTPException(404, "Video not found")
    # Guard against path traversal — must stay inside RAW_VIDEOS_DIR.
    try:
        path.resolve().relative_to(RAW_VIDEOS_DIR.resolve())
    except ValueError:
        raise HTTPException(400, "Invalid path")
    path.unlink()
    return {"ok": True, "deleted": name}


@router.post("/export")
async def export_segments(req: ExportRequest) -> ExportResult:
    source_path = RAW_VIDEOS_DIR / req.source
    if not source_path.exists():
        raise HTTPException(404, "Source video not found")

    if req.kind == "sideline":
        target_dir, r2_category = CUTS_SIDELINE_DIR, "cuts-sideline"
    elif req.kind == "broadcast":
        target_dir, r2_category = CUTS_BROADCAST_DIR, "cuts-broadcast"
    else:
        raise HTTPException(400, f"Invalid kind: {req.kind!r} (expected 'broadcast' or 'sideline')")

    target_dir.mkdir(parents=True, exist_ok=True)

    success = []
    failed = []

    for seg in req.segments:
        output_name = f"{seg.name}.mp4"
        output_path = target_dir / output_name

        # Acquire per-segment so multiple users' exports interleave fairly
        # instead of one user holding a slot for all their segments.
        async with _EXPORT_SEMAPHORE:
            try:
                await export_segment(source_path, seg.start, seg.end, output_path, copy=True)
                sync_to_r2(output_path, r2_category)
                success.append(output_name)
            except FFmpegError:
                failed.append(output_name)

    return ExportResult(success=success, failed=failed)
