"""Video cutter router."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from yp_video.config import CUTS_DIR, VIDEOS_DIR
from yp_video.core.ffmpeg import FFmpegError, export_segment
from yp_video.web.r2_client import serve_video_or_r2_redirect, sync_to_r2

router = APIRouter()


class Segment(BaseModel):
    name: str
    start: float
    end: float


class ExportRequest(BaseModel):
    source: str
    segments: list[Segment]


class ExportResult(BaseModel):
    success: list[str]
    failed: list[str]


@router.get("/videos")
def list_videos() -> list[str]:
    if not VIDEOS_DIR.exists():
        return []
    return sorted(f.name for f in VIDEOS_DIR.glob("*.mp4"))


@router.get("/video/{name}")
def stream_video(name: str):
    response = serve_video_or_r2_redirect(VIDEOS_DIR / name, ("videos",))
    if response:
        return response
    raise HTTPException(404, "Video not found")


@router.post("/export")
def export_segments(req: ExportRequest) -> ExportResult:
    source_path = VIDEOS_DIR / req.source
    if not source_path.exists():
        raise HTTPException(404, "Source video not found")

    CUTS_DIR.mkdir(exist_ok=True)

    success = []
    failed = []

    for seg in req.segments:
        output_name = f"{seg.name}.mp4"
        output_path = CUTS_DIR / output_name

        try:
            export_segment(source_path, seg.start, seg.end, output_path, copy=True)
            sync_to_r2(output_path, "cuts")
            success.append(output_name)
        except FFmpegError:
            failed.append(output_name)

    return ExportResult(success=success, failed=failed)
