"""Video cutter router."""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from utils.ffmpeg import FFmpegError, export_segment

router = APIRouter()

VIDEOS_DIR = Path.home() / "videos"
CUTS_DIR = VIDEOS_DIR / "cuts"


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
    path = VIDEOS_DIR / name
    if not path.exists() or not path.is_file():
        raise HTTPException(404, "Video not found")
    return FileResponse(path, media_type="video/mp4")


@router.post("/export")
def export_segments(req: ExportRequest) -> ExportResult:
    source_path = VIDEOS_DIR / req.source
    if not source_path.exists():
        raise HTTPException(404, "Source video not found")

    CUTS_DIR.mkdir(exist_ok=True)

    stem = source_path.stem
    success = []
    failed = []

    for seg in req.segments:
        output_name = f"{stem}_{seg.name}.mp4"
        output_path = CUTS_DIR / output_name

        try:
            export_segment(source_path, seg.start, seg.end, output_path, copy=True)
            success.append(output_name)
        except FFmpegError:
            failed.append(output_name)

    return ExportResult(success=success, failed=failed)
