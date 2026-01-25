"""Video cutter FastAPI server."""

import signal
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from utils.ffmpeg import FFmpegError, export_segment


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    # Startup
    print("Starting up...")

    # Setup signal handlers for graceful shutdown
    def handle_signal(signum, frame):
        print(f"\nReceived signal {signum}, shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    yield

    # Shutdown
    print("Shutting down...")


app = FastAPI(lifespan=lifespan)

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


@app.get("/api/videos")
def list_videos() -> list[str]:
    if not VIDEOS_DIR.exists():
        return []
    return sorted(f.name for f in VIDEOS_DIR.glob("*.mp4"))


@app.get("/api/video/{name}")
def stream_video(name: str):
    path = VIDEOS_DIR / name
    if not path.exists() or not path.is_file():
        raise HTTPException(404, "Video not found")
    return FileResponse(path, media_type="video/mp4")


@app.post("/api/export")
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
            export_segment(source_path, seg.start, seg.end, output_path)
            success.append(output_name)
        except FFmpegError:
            failed.append(output_name)

    return ExportResult(success=success, failed=failed)


static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


def run_server(host: str = "0.0.0.0", port: int = 8001):
    """Run the video cutter server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
