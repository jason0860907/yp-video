"""Rally Annotator FastAPI server."""

import json
import signal
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    print("Starting Rally Annotator...")

    def handle_signal(signum, frame):
        print(f"\nReceived signal {signum}, shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    yield

    print("Shutting down...")


app = FastAPI(lifespan=lifespan)

VIDEOS_DIR = Path.home() / "videos"
ANNOTATIONS_DIR = VIDEOS_DIR / "annotations"


class Annotation(BaseModel):
    start: float
    end: float
    label: str


class SaveAnnotationsRequest(BaseModel):
    video: str
    duration: float
    annotations: list[Annotation]


@app.get("/api/results")
def list_results() -> list[str]:
    """List all *_results.json files in ~/videos."""
    if not VIDEOS_DIR.exists():
        return []
    return sorted(f.name for f in VIDEOS_DIR.glob("*_results.json"))


@app.get("/api/results/{name}")
def get_result(name: str) -> dict:
    """Get contents of a specific results JSON file."""
    path = VIDEOS_DIR / name
    if not path.exists() or not path.is_file():
        raise HTTPException(404, "Results file not found")

    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON file")


@app.get("/api/video/{path:path}")
def stream_video(path: str):
    """Stream video file from arbitrary path."""
    # Decode URL-encoded path
    decoded_path = unquote(path)

    # Handle absolute paths
    if decoded_path.startswith("/"):
        video_path = Path(decoded_path)
    else:
        video_path = VIDEOS_DIR / decoded_path

    if not video_path.exists() or not video_path.is_file():
        raise HTTPException(404, f"Video not found: {video_path}")

    return FileResponse(video_path, media_type="video/mp4")


@app.post("/api/annotations")
def save_annotations(req: SaveAnnotationsRequest) -> dict:
    """Save annotations to JSON file."""
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Extract video stem from path
    video_path = Path(req.video)
    output_name = f"{video_path.stem}_annotations.json"
    output_path = ANNOTATIONS_DIR / output_name

    data = {
        "video": req.video,
        "duration": req.duration,
        "annotations": [
            {"start": a.start, "end": a.end, "label": a.label}
            for a in req.annotations
        ]
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    return {"saved": str(output_path), "count": len(req.annotations)}


static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


def run_server(host: str = "0.0.0.0", port: int = 8002):
    """Run the Rally Annotator server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
