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
PRE_ANNOTATIONS_DIR = VIDEOS_DIR / "rally-pre-annotations"
ANNOTATIONS_DIR = VIDEOS_DIR / "rally-annotations"


class Annotation(BaseModel):
    start: float
    end: float
    label: str


class SaveAnnotationsRequest(BaseModel):
    video: str
    duration: float
    annotations: list[Annotation]


def read_jsonl(path: Path) -> dict:
    """Read JSONL file and return structure compatible with JSON format.

    JSONL format:
        Line 1: metadata with _meta=true
        Line 2+: one clip result per line

    Returns:
        Dict with metadata fields + "results" list
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        return {"results": []}

    # First line is metadata
    meta = json.loads(lines[0])
    meta.pop("_meta", None)

    # Remaining lines are results
    results = []
    for line in lines[1:]:
        line = line.strip()
        if line:
            results.append(json.loads(line))

    meta["results"] = results
    return meta


@app.get("/api/results")
def list_results() -> list[str]:
    """List .jsonl files from rally-annotations and rally-pre-annotations.

    Files in rally-annotations (human-corrected) take priority over
    rally-pre-annotations (auto-generated) when both exist.
    """
    files: dict[str, None] = {}
    # Add pre-annotations first
    if PRE_ANNOTATIONS_DIR.exists():
        for f in PRE_ANNOTATIONS_DIR.glob("*.jsonl"):
            files[f.name] = None
    # Override with human-corrected annotations
    if ANNOTATIONS_DIR.exists():
        for f in ANNOTATIONS_DIR.glob("*.jsonl"):
            files[f.name] = None
    return sorted(files)


@app.get("/api/results/{name}")
def get_result(name: str) -> dict:
    """Get contents of a specific results JSONL file.

    Checks rally-annotations first, then falls back to rally-pre-annotations.
    """
    # Prefer human-corrected annotations
    path = ANNOTATIONS_DIR / name
    source = "rally-annotations"
    if not path.exists() or not path.is_file():
        path = PRE_ANNOTATIONS_DIR / name
        source = "rally-pre-annotations"
    if not path.exists() or not path.is_file():
        raise HTTPException(404, "Results file not found")

    try:
        data = read_jsonl(path)
        data["source"] = source
        return data
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSONL file")


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
    """Save annotations to JSONL file.

    Format:
        Line 1: metadata with _meta=true
        Line 2+: one annotation per line
    """
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Extract video stem from path
    video_path = Path(req.video)
    output_name = f"{video_path.stem}_annotations.jsonl"
    output_path = ANNOTATIONS_DIR / output_name

    with open(output_path, "w", encoding="utf-8") as f:
        # First line: metadata
        meta = {"_meta": True, "video": req.video, "duration": req.duration}
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

        # Subsequent lines: one annotation per line
        for a in req.annotations:
            annotation = {"start": a.start, "end": a.end, "label": a.label}
            f.write(json.dumps(annotation, ensure_ascii=False) + "\n")

    return {"saved": str(output_path), "count": len(req.annotations)}


static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


def run_server(host: str = "0.0.0.0", port: int = 8003):
    """Run the Rally Annotator server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
