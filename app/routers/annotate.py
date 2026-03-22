"""Rally annotator router."""

import json
from pathlib import Path
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

router = APIRouter()

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
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        return {"results": []}

    meta = json.loads(lines[0])
    meta.pop("_meta", None)

    results = []
    for line in lines[1:]:
        line = line.strip()
        if line:
            results.append(json.loads(line))

    meta["results"] = results
    return meta


@router.get("/results")
def list_results() -> list[str]:
    files: dict[str, None] = {}
    if PRE_ANNOTATIONS_DIR.exists():
        for f in PRE_ANNOTATIONS_DIR.glob("*.jsonl"):
            files[f.name] = None
    if ANNOTATIONS_DIR.exists():
        for f in ANNOTATIONS_DIR.glob("*.jsonl"):
            files[f.name] = None
    return sorted(files)


@router.get("/results/{name}")
def get_result(name: str) -> dict:
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


@router.get("/video/{path:path}")
def stream_video(path: str):
    decoded_path = unquote(path)
    if decoded_path.startswith("/"):
        video_path = Path(decoded_path)
    else:
        video_path = VIDEOS_DIR / decoded_path
    if not video_path.exists() or not video_path.is_file():
        raise HTTPException(404, f"Video not found: {video_path}")
    return FileResponse(video_path, media_type="video/mp4")


@router.post("/annotations")
def save_annotations(req: SaveAnnotationsRequest) -> dict:
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    video_path = Path(req.video)
    output_name = f"{video_path.stem}_annotations.jsonl"
    output_path = ANNOTATIONS_DIR / output_name

    with open(output_path, "w", encoding="utf-8") as f:
        meta = {"_meta": True, "video": req.video, "duration": req.duration}
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

        for a in req.annotations:
            annotation = {"start": a.start, "end": a.end, "label": a.label}
            f.write(json.dumps(annotation, ensure_ascii=False) + "\n")

    return {"saved": str(output_path), "count": len(req.annotations)}
