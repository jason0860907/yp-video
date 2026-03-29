"""Rally annotator router."""

import json
from pathlib import Path
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from yp_video.config import ANNOTATIONS_DIR, PRE_ANNOTATIONS_DIR, VIDEOS_DIR
from yp_video.core.jsonl import read_jsonl
from yp_video.web.r2_client import r2_client, serve_video_or_r2_redirect, sync_to_r2

router = APIRouter()


class Annotation(BaseModel):
    start: float
    end: float
    label: str


class SaveAnnotationsRequest(BaseModel):
    video: str
    duration: float
    annotations: list[Annotation]


def _read_jsonl_as_dict(path: Path) -> dict:
    """Read JSONL and return as {**meta, results: [...]}."""
    meta, records = read_jsonl(path)
    meta["results"] = records
    return meta


@router.get("/results")
def list_results() -> list[dict]:
    files: dict[str, set[str]] = {}  # name -> set of sources
    if PRE_ANNOTATIONS_DIR.exists():
        for f in PRE_ANNOTATIONS_DIR.glob("*.jsonl"):
            files.setdefault(f.name, set()).add("pre-annotation")
    if ANNOTATIONS_DIR.exists():
        for f in ANNOTATIONS_DIR.glob("*.jsonl"):
            files.setdefault(f.name, set()).add("annotation")
    # Include R2-only files
    if r2_client.configured:
        try:
            for obj in r2_client.list_objects(prefix="rally-annotations/"):
                files.setdefault(Path(obj["key"]).name, set()).add("annotation")
            for obj in r2_client.list_objects(prefix="rally-pre-annotations/"):
                files.setdefault(Path(obj["key"]).name, set()).add("pre-annotation")
        except Exception:
            pass
    return sorted(
        [{"name": k, "source": sorted(v)} for k, v in files.items()],
        key=lambda x: x["name"],
    )


@router.get("/results/{name}")
def get_result(name: str) -> dict:
    # Try local files first
    path = ANNOTATIONS_DIR / name
    source = "rally-annotations"
    if not path.exists() or not path.is_file():
        path = PRE_ANNOTATIONS_DIR / name
        source = "rally-pre-annotations"
    if path.exists() and path.is_file():
        try:
            data = _read_jsonl_as_dict(path)
            data["source"] = source
            return data
        except json.JSONDecodeError:
            raise HTTPException(400, "Invalid JSONL file")

    # Fallback: download from R2 and cache locally
    if r2_client.configured:
        for category in ("rally-annotations", "rally-pre-annotations"):
            r2_key = f"{category}/{name}"
            if r2_client.object_exists(r2_key):
                local_dir = ANNOTATIONS_DIR if category == "rally-annotations" else PRE_ANNOTATIONS_DIR
                local_dir.mkdir(parents=True, exist_ok=True)
                local_path = local_dir / name
                r2_client.download_file(r2_key, local_path)
                data = _read_jsonl_as_dict(local_path)
                data["source"] = category
                return data

    raise HTTPException(404, "Results file not found")


@router.get("/video/{path:path}")
def stream_video(path: str):
    decoded_path = unquote(path)
    if decoded_path.startswith("/"):
        video_path = Path(decoded_path)
    else:
        video_path = VIDEOS_DIR / decoded_path
    response = serve_video_or_r2_redirect(video_path, ("cuts", "videos"))
    if response:
        return response
    raise HTTPException(404, f"Video not found: {video_path}")


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

    # Auto-sync to R2
    sync_to_r2(output_path, "rally-annotations")

    return {"saved": str(output_path), "count": len(req.annotations)}
