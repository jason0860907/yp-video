"""Rally annotator router."""

import asyncio
import json
import os
from pathlib import Path
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from yp_video.config import ANNOTATIONS_DIR, PRE_ANNOTATIONS_DIR, RAW_VIDEOS_DIR, VIDEOS_DIR
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
async def get_result(name: str) -> dict:
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

    # Fallback: download from R2 and cache locally.
    # boto3 is synchronous, so run in a thread to avoid blocking the event loop.
    if r2_client.configured:
        for category in ("rally-annotations", "rally-pre-annotations"):
            r2_key = f"{category}/{name}"
            exists = await asyncio.to_thread(r2_client.object_exists, r2_key)
            if exists:
                local_dir = ANNOTATIONS_DIR if category == "rally-annotations" else PRE_ANNOTATIONS_DIR
                local_dir.mkdir(parents=True, exist_ok=True)
                local_path = local_dir / name
                await asyncio.to_thread(r2_client.download_file, r2_key, local_path)
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
        # Subpath like "cuts/foo.mp4" resolves under VIDEOS_DIR. A bare
        # filename is a raw video → falls through to RAW_VIDEOS_DIR.
        video_path = VIDEOS_DIR / decoded_path
        if not video_path.exists():
            alt = RAW_VIDEOS_DIR / decoded_path
            if alt.exists():
                video_path = alt
    response = serve_video_or_r2_redirect(video_path, ("cuts", "videos"))
    if response:
        return response
    raise HTTPException(404, f"Video not found: {video_path}")


def _write_annotations_atomic(output_path: Path, video: str, duration: float, annotations: list[Annotation]) -> None:
    """Write JSONL via tmp file + atomic rename so concurrent writes
    to the same file never corrupt each other — the last rename wins,
    but the file always contains a complete, consistent snapshot."""
    tmp_path = output_path.with_suffix(output_path.suffix + f".tmp.{os.getpid()}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        meta = {"_meta": True, "video": video, "duration": duration}
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for a in annotations:
            annotation = {"start": a.start, "end": a.end, "label": a.label}
            f.write(json.dumps(annotation, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, output_path)


@router.post("/annotations")
async def save_annotations(req: SaveAnnotationsRequest) -> dict:
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    video_path = Path(req.video)
    output_name = f"{video_path.stem}_annotations.jsonl"
    output_path = ANNOTATIONS_DIR / output_name

    # Run file I/O in a thread so we don't block the event loop
    # (fsync can be slow under concurrent load).
    await asyncio.to_thread(
        _write_annotations_atomic,
        output_path,
        req.video,
        req.duration,
        req.annotations,
    )

    # Auto-sync to R2 (fire-and-forget; safe to call from async context)
    sync_to_r2(output_path, "rally-annotations")

    return {"saved": str(output_path), "count": len(req.annotations)}
