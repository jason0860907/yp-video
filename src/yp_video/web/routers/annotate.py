"""Rally annotator router."""

import asyncio
import io
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import NamedTuple
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from starlette.background import BackgroundTask

from yp_video.config import (
    ANNOTATIONS_DIR,
    CUT_R2_CATEGORIES,
    PRE_ANNOTATIONS_DIR,
    RALLY_SPOT_PRE_ANNOTATIONS_DIR,
    RAW_VIDEOS_DIR,
    VIDEOS_DIR,
    cut_kind_of,
    find_cut,
)
from yp_video.app_export import AppExportError, export_one_match
from yp_video.core.annotation_ids import rally_id
from yp_video.core.ffmpeg import FFmpegError, export_segment
from yp_video.core.jsonl import read_jsonl
from yp_video.web.r2_client import r2_client, serve_video_or_r2_redirect, sync_to_r2

router = APIRouter()


class _Source(NamedTuple):
    tag: str
    directory: Path
    r2_category: str


# Where a result file may live. Order is the default load priority: reviewed
# truth first, then the newest ML pass (SPOT), then the VLM pass. The Load UI
# can force one via the ``source`` query param (by tag).
_SOURCES = (
    _Source("annotation", ANNOTATIONS_DIR, "rally-annotations"),
    _Source("spot-pre-annotation", RALLY_SPOT_PRE_ANNOTATIONS_DIR, "rally-spot-pre-annotations"),
    _Source("pre-annotation", PRE_ANNOTATIONS_DIR, "rally-pre-annotations"),
)
_SOURCE_BY_TAG = {s.tag: s for s in _SOURCES}


class Annotation(BaseModel):
    id: str | None = None
    rally_id: int | None = None
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
    video = str(meta.get("video") or path.stem.removesuffix("_annotations"))
    records.sort(key=lambda r: (
        float(r.get("start", r.get("start_time", 0)) or 0),
        float(r.get("end", r.get("end_time", 0)) or 0),
        str(r.get("label", "rally")),
    ))
    records = [_with_rally_id(video, r, i) for i, r in enumerate(records)]
    meta["results"] = records
    return meta


def _with_rally_id(video: str, record: dict, index: int) -> dict:
    normalized = {k: v for k, v in record.items() if k != "id"}
    return {
        **normalized,
        "rally_id": rally_id(video, record, index),
    }


@router.get("/results")
def list_results() -> list[dict]:
    files: dict[str, set[str]] = {}  # name -> set of source tags
    for source in _SOURCES:
        if source.directory.exists():
            for f in source.directory.glob("*.jsonl"):
                files.setdefault(f.name, set()).add(source.tag)
    # Include R2-only files
    if r2_client.configured:
        try:
            for source in _SOURCES:
                for obj in r2_client.list_objects(prefix=f"{source.r2_category}/"):
                    files.setdefault(Path(obj["key"]).name, set()).add(source.tag)
        except Exception:
            pass
    def _kind(name: str) -> str:
        # Strip the conventional "_annotations.jsonl" suffix to get the cut stem.
        stem = name.removesuffix(".jsonl").removesuffix("_annotations")
        cut = find_cut(f"{stem}.mp4")
        return cut_kind_of(cut) if cut else "broadcast"
    return sorted(
        [{"name": k, "source": sorted(v), "kind": _kind(k)} for k, v in files.items()],
        key=lambda x: x["name"],
    )


@router.get("/results/{name}")
async def get_result(name: str, source: str | None = None) -> dict:
    """Load one result file, preferring the highest-priority source.

    ``source`` (a tag from ``_SOURCES``) restricts the lookup to that one
    location — used by the Load UI to open e.g. the VLM pass even when a SPOT
    pass or a saved annotation also exists.
    """
    if source is not None and source not in _SOURCE_BY_TAG:
        raise HTTPException(
            400, f"Unknown source {source!r}; expected one of {[s.tag for s in _SOURCES]}"
        )
    candidates = (_SOURCE_BY_TAG[source],) if source else _SOURCES

    # Try local files first
    for candidate in candidates:
        path = candidate.directory / name
        if path.exists() and path.is_file():
            try:
                data = _read_jsonl_as_dict(path)
            except json.JSONDecodeError:
                raise HTTPException(400, "Invalid JSONL file")
            data["source"] = candidate.r2_category
            return data

    # Fallback: download from R2 and cache locally.
    # boto3 is synchronous, so run in a thread to avoid blocking the event loop.
    if r2_client.configured:
        for candidate in candidates:
            r2_key = f"{candidate.r2_category}/{name}"
            exists = await asyncio.to_thread(r2_client.object_exists, r2_key)
            if exists:
                candidate.directory.mkdir(parents=True, exist_ok=True)
                local_path = candidate.directory / name
                await asyncio.to_thread(r2_client.download_file, r2_key, local_path)
                data = _read_jsonl_as_dict(local_path)
                data["source"] = candidate.r2_category
                return data

    raise HTTPException(404, "Results file not found")


@router.get("/video/{path:path}")
def stream_video(path: str):
    from yp_video.config import find_cut
    decoded_path = unquote(path)
    basename = Path(decoded_path).name
    if decoded_path.startswith("/"):
        video_path = Path(decoded_path)
    else:
        # Try the split cut dirs first (the common case for annotations
        # produced by the detect → review pipeline), then VIDEOS_DIR for
        # historical paths, then raw-videos as a final fallback.
        resolved = find_cut(basename)
        if resolved is not None:
            video_path = resolved
        else:
            video_path = VIDEOS_DIR / decoded_path
            if not video_path.exists():
                alt = RAW_VIDEOS_DIR / decoded_path
                if alt.exists():
                    video_path = alt
    response = serve_video_or_r2_redirect(video_path, (*CUT_R2_CATEGORIES, "videos"))
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
        ordered = sorted(annotations, key=lambda ann: (ann.start, ann.end, ann.label))
        for i, a in enumerate(ordered):
            annotation = {
                "start": a.start,
                "end": a.end,
                "label": a.label,
                "rally_id": rally_id(video, a.model_dump(mode="json"), i),
            }
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


# ── Rally clip download ──────────────────────────────────────────────────
#
# Cut the source video into mp4 clips at the rally annotation boundaries, so
# a reviewer can download the actual rally footage (single clip or a zip).

# Global cap on concurrent FFmpeg cuts, matching the Cut page's policy — each
# FFmpeg process is CPU-heavy and more than 2 at once just thrash the VM.
_CLIP_SEMAPHORE = asyncio.Semaphore(2)


class ClipSegment(BaseModel):
    start: float
    end: float
    label: str = "rally"


class ClipRequest(BaseModel):
    video: str
    segment: ClipSegment


class ClipZipRequest(BaseModel):
    video: str
    segments: list[ClipSegment]


def _resolve_clip_source(video: str) -> Path:
    """Resolve an annotation's stored video path to a real file on disk.

    Annotation files store the source video as the path it was cut from.
    Accept either an absolute path or a bare filename resolved against the
    cut dirs. The file must be present locally — FFmpeg needs to read it.
    """
    p = Path(video)
    if p.is_absolute() and p.is_file():
        return p
    found = find_cut(p.name)
    if found is not None:
        return found
    raise HTTPException(
        404,
        f"Source video not found locally: {video}. "
        "The cut video must be on this machine to export clips.",
    )


def _clip_name(stem: str, seg: ClipSegment, idx: int) -> str:
    """Stable, sortable clip filename: <video>_<label>NNN_<start>-<end>.mp4."""
    return f"{stem}_{seg.label}{idx:03d}_{int(seg.start)}-{int(seg.end)}.mp4"


async def _cut(source: Path, seg: ClipSegment, out: Path) -> None:
    """Stream-copy one segment, surfacing FFmpeg failures as HTTP 500."""
    if seg.end <= seg.start:
        raise HTTPException(400, f"Segment end must be after start ({seg.start}–{seg.end})")
    try:
        async with _CLIP_SEMAPHORE:
            # copy=True: stream copy, fast but cuts at the nearest keyframe —
            # same trade-off the Cut page uses for segment export.
            await export_segment(source, seg.start, seg.end, out, copy=True)
    except FFmpegError as e:
        raise HTTPException(500, f"Clip export failed: {e}")


@router.post("/clip")
async def cut_clip(req: ClipRequest):
    """Cut a single rally segment and return it as an mp4 download."""
    source = _resolve_clip_source(req.video)
    tmp = Path(tempfile.mkdtemp(prefix="rally-clip-"))
    out = tmp / _clip_name(source.stem, req.segment, 1)
    try:
        await _cut(source, req.segment, out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    # BackgroundTask removes the temp dir after the response is fully sent.
    return FileResponse(
        out, media_type="video/mp4", filename=out.name,
        background=BackgroundTask(shutil.rmtree, tmp, ignore_errors=True),
    )


@router.post("/clip-zip")
async def cut_clip_zip(req: ClipZipRequest):
    """Cut multiple rally segments and bundle them into one zip."""
    if not req.segments:
        raise HTTPException(400, "No segments selected")
    source = _resolve_clip_source(req.video)
    tmp = Path(tempfile.mkdtemp(prefix="rally-clips-"))
    try:
        buf = io.BytesIO()
        # ZIP_STORED — mp4 is already compressed, deflating just burns CPU.
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
            for i, seg in enumerate(req.segments, 1):
                out = tmp / _clip_name(source.stem, seg, i)
                await _cut(source, seg, out)
                zf.write(out, out.name)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return Response(
        buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="rally-clips.zip"'},
    )


class PublishRequest(BaseModel):
    video: str


@router.post("/publish")
async def publish_to_app(req: PublishRequest) -> dict:
    """Mark a match complete and push it to the iOS app.

    Uploads the cut video plus a single-match manifest to R2, then returns
    the manifest URL the user pastes into VolleyIQ. Expects the rally
    annotations to have been saved first (the Annotate UI saves before
    calling this). Heavy network I/O runs off the event loop.
    """
    basename = Path(req.video).stem
    try:
        return await asyncio.to_thread(export_one_match, basename)
    except AppExportError as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # noqa: BLE001 — surface R2 / network failures
        raise HTTPException(502, f"Export to app failed: {e}")
