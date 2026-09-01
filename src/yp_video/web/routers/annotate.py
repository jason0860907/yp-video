"""Rally annotator router."""

import asyncio
import io
import json
import logging
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from starlette.background import BackgroundTask

from yp_video.config import (
    CUT_R2_CATEGORIES,
    RALLY_ANNOTATIONS_DIR,
    RAW_VIDEOS_DIR,
    VIDEOS_DIR,
    find_cut,
)
from yp_video.core import label_done
from yp_video.core.ffmpeg import FFmpegError, export_segment
from yp_video.core.jsonl import read_jsonl
from yp_video.core.rallies import (
    RALLY_SOURCES,
    SOURCE_BY_TAG,
    annotation_name,
    resolve_rally_ids,
)
from yp_video.web import audit, worklists
from yp_video.web.r2_client import r2_client, serve_video_or_r2_redirect, sync_to_r2
from yp_video.web.schemas import StrictModel
from yp_video.web.rally_annotations import Annotation, write_annotations_atomic

log = logging.getLogger(__name__)
router = APIRouter()


# Where a result file may live, and in what priority — see core/rallies.py.
# This editor writes the top-priority location; everything downstream reads
# the same table, so "which file counts" has one answer.
_SOURCES = RALLY_SOURCES
_SOURCE_BY_TAG = SOURCE_BY_TAG


class SaveAnnotationsRequest(StrictModel):
    video: str
    duration: float
    annotations: list[Annotation]


def _read_jsonl_as_dict(path: Path) -> dict:
    """Read JSONL and return as {**meta, results: [...]}.

    Ids come from the file (resolve_rally_ids) — the sort is presentation
    order only. Stray ``id`` keys from older pre-annotation writers are
    dropped so the editor's Annotation schema stays the one shape.
    """
    meta, records = read_jsonl(path)
    ids = resolve_rally_ids(records)
    rows = [
        {**{k: v for k, v in record.items() if k != "id"}, "rally_id": rid}
        for record, rid in zip(records, ids)
    ]
    rows.sort(key=lambda r: (
        float(r.get("start", 0) or 0),
        float(r.get("end", 0) or 0),
        str(r.get("label", "rally")),
    ))
    meta["results"] = rows
    return meta


@router.get("/results")
def list_results() -> list[dict]:
    return worklists.rally_results()


class DoneRequest(StrictModel):
    done: bool = True


@router.put("/done/{name}")
def set_done(name: str, req: DoneRequest) -> dict:
    """Persist the human "rally labeling is finished" verdict for one video."""
    flags = label_done.set_done(Path(unquote(name)).stem, "rally", req.done)
    return {"done": flags["rally"]}


@router.get("/results/{name}")
async def get_result(name: str, source: str) -> dict:
    """Load one result file from exactly the requested store.

    ``source`` is a tag from ``_SOURCES`` — deterministic on purpose, no
    priority ladder: the editor's Source select means what it says. A store
    with no file is a 404 (the frontend renders it as an empty editor).
    """
    if source not in _SOURCE_BY_TAG:
        raise HTTPException(
            400, f"Unknown source {source!r}; expected one of {[s.tag for s in _SOURCES]}"
        )
    candidate = _SOURCE_BY_TAG[source]

    # Try the local file first
    path = candidate.directory / name
    if path.exists() and path.is_file():
        try:
            data = _read_jsonl_as_dict(path)
        except json.JSONDecodeError:
            raise HTTPException(400, "Invalid JSONL file")
        # The tag, not the r2_category path — this is the value the
        # editor's Source select and loaded-source badge speak.
        data["source"] = candidate.tag
        return data

    # Fallback: download from R2 and cache locally.
    # boto3 is synchronous, so run in a thread to avoid blocking the event loop.
    if r2_client.configured:
        r2_key = f"{candidate.r2_category}/{name}"
        exists = await asyncio.to_thread(r2_client.object_exists, r2_key)
        if exists:
            candidate.directory.mkdir(parents=True, exist_ok=True)
            local_path = candidate.directory / name
            await asyncio.to_thread(r2_client.download_file, r2_key, local_path)
            data = _read_jsonl_as_dict(local_path)
            data["source"] = candidate.tag
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


@router.post("/annotations")
async def save_annotations(req: SaveAnnotationsRequest) -> dict:
    provided = [a.rally_id for a in req.annotations if a.rally_id is not None]
    if len(set(provided)) != len(provided):
        seen: set[int] = set()
        duplicates = sorted({i for i in provided if i in seen or seen.add(i)})
        raise HTTPException(400, f"Duplicate rally_id(s): {duplicates}")

    RALLY_ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    video_path = Path(req.video)
    output_name = annotation_name(video_path.stem)
    output_path = RALLY_ANNOTATIONS_DIR / output_name

    # Run file I/O in a thread so we don't block the event loop
    # (fsync can be slow under concurrent load).
    saved, before = await asyncio.to_thread(
        write_annotations_atomic,
        output_path,
        req.video,
        req.duration,
        req.annotations,
    )

    # Auto-sync to R2 (fire-and-forget; safe to call from async context)
    sync_to_r2(output_path, "rally-spot/annotations")

    # An unchanged rewrite — the autosave timer, or the tab-close flush —
    # leaves no row at all.
    audit.record_diff(
        target=video_path.stem,
        before=before,
        after=saved,
        key=lambda r: r["rally_id"],
        rallies=len(saved),
    )

    # The rows as written, ids assigned — the editor adopts these so a new
    # rally does not get a fresh id minted on every autosave.
    return {"saved": str(output_path), "count": len(saved), "annotations": saved}


# ── Rally clip download ──────────────────────────────────────────────────
#
# Cut the source video into mp4 clips at the rally annotation boundaries, so
# a reviewer can download the actual rally footage (single clip or a zip).

# Global cap on concurrent FFmpeg cuts, matching the Cut page's policy — each
# FFmpeg process is CPU-heavy and more than 2 at once just thrash the VM.
_CLIP_SEMAPHORE = asyncio.Semaphore(2)


class ClipSegment(StrictModel):
    start: float
    end: float
    label: str = "rally"


class ClipRequest(StrictModel):
    video: str
    segment: ClipSegment


class ClipZipRequest(StrictModel):
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
    audit.detail(target=source.stem, segments=1)
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
    audit.detail(target=source.stem, segments=len(req.segments))
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
