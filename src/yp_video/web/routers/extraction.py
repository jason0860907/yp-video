"""Player detection: who is on screen when each action happened.

RF-DETR Seg on every annotated action frame, keeping ALL person boxes.
Decides nothing — which of those people acted is
routers/actor_association.py, and it re-decides among these boxes without
ever opening the video again.

The sparse sibling of routers/tracklets.py: tracking detects every frame of
every rally and links the results, this detects the ~300 frames an action
happened on and keeps their boxes. Two perception stages, two upstreams
(rally spans and action labels), neither waiting on the other — this endpoint
does not require tracklets, because it never reads one.

Detection used to pick and crop here too, which made the first association
pass a different code path from every later one. The records and crops it
still serves are shared with association, which writes the pick into the same
record — hence the neutral `extraction` name for where that data lives.
"""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import Field

from yp_video.actor import labels as actor_labels
from yp_video.config import cut_kind_of, find_cut, iter_all_cuts
from yp_video.core.cache import StatCache
from yp_video.core.jsonl import read_jsonl, read_jsonl_cached
from yp_video.extraction import pipeline
from yp_video.extraction import store as extraction_store
from yp_video.extraction.prerequisites import prerequisites
from yp_video.person.detector import DETECTOR_NAME
from yp_video.web.job_helpers import init_batch_items, spawn_batch_video_job
from yp_video.web.jobs import JobSummary, JobType, job_manager
from yp_video.web.schemas import StrictModel

log = logging.getLogger(__name__)
router = APIRouter()


# Slimmed UI payload, rebuilt when the detector output OR either annotation
# source changes. Values are shared across requests — read-only, like
# everything cached.
_slim_records_cache: StatCache = StatCache()


def _has_current_detections(path: Path) -> bool:
    """Whether records were produced by the detector this build expects."""
    if not path.exists():
        return False
    header, _ = read_jsonl_cached(path)
    return (header.get("source") or {}).get("detector") == DETECTOR_NAME


@router.get("/videos")
def list_videos() -> list[dict]:
    """Cut videos that have action events — the detection work list."""
    results = []
    for f in sorted(iter_all_cuts(), key=lambda p: p.name):
        events = pipeline.load_events(f.stem)
        if not events:
            continue
        path = extraction_store.records_path(f.stem)
        header = read_jsonl_cached(path)[0] if path.exists() else None
        current = _has_current_detections(path)
        results.append({
            "name": f.name,
            "kind": cut_kind_of(f),
            "event_count": len(events),
            # Retired detector output is deliberately pending: running the
            # default job migrates it without requiring Overwrite.
            "has_records": current,
            # This stage's own outcome. What was DECIDED about the detections
            # is the association listing's to report.
            "detections": int(header.get("detections") or 0) if header else None,
            "detector": (header.get("source") or {}).get("detector") if header else None,
            "pipeline": prerequisites(f.stem).payload(),
        })
    return results


class DetectRequest(StrictModel):
    videos: list[str] = Field(min_length=1)
    #: Re-detect videos that already have detections. Off = skip them; the
    #: candidate list only changes when the detector does.
    overwrite: bool = False
    stop_vllm: bool = False


@router.post("/detect", response_model=JobSummary)
async def detect(req: DetectRequest) -> dict:
    video_paths: list[Path] = []
    skipped: list[str] = []
    for name in req.videos:
        path = find_cut(name)
        if path is None:
            raise HTTPException(404, f"Video not found: {name}")
        # The action labels say WHICH frames to look at. Nothing else is a
        # prerequisite: detection reads the video and decides nothing, so
        # tracklets — which only the association stage needs, and which that
        # stage gates on itself — must not hold it up. Requiring them here
        # was left over from when this endpoint also picked the actor.
        if extraction_store.action_annotation_path(path.stem) is None:
            raise HTTPException(400, f"No action annotations for: {name}")
        if (
            not req.overwrite
            and _has_current_detections(extraction_store.records_path(path.stem))
        ):
            skipped.append(path.stem)
            continue
        video_paths.append(path)

    if not video_paths:
        raise HTTPException(400, "All selected videos already have detections (enable overwrite)")

    job = job_manager.create_job(
        JobType.PLAYER_DETECTION,
        {
            "videos": [p.name for p in video_paths],
            "skipped_existing": skipped,
            "items": init_batch_items([p.name for p in video_paths]),
        },
        name=f"Player Detection ({len(video_paths)} videos)",
    )
    spawn_batch_video_job(
        job,
        video_paths,
        stop_vllm=req.stop_vllm,
        work=lambda p, cb: pipeline.detect_video(p, on_progress=cb),
        done_message=lambda c: (
            f"{c['detections']} people over {c['events']} events"
            + (f" · {c['undecodable']} frames undecodable" if c["undecodable"] else "")
        ),
        start_message="detecting players...",
    )
    return job.to_dict()


@router.get("/records/{name}")
def records(name: str) -> dict:
    """One video's extraction records (UI payload)."""
    stem = Path(unquote(name)).stem
    path = extraction_store.records_path(stem)
    if not path.exists():
        raise HTTPException(404, f"No extraction records for {stem}")
    # Cached parse shares objects across requests — filter into copies, never
    # mutate what read_jsonl_cached hands out.
    meta, _records = read_jsonl_cached(path)
    meta = dict(meta)
    # The video-sync overlay needs fps (frame ↔ time) and the rally spans for
    # its rally navigator — both live in the annotation header, not the
    # extraction header.
    ann = extraction_store.action_annotation_path(stem)
    if ann is not None:
        ann_meta, _ = read_jsonl(ann)
        if not meta.get("fps") and ann_meta.get("fps"):
            meta["fps"] = ann_meta["fps"]
        meta["rallies"] = ann_meta.get("rallies") or []
    sources = [path, *extraction_store.action_source_paths(stem)]
    rows = _slim_records_cache.get(
        stem, sources, lambda: _slim_records(path, stem)
    )
    labels = actor_labels.load(stem)
    return {
        "meta": meta,
        "records": [
            {
                **record,
                "actor_review": (
                    labels[record["id"]].verdict.value
                    if record["id"] in labels
                    else "unreviewed"
                ),
            }
            for record in rows
        ],
    }


def _slim_records(path: Path, stem: str) -> list[dict]:
    meta, records = read_jsonl_cached(path)
    out = []
    # Only the events somebody can actually be identified in — a score has no
    # player, and nobody is tracked between rallies (extraction/store.py).
    for r in extraction_store.labelable(records, stem, float(meta.get("fps") or 0)):
        r = dict(r)
        # Exclude legacy pose fields from records created before their removal.
        r.pop("keypoints", None)
        if r.get("detections"):
            r["detections"] = [{k: v for k, v in d.items() if k != "keypoints"} for d in r["detections"]]
        out.append(r)
    return out


@router.get("/crop/{name}/{crop_file}")
def crop(name: str, crop_file: str, masked: bool = False) -> FileResponse:
    """One crop jpg. ``masked=True`` serves the background-suppressed variant
    the masked embedders saw, falling back to the original while that video's
    masked embed hasn't run yet."""
    stem = Path(unquote(name)).stem
    fname = Path(unquote(crop_file)).name
    path = extraction_store.masked_crop_dir(stem) / fname if masked else extraction_store.crop_dir(stem) / fname
    if masked and not path.exists():
        path = extraction_store.crop_dir(stem) / fname
    if not path.exists():
        raise HTTPException(404, "Crop not found")
    return FileResponse(path, media_type="image/jpeg")
