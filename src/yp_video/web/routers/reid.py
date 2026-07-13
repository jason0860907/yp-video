"""Player ReID router.

Runs the tracking-free ReID extraction (RF-DETR person detection → contact
point association → OSNet embedding) over the annotated action events of
selected cut videos. Results land in player-reid/ as per-video jsonl +
crop images; identity matching consumes them later.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from yp_video.config import cut_kind_of, find_cut, iter_all_cuts
from yp_video.core.jsonl import read_jsonl, read_jsonl_cached
from yp_video.reid import identity, pipeline, tracking
from yp_video.reid.embedder import DEFAULT_EMBEDDER, EMBEDDER_WEIGHTS
from yp_video.web.job_helpers import (
    TERMINAL_ITEM_STATUSES,
    batch_items_params,
    batch_message,
    batch_progress,
    fail_job_from_exc,
    finalize_batch_job,
    init_batch_items,
    mark_batch_item,
    stop_vllm_for_job,
    update_batch_item,
)
from yp_video.web.jobs import job_manager

log = logging.getLogger(__name__)
router = APIRouter()


class ReidStartRequest(BaseModel):
    videos: list[str] = Field(min_length=1)
    overwrite: bool = False
    stop_vllm: bool = False
    # Keypoint source from the registry (see /reid/options); detection
    # itself is always RF-DETR.
    keypoints: str = "rf-detr"


def _read_header(stem: str) -> dict | None:
    path = pipeline.reid_path(stem)
    if not path.exists():
        return None
    import json

    with open(path, encoding="utf-8") as f:
        line = f.readline().strip()
    if not line:
        return None
    header = json.loads(line)
    header.pop("_meta", None)
    return header


@router.get("/videos")
def list_videos() -> list[dict]:
    """Cut videos that have action events — the ReID work list."""
    results = []
    for f in sorted(iter_all_cuts(), key=lambda p: p.name):
        events = pipeline.load_events(f.stem)
        if not events:
            continue
        header = _read_header(f.stem)
        results.append({
            "name": f.name,
            "kind": cut_kind_of(f),
            "event_count": len(events),
            "has_reid": header is not None,
            "reid_counts": (
                {k: header.get(k, 0) for k in ("ok", "multi", "miss")} if header else None
            ),
            "player_count": len(set(identity.load_assignments(f.stem).values())),
        })
    return results


@router.get("/options")
def options() -> dict:
    """Available keypoint-source / embedder choices for the Predict / Label pages."""
    return {"keypoint_sources": list(pipeline.keypoint_sources), "embedders": list(pipeline.embedders)}


@router.post("/start")
async def start(req: ReidStartRequest) -> dict:
    if req.keypoints not in pipeline.keypoint_sources:
        raise HTTPException(
            400,
            f"Unknown keypoint source: {req.keypoints} (available: {', '.join(pipeline.keypoint_sources)} — "
            "sam-3d-body needs its gated HF checkpoint downloaded first)",
        )
    video_paths: list[Path] = []
    skipped: list[str] = []
    for name in req.videos:
        path = find_cut(name)
        if path is None:
            raise HTTPException(404, f"Video not found: {name}")
        if pipeline.action_annotation_path(path.stem) is None:
            raise HTTPException(400, f"No action annotations for: {name}")
        if not req.overwrite and pipeline.reid_path(path.stem).exists():
            skipped.append(path.stem)
            continue
        video_paths.append(path)

    if not video_paths:
        raise HTTPException(400, "All selected videos already have ReID results (enable overwrite)")

    job = job_manager.create_job(
        "player_reid",
        {
            "videos": [p.name for p in video_paths],
            "skipped_existing": skipped,
            "items": init_batch_items([p.name for p in video_paths]),
        },
        name=f"Player ReID ({len(video_paths)} videos)",
    )
    _spawn_batch_job(
        job,
        video_paths,
        stop_vllm=req.stop_vllm,
        work=lambda p, cb: pipeline.extract_video(p, keypoints=req.keypoints, on_progress=cb),
        done_message=lambda c: f"{c['ok']} ok · {c['multi']} multi · {c['miss']} miss",
        progress_noun="event",
        start_message="detecting players...",
    )
    return job.to_dict()


def _spawn_batch_job(job, video_paths: list[Path], *, stop_vllm: bool, work, done_message, progress_noun: str, start_message: str) -> None:
    """Run ``work(video_path, on_progress)`` per video inside the standard
    batch-job scaffolding: inference lock, throttled per-item progress,
    cancellation, and the final ok/failed roll-up. ``work`` is synchronous
    and GPU-bound — it runs in an executor thread.
    """
    total = len(video_paths)

    async def run_job() -> None:
        items = job.params["items"]
        loop = asyncio.get_event_loop()
        failed = 0
        try:
            await job_manager.update_job(
                job.id, status="running", message="Waiting for inference slot..."
            )
            async with stop_vllm_for_job(job.id, when=stop_vllm):
                async with job_manager.inference_lock:
                    for i, video_path in enumerate(video_paths):
                        await update_batch_item(
                            job.id, items, i, status="running", message=start_message,
                            overall_progress=batch_progress(i, 0.0, total),
                            overall_message=batch_message(i, total, video_path.name, start_message),
                        )

                        last_push = {"t": 0.0}

                        def on_progress(done, total_units, _status, *, index=i, name=video_path.name):
                            # Executor thread → schedule onto the loop; throttle
                            # to ~1/s except the final unit.
                            now = time.monotonic()
                            if done != total_units and now - last_push["t"] < 1.0:
                                return
                            last_push["t"] = now
                            frac = done / total_units if total_units else 0.0
                            detail = f"{progress_noun} {done}/{total_units}"
                            loop.call_soon_threadsafe(
                                asyncio.ensure_future,
                                update_batch_item(
                                    job.id, items, index, progress=frac, message=detail,
                                    overall_progress=batch_progress(index, frac, total),
                                    overall_message=batch_message(index, total, name, detail),
                                ),
                            )

                        try:
                            counts = await loop.run_in_executor(
                                None,
                                lambda p=video_path, cb=on_progress: work(p, cb),
                            )
                            await update_batch_item(
                                job.id, items, i, status="completed", progress=1.0,
                                message=done_message(counts),
                                overall_progress=batch_progress(i, 1.0, total),
                                overall_message=batch_message(i, total, video_path.name, "done"),
                            )
                        except Exception as exc:  # noqa: BLE001
                            failed += 1
                            log.exception("%s failed for %s", job.name, video_path.name)
                            await update_batch_item(
                                job.id, items, i, status="failed",
                                message=f"{type(exc).__name__}: {exc}", error=str(exc),
                                overall_message=batch_message(i, total, video_path.name, f"failed — {exc}"),
                            )
            await finalize_batch_job(job.id, total, failed)
        except asyncio.CancelledError:
            for idx in range(len(items)):
                if items[idx].get("status") not in TERMINAL_ITEM_STATUSES:
                    mark_batch_item(items, idx, status="cancelled", message="Cancelled")
            current = job_manager.get_job(job.id)
            await job_manager.update_job(
                job.id, status="cancelled", message="Cancelled",
                params={**(current.params if current else {}), **batch_items_params(items)},
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("%s job failed", job.name)
            await fail_job_from_exc(job.id, exc)

    task = asyncio.create_task(run_job())
    job_manager.attach_task(job, task)


class TrackStartRequest(BaseModel):
    videos: list[str] = Field(min_length=1)
    overwrite: bool = False
    stop_vllm: bool = False
    # Detect every Nth rally frame; ByteTrack is told the effective rate.
    stride: int = Field(1, ge=1, le=10)


@router.post("/track")
async def track(req: TrackStartRequest) -> dict:
    """Dense per-rally detection + ByteTrack (see reid/tracking.py)."""
    video_paths: list[Path] = []
    skipped: list[str] = []
    for name in req.videos:
        path = find_cut(name)
        if path is None:
            raise HTTPException(404, f"Video not found: {name}")
        if pipeline.action_annotation_path(path.stem) is None:
            raise HTTPException(400, f"No action annotations for: {name}")
        if not req.overwrite and tracking.tracks_path(path.stem).exists():
            skipped.append(path.stem)
            continue
        video_paths.append(path)

    if not video_paths:
        raise HTTPException(400, "All selected videos already have tracking (enable overwrite)")

    job = job_manager.create_job(
        "player_tracking",
        {
            "videos": [p.name for p in video_paths],
            "skipped_existing": skipped,
            "items": init_batch_items([p.name for p in video_paths]),
        },
        name=f"Rally Tracking ({len(video_paths)} videos)",
    )
    _spawn_batch_job(
        job,
        video_paths,
        stop_vllm=req.stop_vllm,
        work=lambda p, cb: tracking.track_video(p, stride=req.stride, on_progress=cb),
        done_message=lambda c: f"{c['tracklets']} tracklets over {c['frames']} frames",
        progress_noun="frame",
        start_message="tracking rallies...",
    )
    return job.to_dict()


@router.get("/tracks/{name}")
def tracks(name: str) -> dict:
    """Tracklets (for the video overlay) + event→tracklet links (for crop
    badges and propagation). Boxes are truncated to whole pixels — the
    overlay doesn't need tenths and the payload holds ~100k boxes."""
    stem = Path(unquote(name)).stem
    if not tracking.tracks_path(stem).exists():
        raise HTTPException(404, f"No tracking for {stem} — run tracking on the ReID Predict page first")
    if not pipeline.reid_path(stem).exists():
        raise HTTPException(404, f"No ReID results for {stem}")
    _meta, tracklets = read_jsonl_cached(tracking.tracks_path(stem))  # read-only — copy, never mutate
    slim = [
        {
            "rally_id": t["rally_id"],
            "track_id": t["track_id"],
            "frames": t["frames"],
            "boxes": [[int(v) for v in b] for b in t["boxes"]],
        }
        for t in tracklets
    ]
    return {"tracklets": slim, "links": tracking.link_events(stem)}


@router.get("/results/{name}")
def results(name: str) -> dict:
    """One video's extraction records, embeddings stripped (UI payload)."""
    stem = Path(unquote(name)).stem
    path = pipeline.reid_path(stem)
    if not path.exists():
        raise HTTPException(404, f"No ReID results for {stem}")
    # Cached parse shares objects across requests — strip into copies, never
    # mutate what read_jsonl_cached hands out.
    meta, records = read_jsonl_cached(path)
    meta = dict(meta)
    # The video-sync overlay needs fps (frame ↔ time) and the rally spans for
    # its rally navigator — both live in the annotation header, not the
    # extraction header.
    ann = pipeline.action_annotation_path(stem)
    if ann is not None:
        ann_meta, _ = read_jsonl(ann)
        if not meta.get("fps") and ann_meta.get("fps"):
            meta["fps"] = ann_meta["fps"]
        meta["rallies"] = ann_meta.get("rallies") or []
    out = []
    # Drop score events from old extractions too (see pipeline.SKIP_LABELS).
    for r in records:
        if r.get("label") in pipeline.SKIP_LABELS:
            continue
        r = {k: v for k, v in r.items() if k != "embeddings"}
        # The actor picker only needs boxes + scores; skeletons stay server-side.
        if r.get("detections"):
            r["detections"] = [{k: v for k, v in d.items() if k != "keypoints"} for d in r["detections"]]
        out.append(r)
    return {"meta": meta, "records": out}


@router.get("/crop/{name}/{crop_file}")
def crop(name: str, crop_file: str) -> FileResponse:
    stem = Path(unquote(name)).stem
    path = pipeline.crop_dir(stem) / Path(unquote(crop_file)).name
    if not path.exists():
        raise HTTPException(404, "Crop not found")
    return FileResponse(path, media_type="image/jpeg")


def _validated_model(model: str) -> str:
    if model not in EMBEDDER_WEIGHTS:
        raise HTTPException(400, f"Unknown embedder: {model} (have: {', '.join(EMBEDDER_WEIGHTS)})")
    return model


@router.get("/clusters/{name}")
def clusters(
    name: str,
    threshold: float = identity.DEFAULT_CLUSTER_THRESHOLD,
    model: str = DEFAULT_EMBEDDER,
) -> dict:
    """Unsupervised grouping of one video's embeddings (event ids per cluster)."""
    stem = Path(unquote(name)).stem
    try:
        records, matrix = identity.load_embeddings(stem, model=_validated_model(model))
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    labels = identity.cluster(matrix, threshold=threshold)
    grouped: dict[int, list[str]] = {}
    for record, label in zip(records, labels):
        grouped.setdefault(int(label), []).append(record["id"])
    return {
        "threshold": threshold,
        "model": model,
        "clusters": [
            {"id": label, "size": len(ids), "event_ids": ids}
            for label, ids in sorted(grouped.items())
        ],
    }


class SaveAssignmentsRequest(BaseModel):
    assignments: dict[str, str]


@router.get("/players/{name}")
def get_players(name: str, model: str = DEFAULT_EMBEDDER) -> dict:
    """Saved identities + nearest-centroid match for every embedded event."""
    stem = Path(unquote(name)).stem
    assignments = identity.load_assignments(stem)
    matches: dict[str, dict] = {}
    if assignments:
        try:
            records, matrix = identity.load_embeddings(stem, model=_validated_model(model))
            matches = identity.match(records, matrix, assignments)
        except FileNotFoundError:
            pass
    return {
        "assignments": assignments,
        "players": sorted(set(assignments.values())),
        "matches": matches,
    }


@router.put("/players/{name}")
def put_players(name: str, req: SaveAssignmentsRequest, model: str = DEFAULT_EMBEDDER) -> dict:
    stem = Path(unquote(name)).stem
    if not pipeline.reid_path(stem).exists():
        raise HTTPException(404, f"No ReID results for {stem}")
    identity.save_assignments(stem, req.assignments)
    return get_players(name, model=model)


class SeedClusterRequest(BaseModel):
    # Seed key (the UI's group row key) -> event ids anchoring that group.
    seeds: dict[str, list[str]]
    threshold: float = identity.DEFAULT_CLUSTER_THRESHOLD
    model: str = DEFAULT_EMBEDDER


@router.post("/seed-cluster/{name}")
def seed_cluster(name: str, req: SeedClusterRequest) -> dict:
    """Distribute unassigned events to the nearest user-seeded group.

    Events farther than ``threshold`` from every seed centroid stay out;
    they come back agglomeratively clustered (same threshold) so the UI can
    show them as leftover pools for further seeding.
    """
    stem = Path(unquote(name)).stem
    try:
        records, matrix = identity.load_embeddings(stem, model=_validated_model(req.model))
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    groups, leftover_ids = identity.seeded_groups(records, matrix, req.seeds, req.threshold)
    leftover_clusters: list[list[str]] = []
    if leftover_ids:
        index = {r["id"]: i for i, r in enumerate(records)}
        rows = [index[i] for i in leftover_ids]
        labels = identity.cluster(matrix[rows], threshold=req.threshold)
        grouped: dict[int, list[str]] = {}
        for event_id, label in zip(leftover_ids, labels):
            grouped.setdefault(int(label), []).append(event_id)
        leftover_clusters = [ids for _, ids in sorted(grouped.items())]
    return {"groups": groups, "leftover_clusters": leftover_clusters}


class ActorFixRequest(BaseModel):
    event_id: str
    # box = manual pick; none = nobody is the actor; neither = revert to auto.
    box: list[float] | None = Field(default=None, min_length=4, max_length=4)
    none: bool = False


@router.post("/actor-fix/{name}")
def actor_fix(name: str, req: ActorFixRequest) -> dict:
    """Re-point one event at the person the user clicked (or nobody / auto).

    The fix lands in the players file (the durable human record, replayed on
    re-extraction) and is applied to the extraction jsonl immediately: the
    chosen box is cropped and re-embedded, so clusters/centroids follow.
    """
    video_path = find_cut(unquote(name))
    if video_path is None:
        raise HTTPException(404, f"Video not found: {name}")
    stem = video_path.stem
    if not pipeline.reid_path(stem).exists():
        raise HTTPException(404, f"No ReID results for {stem}")
    try:
        if req.box is not None:
            identity.save_actor_fix(stem, req.event_id, req.box)
        elif req.none:
            identity.save_actor_fix(stem, req.event_id, None)
        else:
            identity.remove_actor_fix(stem, req.event_id)
        record = pipeline.apply_actor_fix(video_path, req.event_id, req.box, none=req.none)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    for d in record.get("detections") or []:
        d.pop("keypoints", None)
    return {"record": record}
