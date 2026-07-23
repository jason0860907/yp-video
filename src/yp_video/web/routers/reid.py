"""Player ReID router.

Runs the tracking-free ReID extraction (RF-DETR person detection → contact
point association → OSNet embedding) over the annotated action events of
selected cut videos. Results land in reid/ as per-video jsonl +
crop images; identity matching consumes them later.
"""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from yp_video.config import cut_kind_of, find_cut, iter_all_cuts
from yp_video.core.cache import StatCache
from yp_video.core.jsonl import read_jsonl, read_jsonl_cached
from yp_video.reid import identity, pipeline, store, tracking
from yp_video.reid.detector import build_keypoint_sources
from yp_video.reid.embedder import DEFAULT_EMBEDDER, EMBEDDER_NAMES, build_embedders, threshold_calibration
from yp_video.web.job_helpers import init_batch_items, spawn_batch_video_job
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
    path = store.reid_path(stem)
    if not path.exists():
        return None
    return read_jsonl_cached(path)[0] or None  # read-only — shared cached object


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
            "embedded_models": store.embedded_models(f.stem),
            "player_count": len(set(identity.load_assignments(f.stem).values())),
            "done": identity.load_done(f.stem),
        })
    return results


@router.get("/options")
def options() -> dict:
    """Available keypoint-source / embedder choices for the Predict / Label
    pages. Each embedder ships its cluster-threshold slider calibration, so
    adding a model server-side never needs a frontend edit."""
    registry = build_embedders()
    return {
        "keypoint_sources": list(build_keypoint_sources()),
        "default_embedder": DEFAULT_EMBEDDER if DEFAULT_EMBEDDER in registry else next(iter(registry)),
        "embedders": [
            # masked → the crop viewer should show the crops-masked variant.
            {"name": n, "threshold": threshold_calibration(n), "masked": getattr(e, "masked_input", False)}
            for n, e in registry.items()
        ],
    }


@router.post("/start")
async def start(req: ReidStartRequest) -> dict:
    keypoint_sources = build_keypoint_sources()
    if req.keypoints not in keypoint_sources:
        raise HTTPException(
            400,
            f"Unknown keypoint source: {req.keypoints} (available: {', '.join(keypoint_sources)} — "
            "sam-3d-body needs its gated HF checkpoint downloaded first)",
        )
    video_paths: list[Path] = []
    skipped: list[str] = []
    for name in req.videos:
        path = find_cut(name)
        if path is None:
            raise HTTPException(404, f"Video not found: {name}")
        if store.action_annotation_path(path.stem) is None:
            raise HTTPException(400, f"No action annotations for: {name}")
        if not req.overwrite and store.reid_path(path.stem).exists():
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
    spawn_batch_video_job(
        job,
        video_paths,
        stop_vllm=req.stop_vllm,
        work=lambda p, cb: pipeline.extract_video(p, keypoints=req.keypoints, on_progress=cb),
        done_message=lambda c: f"{c['ok']} ok · {c['multi']} multi · {c['miss']} miss",
        start_message="detecting players...",
    )
    return job.to_dict()


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
        if store.action_annotation_path(path.stem) is None:
            raise HTTPException(400, f"No action annotations for: {name}")
        if not req.overwrite and store.tracks_path(path.stem).exists():
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
    spawn_batch_video_job(
        job,
        video_paths,
        stop_vllm=req.stop_vllm,
        work=lambda p, cb: tracking.track_video(p, stride=req.stride, on_progress=cb),
        done_message=lambda c: f"{c['tracklets']} tracklets over {c['frames']} frames",
        start_message="tracking rallies...",
    )
    return job.to_dict()


class EmbedStartRequest(BaseModel):
    videos: list[str] = Field(min_length=1)
    # None = every registered embedder; missing matrices only unless overwrite.
    models: list[str] | None = None
    overwrite: bool = False
    stop_vllm: bool = False


@router.post("/embed")
async def embed(req: EmbedStartRequest) -> dict:
    """Backfill embedding matrices from the saved crops (see pipeline.embed_video).

    This is how a newly registered embedder covers already-extracted videos —
    no re-extraction, the video file is never opened.
    """
    registry = build_embedders()
    unknown = set(req.models or ()) - set(registry)
    if unknown:
        raise HTTPException(400, f"Unknown embedders: {', '.join(sorted(unknown))} (have: {', '.join(registry)})")
    video_paths: list[Path] = []
    for name in req.videos:
        path = find_cut(name)
        if path is None:
            raise HTTPException(404, f"Video not found: {name}")
        if not store.reid_path(path.stem).exists():
            raise HTTPException(400, f"No ReID results for {name} — run extraction first")
        video_paths.append(path)

    job = job_manager.create_job(
        "player_embed",
        {"videos": [p.name for p in video_paths], "items": init_batch_items([p.name for p in video_paths])},
        name=f"Embeddings ({len(video_paths)} videos)",
    )
    spawn_batch_video_job(
        job,
        video_paths,
        stop_vllm=req.stop_vllm,
        work=lambda p, cb: pipeline.embed_video(p.stem, models=req.models, overwrite=req.overwrite, on_progress=cb),
        done_message=lambda c: (
            f"{', '.join(c['models'])} over {c['crops']} crops" if c["models"] else "already embedded"
        ),
        start_message="embedding crops...",
    )
    return job.to_dict()


# Slimmed UI payloads, rebuilt only when their source files change. Values
# are shared across requests — read-only, like everything cached.
_slim_tracks_cache: StatCache = StatCache()
_slim_records_cache: StatCache = StatCache()


@router.get("/tracks/{name}")
def tracks(name: str) -> dict:
    """Tracklets (for the video overlay) + event→tracklet links (for crop
    badges and propagation). Scores stay server-side — the overlay only
    draws boxes, and the payload holds ~286k of them (8.5 MB of JSON, but
    ~1.5 MB over the wire once GZipMiddleware has had it)."""
    stem = Path(unquote(name)).stem
    if not store.tracks_path(stem).exists():
        raise HTTPException(404, f"No tracking for {stem} — run tracking on the ReID Predict page first")
    if not store.reid_path(stem).exists():
        raise HTTPException(404, f"No ReID results for {stem}")

    def slim() -> list[dict]:
        _meta, tracklets = read_jsonl_cached(store.tracks_path(stem))  # read-only — copy, never mutate
        return [{k: t[k] for k in ("rally_id", "track_id", "frames", "boxes")} for t in tracklets]

    return {
        "tracklets": _slim_tracks_cache.get(stem, [store.tracks_path(stem)], slim),
        "links": tracking.link_events(stem),
    }


@router.get("/track-masks/{name}")
def track_masks(name: str, rally: int) -> dict:
    """One rally's instance masks, whole tracklets at once — the overlay
    silhouettes. Each entry is the tracklet's packed mask rows (base64,
    box-crop space, see store.save_track_masks), row i ↔ the tracklet's
    i-th frame in the tracks jsonl the client already holds."""
    import base64

    import numpy as np

    stem = Path(unquote(name)).stem
    masks_path = store.tracks_masks_path(stem)
    if not masks_path.exists():
        raise HTTPException(404, f"No track masks for {stem} — re-run tracking")
    _meta, tracklets = read_jsonl_cached(store.tracks_path(stem))  # read-only
    tracks: dict[str, str] = {}
    with np.load(masks_path) as z:
        h, w = (int(v) for v in z["_shape"])
        for t in tracklets:
            key = f"{t['rally_id']}:{t['track_id']}"
            if t["rally_id"] == rally and key in z:
                tracks[key] = base64.b64encode(z[key].tobytes()).decode()
    return {"mask_hw": [h, w], "tracks": tracks}


@router.get("/results/{name}")
def results(name: str) -> dict:
    """One video's extraction records (UI payload)."""
    stem = Path(unquote(name)).stem
    path = store.reid_path(stem)
    if not path.exists():
        raise HTTPException(404, f"No ReID results for {stem}")
    # Cached parse shares objects across requests — filter into copies, never
    # mutate what read_jsonl_cached hands out.
    meta, _records = read_jsonl_cached(path)
    meta = dict(meta)
    # The video-sync overlay needs fps (frame ↔ time) and the rally spans for
    # its rally navigator — both live in the annotation header, not the
    # extraction header.
    ann = store.action_annotation_path(stem)
    if ann is not None:
        ann_meta, _ = read_jsonl(ann)
        if not meta.get("fps") and ann_meta.get("fps"):
            meta["fps"] = ann_meta["fps"]
        meta["rallies"] = ann_meta.get("rallies") or []
    return {"meta": meta, "records": _slim_records_cache.get(stem, [path], lambda: _slim_records(path))}


def _slim_records(path: Path) -> list[dict]:
    _meta, records = read_jsonl_cached(path)
    out = []
    # Drop score events from old extractions too (see store.SKIP_LABELS).
    for r in records:
        if r.get("label") in store.SKIP_LABELS:
            continue
        r = dict(r)
        # The actor picker only needs boxes + scores; skeletons stay server-side.
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
    path = store.masked_crop_dir(stem) / fname if masked else store.crop_dir(stem) / fname
    if masked and not path.exists():
        path = store.crop_dir(stem) / fname
    if not path.exists():
        raise HTTPException(404, "Crop not found")
    return FileResponse(path, media_type="image/jpeg")


def _validated_model(model: str) -> str:
    if model not in EMBEDDER_NAMES:
        raise HTTPException(400, f"Unknown embedder: {model} (have: {', '.join(EMBEDDER_NAMES)})")
    return model


def _load_or_http(loader):
    """Run a reid data loader with its failures mapped to actionable HTTP
    errors: matrix file missing → 404, matrix/record row mismatch → 409."""
    try:
        return loader()
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/clusters/{name}")
def clusters(
    name: str,
    threshold: float = identity.DEFAULT_CLUSTER_THRESHOLD,
    model: str = DEFAULT_EMBEDDER,
) -> dict:
    """Unsupervised grouping of one video's embeddings (event ids per cluster)."""
    stem = Path(unquote(name)).stem
    records, labels = _load_or_http(lambda: identity.cluster_video(stem, _validated_model(model), threshold))
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
        records, matrix = _load_or_http(lambda: identity.load_embeddings(stem, model=_validated_model(model)))
        matches = identity.match(records, matrix, assignments)
    return {
        "assignments": assignments,
        "players": sorted(set(assignments.values())),
        "matches": matches,
    }


class DoneRequest(BaseModel):
    done: bool = True


@router.put("/done/{name}")
def put_done(name: str, req: DoneRequest) -> dict:
    """Mark (or unmark) a video's labeling as finished — the Label page's
    Done button. A human verdict, stored alongside the assignments."""
    stem = Path(unquote(name)).stem
    if not store.reid_path(stem).exists():
        raise HTTPException(404, f"No ReID results for {stem}")
    identity.save_done(stem, req.done)
    return {"done": req.done}


@router.put("/players/{name}")
def put_players(name: str, req: SaveAssignmentsRequest) -> dict:
    """Persist assignments. Returns them without matches — a save must
    succeed even when the current model's matrix is missing."""
    stem = Path(unquote(name)).stem
    if not store.reid_path(stem).exists():
        raise HTTPException(404, f"No ReID results for {stem}")
    identity.save_assignments(stem, req.assignments)
    return {
        "assignments": req.assignments,
        "players": sorted(set(req.assignments.values())),
    }


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
    records, matrix = _load_or_http(lambda: identity.load_embeddings(stem, model=_validated_model(req.model)))
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
    # Cross-frame pick: the box lives on this frame, not the event's — the
    # crop is cut from here (actor undetected on the event frame).
    frame: int | None = None
    # False = the client's mask arbitration ruled that NO stored detection is
    # this player — embed the box as drawn, never IoU-snap onto an occluder.
    snap: bool = True


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
    if not store.reid_path(stem).exists():
        raise HTTPException(404, f"No ReID results for {stem}")
    try:
        if req.box is not None:
            identity.save_actor_fix(stem, req.event_id, req.box, frame=req.frame, snap=req.snap)
        elif req.none:
            identity.save_actor_fix(stem, req.event_id, None)
        else:
            identity.remove_actor_fix(stem, req.event_id)
        # Any re-pick invalidates the event's player assignment: the crop is a
        # different person now, so it returns to the unassigned pool.
        identity.remove_assignment(stem, req.event_id)
        record = pipeline.apply_actor_fix(video_path, req.event_id, req.box, none=req.none, frame=req.frame, snap=req.snap)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    for d in record.get("detections") or []:
        d.pop("keypoints", None)
    return {"record": record}
