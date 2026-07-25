"""Player ReID router.

Runs the tracking-free extraction (RF-DETR person detection → contact point
association → embedding) over the annotated action events of selected cut
videos, and serves the ReID Label page: crops, clusters and player
assignments. Results land in reid/ as per-video jsonl + crop images.

Reviewing WHICH person performed each action is the Association Label page's
job and lives in routers/actor_association.py; the only thing the two share
is the extraction records both read.
"""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from yp_video.actor import labels as actor_labels
from yp_video.config import cut_kind_of, find_cut, iter_all_cuts
from yp_video.core.cache import StatCache
from yp_video.core.jsonl import read_jsonl, read_jsonl_cached
from yp_video.core.rallies import rally_sources
from yp_video.extraction import done, links, pipeline
from yp_video.extraction.prerequisites import prerequisites
from yp_video.extraction import store as extraction_store
from yp_video.person.detector import build_keypoint_sources
from yp_video.reid import checkpoints, identity, store
from yp_video.tracklets import store as tracks_store
from yp_video.tracklets import tracking
from yp_video.reid.embedder import DEFAULT_EMBEDDER, EMBEDDER_NAMES, build_embedders, threshold_calibration
from yp_video.web.job_helpers import init_batch_items, spawn_batch_video_job
from yp_video.web.jobs import job_manager

log = logging.getLogger(__name__)
router = APIRouter()


def _resolve_optional_checkpoint(ref: str | None) -> Path | None:
    """A checkpoint ref from the UI; None keeps the official default."""
    if not ref:
        return None
    try:
        return checkpoints.resolve_checkpoint(ref)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        raise HTTPException(400, f"Bad checkpoint: {exc}") from exc


class ReidStartRequest(BaseModel):
    videos: list[str] = Field(min_length=1)
    overwrite: bool = False
    stop_vllm: bool = False
    # Keypoint source from the registry (see /reid/options); detection
    # itself is always RF-DETR.
    keypoints: str = "rf-detr"
    # Checkpoint package ref for the clip-reident embedder; None = official
    # default. Only affects the clip-reident family.
    checkpoint: str | None = None


def _read_header(stem: str) -> dict | None:
    path = extraction_store.records_path(stem)
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
        players = identity.load_players(f.stem)
        results.append({
            "name": f.name,
            "kind": cut_kind_of(f),
            "event_count": len(events),
            "has_reid": header is not None,
            "reid_counts": (
                {k: header.get(k, 0) for k in ("ok", "multi", "miss")} if header else None
            ),
            "embedded_models": store.embedded_models(f.stem),
            "stale_embedding_models": store.stale_embedding_models(f.stem),
            "player_count": len(
                set(players.tracks.values()) | set(players.assignments.values())
            ),
            "done": players.done,
            "pipeline": prerequisites(f.stem).payload(),
        })
    return results


@router.get("/options")
def options() -> dict:
    """Available keypoint-source / embedder choices for the Predict / Label
    pages. Each embedder ships its cluster-threshold slider calibration, so
    adding a model server-side never needs a frontend edit."""
    registry = build_embedders()
    runs = checkpoints.list_checkpoints()
    return {
        "keypoint_sources": list(build_keypoint_sources()),
        "default_embedder": DEFAULT_EMBEDDER if DEFAULT_EMBEDDER in registry else next(iter(registry)),
        "embedders": [
            # masked → the crop viewer should show the crops-masked variant.
            {"name": n, "threshold": threshold_calibration(n), "masked": getattr(e, "masked_input", False)}
            for n, e in registry.items()
        ],
        "checkpoints": [
            {
                "ref": r["path"],
                "run_name": r["run_name"],
                "active": r["active"],
            }
            for r in runs
        ],
    }


@router.post("/start")
async def start(req: ReidStartRequest) -> dict:
    checkpoint = _resolve_optional_checkpoint(req.checkpoint)
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
        if extraction_store.action_annotation_path(path.stem) is None:
            raise HTTPException(400, f"No action annotations for: {name}")
        # Association resolves an actor to a tracklet, so extraction without
        # tracking would produce records nobody can label at the current unit.
        if not tracks_store.tracks_path(path.stem).exists():
            raise HTTPException(
                400, f"No tracking for: {name} — run Rally Tracking first"
            )
        if not req.overwrite and extraction_store.records_path(path.stem).exists():
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
        work=lambda p, cb: pipeline.extract_video(p, keypoints=req.keypoints, checkpoint=checkpoint, on_progress=cb),
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
    """Dense per-rally detection + ByteTrack (see tracklets/tracking.py)."""
    video_paths: list[Path] = []
    skipped: list[str] = []
    for name in req.videos:
        path = find_cut(name)
        if path is None:
            raise HTTPException(404, f"Video not found: {name}")
        # Tracking needs rally spans and nothing else — deliberately NOT the
        # action annotation, so it can run alongside action labeling.
        if not rally_sources(path.stem):
            raise HTTPException(
                400,
                f"No rally spans for: {name} — label rallies or run Rally SPOT Predict",
            )
        if not req.overwrite and tracks_store.tracks_path(path.stem).exists():
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
    # Checkpoint package ref for the clip-reident embedder; None = official
    # default. Only affects the clip-reident family.
    checkpoint: str | None = None


@router.post("/embed")
async def embed(req: EmbedStartRequest) -> dict:
    """Backfill embedding matrices from the saved crops (see pipeline.embed_video).

    This is how a newly registered embedder covers already-extracted videos —
    no re-extraction, the video file is never opened.
    """
    checkpoint = _resolve_optional_checkpoint(req.checkpoint)
    registry = build_embedders()
    unknown = set(req.models or ()) - set(registry)
    if unknown:
        raise HTTPException(400, f"Unknown embedders: {', '.join(sorted(unknown))} (have: {', '.join(registry)})")
    video_paths: list[Path] = []
    for name in req.videos:
        path = find_cut(name)
        if path is None:
            raise HTTPException(404, f"Video not found: {name}")
        if not extraction_store.records_path(path.stem).exists():
            raise HTTPException(400, f"No extraction records for {name} — run extraction first")
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
        work=lambda p, cb: pipeline.embed_video(p.stem, models=req.models, overwrite=req.overwrite, checkpoint=checkpoint, on_progress=cb),
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
    if not tracks_store.tracks_path(stem).exists():
        raise HTTPException(404, f"No tracking for {stem} — run tracking on the ReID Predict page first")
    if not extraction_store.records_path(stem).exists():
        raise HTTPException(404, f"No extraction records for {stem}")

    def slim() -> list[dict]:
        _meta, tracklets = read_jsonl_cached(tracks_store.tracks_path(stem))  # read-only — copy, never mutate
        return [{k: t[k] for k in ("rally_id", "track_id", "frames", "boxes")} for t in tracklets]

    return {
        "tracklets": _slim_tracks_cache.get(stem, [tracks_store.tracks_path(stem)], slim),
        "links": links.link_payload(stem),
    }


@router.get("/track-masks/{name}")
def track_masks(name: str, rally: int) -> dict:
    """One rally's instance masks, whole tracklets at once — the overlay
    silhouettes. Each entry is the tracklet's packed mask rows (base64,
    box-crop space, see tracklets/store.save_track_masks), row i ↔ the tracklet's
    i-th frame in the tracks jsonl the client already holds."""
    import base64

    import numpy as np

    stem = Path(unquote(name)).stem
    masks_path = tracks_store.tracks_masks_path(stem)
    if not masks_path.exists():
        raise HTTPException(404, f"No track masks for {stem} — re-run tracking")
    _meta, tracklets = read_jsonl_cached(tracks_store.tracks_path(stem))  # read-only
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
    records = _slim_records_cache.get(
        stem, [path], lambda: _slim_records(path)
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
            for record in records
        ],
    }


def _slim_records(path: Path) -> list[dict]:
    _meta, records = read_jsonl_cached(path)
    out = []
    # Drop score events from old extractions too (see extraction_store.SKIP_LABELS).
    for r in records:
        if r.get("label") in extraction_store.SKIP_LABELS:
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
    path = extraction_store.masked_crop_dir(stem) / fname if masked else extraction_store.crop_dir(stem) / fname
    if masked and not path.exists():
        path = extraction_store.crop_dir(stem) / fname
    if not path.exists():
        raise HTTPException(404, "Crop not found")
    return FileResponse(path, media_type="image/jpeg")


def _validated_model(model: str) -> str:
    if model not in EMBEDDER_NAMES:
        raise HTTPException(400, f"Unknown embedder: {model} (have: {', '.join(EMBEDDER_NAMES)})")
    return model


def _validated_fresh_model(stem: str, model: str) -> str:
    model = _validated_model(model)
    if not store.embedding_is_fresh(stem, model):
        if store.embedding_path(stem, model).exists():
            raise HTTPException(
                409,
                f"{model} embeddings for {stem} are refreshing after an actor fix",
            )
        raise HTTPException(
            404, f"No {model} embeddings for {stem} — backfill the model first"
        )
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
    """Unsupervised grouping of one video's units (see reid/identity.py).

    A unit is a tracklet where one exists and a lone event otherwise, so the
    board can render crops without a second round trip: ``units`` carries the
    membership every cluster entry refers to.
    """
    stem = Path(unquote(name)).stem
    unit_links = links.track_keys(stem)
    units, labels = _load_or_http(
        lambda: identity.cluster_video(
            stem, _validated_fresh_model(stem, model), threshold, unit_links
        )
    )
    grouped: dict[int, list[str]] = {}
    for unit, label in zip(units, labels):
        grouped.setdefault(int(label), []).append(unit.key)
    return {
        "threshold": threshold,
        "model": model,
        "units": {u.key: {"event_ids": list(u.event_ids)} for u in units},
        "clusters": [
            {"id": label, "size": len(keys), "unit_keys": keys}
            for label, keys in sorted(grouped.items())
        ],
    }


class SavePlayersRequest(BaseModel):
    """The naming maps as the board holds them (see identity.PlayersFile)."""

    tracks: dict[str, str] = Field(default_factory=dict)
    assignments: dict[str, str] = Field(default_factory=dict)


@router.get("/players/{name}")
def get_players(name: str, model: str = DEFAULT_EMBEDDER) -> dict:
    """Saved identities + nearest-centroid match for every unit."""
    stem = Path(unquote(name)).stem
    players = identity.load_players(stem)
    unit_links = links.track_keys(stem)
    matches: dict[str, dict] = {}
    names: dict[str, str] = {}
    if players.tracks or players.assignments:
        records, matrix = _load_or_http(
            lambda: identity.load_embeddings(
                stem, model=_validated_fresh_model(stem, model)
            )
        )
        units, unit_matrix = identity.unit_embeddings(records, matrix, unit_links)
        names = identity.unit_names(units, players)
        matches = identity.match(units, unit_matrix, names)
    return {
        "tracks": players.tracks,
        "assignments": players.assignments,
        "unit_names": names,
        "players": sorted(set(players.tracks.values()) | set(players.assignments.values())),
        "matches": matches,
    }


class DoneRequest(BaseModel):
    done: bool = True
    confirm_auto_actors: bool = False


@router.put("/done/{name}")
def put_done(name: str, req: DoneRequest) -> dict:
    """Mark (or unmark) a video's labeling as finished — the Label page's
    Done button. A human verdict, stored alongside the assignments."""
    stem = Path(unquote(name)).stem
    if not extraction_store.records_path(stem).exists():
        raise HTTPException(404, f"No extraction records for {stem}")
    confirmed = done.mark_done(
        stem,
        req.done,
        confirm_auto=req.confirm_auto_actors,
    )
    return {"done": req.done, "confirmed_auto_actors": confirmed}


@router.put("/players/{name}")
def put_players(name: str, req: SavePlayersRequest) -> dict:
    """Persist the naming maps. Returns them without matches — a save must
    succeed even when the current model's matrix is missing."""
    stem = Path(unquote(name)).stem
    if not extraction_store.records_path(stem).exists():
        raise HTTPException(404, f"No extraction records for {stem}")
    identity.save_players(stem, tracks=req.tracks, assignments=req.assignments)
    return {
        "tracks": req.tracks,
        "assignments": req.assignments,
        "players": sorted(set(req.tracks.values()) | set(req.assignments.values())),
    }


class SeedClusterRequest(BaseModel):
    # Seed key (the UI's group row key) -> unit keys anchoring that group.
    seeds: dict[str, list[str]]
    threshold: float = identity.DEFAULT_CLUSTER_THRESHOLD
    model: str = DEFAULT_EMBEDDER


@router.post("/seed-cluster/{name}")
def seed_cluster(name: str, req: SeedClusterRequest) -> dict:
    """Distribute unnamed units to the nearest user-seeded group.

    Units farther than ``threshold`` from every seed centroid stay out; they
    come back agglomeratively clustered (same threshold) so the UI can show
    them as leftover pools for further seeding.
    """
    stem = Path(unquote(name)).stem
    records, matrix = _load_or_http(lambda: identity.load_embeddings(stem, model=_validated_model(req.model)))
    units, matrix = identity.unit_embeddings(records, matrix, links.track_keys(stem))
    groups, leftover_ids = identity.seeded_groups(units, matrix, req.seeds, req.threshold)
    leftover_clusters: list[list[str]] = []
    if leftover_ids:
        index = {u.key: i for i, u in enumerate(units)}
        rows = [index[i] for i in leftover_ids]
        labels = identity.cluster(matrix[rows], threshold=req.threshold)
        grouped: dict[int, list[str]] = {}
        for event_id, label in zip(leftover_ids, labels):
            grouped.setdefault(int(label), []).append(event_id)
        leftover_clusters = [ids for _, ids in sorted(grouped.items())]
    return {"groups": groups, "leftover_clusters": leftover_clusters}
