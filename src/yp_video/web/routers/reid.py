"""Player ReID: putting the same person in the same group.

That is the whole job. Given crops somebody has already agreed show the right
person, turn them into vectors and group the vectors — clusters, seeded
groups, nearest-centroid matches, and the player names a human puts on them.

Everything else that used to live here answered a different question and now
lives where it belongs: finding the people and choosing who acted is
routers/extraction.py, who is on court over time is routers/tracklets.py,
and reviewing whether the right person was cropped is
routers/actor_association.py. They were all under /reid because they happened
to feed it, which is not the same as being it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from yp_video.config import cut_kind_of, find_cut, iter_all_cuts
from yp_video.extraction import done, links, pipeline
from yp_video.extraction import store as extraction_store
from yp_video.extraction.prerequisites import prerequisites
from yp_video.reid import checkpoints, identity, store
from yp_video.reid.embedder import (
    DEFAULT_EMBEDDER,
    EMBEDDER_NAMES,
    build_embedders,
    threshold_calibration,
)
from yp_video.web.job_helpers import init_batch_items, spawn_batch_video_job
from yp_video.web.jobs import JobSummary, JobType, job_manager

log = logging.getLogger(__name__)
router = APIRouter()


@router.get("/videos")
def list_videos() -> list[dict]:
    """Extracted videos and how far their player naming has got."""
    results = []
    for f in sorted(iter_all_cuts(), key=lambda p: p.name):
        events = pipeline.load_events(f.stem)
        if not events:
            continue
        players = store.load_players(f.stem)
        results.append({
            "name": f.name,
            "kind": cut_kind_of(f),
            "event_count": len(events),
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
    """Available embedders for the Embed / Label pages. Each ships its
    cluster-threshold slider calibration, so adding a model server-side never
    needs a frontend edit."""
    registry = build_embedders()
    runs = checkpoints.list_checkpoints()
    return {
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


class EmbedRequest(BaseModel):
    videos: list[str] = Field(min_length=1)
    # None = every registered embedder; missing matrices only unless overwrite.
    models: list[str] | None = None
    overwrite: bool = False
    stop_vllm: bool = False
    # Checkpoint package ref for the clip-reident embedder; None = official
    # default. Only affects the clip-reident family.
    checkpoint: str | None = None


@router.post("/embed", response_model=JobSummary)
async def embed(req: EmbedRequest) -> dict:
    """Saved crops → one embedding matrix per model (see pipeline.embed_video).

    Run once the actors are reviewed. The video file is never opened, so this
    is also how a newly registered embedder covers already-extracted videos.
    """
    checkpoint = None
    if req.checkpoint:
        try:
            checkpoint = checkpoints.resolve_checkpoint(req.checkpoint)
        except (ValueError, FileNotFoundError, KeyError) as exc:
            raise HTTPException(400, f"Bad checkpoint: {exc}") from exc
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
            raise HTTPException(400, f"No extraction records for {name} — run Extraction first")
        video_paths.append(path)

    job = job_manager.create_job(
        JobType.PLAYER_EMBED,
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


def _validated_fresh_model(stem: str, model: str) -> str:
    """A registered embedder whose matrix is current for this video."""
    if model not in EMBEDDER_NAMES:
        raise HTTPException(400, f"Unknown embedder: {model} (have: {', '.join(EMBEDDER_NAMES)})")
    if not store.embedding_is_fresh(stem, model):
        if store.embedding_path(stem, model).exists():
            raise HTTPException(
                409,
                f"{model} embeddings for {stem} are refreshing after an actor fix",
            )
        raise HTTPException(
            404, f"No {model} embeddings for {stem} — run Embedding first"
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


def _grouped(keys: list[str], labels) -> dict[int, list[str]]:
    """Cluster label → its members, in the order the clusterer emitted them."""
    out: dict[int, list[str]] = {}
    for key, label in zip(keys, labels):
        out.setdefault(int(label), []).append(key)
    return out


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
    grouped = _grouped([u.key for u in units], labels)
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
    """The naming maps as the board holds them (see store.PlayersFile)."""

    tracks: dict[str, str] = Field(default_factory=dict)
    assignments: dict[str, str] = Field(default_factory=dict)


@router.get("/players/{name}")
def get_players(name: str, model: str = DEFAULT_EMBEDDER) -> dict:
    """Saved identities + nearest-centroid match for every unit."""
    stem = Path(unquote(name)).stem
    players = store.load_players(stem)
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
    store.save_players(stem, tracks=req.tracks, assignments=req.assignments)
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
    records, matrix = _load_or_http(
        lambda: identity.load_embeddings(
            stem, model=_validated_fresh_model(stem, req.model)
        )
    )
    units, matrix = identity.unit_embeddings(records, matrix, links.track_keys(stem))
    groups, leftover_ids = identity.seeded_groups(units, matrix, req.seeds, req.threshold)
    leftover_clusters: list[list[str]] = []
    if leftover_ids:
        index = {u.key: i for i, u in enumerate(units)}
        rows = [index[i] for i in leftover_ids]
        labels = identity.cluster(matrix[rows], threshold=req.threshold)
        grouped = _grouped(leftover_ids, labels)
        leftover_clusters = [ids for _, ids in sorted(grouped.items())]
    return {"groups": groups, "leftover_clusters": leftover_clusters}
