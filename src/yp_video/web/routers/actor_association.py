"""Actor association: the labeling work list, the fix endpoint, and learning.

Serves the Association Label page — which video still has unreviewed actors,
and the one write that answers "this person performed this action". Player
identity is the ReID router's business; the only thing the two share is the
extraction records they both read.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import unquote

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from yp_video.actor import checkpoints as actor_checkpoints
from yp_video.actor import dataset as actor_dataset
from yp_video.actor import evaluate as actor_evaluate
from yp_video.actor import labels as actor_labels
from yp_video.actor import policy as actor_policy
from yp_video.actor import train as actor_train
from yp_video.actor.ranking import RULE_BASED
from yp_video.actor.service import shadow_rejection
from yp_video.config import cut_kind_of, find_cut, iter_all_cuts
from yp_video.core.cache import StatCache
from yp_video.core.jsonl import read_jsonl_cached
from yp_video.action import prelabel
from yp_video.actor import spot_associate
from yp_video.extraction import actor_fix, links, reassociate
from yp_video.extraction.prerequisites import prerequisites
from yp_video.extraction import store as extraction_store
from yp_video.reid import store as reid_store
from yp_video.tracklets import store as tracks_store
from yp_video.tracklets.geometry import TrackRef
from yp_video.reid.embedder import DEFAULT_EMBEDDER, base_embedder_name
from yp_video.web.job_helpers import (
    fail_job_from_exc,
    init_batch_items,
    spawn_batch_video_job,
)
from yp_video.web.jobs import JobStatus, job_manager

log = logging.getLogger(__name__)
router = APIRouter()

TRAIN_JOB_TYPE = "actor_association_train"
PREDICT_JOB_TYPE = "actor_association_predict"
_evaluation_cache: StatCache = StatCache()


@router.get("/videos")
def list_videos() -> list[dict]:
    """Extracted videos and how much of their actor review is left.

    Counts come from the record header and the label file — never from
    parsing every record — so this stays cheap enough for a page load.
    """
    results = []
    for path in sorted(iter_all_cuts(), key=lambda p: p.name):
        records = extraction_store.records_path(path.stem)
        if not records.exists():
            continue
        header = read_jsonl_cached(records)[0] or {}
        labels = actor_labels.load(path.stem)
        verdicts: dict[str, int] = {}
        for label in labels.values():
            verdicts[label.verdict.value] = verdicts.get(label.verdict.value, 0) + 1
        event_count = int(header.get("events") or 0)
        results.append(
            {
                "name": path.name,
                "kind": cut_kind_of(path),
                "event_count": event_count,
                "reviewed": len(labels),
                "unreviewed": max(event_count - len(labels), 0),
                "verdicts": verdicts,
                # The automatic policy's own outcome, for context on how much
                # of the remainder is likely to just need confirming.
                "auto_counts": {
                    key: int(header.get(key) or 0) for key in ("ok", "multi", "miss")
                },
                "pipeline": prerequisites(path.stem).payload(),
            }
        )
    return results


def _active_job() -> dict | None:
    return next(
        (
            job.to_dict()
            for job in job_manager.jobs.values()
            if job.type == TRAIN_JOB_TYPE
            and job.status in (JobStatus.PENDING, JobStatus.RUNNING)
        ),
        None,
    )


def _shadow_blocked_on(name: str) -> str | None:
    """Why this checkpoint cannot be the extraction shadow, or None.

    A checkpoint that no longer LOADS is one of the answers. Feature contracts
    get retired, and the checkpoints trained against them stay on disk; a
    listing that let that raise would take down the page for every other
    checkpoint too, over one file nobody can activate anyway.
    """
    try:
        return shadow_rejection(actor_checkpoints.load(name))
    except (OSError, ValueError, KeyError) as exc:
        return str(exc)


@router.get("/status")
def status() -> dict:
    dataset = actor_dataset.load_dataset()
    return {
        "dataset": dataset.payload(),
        "checkpoints": [
            {
                **candidate,
                # Activatability is the service's judgement, not the
                # repository's — the page needs it to disable the button
                # instead of offering a 400.
                "shadow_blocked_on": _shadow_blocked_on(candidate["name"]),
            }
            for candidate in actor_checkpoints.list_candidates()
        ],
        # yp-spot models that answer the same question by looking at pixels.
        # A separate list because they are a different kind of thing: no
        # grouped-OOF metrics, no shadow activation, and they are selected
        # through `spot_checkpoint` rather than `checkpoint`.
        "spot_checkpoints": spot_associate.list_actor_checkpoints(),
        "active_shadow": actor_checkpoints.active_shadow_name(),
        "active_job": _active_job(),
    }


@router.get("/performance")
async def performance() -> dict:
    dataset = actor_dataset.load_dataset()
    candidates = actor_checkpoints.list_candidates()
    sources = list(dataset.sources)
    for candidate in candidates:
        sources.append(
            actor_checkpoints.checkpoint_dir(candidate["name"])
            / actor_checkpoints.MANIFEST_FILE
        )

    def compute() -> dict:
        rules = actor_evaluate.evaluate_dataset(dataset)
        return {
            **rules,
            "candidates": {
                candidate["name"]: (
                    candidate.get("metrics", {}).get("grouped_oof")
                )
                for candidate in candidates
            },
        }

    return await asyncio.to_thread(
        _evaluation_cache.get,
        ("actor-association", dataset.stems),
        sources,
        compute,
    )


class AssociationTrainRequest(BaseModel):
    run_name: str | None = None
    seed: int = 42
    folds: int = Field(default=5, ge=2, le=10)
    l2: float = Field(default=0.05, gt=0)
    target_precision: float = Field(default=0.9, gt=0, le=1)
    min_occluded_rejection: float = Field(default=0.5, ge=0, le=1)


@router.post("/train")
async def train(req: AssociationTrainRequest) -> dict:
    dataset = actor_dataset.load_dataset()
    if len(dataset.stems) < 2:
        raise HTTPException(
            400,
            "Association training needs completed reviews from at least "
            "two videos",
        )
    name = req.run_name or f"association_{time.strftime('%Y%m%d-%H%M%S')}"
    try:
        root = actor_checkpoints.checkpoint_dir(name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if root.exists():
        raise HTTPException(
            409,
            f"Association checkpoint {name} exists; run names are immutable",
        )
    config = actor_train.TrainingConfig(
        seed=req.seed,
        folds=req.folds,
        l2=req.l2,
        target_precision=req.target_precision,
        min_occluded_rejection=req.min_occluded_rejection,
    )
    job = job_manager.create_job(
        TRAIN_JOB_TYPE,
        {
            "run_name": name,
            "dataset": dataset.payload(),
            "config": {
                "seed": req.seed,
                "folds": req.folds,
                "l2": req.l2,
                "target_precision": req.target_precision,
                "min_occluded_rejection": req.min_occluded_rejection,
            },
        },
        name=f"Actor association train ({name})",
    )

    async def run_job() -> None:
        try:
            await job_manager.update_job(
                job.id,
                status=JobStatus.RUNNING,
                message="Grouped association training…",
            )
            result = await asyncio.to_thread(
                actor_train.train_candidate,
                dataset,
                name,
                config=config,
            )
            metrics = result["metrics"]["grouped_oof"]
            await job_manager.update_job(
                job.id,
                status=JobStatus.COMPLETED,
                progress=1.0,
                message=(
                    f"Association candidate ready: {name} · "
                    f"coverage {metrics['auto_coverage']:.1%} · "
                    f"precision {metrics['selective_accuracy']:.1%}"
                ),
                params={
                    **job.params,
                    "checkpoint": name,
                    "metrics": metrics,
                },
            )
        except asyncio.CancelledError:
            await job_manager.update_job(
                job.id,
                status=JobStatus.CANCELLED,
                message="Association training cancelled",
            )
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("Association training failed")
            await fail_job_from_exc(job.id, exc)

    job_manager.attach_task(job, asyncio.create_task(run_job()))
    return job.to_dict()


class _SpotPlan:
    """Stands in for a policy until the video has actually been scored.

    The spot model decides per VIDEO, so the real policy only exists after its
    subprocess has run. This carries the two things the request handler needs
    before then: a name for the job card, and the fact that it picks among
    tracklets — so an untracked video is refused up front rather than
    abstaining on every event.
    """

    needs_tracklets = True

    def __init__(self, checkpoint: Path):
        self.name = f"spot:{checkpoint.parent.name}"


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    videos: list[str]
    #: None (or "rule-based") is the rule; anything else names a checkpoint.
    checkpoint: str | None = None
    #: A yp-spot checkpoint carrying the actor head, under the action
    #: checkpoints root. Mutually exclusive with `checkpoint`: they are two
    #: different models answering the same question, and picking both would
    #: leave which one decided to argument order.
    spot_checkpoint: str | None = None
    stop_vllm: bool = False

    @model_validator(mode="after")
    def one_model_only(self):
        if self.spot_checkpoint and self.checkpoint not in (None, RULE_BASED):
            raise ValueError("checkpoint and spot_checkpoint are mutually exclusive")
        return self


@router.post("/predict")
async def predict(req: PredictRequest) -> dict:
    """Re-decide the automatic actor picks, without re-detecting anybody.

    Every human verdict survives untouched — see extraction/reassociate.py.
    """
    spot_checkpoint: Path | None = None
    if req.spot_checkpoint:
        try:
            spot_checkpoint = prelabel.resolve_checkpoint(req.spot_checkpoint)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(404, str(exc)) from exc
        reason = spot_associate.rejection(spot_checkpoint)
        if reason is not None:
            raise HTTPException(400, reason)
        policy = _SpotPlan(spot_checkpoint)
    else:
        try:
            policy = actor_policy.build_policy(req.checkpoint)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(404, str(exc)) from exc

    video_paths: list[Path] = []
    for name in req.videos:
        path = find_cut(name)
        if path is None:
            raise HTTPException(404, f"Video not found: {name}")
        if not extraction_store.records_path(path.stem).exists():
            raise HTTPException(
                400, f"No extraction records for: {name} — run ReID Predict first"
            )
        if policy.needs_tracklets and not tracks_store.tracks_path(path.stem).exists():
            raise HTTPException(
                400,
                f"{policy.name} picks among tracklets, and {name} has not been "
                "tracked — run Rally Tracking first",
            )
        video_paths.append(path)
    if not video_paths:
        raise HTTPException(400, "Select at least one video")

    job = job_manager.create_job(
        PREDICT_JOB_TYPE,
        {
            "policy": policy.name,
            "videos": [p.name for p in video_paths],
            "items": init_batch_items([p.name for p in video_paths]),
        },
        name=f"Association Predict ({len(video_paths)} videos · {policy.name})",
    )
    def decide(path: Path, on_progress) -> dict:
        """Score the video, then rewrite the picks it changed.

        The spot model runs per VIDEO — one subprocess, one pass over the
        frames — and only then does the per-event policy exist. The rule and
        the ranker skip straight to the second half.
        """
        chosen = policy
        if spot_checkpoint is not None:
            answers = spot_associate.run(
                path, spot_checkpoint, on_progress=on_progress
            )
            chosen = actor_policy.SpotActorPolicy(
                answers, name=spot_checkpoint.parent.name
            )
        return reassociate.reassociate_video(path, chosen, on_progress=on_progress)

    spawn_batch_video_job(
        job,
        video_paths,
        stop_vllm=req.stop_vllm,
        work=decide,
        done_message=lambda c: (
            f"{c['changed']} moved · {c['unchanged']} unchanged · "
            f"{c['labeled']} labeled kept"
        ),
        start_message="re-deciding actors...",
    )
    return job.to_dict()


class ShadowRequest(BaseModel):
    checkpoint: str | None


@router.put("/shadow")
def set_shadow(req: ShadowRequest) -> dict:
    """Explicitly select learned diagnostics; the rule remains production."""
    try:
        if req.checkpoint is not None:
            reason = shadow_rejection(actor_checkpoints.load(req.checkpoint))
            if reason is not None:
                raise HTTPException(400, reason)
        actor_checkpoints.set_active_shadow(req.checkpoint)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"active_shadow": req.checkpoint}


class ConfirmRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: None = every automatic pick in the video that has no verdict yet.
    event_ids: list[str] | None = None


@router.post("/confirm/{name}")
def confirm(name: str, req: ConfirmRequest) -> dict:
    """Endorse automatic picks: "the policy already got these right".

    Purely an annotation write — the record, the crop and every embedding
    stay exactly as they are, because agreeing with a pick changes nothing
    about it. That is what separates this from a fix, and why it needs none
    of the fix's transaction machinery.

    Events already carrying a verdict are left alone (a human correction
    outranks a bulk confirmation), so the count reports what actually landed.
    """
    stem = Path(unquote(name)).stem
    path = extraction_store.records_path(stem)
    if not path.exists():
        raise HTTPException(404, f"No extraction records for {stem}")

    _meta, records = read_jsonl_cached(path)
    confirmable = actor_labels.confirmations_for(records)
    if req.event_ids is not None:
        wanted = set(req.event_ids)
        unknown = sorted(wanted - set(confirmable))
        if unknown:
            # Silently dropping these would report a success that did not
            # happen; a miss needs a real verdict, not a confirmation.
            raise HTTPException(
                400,
                f"Not automatic picks (nothing to confirm): {', '.join(unknown[:5])}"
                + (f" (+{len(unknown) - 5} more)" if len(unknown) > 5 else ""),
            )
        confirmable = {k: v for k, v in confirmable.items() if k in wanted}

    return {"confirmed": actor_labels.confirm_auto(stem, confirmable)}


class ActorFixBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1)


class PickActorRequest(ActorFixBase):
    mode: Literal["pick"]
    box: tuple[float, float, float, float]
    # The tracklet clicked, as "{rally_id}:{track_id}". When present the box
    # is only the anchor — the server re-resolves the tracklet to a croppable
    # detection itself, so the crop is reproducible from the label alone.
    track: str | None = Field(default=None, pattern=r"^\d+:\d+$")
    # Cross-frame pick: the box lives on this frame, not the event's — the
    # crop is cut from here (actor undetected on the event frame). Tracklet
    # picks do not need it; the tracklet already spans frames.
    frame: int | None = Field(default=None, ge=0)
    # False = no stored detection is this player, so embed the box as drawn
    # rather than IoU-snapping onto an occluder. Box picks only.
    snap: bool = True

    @property
    def command(self) -> actor_fix.PickActor:
        return actor_fix.PickActor(
            mode="pick",
            event_id=self.event_id,
            box=self.box,
            track=TrackRef.parse(self.track) if self.track else None,
            frame=self.frame,
            snap=self.snap,
        )


class OccludedActorRequest(ActorFixBase):
    mode: Literal["occluded"]

    @property
    def command(self) -> actor_fix.MarkOccluded:
        return actor_fix.MarkOccluded(mode="occluded", event_id=self.event_id)


class AutoActorRequest(ActorFixBase):
    mode: Literal["auto"]

    @property
    def command(self) -> actor_fix.RevertActor:
        return actor_fix.RevertActor(mode="auto", event_id=self.event_id)


ActorFixRequest = Annotated[
    PickActorRequest | OccludedActorRequest | AutoActorRequest,
    Field(discriminator="mode"),
]


def _synchronous_model(stem: str) -> str:
    """The embedding family refreshed before the response returns.

    A fix invalidates every matrix, but refreshing them all inline would make
    the click feel broken. The default embedder's family goes first because
    that is what the ReID Label page opens with; the rest follow in the
    background. Which model that is stays server-side — reviewing an actor is
    not a question about embeddings, so the page never has to name one.
    """
    embedded = reid_store.embedded_models(stem)
    if not embedded:
        raise HTTPException(
            404,
            f"No embeddings for {stem} — run extraction or backfill first",
        )
    family = base_embedder_name(DEFAULT_EMBEDDER)
    return next(
        (name for name in embedded if base_embedder_name(name) == family),
        embedded[0],
    )


@router.post("/fix/{name}")
def fix(
    name: str, req: ActorFixRequest, background_tasks: BackgroundTasks
) -> dict:
    """Re-point one event at the person the user clicked (or nobody / auto).

    The verdict lands in the video's actor labels (the durable human record,
    replayed on re-extraction) and is applied to the extraction record
    immediately: the chosen box is cropped and re-embedded, so the identity
    clusters that read those crops follow.
    """
    video_path = find_cut(unquote(name))
    if video_path is None:
        raise HTTPException(404, f"Video not found: {name}")
    stem = video_path.stem
    if not extraction_store.records_path(stem).exists():
        raise HTTPException(404, f"No extraction records for {stem}")

    command: actor_fix.ActorFixCommand = req.command
    try:
        result = actor_fix.apply(
            video_path, command, active_model=_synchronous_model(stem)
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    record = result.record
    label = command.label
    record["actor_review"] = label.verdict.value if label else "unreviewed"
    for detection in record.get("detections") or []:
        detection.pop("keypoints", None)
    track_link = None
    if tracks_store.tracks_path(stem).exists():
        ref = links.event_tracks(stem).get(req.event_id)
        track_link = ref.payload() if ref else None
    background_tasks.add_task(
        actor_fix.refresh_deferred,
        stem,
        req.event_id,
        models=result.refreshing_models,
        expected_revision=result.actor_revision,
    )
    return {
        "record": record,
        "track_link": track_link,
        "refreshing_models": result.refreshing_models,
    }
