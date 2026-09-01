"""Actor association: the labeling work list, the fix endpoint, and learning.

Serves the Association Label page — which video still has unreviewed actors,
and the one write that answers "this person performed this action". Player
identity is the ReID router's business; the only thing the two share is the
extraction records they both read.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import unquote

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from yp_video.action import prelabel
from yp_video.actor import dataset as actor_dataset
from yp_video.actor import evaluate as actor_evaluate
from yp_video.actor import labels as actor_labels
from yp_video.actor import policy as actor_policy
from yp_video.actor import spot_associate, spot_predictions
from yp_video.actor.ranking import RULE_BASED
from yp_video.config import (
    ACTION_FRAMES_DIR,
    SPOT_DIR,
    SPOT_PYTHON,
    find_cut,
)
from yp_video.core import label_done
from yp_video.core.cache import StatCache
from yp_video.core.jsonl import read_jsonl_cached
from yp_video.extraction import actor_fix, links, reassociate
from yp_video.extraction import done as extraction_done
from yp_video.extraction import store as extraction_store
from yp_video.reid import store as reid_store
from yp_video.reid.embedder import DEFAULT_EMBEDDER, base_embedder_name
from yp_video.tracklets import store as tracks_store
from yp_video.tracklets.geometry import TrackRef
from yp_video.web import audit, worklists
from yp_video.web.job_helpers import (
    init_batch_items,
    spawn_batch_video_job,
)
from yp_video.web.jobs import JobSummary, JobType, job_manager
from yp_video.web.r2_client import sync_to_r2
from yp_video.web.schemas import StrictModel

log = logging.getLogger(__name__)
router = APIRouter()

_evaluation_cache: StatCache = StatCache()


@router.get("/videos")
def list_videos() -> list[dict]:
    return worklists.association_videos()


class DoneRequest(StrictModel):
    done: bool = True


@router.put("/done/{name:path}")
def set_done(name: str, req: DoneRequest) -> dict:
    """Persist the human "actor review is finished" verdict for one video.

    Done also sweep-confirms every current automatic answer (same write as
    the per-rally sweep, video-wide), and reassociation keeps honouring the
    flag afterwards — a predict re-run confirms the answers it invents
    instead of un-reviewing a finished video (extraction/done.py).
    """
    video = find_cut(Path(unquote(name)).name)
    if video is None:
        raise HTTPException(404, "Video not found")
    flags = label_done.set_done(video.stem, "association", req.done)
    confirmed = extraction_done.confirm_reviewed(video.stem) if req.done else 0
    return {"done": flags["association"], "confirmed": confirmed}


@router.get("/status")
def status() -> dict:
    """Which fusion actor heads exist.

    Cheap on purpose, and it must stay that way: this is the single query the
    Association Predict pickers wait on. It used to also return the training
    corpus summary, which meant building the dataset — decompressing a
    silhouette archive per labelled video — before anyone could choose a
    model. The corpus belongs to /performance, which is already the slow,
    cached one and is read by the page that actually wants it.
    """
    association_checkpoints = spot_associate.list_association_checkpoints()
    return {
        # Visual models answer by looking at pixels and choosing among the
        # tracked candidates; this also includes fusion actor heads.
        "association_checkpoints": association_checkpoints,
        "spot_available": SPOT_DIR.exists() and SPOT_PYTHON.exists(),
        "frame_dir": str(ACTION_FRAMES_DIR),
    }


@router.get("/performance")
async def performance() -> dict:
    """Every policy that can answer, on the reviewed events, sliced.

    The rule and persisted yp-actor answers are scored on the same reviewed
    events. The `hard` and `manual` slices are the point: the aggregate is
    dominated by events the rule already gets right, so a model can move it
    without touching a single case anyone cares about.

    Note the yp-spot column is scored on answers ALREADY on disk from an
    earlier Association Predict run, not by re-running the head — scoring
    would mean a GPU pass per video from inside a web request.
    """
    stems = list(actor_labels.labeled_stems())
    spot_runs = sorted(spot_predictions.available_runs(stems))

    sources = [
        *actor_dataset.source_paths(stems),
        spot_predictions.ACTOR_PREDICTIONS_DIR,
    ]

    def compute() -> dict:
        dataset = actor_dataset.load_track_dataset(stems)
        builders: dict = {RULE_BASED: lambda _stem: actor_policy.RulePolicy()}
        for run in spot_runs:
            builders[f"spot:{run}"] = (
                lambda stem, r=run: spot_predictions.policy_for(stem, r)
            )
        return {
            "dataset": dataset.payload(),
            "slices": list(actor_evaluate.SLICES),
            "policies": actor_evaluate.evaluate_policies(builders, stems),
            "candidates": {},
        }

    return await asyncio.to_thread(
        _evaluation_cache.get,
        ("actor-association", tuple(stems)),
        sources,
        compute,
    )


class PredictRequest(StrictModel):
    model_config = ConfigDict(extra="forbid")

    videos: list[str]
    #: A fusion actor-head checkpoint package. None selects the rule.
    association_checkpoint: str | None = None
    stop_vllm: bool = False


@router.post("/predict", response_model=JobSummary)
async def predict(req: PredictRequest) -> dict:
    """Re-decide the automatic actor picks, without re-detecting anybody.

    Every human verdict survives untouched — see extraction/reassociate.py.
    """
    if req.association_checkpoint:
        try:
            association_checkpoint = prelabel.resolve_checkpoint(
                req.association_checkpoint
            )
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(404, str(exc)) from exc
        reason = spot_associate.rejection(association_checkpoint)
        if reason is not None:
            raise HTTPException(400, reason)
        plan: actor_policy.PolicyPlan = actor_policy.SpotPlan(
            association_checkpoint
        )
    else:
        plan = actor_policy.RulePolicy()

    video_paths: list[Path] = []
    for name in req.videos:
        path = find_cut(name)
        if path is None:
            raise HTTPException(404, f"Video not found: {name}")
        if not extraction_store.records_path(path.stem).exists():
            raise HTTPException(
                400, f"No extraction records for: {name} — run ReID Predict first"
            )
        if plan.needs_tracklets and not tracks_store.tracks_path(path.stem).exists():
            raise HTTPException(
                400,
                f"{plan.name} picks among tracklets, and {name} has not been "
                "tracked — run Rally Tracking first",
            )
        video_paths.append(path)
    if not video_paths:
        raise HTTPException(400, "Select at least one video")

    job = job_manager.create_job(
        JobType.ACTOR_ASSOCIATION_PREDICT,
        {
            "policy": plan.name,
            "videos": [p.name for p in video_paths],
            "items": init_batch_items([p.name for p in video_paths]),
        },
        name=f"Association Predict ({len(video_paths)} videos · {plan.name})",
    )
    spawn_batch_video_job(
        job,
        video_paths,
        stop_vllm=req.stop_vllm,
        # Whether the policy exists yet is the plan's business: the rule and
        # the ranker hand back themselves, the spot head scores the video
        # first (see actor/policy.SpotPlan).
        work=lambda path, cb: reassociate.reassociate_video(
            path, plan.build(path, cb), on_progress=cb
        ),
        done_message=lambda c: (
            f"{c['changed']} moved · {c['unchanged']} unchanged · "
            f"{c['labeled']} labeled kept"
            + (f" · {c['confirmed']} auto-confirmed (video marked done)" if c.get("confirmed") else "")
        ),
        start_message="re-deciding actors...",
    )
    return job.to_dict()


class ConfirmRequest(StrictModel):
    model_config = ConfigDict(extra="forbid")

    #: None = every automatic pick in the video that has no verdict yet.
    event_ids: list[str] | None = None


@router.post("/confirm/{name}")
def confirm(name: str, req: ConfirmRequest) -> dict:
    """Endorse the policy's answer: "it already got these right".

    Two answers are endorsable and land as different verdicts — a pick
    becomes ``confirmed_auto``, an explicit "nobody is visible" becomes
    ``occluded`` (see actor/labels.confirmations_for).

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

    meta, records = read_jsonl_cached(path)
    records = extraction_store.labelable(
        records, stem, float(meta.get("fps") or 0)
    )
    confirmable = actor_labels.confirmations_for(records)
    if req.event_ids is not None:
        wanted = set(req.event_ids)
        unknown = sorted(wanted - set(confirmable))
        if unknown:
            # Silently dropping these would report a success that did not
            # happen; a miss needs a real verdict, not a confirmation.
            raise HTTPException(
                400,
                "Nothing to endorse — the policy neither picked anybody nor "
                f"called it occluded: {', '.join(unknown[:5])}"
                + (f" (+{len(unknown) - 5} more)" if len(unknown) > 5 else ""),
            )
        confirmable = {k: v for k, v in confirmable.items() if k in wanted}

    # Which VERDICT each event got, not just that it landed: endorsing a
    # pick and endorsing an occlusion are two different answers, and a caller
    # that assumes one of them shows the wrong badge for the other.
    before = _actor_rows(actor_labels.load(stem))
    landed = actor_labels.confirm_auto(stem, confirmable)
    # A bulk endorsement writes many durable verdicts at once. Not folded into
    # a session: it is one click, not a stretch of work.
    audit.record_diff(
        target=stem,
        before=before,
        after=_actor_rows(actor_labels.load(stem)),
        key=lambda r: r["id"],
        confirmed=len(landed),
    )
    return {"confirmed": {event_id: confirmable[event_id].verdict.value for event_id in landed}}


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


def _synchronous_model(stem: str) -> str | None:
    """The embedding family refreshed before the response returns.

    None when nothing is embedded yet, which is the ordinary case: actor
    review is what decides whether a crop is worth embedding, so it runs
    first. Refusing the fix there would have made this page depend on the
    stage that depends on it.

    Once vectors do exist a fix invalidates every matrix, but refreshing them
    all inline would make the click feel broken. The default embedder's family
    goes first because that is what the ReID Label page opens with; the rest
    follow in the background. Which model that is stays server-side —
    reviewing an actor is not a question about embeddings, so the page never
    has to name one.
    """
    embedded = reid_store.embedded_models(stem)
    if not embedded:
        return None
    family = base_embedder_name(DEFAULT_EMBEDDER)
    return next(
        (name for name in embedded if base_embedder_name(name) == family),
        embedded[0],
    )


def _actor_rows(labels) -> list[dict]:
    """The video's actor verdicts as records, for auditing.

    One per event that carries a human verdict. The payload is what actually
    lands on disk, so a diff of two of these is exactly what the save changed.
    """
    return [{"id": event_id, **label.payload()} for event_id, label in labels.items()]


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

    before = _actor_rows(actor_labels.load(stem))
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
    sync_to_r2(actor_labels.actors_path(stem), "association/annotations")

    current = extraction_store.with_current_actions([result.record], stem)
    record = current[0] if current else result.record
    label = command.label
    record["actor_review"] = label.verdict.value if label else "unreviewed"
    # Sparse, recomputed AFTER the fix landed: present only when the fresh
    # label still resolves to no tracklet (see links.unresolved_labels) — a
    # box pick that lands on a tracked player clears the flag by resolving.
    if label is not None and req.event_id in links.unresolved_labels(stem):
        record["actor_review_unresolved"] = True
    for detection in record.get("detections") or []:
        detection.pop("keypoints", None)
    track_link = None
    if tracks_store.tracks_path(stem).exists():
        ref = links.event_tracks(stem).get(req.event_id)
        track_link = ref.payload() if ref else None
    # Association is labeling work like the other three panels: one call per
    # event the reviewer re-points. Folded into a session (see audit's
    # _COALESCING) so an afternoon of it reads as hours worked rather than as
    # hundreds of instantaneous rows totalling nothing.
    audit.record_diff(
        target=stem,
        before=before,
        after=_actor_rows(actor_labels.load(stem)),
        key=lambda r: r["id"],
        event=req.event_id,
    )
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
