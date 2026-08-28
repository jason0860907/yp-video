"""Actor association: the labeling work list, the fix endpoint, and learning.

Serves the Association Label page — which video still has unreviewed actors,
and the one write that answers "this person performed this action". Player
identity is the ReID router's business; the only thing the two share is the
extraction records they both read.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import unquote

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from yp_video.action import prelabel
from yp_video.action.frames import ensure_action_frame_caches
from yp_video.action.training import materialize_holdout_split
from yp_video.actor import candidates as actor_candidates
from yp_video.actor import dataset as actor_dataset
from yp_video.actor import evaluate as actor_evaluate
from yp_video.actor import labels as actor_labels
from yp_video.actor import policy as actor_policy
from yp_video.actor import spot_associate, spot_predictions
from yp_video.actor.ranking import RULE_BASED
from yp_video.actor.training_labels import prepare_action_training_labels
from yp_video.config import (
    ACTION_FRAMES_DIR,
    SPOT_CHECKPOINTS_DIR,
    SPOT_DIR,
    SPOT_PYTHON,
    find_cut,
)
from yp_video.contracts.action import (
    ACTION_CONTRACT_VERSION,
    ACTION_CONTRACT_VERSION_ENV,
    ACTOR_FILE_GLOB,
    ACTOR_LABEL_SUBDIR,
    ASSOCIATION_PACKAGE_TYPE,
)
from yp_video.core import label_done
from yp_video.core.cache import StatCache
from yp_video.core.jsonl import read_jsonl_cached
from yp_video.extraction import actor_fix, links, reassociate
from yp_video.extraction import done as extraction_done
from yp_video.extraction import store as extraction_store
from yp_video.reid import store as reid_store
from yp_video.web.r2_client import sync_to_r2
from yp_video.reid.embedder import DEFAULT_EMBEDDER, base_embedder_name
from yp_video.tracklets import store as tracks_store
from yp_video.tracklets.geometry import TrackRef
from yp_video.web import audit, worklists
from yp_video.web.job_helpers import (
    ProgressParser,
    fail_job_from_exc,
    init_batch_items,
    spawn_batch_video_job,
    stop_vllm_for_job,
    stream_subprocess,
    terminal_prefix,
)
from yp_video.web.jobs import JobSummary, JobType, job_manager
from yp_video.web.schemas import StrictModel
from yp_video.web.spot_runs import (
    PackageExporter,
    actor_task_metrics,
    export_checkpoint_package,
    performance_payload,
)
from yp_video.web.train_requests import AssociationTrainRequest

log = logging.getLogger(__name__)
router = APIRouter()

_evaluation_cache: StatCache = StatCache()
_train_start_lock = asyncio.Lock()


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
    """Which models exist, and whether a training job is running.

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
        "init_checkpoints": [
            {
                "value": row["path"],
                "label": (
                    f"{row['name']} (Top-1 "
                    f"{float(((row.get('best') or {}).get('value') or 0)):.1%})"
                ),
            }
            for row in association_checkpoints
            if row["family"] == spot_associate.INDEPENDENT_FORMAT
        ],
        "frame_dir": str(ACTION_FRAMES_DIR),
        "active_job": active.to_dict() if (active := job_manager.active_job(JobType.ACTOR_ASSOCIATION_TRAIN)) else None,
    }


@router.get("/train-performance")
def train_performance(run: str | None = None) -> dict:
    """Per-epoch metrics for independent association runs, in the shared
    performance shape — the same card every other Train page renders."""
    return performance_payload(
        SPOT_CHECKPOINTS_DIR, run, package_types=(ASSOCIATION_PACKAGE_TYPE,)
    )


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


def _association_training_items(
    names: list[str],
) -> list[tuple[Path, Path]]:
    """Resolve the exact action-label/video pairs selected on this page.

    Association supervision may sit on top of either manual action labels or
    action predictions.  Reusing Action Train's global annotation scan would
    silently drop the latter, so the association surface resolves each video
    through the same manual-first source rule used by inference.
    """
    items: list[tuple[Path, Path]] = []
    seen: set[str] = set()
    for name in names:
        path = find_cut(name)
        if path is None:
            raise HTTPException(404, f"Video not found: {name}")
        if path.stem in seen:
            continue
        seen.add(path.stem)
        label_path = spot_associate.action_label_path(path.stem)
        if label_path is None:
            raise HTTPException(
                400,
                f"No action labels for {name}; run Action Predict or label it first",
            )
        if not actor_labels.load(path.stem):
            raise HTTPException(
                400,
                f"No reviewed actors for {name}; review it in Association Label first",
            )
        _meta, events = read_jsonl_cached(label_path)
        actor_rows, _tally = actor_candidates.build(path.stem, events)
        if not actor_rows:
            raise HTTPException(
                400,
                f"No usable yp-actor targets for {name}; it needs reviewed "
                "tracklet labels and Rally Tracking",
            )
        items.append((label_path, path))
    return items


@router.post("/train", response_model=JobSummary)
async def train(req: AssociationTrainRequest) -> dict:
    async with _train_start_lock:
        return await _train_locked(req)


async def _train_locked(req: AssociationTrainRequest) -> dict:
    active = job_manager.active_job(JobType.ACTOR_ASSOCIATION_TRAIN)
    if active is not None:
        raise HTTPException(
            409,
            f"Association training is already active: {active.name}",
        )
    name = req.run_name or f"yp_actor_{time.strftime('%Y%m%d-%H%M%S')}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name.startswith("."):
        raise HTTPException(
            400,
            "Run name may contain only letters, numbers, dot, underscore and dash",
        )
    checkpoint_dir = SPOT_CHECKPOINTS_DIR / name
    save_dir = SPOT_DIR / "exp" / name
    if checkpoint_dir.exists() or save_dir.exists():
        raise HTTPException(
            409,
            f"Association run {name} already exists; run names are immutable",
        )
    train_items = _association_training_items(req.train_videos)
    val_items = _association_training_items(req.val_videos)
    init_checkpoint = None
    if req.init_checkpoint:
        try:
            init_checkpoint = prelabel.resolve_checkpoint(req.init_checkpoint)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(404, str(exc)) from exc
        reason = spot_associate.rejection(init_checkpoint)
        if reason:
            raise HTTPException(400, reason)
    return await _start_association_training(
        req,
        save_dir=save_dir,
        checkpoint_dir=checkpoint_dir,
        train_items=train_items,
        val_items=val_items,
        init_checkpoint=init_checkpoint,
    )


def _export_association_package(
    *,
    run_dir: Path,
    package_dir: Path,
    req: AssociationTrainRequest,
    cmd: list[str],
    label_summary: dict,
) -> dict:
    summary = export_checkpoint_package(
        run_dir=run_dir,
        package_dir=package_dir,
        checkpoints_root=SPOT_CHECKPOINTS_DIR,
        package_type=ASSOCIATION_PACKAGE_TYPE,
        label_subdirs=(TASKS["action"].label_subdir, TASKS["actor"].label_subdir),
        training={
            "purpose": "association",
            "frame_dir": str(ACTION_FRAMES_DIR),
            "selection_metric": "player_top1",
            "label_summary": label_summary,
        },
        cmd=cmd,
    )
    manifest_path = package_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    best = manifest.get("best") or {}
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists() and isinstance(best.get("epoch"), int):
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("epoch") == best["epoch"]:
                best["metrics"] = record.get("val") or {}
                break
    manifest["best"] = best
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary["best"] = best
    return summary


async def _start_association_training(
    req: AssociationTrainRequest,
    *,
    save_dir: Path,
    checkpoint_dir: Path,
    train_items: list[tuple[Path, Path]],
    val_items: list[tuple[Path, Path]],
    init_checkpoint: Path | None,
) -> dict:
    job = job_manager.create_job(
        JobType.ACTOR_ASSOCIATION_TRAIN,
        {
            "save_dir": str(save_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "train_videos": [video.name for _label, video in train_items],
            "val_videos": [video.name for _label, video in val_items],
            "backbone": req.backbone,
            "epochs": req.num_epochs,
        },
        name=f"Association Train ({save_dir.name})",
    )

    async def run_job() -> None:
        exporter: PackageExporter | None = None
        try:
            await job_manager.update_job(
                job.id, status="running", message="Preparing association events..."
            )
            items = [*train_items, *val_items]
            frame_summary = await asyncio.to_thread(
                ensure_action_frame_caches,
                [(video, None) for _label, video in items],
                cache_root=ACTION_FRAMES_DIR,
            )
            label_summary = await asyncio.to_thread(
                prepare_action_training_labels,
                items=items,
                frame_dir=ACTION_FRAMES_DIR,
                save_dir=save_dir,
                tasks=("action", "location", "actor"),
                camera_view="all",
            )
            val_stems = {label.stem.removesuffix("_actions") for label, _video in val_items}
            split = await asyncio.to_thread(
                materialize_holdout_split,
                Path(label_summary["label_dir"]),
                val_stems,
                known_stems=val_stems,
            )
            label_summary = {**label_summary, **split}
            cmd = [
                str(SPOT_PYTHON),
                "-m",
                "yp_spot.association.train",
                "--train-labels",
                str(save_dir / "labels" / "train"),
                "--val-labels",
                str(save_dir / "labels" / "val"),
                "--actor-dir",
                str(save_dir / "labels" / ACTOR_LABEL_SUBDIR),
                "--frame-dir",
                str(ACTION_FRAMES_DIR),
                "--save-dir",
                str(save_dir),
                "--backbone",
                req.backbone,
                "--batch-size",
                str(req.batch_size),
                "--epochs",
                str(req.num_epochs),
                "--learning-rate",
                str(req.learning_rate),
                "--backbone-learning-rate",
                str(req.backbone_learning_rate),
                "--warmup-epochs",
                str(req.warm_up_epochs),
                "--num-workers",
                str(req.num_workers),
                "--crop-dim",
                str(req.crop_dim),
            ]
            if init_checkpoint:
                cmd.extend(["--init-checkpoint", str(init_checkpoint)])
            await job_manager.update_job(
                job.id,
                progress=0.2,
                message="Waiting for GPU...",
                params={
                    **job.params,
                    "frame_cache": frame_summary,
                    "training_labels": label_summary,
                    "command": cmd,
                },
            )
            exporter = PackageExporter(
                job.id,
                save_dir,
                lambda: _export_association_package(
                    run_dir=save_dir,
                    package_dir=checkpoint_dir,
                    req=req,
                    cmd=cmd,
                    label_summary=label_summary,
                ),
            )

            best_state: dict = {}

            def on_metrics(match: re.Match) -> dict:
                record = json.loads(match.group(1))
                epoch = int(record["epoch"])
                validation = record.get("val") or {}
                if record.get("best"):
                    exporter.schedule(epoch, "new_best")
                top1 = validation.get("player_top1")
                overall = validation.get("overall_exact")
                progress = 0.2 + 0.79 * ((epoch + 1) / req.num_epochs)
                loss = record.get("loss") or {}
                # The shared live-progress schema (see spot_runs.TrainProgress):
                # one Train page job card renders every trainer, so this
                # trainer reports itself as its one task.
                task_metrics = actor_task_metrics(record)
                snapshot = {
                    "epoch": epoch,
                    "epoch_display": epoch + 1,
                    "epochs": req.num_epochs,
                    "completed_epoch": epoch,
                    "latest_train_loss": loss.get("train"),
                    "latest_val_loss": loss.get("val"),
                    "latest_val_map": None,
                    "latest_task_metrics": task_metrics,
                }
                if record.get("best"):
                    best_state.update(
                        best_epoch=epoch,
                        best_value=top1,
                        best_task_metrics=task_metrics,
                    )
                return {
                    "progress": min(progress, 0.99),
                    "message": (
                        f"Epoch {epoch + 1}/{req.num_epochs} · player Top-1 "
                        f"{float(top1 or 0):.1%} · overall "
                        f"{float(overall or 0):.1%}"
                    ),
                    "params": {
                        "association_train_progress": {
                            **snapshot,
                            **best_state,
                        }
                    },
                }

            parsers = [
                ProgressParser(r"ASSOCIATION_METRICS (\{.*\})", on_metrics)
            ]
            env = {
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": (
                    f"{SPOT_DIR}{os.pathsep}{os.environ['PYTHONPATH']}"
                    if os.environ.get("PYTHONPATH")
                    else str(SPOT_DIR)
                ),
                "CUDA_VISIBLE_DEVICES": str(req.gpu),
                ACTION_CONTRACT_VERSION_ENV: ACTION_CONTRACT_VERSION,
            }
            async with stop_vllm_for_job(job.id, when=req.stop_vllm):
                async with job_manager.gpu_lock:
                    await job_manager.update_job(
                        job.id, message="Training independent association model..."
                    )
                    rc, last_line = await stream_subprocess(
                        job.id,
                        cmd,
                        cwd=SPOT_DIR,
                        env=env,
                        parsers=parsers,
                        is_key_line=lambda line: (
                            "ASSOCIATION_METRICS " in line
                            or "Best epoch:" in line
                        ),
                        tee_to_terminal=True,
                        log_path=save_dir / "terminal.log",
                    )
            if rc != 0:
                raise RuntimeError(
                    last_line or f"Association training exited with code {rc}"
                )
            checkpoint_summary = await exporter.export_once(
                expected_epoch=None, reason="completed", update_job=False
            )
            if checkpoint_summary is None:
                raise RuntimeError("Training completed without a best checkpoint")
            await job_manager.update_job(
                job.id,
                status="completed",
                progress=1.0,
                message=f"Association model ready: {checkpoint_dir}",
                params={
                    **job.params,
                    "checkpoint_package": checkpoint_summary,
                },
            )
        except asyncio.CancelledError:
            checkpoint_summary = None
            if (
                exporter is not None
                and (save_dir / "checkpoint_best.pt").exists()
            ):
                checkpoint_summary = await exporter.export_once(
                    expected_epoch=None, reason="cancelled", update_job=False
                )
            if checkpoint_summary:
                await job_manager.update_job(
                    job.id,
                    params={**job.params, "checkpoint_package": checkpoint_summary},
                )
            raise
        except Exception as exc:  # noqa: BLE001
            print(
                f"{terminal_prefix(job)}Failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
            log.exception("Association training failed")
            await fail_job_from_exc(job.id, exc)

    task = asyncio.create_task(run_job())
    job_manager.attach_task(job, task)
    return job.to_dict()


class PredictRequest(StrictModel):
    model_config = ConfigDict(extra="forbid")

    videos: list[str]
    #: An independent yp-association checkpoint. None selects the rule.
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
