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
from pydantic import BaseModel, ConfigDict, Field, model_validator

from yp_video.actor import dataset as actor_dataset
from yp_video.actor import evaluate as actor_evaluate
from yp_video.actor import labels as actor_labels
from yp_video.actor import policy as actor_policy
from yp_video.actor import review as actor_review
from yp_video.actor.ranking import RULE_BASED
from yp_video.config import (
    ACTION_CHECKPOINTS_DIR,
    ACTION_FRAMES_DIR,
    SPOT_DIR,
    SPOT_PYTHON,
    cut_kind_of,
    find_cut,
    iter_all_cuts,
)
from yp_video.contracts.action import (
    ACTION_CONTRACT_VERSION,
    ACTION_CONTRACT_VERSION_ENV,
    ACTOR_FILE_GLOB,
    ACTOR_LABEL_SUBDIR,
)
from yp_video.action.frames import ensure_action_frame_caches
from yp_video.core.cache import StatCache
from yp_video.core.jsonl import read_jsonl_cached, read_jsonl_header
from yp_video.action import actor_labels as spot_actor_labels
from yp_video.action import prelabel
from yp_video.actor import spot_associate
from yp_video.actor import spot_predictions
from yp_video.extraction import actor_fix, links, reassociate
from yp_video.extraction.prerequisites import prerequisites
from yp_video.extraction import store as extraction_store
from yp_video.reid import store as reid_store
from yp_video.tracklets import store as tracks_store
from yp_video.tracklets.geometry import TrackRef
from yp_video.reid.embedder import DEFAULT_EMBEDDER, base_embedder_name
from yp_video.web.job_helpers import (
    ProgressParser,
    fail_job_from_exc,
    init_batch_items,
    spawn_batch_video_job,
    stop_vllm_for_job,
    stream_subprocess,
    terminal_prefix,
)
from yp_video.web.jobs import JobStatus, job_manager
from yp_video.web.routers import action_train as action_train_router
from yp_video.web.spot_runs import PackageExporter, export_checkpoint_package

log = logging.getLogger(__name__)
router = APIRouter()

TRAIN_JOB_TYPE = "actor_association_train"
PREDICT_JOB_TYPE = "actor_association_predict"
_evaluation_cache: StatCache = StatCache()
_train_start_lock = asyncio.Lock()


@router.get("/videos")
def list_videos() -> list[dict]:
    """Extracted videos and how much of their actor review is left.

    Action annotations own event membership and labels. Extraction records
    only say which of those events have detector output, and actor labels say
    which current, labelable ids a human reviewed.

    A video missing anything association is built on is left out entirely
    rather than listed as a row with nothing in it: actions own which events
    exist, rallies own which of them are in play (and namespace every tracklet
    an answer can name), and records hold the detections a pick chooses among.
    Producing any of the three is another page's job, so a row here would be a
    dead end — the pipeline chips on those pages are where the gap belongs.
    """
    results = []
    for path in sorted(iter_all_cuts(), key=lambda p: p.name):
        # Cheapest gate first: this walks every cut on every page load, and
        # only a minority have been extracted at all.
        records = extraction_store.records_path(path.stem)
        if not records.exists():
            continue
        pipeline = prerequisites(path.stem)
        if not (pipeline.rally_sources and pipeline.has_action):
            continue
        header = read_jsonl_header(records)
        progress = actor_review.review_progress(
            path.stem, float(header.get("fps") or 0)
        )
        results.append(
            {
                "name": path.name,
                "kind": cut_kind_of(path),
                "event_count": progress.event_count,
                "reviewed": progress.reviewed,
                "unreviewed": progress.unreviewed,
                "verdicts": progress.verdicts,
                # The automatic policy's own outcome, for context on how much
                # of the remainder is likely to just need confirming. These
                # are detector-run diagnostics; unlike progress above they
                # deliberately describe that immutable run.
                "auto_counts": {
                    key: int(header.get(key) or 0)
                    for key in ("ok", "multi", "miss")
                },
                "pipeline": pipeline.payload(),
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
        # Kept as an empty field until the hand-written frontend response type
        # is retired. Association no longer trains or offers the linear
        # tracklet ranker.
        "checkpoints": [],
        # Visual models answer by looking at pixels and choosing among the
        # tracked candidates; this also includes supported legacy actor heads.
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
        "active_job": _active_job(),
    }


def _association_history(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    history = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record.get("epoch"), int):
            continue
        train_metrics = record.get("train") or {}
        val_metrics = record.get("val") or {}
        losses = record.get("loss") or {}
        history.append(
            {
                "epoch": record["epoch"] + 1,
                "train_player_top1": train_metrics.get("player_top1"),
                "val_player_top1": val_metrics.get("player_top1"),
                "train_overall_exact": train_metrics.get("overall_exact"),
                "val_overall_exact": val_metrics.get("overall_exact"),
                "train_loss": losses.get("train"),
                "val_loss": losses.get("val"),
                "best": bool(record.get("best")),
            }
        )
    return history


@router.get("/train-history")
def train_history(run: str | None = None) -> dict:
    """Per-epoch metrics for an active or packaged Association run."""
    if run is not None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", run) or run.startswith("."):
            raise HTTPException(400, "Invalid Association run name")
        candidates = (
            SPOT_DIR / "exp" / run / "metrics.jsonl",
            ACTION_CHECKPOINTS_DIR / run / "metrics.jsonl",
        )
    else:
        active = _active_job()
        if active is None:
            return {"run": None, "history": []}
        save_dir = (active.get("params") or {}).get("save_dir")
        if not isinstance(save_dir, str):
            return {"run": None, "history": []}
        root = Path(save_dir)
        run = root.name
        candidates = (root / "metrics.jsonl",)
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    return {
        "run": run,
        "history": _association_history(path) if path is not None else [],
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


class AssociationTrainRequest(BaseModel):
    """An independent event-level association experiment."""

    model_config = ConfigDict(extra="forbid")

    train_videos: list[str] = Field(min_length=1)
    val_videos: list[str] = Field(min_length=1)
    run_name: str | None = None
    init_checkpoint: str | None = None
    gpu: int = Field(default=0, ge=0, le=7)
    num_epochs: int = Field(default=40, ge=1, le=1000)
    batch_size: int = Field(default=8, ge=1, le=64)
    learning_rate: float = Field(default=0.0003, gt=0)
    warm_up_epochs: int = Field(default=3, ge=0, le=100)
    backbone: Literal[
        "rny002",
        "rny002_gsm",
        "rny008",
        "rny008_gsm",
        "rn18",
        "rn50",
    ] = "rny002"
    backbone_learning_rate: float = Field(default=0.00003, gt=0)
    crop_dim: int = Field(default=224, ge=64, le=512)
    num_workers: int = Field(default=4, ge=0, le=32)
    stop_vllm: bool = False

    @model_validator(mode="after")
    def distinct_splits(self):
        train = set(self.train_videos)
        validation = set(self.val_videos)
        overlap = sorted(train & validation)
        if overlap:
            raise ValueError(
                "Train and validation videos must be disjoint: "
                + ", ".join(overlap)
            )
        return self


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
        actor_rows, _tally = spot_actor_labels.build(path.stem, events)
        if not actor_rows:
            raise HTTPException(
                400,
                f"No usable yp-actor targets for {name}; it needs reviewed "
                "tracklet labels and Rally Tracking",
            )
        items.append((label_path, path))
    return items


@router.post("/train")
async def train(req: AssociationTrainRequest) -> dict:
    async with _train_start_lock:
        return await _train_locked(req)


async def _train_locked(req: AssociationTrainRequest) -> dict:
    active = _active_job()
    if active is not None:
        raise HTTPException(
            409,
            f"Association training is already active: {active['name']}",
        )
    name = req.run_name or f"yp_actor_{time.strftime('%Y%m%d-%H%M%S')}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name.startswith("."):
        raise HTTPException(
            400,
            "Run name may contain only letters, numbers, dot, underscore and dash",
        )
    checkpoint_dir = ACTION_CHECKPOINTS_DIR / name
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
        checkpoints_root=ACTION_CHECKPOINTS_DIR,
        package_type="yp-video-association-checkpoint",
        label_subdir="action-annotations",
        label_glob="*_actions.jsonl",
        training={
            "purpose": "association",
            "frame_dir": str(ACTION_FRAMES_DIR),
            "selection_metric": "player_top1",
            "label_summary": label_summary,
        },
        cmd=cmd,
    )
    source = run_dir / "labels" / ACTOR_LABEL_SUBDIR
    destination = package_dir / "labels" / ACTOR_LABEL_SUBDIR
    if source.exists():
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
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
    manifest["files"] = [
        *manifest.get("files", []),
        *(
            str(path.relative_to(package_dir))
            for path in sorted(destination.glob(ACTOR_FILE_GLOB))
        ),
    ]
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
        TRAIN_JOB_TYPE,
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
                action_train_router._prepare_action_training_labels,
                items=items,
                frame_dir=ACTION_FRAMES_DIR,
                save_dir=save_dir,
                camera_view="all",
            )
            split = await asyncio.to_thread(
                action_train_router._materialize_holdout_split,
                Path(label_summary["label_dir"]),
                [label.name for label, _video in val_items],
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

            def on_metrics(match: re.Match) -> dict:
                record = json.loads(match.group(1))
                epoch = int(record["epoch"])
                validation = record.get("val") or {}
                if record.get("best"):
                    exporter.schedule(epoch, "new_best")
                top1 = validation.get("player_top1")
                overall = validation.get("overall_exact")
                progress = 0.2 + 0.79 * ((epoch + 1) / req.num_epochs)
                return {
                    "progress": min(progress, 0.99),
                    "message": (
                        f"Epoch {epoch + 1}/{req.num_epochs} · player Top-1 "
                        f"{float(top1 or 0):.1%} · overall "
                        f"{float(overall or 0):.1%}"
                    ),
                    "params": {
                        "association_train_progress": {
                            "epoch": epoch,
                            "epoch_display": epoch + 1,
                            "epochs": req.num_epochs,
                            "train_loss": (record.get("loss") or {}).get("train"),
                            "val_loss": (record.get("loss") or {}).get("val"),
                            "train": record.get("train"),
                            "val": validation,
                            "best": bool(record.get("best")),
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
            await job_manager.update_job(
                job.id,
                status="cancelled",
                message="Cancelled",
                params={
                    **job.params,
                    **(
                        {"checkpoint_package": checkpoint_summary}
                        if checkpoint_summary
                        else {}
                    ),
                },
            )
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


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    videos: list[str]
    #: An independent yp-association checkpoint. None selects the rule.
    association_checkpoint: str | None = None
    stop_vllm: bool = False


@router.post("/predict")
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
        PREDICT_JOB_TYPE,
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
        ),
        start_message="re-deciding actors...",
    )
    return job.to_dict()


class ConfirmRequest(BaseModel):
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
    landed = actor_labels.confirm_auto(stem, confirmable)
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

    current = extraction_store.with_current_actions([result.record], stem)
    record = current[0] if current else result.record
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
