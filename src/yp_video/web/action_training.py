"""The shared SPOT action-training launcher.

Action Train and Fusion Train are the same trainer with a different flavor
(job type, package type, run naming); Association Train reuses the label
snapshot machinery. This module owns the request models, the command builder
and ``start_training_job`` once, so the routers stay HTTP-thin and never
import each other. Domain rules about the label corpus live below the web
layer, in ``yp_video.action.training`` / ``yp_video.actor.training_labels``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from yp_video.action import training
from yp_video.action.frames import ensure_action_frame_caches
from yp_video.action.prelabel import resolve_checkpoint_path
from yp_video.actor.training_labels import prepare_action_training_labels
from yp_video.config import (
    ACTION_ANNOTATIONS_DIR,
    ACTION_AUDIO_DIR,
    ACTION_CHECKPOINTS_DIR,
    ACTION_FRAMES_DIR,
    ACTION_VAL_SET_FILE,
    CUTS_DIRS,
    SPOT_AUDIO_PRECOMPUTE_MODULE,
    SPOT_DIR,
    SPOT_PYTHON,
    SPOT_TRAIN_MODULE,
)
from yp_video.contracts.action import (
    ACTION_CONTRACT_VERSION,
    ACTION_CONTRACT_VERSION_ENV,
    ACTOR_FILE_GLOB,
    ACTOR_LABEL_SUBDIR,
)
from yp_video.web.job_helpers import (
    fail_job_from_exc,
    stop_vllm_for_job,
    stream_subprocess,
    terminal_prefix,
)
from yp_video.web.jobs import JobType, job_manager
from yp_video.web.spot_runs import (
    PackageExporter,
    TrainProgress,
    export_checkpoint_package,
    last_resumable_epoch,
    make_train_parsers,
    validate_checkpoint_dir,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingFlavor:
    """What differs between SPOT action-training variants."""

    job_type: JobType
    job_name: str
    package_type: str
    progress_key: str
    subject: str


ACTION_TRAINING = TrainingFlavor(
    job_type=JobType.ACTION_TRAIN,
    job_name="Action Train",
    package_type="yp-video-action-checkpoint",
    progress_key="action_train_progress",
    subject="action",
)


class ActionTrainBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str | None = None
    frame_dir: str | None = None
    save_dir: str | None = None
    checkpoint_dir: str | None = None
    # None / "" → train from scratch; an explicit path → that checkpoint
    # (selected from ACTION_CHECKPOINTS_DIR).
    init_checkpoint: str | None = None
    # Continue an interrupted run: restore weights + optimizer/scheduler/history
    # from `save_dir` and keep training toward num_epochs. Requires `save_dir`
    # to point at an existing run with optimizer state; `init_checkpoint` is
    # ignored (SPOT loads from the checkpoint instead).
    resume: bool = False
    gpu: int = Field(default=0, ge=0)
    # "logmel" → late-fusion audio (precomputed before training); "none" →
    # pure-visual model (no audio, no precompute). Must match at inference.
    audio_backend: str = Field(default="logmel", pattern="^(logmel|none)$")
    feature_arch: str = "rny008_gsm"
    temporal_arch: str = "gru"
    pred_loc_arch: str = "mlp"
    clip_len: int = Field(default=64, ge=8, le=256)
    # Per-video frame stride targeting this sampling rate, so clip_len spans
    # the same wall-clock time on 30fps and 60fps sources. 0 = every frame.
    sample_fps: float = Field(default=30.0, ge=0, le=120)
    batch_size: int = Field(default=8, ge=1, le=64)
    num_epochs: int = Field(default=50, ge=1, le=1000)
    warm_up_epochs: int = Field(default=3, ge=0, le=100)
    learning_rate: float = Field(default=0.0003, gt=0)
    num_workers: int = Field(default=4, ge=0, le=32)
    criterion: str = Field(default="map", pattern="^(map|loss)$")
    start_val_epoch: int = Field(default=0, ge=0)
    epoch_num_frames: int | None = Field(default=None, ge=1)
    predict_location: bool = True
    # Also learn WHICH PLAYER acted, from the actor-candidate sidecar the
    # label snapshot carries. Rejected up front when the snapshot has no
    # actor work — see the flag's use in the command builder.
    predict_actor: bool = False
    stop_vllm: bool = False

    @model_validator(mode="after")
    def validate_resume_mode(self):
        if self.resume and self.init_checkpoint:
            raise ValueError("resume and init_checkpoint are mutually exclusive")
        if self.resume and not self.save_dir:
            raise ValueError("resume requires save_dir")
        return self


class VnlActionTrainRequest(ActionTrainBase):
    """Built-in VNL data owns its train/val split and has no camera-view mode."""

    source: Literal["vnl_1_5"] = "vnl_1_5"


class AnnotationActionTrainRequest(ActionTrainBase):
    source: Literal["action_annotations"]
    training_mode: Literal["split", "all", "holdout"] = "split"
    val_ratio: float = Field(default=0.2, gt=0, lt=1)
    split_seed: int = 42
    # holdout mode: the exact videos to hold out as the validation set; every
    # other labelled video trains. Entries may be the raw stem, `<stem>.mp4`, or
    # `<stem>_actions.jsonl` — matched against the run-local label snapshot.
    holdout_videos: list[str] = Field(default_factory=list)
    # "all" trains every view together; "broadcast"/"sideline" restrict to one
    # camera view (labels carry a camera_view tag from prepare_action_training_labels).
    camera_view: Literal["all", "broadcast", "sideline"] = "all"


ActionTrainRequest = Annotated[
    VnlActionTrainRequest | AnnotationActionTrainRequest,
    Field(discriminator="source"),
]


def spot_path(path: str | Path) -> Path:
    p = Path(os.path.expanduser(str(path)))
    if not p.is_absolute():
        p = SPOT_DIR / p
    return p


def default_frame_dir(source: str) -> str:
    if source == "vnl_1_5":
        return "data/vnl_1.5/frames_224p"
    return str(ACTION_FRAMES_DIR)


def default_dataset(source: str) -> str:
    return "vnl_1.5" if source == "vnl_1_5" else "yp_actions"


def _safe_run_name(dataset: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset).strip("._") or "actions"


def _audio_tag(req: ActionTrainBase) -> str:
    """Run-name fragment marking the modality: 'visual' or 'fusion'.

    Makes a run dir self-describing (e.g. yp_actions_fusion_<stamp> vs
    yp_actions_visual_<stamp>) so visual-only and audio late-fusion runs are
    distinguishable at a glance in exp/ and action-checkpoints/.
    """
    return "visual" if req.audio_backend == "none" else "fusion"


def _resolve_save_dir(req: ActionTrainRequest, dataset: str | None = None) -> Path:
    dataset = dataset or req.dataset or default_dataset(req.source)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    view = req.camera_view if isinstance(req, AnnotationActionTrainRequest) else "all"
    name = f"{_safe_run_name(dataset)}_{view}_{_audio_tag(req)}_{stamp}"
    return spot_path(req.save_dir or (Path("exp") / name))


def _action_checkpoint_path(path: str | Path) -> Path:
    return validate_checkpoint_dir(
        resolve_checkpoint_path(path), root=ACTION_CHECKPOINTS_DIR
    )


def _resolve_checkpoint_dir(req: ActionTrainBase, *, save_dir: Path) -> Path:
    if req.checkpoint_dir:
        return _action_checkpoint_path(req.checkpoint_dir)
    return validate_checkpoint_dir(
        ACTION_CHECKPOINTS_DIR / save_dir.name, root=ACTION_CHECKPOINTS_DIR
    )


def _resolve_holdout_videos(req: AnnotationActionTrainRequest) -> list[str]:
    """Explicit request list wins; otherwise fall back to the val-set file."""
    names = req.holdout_videos or training.read_val_set_file()
    if not names:
        raise ValueError(
            "holdout mode needs a validation set. Add one video filename per line "
            f"to {ACTION_VAL_SET_FILE}"
        )
    return names


def _resolve_audio_dir(req: ActionTrainRequest, *, frame_dir: Path) -> Path | None:
    """Per-frame audio feature dir for this run's backend, or None for visual-only.

    Action labels precompute into a managed per-backend cache (built here, see
    ``_audio_precompute_command``). The VNL dataset's source videos aren't local,
    so its features must be precomputed offline next to the frame dir — fail loud
    if absent rather than silently training without audio.
    """
    if req.audio_backend == "none":
        return None
    if req.source == "vnl_1_5":
        audio_dir = frame_dir.parent / f"audio_{req.audio_backend}"
        if not audio_dir.exists():
            raise RuntimeError(
                f"VNL audio features not found: {audio_dir}. Precompute them with "
                f"`python -m yp_spot.audio.precompute --label-file data/vnl_1.5/*.jsonl "
                f"--video-root <vnl videos> --out {audio_dir} --backend {req.audio_backend}`, "
                "or set the audio backend to none for a visual-only model."
            )
        return audio_dir
    return ACTION_AUDIO_DIR / req.audio_backend


def _audio_precompute_command(
    req: ActionTrainBase, *, label_dir: Path, audio_dir: Path
) -> list[str]:
    """Build the ``yp_spot.audio.precompute`` command for the run-local labels.

    Features are keyed by video name and reused across runs (precompute skips
    already-cached videos), so re-training the same set is cheap.
    """
    label_files = sorted(label_dir.glob("*_actions.jsonl"))
    if not label_files:
        raise RuntimeError(f"No action labels to precompute audio from in {label_dir}")
    return [
        str(SPOT_PYTHON),
        "-m",
        SPOT_AUDIO_PRECOMPUTE_MODULE,
        "--label-file",
        *(str(p) for p in label_files),
        "--video-root",
        *(str(d) for d in CUTS_DIRS),
        "--out",
        str(audio_dir),
        "--backend",
        req.audio_backend,
    ]


def _export_action_checkpoint_package(
    *,
    run_dir: Path,
    package_dir: Path,
    req: ActionTrainRequest,
    cmd: list[str],
    label_summary: dict | None,
    flavor: TrainingFlavor = ACTION_TRAINING,
) -> dict:
    return export_checkpoint_package(
        run_dir=run_dir,
        package_dir=package_dir,
        checkpoints_root=ACTION_CHECKPOINTS_DIR,
        package_type=flavor.package_type,
        label_subdir="action-annotations",
        label_glob="*_actions.jsonl",
        training={
            "source": req.source,
            "training_mode": (
                req.training_mode
                if isinstance(req, AnnotationActionTrainRequest)
                else "dataset_split"
            ),
            "dataset": req.dataset or default_dataset(req.source),
            "frame_dir": str(spot_path(req.frame_dir or default_frame_dir(req.source))),
            "init_checkpoint": req.init_checkpoint or "",
            "purpose": flavor.subject,
            "label_summary": label_summary,
        },
        cmd=cmd,
    )


def _build_command(
    req: ActionTrainRequest,
    *,
    save_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
    action_label_dir: Path | None = None,
    audio_dir: Path | None = None,
    actor_dir: Path | None = None,
) -> tuple[list[str], Path, dict]:
    if not SPOT_DIR.exists():
        raise HTTPException(503, "SPOT is not available at ~/yp-spot")
    if not SPOT_PYTHON.exists():
        raise HTTPException(503, f"SPOT python not found: {SPOT_PYTHON}")

    dataset = req.dataset or default_dataset(req.source)
    frame_dir_value = req.frame_dir or default_frame_dir(req.source)
    frame_dir = spot_path(frame_dir_value)
    if not frame_dir.exists():
        raise HTTPException(400, f"Frame directory not found: {frame_dir}")

    if req.source == "vnl_1_5":
        for rel in ("data/vnl_1.5/train.jsonl", "data/vnl_1.5/val.jsonl"):
            if not (SPOT_DIR / rel).exists():
                raise HTTPException(400, f"Missing VNL JSONL labels: {SPOT_DIR / rel}")
    if req.init_checkpoint:
        init_checkpoint = spot_path(req.init_checkpoint)
        if not init_checkpoint.exists():
            raise HTTPException(400, f"Init checkpoint not found: {init_checkpoint}")
    else:
        init_checkpoint = None

    save_dir = save_dir or _resolve_save_dir(req, dataset)
    checkpoint_dir = checkpoint_dir or _resolve_checkpoint_dir(req, save_dir=save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(SPOT_PYTHON),
        "-m",
        SPOT_TRAIN_MODULE,
        dataset,
        str(frame_dir),
        # Second -m is yp_spot.train's own feature-arch flag, not python's.
        "-m",
        req.feature_arch,
        "-t",
        req.temporal_arch,
        "-p",
        req.pred_loc_arch,
        "--clip_len",
        str(req.clip_len),
        "--sample_fps",
        str(req.sample_fps),
        "--batch_size",
        str(req.batch_size),
        "--num_epochs",
        str(req.num_epochs),
        "--warm_up_epochs",
        str(req.warm_up_epochs),
        "--learning_rate",
        str(req.learning_rate),
        "--num_workers",
        str(req.num_workers),
        "--criterion",
        req.criterion,
        "--start_val_epoch",
        str(req.start_val_epoch),
        "-s",
        str(save_dir),
    ]
    cmd.extend(["--audio_backend", req.audio_backend])
    if req.audio_backend != "none":
        if audio_dir is None:
            raise HTTPException(400, "Audio features missing for late-fusion training")
        cmd.extend(["--audio_dir", str(audio_dir)])
    if isinstance(req, AnnotationActionTrainRequest) and req.camera_view != "all":
        cmd.extend(["--camera_view", req.camera_view])
    if req.predict_location:
        cmd.append("--predict_location")
    if req.predict_actor:
        # Refuse rather than train a head with nothing to learn from. This
        # used to fall through silently, which was tolerable while the flag
        # was API-only; it is a checkbox now, and a run that quietly produces
        # no actor head costs hours before anyone finds out.
        if actor_dir is None or not any(actor_dir.glob(ACTOR_FILE_GLOB)):
            raise HTTPException(
                400,
                "Predict actor needs reviewed actor labels in the snapshot; "
                "none of the selected videos have any. Review actors in "
                "Association Label first.",
            )
        cmd.extend(["--predict_actor", "--actor_dir", str(actor_dir)])
    if req.resume:
        if last_resumable_epoch(save_dir) is None:
            raise HTTPException(
                400,
                f"Cannot resume: no optimizer checkpoint (optim_*.pt) in {save_dir}",
            )
        cmd.append("--resume")
    elif init_checkpoint is not None:
        cmd.extend(["--init_checkpoint", str(init_checkpoint)])
    if req.epoch_num_frames is not None:
        cmd.extend(["--epoch_num_frames", str(req.epoch_num_frames)])
    if isinstance(req, AnnotationActionTrainRequest):
        label_dir = action_label_dir or ACTION_ANNOTATIONS_DIR
        if not any(label_dir.glob("*_actions.jsonl")):
            raise HTTPException(400, f"No action JSONL labels found in {label_dir}")
        if req.training_mode == "all":
            cmd.extend([
                "--train_labels",
                str(label_dir),
                "--val_labels",
                str(label_dir),
            ])
        elif req.training_mode == "holdout":
            # train/ and val/ are symlink dirs materialized next to the flat
            # snapshot by materialize_holdout_split before training starts.
            cmd.extend([
                "--train_labels",
                str(label_dir.parent / "train"),
                "--val_labels",
                str(label_dir.parent / "val"),
            ])
        else:
            cmd.extend([
                "--label_dir",
                str(label_dir),
                "--val_ratio",
                str(req.val_ratio),
                "--split_seed",
                str(req.split_seed),
            ])

    params = {
        "source": req.source,
        "dataset": dataset,
        "frame_dir": str(frame_dir),
        "save_dir": str(save_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "init_checkpoint": str(init_checkpoint) if init_checkpoint else "",
        "resume": req.resume,
        "gpu": req.gpu,
        "epochs": req.num_epochs,
        "feature_arch": req.feature_arch,
        "criterion": req.criterion,
        "audio_backend": req.audio_backend,
    }
    if audio_dir is not None:
        params["audio_dir"] = str(audio_dir)
    if isinstance(req, AnnotationActionTrainRequest):
        params["label_dir"] = str(action_label_dir or ACTION_ANNOTATIONS_DIR)
        params["training_mode"] = req.training_mode
        params["camera_view"] = req.camera_view
        if req.training_mode == "split":
            params["val_ratio"] = req.val_ratio
            params["split_seed"] = req.split_seed
        elif req.training_mode == "holdout":
            # Resolved val list lands in training_labels.val_videos; record the
            # source here so the manifest shows where the split came from.
            params["holdout_videos"] = req.holdout_videos
            params["val_set_file"] = str(ACTION_VAL_SET_FILE)
    return cmd, save_dir, params


async def start_training_job(
    req: ActionTrainRequest,
    *,
    flavor: TrainingFlavor = ACTION_TRAINING,
    label_items: list[tuple[Path, Path]] | None = None,
    reuse_existing_labels: bool = False,
    require_actor_targets: bool = False,
) -> dict:
    """Start one SPOT training job using the shared action/actor pipeline.

    ``label_items`` lets Association Train provide the exact videos selected
    on its page, including action pre-annotations. ``reuse_existing_labels``
    resumes a joint-head run against its frozen label/actor snapshot instead
    of silently replacing it with today's annotation corpus.
    """
    dataset = req.dataset or default_dataset(req.source)
    save_dir = _resolve_save_dir(req, dataset)
    checkpoint_dir = _resolve_checkpoint_dir(req, save_dir=save_dir)
    initial_params = {
        "source": req.source,
        "dataset": dataset,
        "frame_dir": str(spot_path(req.frame_dir or default_frame_dir(req.source))),
        "save_dir": str(save_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "gpu": req.gpu,
        "epochs": req.num_epochs,
        "feature_arch": req.feature_arch,
        "criterion": req.criterion,
    }
    if isinstance(req, AnnotationActionTrainRequest):
        initial_params["training_mode"] = req.training_mode
    job = job_manager.create_job(
        flavor.job_type,
        initial_params,
        name=f"{flavor.job_name} ({save_dir.name})",
    )

    async def run_job() -> None:
        exporter: PackageExporter | None = None
        try:
            await job_manager.update_job(
                job.id,
                status="running",
                message=f"Preparing {flavor.subject} training...",
            )
            frame_dir = spot_path(req.frame_dir or default_frame_dir(req.source))
            action_label_dir = None
            label_summary = None
            if isinstance(req, AnnotationActionTrainRequest):
                if reuse_existing_labels:
                    action_label_dir = (
                        save_dir / "labels" / "action-annotations"
                    )
                    actor_dir = save_dir / "labels" / ACTOR_LABEL_SUBDIR
                    if not any(action_label_dir.glob("*_actions.jsonl")):
                        raise RuntimeError(
                            f"Resume label snapshot is missing: {action_label_dir}"
                        )
                    if not any(actor_dir.glob(ACTOR_FILE_GLOB)):
                        raise RuntimeError(
                            f"Resume actor snapshot is missing: {actor_dir}"
                        )
                    label_summary = {
                        "label_dir": str(action_label_dir),
                        "actor_dir": str(actor_dir),
                        "reused": True,
                    }
                    await job_manager.update_job(
                        job.id,
                        progress=0.2,
                        message="Reusing frozen training label snapshot.",
                        params={
                            **job.params,
                            "training_labels": label_summary,
                        },
                    )
                else:
                    items = (
                        list(label_items)
                        if label_items is not None
                        else await asyncio.to_thread(training.label_items)
                    )
                    if not items:
                        raise RuntimeError("No action labels selected for training")

                    loop = asyncio.get_running_loop()

                    def frame_progress(done: int, total: int, message: str) -> None:
                        progress = 0.02 + (0.16 * done / total if total else 0.0)
                        loop.call_soon_threadsafe(
                            lambda progress=progress, message=message: asyncio.ensure_future(
                                job_manager.update_job(
                                    job.id,
                                    progress=progress,
                                    message=message,
                                )
                            )
                        )

                    # Action JSONL metadata can inherit an over-reported MP4 frame
                    # count, so expected_frames is None — the training labels are
                    # normalized against the extracted cache in the next step.
                    summary = await asyncio.to_thread(
                        ensure_action_frame_caches,
                        [(video_path, None) for _label, video_path in items],
                        cache_root=frame_dir,
                        progress=frame_progress,
                    )
                    await job_manager.update_job(
                        job.id,
                        progress=0.18,
                        message="Frame cache ready.",
                        params={**job.params, "frame_cache": summary},
                    )
                    label_summary = await asyncio.to_thread(
                        prepare_action_training_labels,
                        items=items,
                        frame_dir=frame_dir,
                        save_dir=save_dir,
                        camera_view=req.camera_view,
                        require_actor_targets=require_actor_targets,
                    )
                    action_label_dir = Path(label_summary["label_dir"])
                    if req.training_mode == "holdout":
                        holdout_videos = _resolve_holdout_videos(req)
                        split = await asyncio.to_thread(
                            training.materialize_holdout_split,
                            action_label_dir,
                            holdout_videos,
                        )
                        label_summary = {**label_summary, **split}
                    await job_manager.update_job(
                        job.id,
                        progress=0.2,
                        message="Training labels validated.",
                        params={
                            **job.params,
                            "training_labels": label_summary,
                        },
                    )

            # Resolve / build audio features for late fusion (no-op visual-only).
            audio_dir = await asyncio.to_thread(
                _resolve_audio_dir, req, frame_dir=frame_dir
            )
            if audio_dir is not None and isinstance(req, AnnotationActionTrainRequest):
                if action_label_dir is None:
                    raise RuntimeError("Action label snapshot was not prepared")
                audio_dir.mkdir(parents=True, exist_ok=True)
                pre_cmd = _audio_precompute_command(
                    req, label_dir=action_label_dir, audio_dir=audio_dir
                )
                await job_manager.update_job(
                    job.id,
                    message=f"Precomputing {req.audio_backend} audio features...",
                )
                rc, last_line = await stream_subprocess(job.id, pre_cmd, cwd=SPOT_DIR)
                if rc != 0:
                    raise RuntimeError(
                        f"Audio precompute failed (rc={rc}): {last_line}"
                    )

            cmd, resolved_save_dir, params = _build_command(
                req,
                save_dir=save_dir,
                checkpoint_dir=checkpoint_dir,
                action_label_dir=action_label_dir,
                audio_dir=audio_dir,
                actor_dir=save_dir / "labels" / ACTOR_LABEL_SUBDIR,
            )
            await job_manager.update_job(
                job.id,
                params={**job.params, **params},
                message="Waiting for GPU...",
            )
            async with stop_vllm_for_job(job.id, when=req.stop_vllm):
                async with job_manager.gpu_lock:
                    await job_manager.update_job(
                        job.id,
                        message=f"Starting SPOT {flavor.subject} training...",
                    )
                    ctx = TrainProgress(epochs=req.num_epochs)
                    exporter = PackageExporter(
                        job.id,
                        resolved_save_dir,
                        lambda: _export_action_checkpoint_package(
                            run_dir=resolved_save_dir,
                            package_dir=checkpoint_dir,
                            req=req,
                            cmd=cmd,
                            label_summary=label_summary,
                            flavor=flavor,
                        ),
                    )

                    parsers, is_key_line = make_train_parsers(
                        ctx,
                        params_key=flavor.progress_key,
                        criterion=req.criterion,
                        headline_pattern=(
                            r"Harmonic mean \(temporal and spatial mAPs\):\s*([0-9.]+)%"
                        ),
                        on_new_best=lambda: exporter.schedule(
                            ctx.best_epoch, "new_best"
                        ),
                    )

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
                    rc, last_line = await stream_subprocess(
                        job.id,
                        cmd,
                        cwd=SPOT_DIR,
                        env=env,
                        parsers=parsers,
                        is_key_line=is_key_line,
                        tee_to_terminal=True,
                        log_path=resolved_save_dir / "terminal.log",
                    )
            if rc == 0:
                if exporter is None:
                    raise RuntimeError("Checkpoint package exporter was not initialized")
                checkpoint_summary = await exporter.export_once(
                    expected_epoch=None,
                    reason="completed",
                    update_job=False,
                )
                if checkpoint_summary is None:
                    raise RuntimeError(f"Training finished but no checkpoint package was exported to {checkpoint_dir}")
                await job_manager.update_job(
                    job.id,
                    status="completed",
                    progress=1.0,
                    message=f"{flavor.job_name} complete: {checkpoint_dir}",
                    params={**job.params, "checkpoint_package": checkpoint_summary},
                )
            else:
                raise RuntimeError(last_line or f"SPOT training exited with code {rc}")
        except asyncio.CancelledError:
            checkpoint_summary = None
            if exporter is not None:
                try:
                    checkpoint_summary = await exporter.export_once(
                        expected_epoch=None,
                        reason="cancelled",
                        update_job=False,
                    )
                except Exception:  # noqa: BLE001
                    log.exception("Failed to export action checkpoint package after cancellation")
            if checkpoint_summary:
                await job_manager.update_job(
                    job.id,
                    params={**job.params, "checkpoint_package": checkpoint_summary},
                )
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"{terminal_prefix(job)}Failed: {type(exc).__name__}: {exc}", flush=True)
            log.exception("Action training failed")
            checkpoint_summary = None
            if exporter is not None:
                try:
                    checkpoint_summary = await exporter.export_once(
                        expected_epoch=None,
                        reason="failed",
                        update_job=False,
                    )
                except Exception:  # noqa: BLE001
                    log.exception("Failed to export action checkpoint package after failure")
            if checkpoint_summary:
                job_obj = job_manager.get_job(job.id)
                await job_manager.update_job(
                    job.id,
                    params={
                        **(job_obj.params if job_obj else job.params),
                        "checkpoint_package": checkpoint_summary,
                    },
                )
            await fail_job_from_exc(job.id, exc)

    task = asyncio.create_task(run_job())
    job_manager.attach_task(job, task)
    return job.to_dict()
