"""SPOT rally (segment) training router.

Trains the yp-spot model on rally annotations as dense segments — every frame
between a rally's start and end is the "rally" class. See ``yp_video.rally_spot``
for the reduced-fps frame-space contract.
"""

from __future__ import annotations

import asyncio
import logging
import os
import traceback
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from yp_video import rally_spot
from yp_video.config import (
    ACTION_CHECKPOINTS_DIR,
    RALLY_SPOT_CHECKPOINTS_DIR,
    RALLY_SPOT_FRAMES_DIR,
    SPOT_DIR,
    SPOT_PYTHON,
    SPOT_TRAIN_MODULE,
)
from yp_video.contracts.action import (
    ACTION_CONTRACT_VERSION,
    ACTION_CONTRACT_VERSION_ENV,
)
from yp_video.action.frames import ensure_action_frame_caches
from yp_video.web.job_helpers import stop_vllm_for_job, stream_subprocess
from yp_video.web.jobs import JobStatus, job_manager
from yp_video.web.spot_runs import (
    PackageExporter,
    TrainProgress,
    checkpoint_package_options,
    export_checkpoint_package,
    last_resumable_epoch,
    make_train_parsers,
    performance_payload,
    resumable_run_options,
    validate_checkpoint_dir,
)

log = logging.getLogger(__name__)
router = APIRouter()

RALLY_RUN_PREFIX = "yp_rally"
SEGMENT_MAP_PATTERN = r"Segment mAP \(mean over tIoU\):\s*([0-9.]+)%"


class RallyTrainRequest(BaseModel):
    # Frame-extraction rate; also the model's sample_fps. 2fps keeps the whole
    # 800-video library around ~30 GB of JPEGs and gives clip_len=64 a 32 s
    # temporal window — rally on/off is a slow signal, it does not need 30fps.
    extract_fps: float = Field(default=2.0, ge=0.5, le=10)
    # 0 = every annotated video. A positive limit takes a seeded-shuffle subset
    # (stable across runs) so a quick experiment doesn't extract 300 hours of
    # frames first.
    video_limit: int = Field(default=100, ge=0)
    camera_view: str = Field(default="all", pattern="^(all|broadcast|sideline)$")
    save_dir: str | None = None
    # None / "" → train from scratch; an explicit path → that checkpoint
    # (a rally or action package — mismatched heads are skipped on load).
    init_checkpoint: str | None = None
    resume: bool = False
    gpu: int = Field(default=0, ge=0)
    feature_arch: str = "rny008_gsm"
    temporal_arch: str = "gru"
    clip_len: int = Field(default=64, ge=8, le=256)
    batch_size: int = Field(default=8, ge=1, le=64)
    num_epochs: int = Field(default=30, ge=1, le=1000)
    warm_up_epochs: int = Field(default=2, ge=0, le=100)
    learning_rate: float = Field(default=0.0003, gt=0)
    num_workers: int = Field(default=4, ge=0, le=32)
    criterion: str = Field(default="map", pattern="^(map|loss)$")
    start_val_epoch: int = Field(default=0, ge=0)
    epoch_num_frames: int | None = Field(default=None, ge=1)
    val_ratio: float = Field(default=0.2, gt=0, lt=1)
    split_seed: int = 42
    stop_vllm: bool = False


def _resolve_save_dir(req: RallyTrainRequest) -> Path:
    if req.save_dir:
        path = Path(os.path.expanduser(req.save_dir))
        return path if path.is_absolute() else SPOT_DIR / path
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{RALLY_RUN_PREFIX}_{req.camera_view}_fps{req.extract_fps:g}_{stamp}"
    return SPOT_DIR / "exp" / name


def _resolve_init_checkpoint(req: RallyTrainRequest) -> Path | None:
    if not req.init_checkpoint:
        return None
    path = Path(os.path.expanduser(req.init_checkpoint))
    if not path.is_absolute():
        path = SPOT_DIR / path
    if not path.exists():
        raise HTTPException(400, f"Init checkpoint not found: {path}")
    return path


def _frame_cache_stats() -> list[dict]:
    """Cached-video counts per extraction rate (rally-spot-frames/fps*/)."""
    if not RALLY_SPOT_FRAMES_DIR.exists():
        return []
    return [
        {
            "fps": d.name.removeprefix("fps"),
            "videos": sum(1 for c in d.iterdir() if c.is_dir()),
        }
        for d in sorted(RALLY_SPOT_FRAMES_DIR.iterdir())
        if d.is_dir() and d.name.startswith("fps")
    ]


def _rally_checkpoint_stats() -> dict:
    count = 0
    if RALLY_SPOT_CHECKPOINTS_DIR.exists():
        count = sum(
            1
            for path in RALLY_SPOT_CHECKPOINTS_DIR.glob("*/checkpoint_best.pt")
            if path.is_file()
        )
    return {
        "dir": str(RALLY_SPOT_CHECKPOINTS_DIR),
        "runs": count,
        "exists": RALLY_SPOT_CHECKPOINTS_DIR.exists(),
    }


def _init_checkpoint_options() -> list[dict]:
    """Rally packages first, then action packages (backbone warm start)."""
    return [
        {**option, "label": f"{kind}: {option['label']}"}
        for kind, checkpoints_dir in (
            ("rally", RALLY_SPOT_CHECKPOINTS_DIR),
            ("action", ACTION_CHECKPOINTS_DIR),
        )
        for option in checkpoint_package_options(checkpoints_dir)
    ]


def _active_job() -> dict | None:
    for job in job_manager.jobs.values():
        if job.type == "rally_spot_train" and job.status == JobStatus.RUNNING:
            return job.to_dict()
    return None


@router.get("/status")
def status() -> dict:
    items, missing = rally_spot.select_training_items(0)
    return {
        "spot_available": SPOT_DIR.exists() and SPOT_PYTHON.exists(),
        "spot_dir": str(SPOT_DIR),
        "rally_annotations": {
            **rally_spot.rally_stats(),
            "with_local_video": len(items),
            "missing_videos": len(missing),
        },
        "frame_caches": _frame_cache_stats(),
        "init_checkpoints": _init_checkpoint_options(),
        "resumable_runs": resumable_run_options(RALLY_RUN_PREFIX),
        "rally_checkpoints": _rally_checkpoint_stats(),
        "active_job": _active_job(),
    }


@router.get("/performance")
def performance(run: str | None = None) -> dict:
    """Per-epoch validation metrics for a rally-spot-checkpoints run."""
    return performance_payload(RALLY_SPOT_CHECKPOINTS_DIR, run)


def _build_command(
    req: RallyTrainRequest,
    *,
    save_dir: Path,
    frame_root: Path,
    label_dir: Path,
    init_checkpoint: Path | None,
) -> list[str]:
    cmd = [
        str(SPOT_PYTHON),
        "-m",
        SPOT_TRAIN_MODULE,
        "yp_rally",
        str(frame_root),
        # Second -m is yp_spot.train's own feature-arch flag, not python's.
        "-m",
        req.feature_arch,
        "-t",
        req.temporal_arch,
        "--clip_len",
        str(req.clip_len),
        # Frames are already extracted at extract_fps, so training strides by 1;
        # recording it in config.json makes inference re-sample native video to
        # the same temporal density automatically.
        "--sample_fps",
        str(req.extract_fps),
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
        "--audio_backend",
        "none",
        "--label_dir",
        str(label_dir),
        "--val_ratio",
        str(req.val_ratio),
        "--split_seed",
        str(req.split_seed),
    ]
    if req.camera_view != "all":
        cmd.extend(["--camera_view", req.camera_view])
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
    return cmd


def _export_rally_checkpoint_package(
    *,
    run_dir: Path,
    package_dir: Path,
    req: RallyTrainRequest,
    cmd: list[str],
    label_summary: dict | None,
) -> dict:
    return export_checkpoint_package(
        run_dir=run_dir,
        package_dir=package_dir,
        checkpoints_root=RALLY_SPOT_CHECKPOINTS_DIR,
        package_type="yp-video-rally-spot-checkpoint",
        label_subdir="rally-annotations",
        label_glob=f"*{rally_spot.RALLY_LABEL_FILE_SUFFIX}",
        training={
            "source": "rally_annotations",
            "extract_fps": req.extract_fps,
            "video_limit": req.video_limit,
            "camera_view": req.camera_view,
            "init_checkpoint": req.init_checkpoint or "",
            "label_summary": label_summary,
        },
        cmd=cmd,
    )


@router.post("/start")
async def start(req: RallyTrainRequest) -> dict:
    if not SPOT_DIR.exists() or not SPOT_PYTHON.exists():
        raise HTTPException(503, f"SPOT is not available at {SPOT_DIR}")
    init_checkpoint = _resolve_init_checkpoint(req)
    save_dir = _resolve_save_dir(req)
    checkpoint_dir = validate_checkpoint_dir(
        RALLY_SPOT_CHECKPOINTS_DIR / save_dir.name, root=RALLY_SPOT_CHECKPOINTS_DIR
    )
    frame_root = rally_spot.frame_cache_root(req.extract_fps)

    job = job_manager.create_job(
        "rally_spot_train",
        {
            "extract_fps": req.extract_fps,
            "video_limit": req.video_limit,
            "camera_view": req.camera_view,
            "frame_dir": str(frame_root),
            "save_dir": str(save_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "gpu": req.gpu,
            "epochs": req.num_epochs,
            "feature_arch": req.feature_arch,
            "criterion": req.criterion,
        },
        name=f"SPOT rally training (fps{req.extract_fps:g})",
    )

    async def run_job() -> None:
        exporter: PackageExporter | None = None
        try:
            await job_manager.update_job(
                job.id, status="running", message="Preparing rally training..."
            )
            items, missing = await asyncio.to_thread(
                rally_spot.select_training_items, req.video_limit
            )
            if not items:
                raise RuntimeError("No rally annotations with a local cut video")

            loop = asyncio.get_running_loop()

            def frame_progress(done: int, total: int, message: str) -> None:
                progress = 0.02 + (0.16 * done / total if total else 0.0)
                loop.call_soon_threadsafe(
                    lambda progress=progress, message=message: asyncio.ensure_future(
                        job_manager.update_job(
                            job.id, progress=progress, message=message
                        )
                    )
                )

            frame_summary = await asyncio.to_thread(
                ensure_action_frame_caches,
                [(video_path, None) for _ann, video_path in items],
                cache_root=frame_root,
                fps=req.extract_fps,
                progress=frame_progress,
            )
            await job_manager.update_job(
                job.id,
                progress=0.18,
                message="Frame cache ready.",
                params={
                    **job.params,
                    "frame_cache": frame_summary,
                    "missing_videos": missing,
                },
            )

            label_summary = await asyncio.to_thread(
                rally_spot.write_training_labels,
                items,
                cache_root=frame_root,
                extract_fps=req.extract_fps,
                label_dir=save_dir / "labels" / "rally-annotations",
            )
            await job_manager.update_job(
                job.id,
                progress=0.2,
                message="Training labels written.",
                params={**job.params, "training_labels": label_summary},
            )

            cmd = _build_command(
                req,
                save_dir=save_dir,
                frame_root=frame_root,
                label_dir=Path(label_summary["label_dir"]),
                init_checkpoint=init_checkpoint,
            )
            await job_manager.update_job(job.id, message="Waiting for GPU...")
            async with stop_vllm_for_job(job.id, when=req.stop_vllm):
                async with job_manager.gpu_lock:
                    await job_manager.update_job(
                        job.id, message="Starting SPOT rally training..."
                    )
                    ctx = TrainProgress(epochs=req.num_epochs)
                    exporter = PackageExporter(
                        job.id,
                        save_dir,
                        lambda: _export_rally_checkpoint_package(
                            run_dir=save_dir,
                            package_dir=checkpoint_dir,
                            req=req,
                            cmd=cmd,
                            label_summary=label_summary,
                        ),
                    )
                    parsers, is_key_line = make_train_parsers(
                        ctx,
                        params_key="rally_train_progress",
                        criterion=req.criterion,
                        headline_pattern=SEGMENT_MAP_PATTERN,
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
                        terminal_prefix="[rally-train] ",
                        log_path=save_dir / "terminal.log",
                    )
            if rc == 0:
                checkpoint_summary = await exporter.export_once(
                    expected_epoch=None, reason="completed", update_job=False
                )
                if checkpoint_summary is None:
                    raise RuntimeError(
                        f"Training finished but no checkpoint package was exported to {checkpoint_dir}"
                    )
                await job_manager.update_job(
                    job.id,
                    status="completed",
                    progress=1.0,
                    message=f"Training complete: {checkpoint_dir}",
                    params={**job.params, "checkpoint_package": checkpoint_summary},
                )
            else:
                raise RuntimeError(last_line or f"SPOT training exited with code {rc}")
        except asyncio.CancelledError:
            checkpoint_summary = None
            if exporter is not None:
                try:
                    checkpoint_summary = await exporter.export_once(
                        expected_epoch=None, reason="cancelled", update_job=False
                    )
                except Exception:  # noqa: BLE001
                    log.exception("Failed to export rally checkpoint package after cancellation")
            await job_manager.update_job(
                job.id,
                status="cancelled",
                message="Training cancelled",
                params={
                    **job.params,
                    **({"checkpoint_package": checkpoint_summary} if checkpoint_summary else {}),
                },
            )
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            print(f"\n[rally-train] Failed:\n{tb}", flush=True)
            log.error("Rally training failed:\n%s", tb)
            job_obj = job_manager.get_job(job.id)
            if job_obj:
                job_obj.logs.append(f"{type(exc).__name__}: {exc}")
                job_obj.logs.extend(tb.splitlines())
            await job_manager.update_job(
                job.id,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                message="SPOT rally training failed",
            )

    task = asyncio.create_task(run_job())
    job_manager.attach_task(job, task)
    return job.to_dict()
