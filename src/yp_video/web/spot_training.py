"""The one SPOT training launcher behind Fusion Train.

Every recipe — rally, rally + winner, action, association + action — is the
same trainer with a different task list. This module owns the run naming,
the command builder and ``start_training_job`` once; which labels a recipe
draws from is ``label_sources``, and what the recipe trains is the contract
registry (``RECIPES`` / ``TASKS``). The router stays HTTP-thin.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path

from fastapi import HTTPException

from yp_video.action import training
from yp_video.action.prelabel import resolve_checkpoint_path
from yp_video.config import (
    ACTION_AUDIO_DIR,
    CUTS_DIRS,
    SPOT_AUDIO_PRECOMPUTE_MODULE,
    SPOT_CHECKPOINTS_DIR,
    SPOT_DIR,
    SPOT_PYTHON,
    SPOT_TRAIN_MODULE,
)
from yp_video.contracts.action import (
    ACTION_CONTRACT_VERSION,
    ACTION_CONTRACT_VERSION_ENV,
    RECIPES,
    SPOT_PACKAGE_TYPE,
    TASKS,
    Recipe,
    spotting_task,
)
from yp_video.web.job_helpers import (
    fail_job_from_exc,
    stop_vllm_for_job,
    stream_subprocess,
    terminal_prefix,
)
from yp_video.web.jobs import JobType, job_manager
from yp_video.web.label_sources import (
    PreparedLabels,
    check_task_supervision,
    label_stem,
    source_for,
)
from yp_video.web.spot_runs import (
    PackageExporter,
    TrainProgress,
    dedupe_run_name,
    export_checkpoint_package,
    make_train_parsers,
    spot_run_name,
    validate_checkpoint_dir,
)
from yp_video.web.train_requests import FusionTrainRequest

log = logging.getLogger(__name__)

PROGRESS_KEY = "spot_train_progress"

#: The per-epoch headline line ``make_train_parsers`` reads, per spotting task.
HEADLINE_PATTERNS = {
    "rally": r"Segment mAP \(mean over tIoU\):\s*([0-9.]+)%",
    "action": r"Harmonic mean \(temporal and spatial mAPs\):\s*([0-9.]+)%",
}

_RUN_NAME_RE = re.compile(r"[A-Za-z0-9_.-]+")


def recipe_token(recipe: Recipe) -> str:
    """Run-name token: ``ral`` / ``ral_win`` / ``act`` / ``ass_act``."""
    tasks = recipe.tasks
    if "action" in tasks:
        return "ass_act" if "actor" in tasks else "act"
    return "ral_win" if "winner" in tasks else "ral"


def resolve_run_name(req: FusionTrainRequest, recipe: Recipe) -> str:
    name = req.run_name or dedupe_run_name(
        spot_run_name(
            view=req.camera_view, task=recipe_token(recipe), feature_arch=req.feature_arch
        ),
        SPOT_DIR / "exp",
    )
    if not _RUN_NAME_RE.fullmatch(name) or name.startswith("."):
        raise HTTPException(
            400, "Run name may contain only letters, numbers, dot, underscore and dash"
        )
    return name


def _resolve_init_checkpoint(req: FusionTrainRequest) -> Path | None:
    if not req.init_checkpoint:
        return None
    path = resolve_checkpoint_path(req.init_checkpoint, root=SPOT_CHECKPOINTS_DIR)
    if not path.exists():
        raise HTTPException(400, f"Init checkpoint not found: {path}")
    return path


def _audio_backend(req: FusionTrainRequest, recipe: Recipe) -> str:
    """Rally recipes are visual-only: rally spans have no audio cue worth a
    late-fusion branch, and the reduced-fps cache has no audio features."""
    return req.audio_backend if spotting_task(recipe.tasks) == "action" else "none"


def _audio_precompute_command(
    backend: str, *, label_dir: Path, audio_dir: Path
) -> list[str]:
    """``yp_spot.audio.precompute`` over the run-local labels. Features are
    keyed by video and reused across runs (already-cached videos are skipped)."""
    label_files = sorted(label_dir.glob(TASKS["action"].label_glob))
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
        backend,
    ]


def build_command(
    req: FusionTrainRequest,
    recipe: Recipe,
    prepared: PreparedLabels,
    *,
    save_dir: Path,
    init_checkpoint: Path | None,
    audio_dir: Path | None,
) -> list[str]:
    rally = spotting_task(recipe.tasks) == "rally"
    cmd = [
        str(SPOT_PYTHON),
        "-m",
        SPOT_TRAIN_MODULE,
        prepared.dataset,
        str(prepared.frame_dir),
        # Second -m is yp_spot.train's own feature-arch flag, not python's.
        "-m",
        req.feature_arch,
        "-t",
        req.temporal_arch,
        "--tasks",
        ",".join(recipe.tasks),
        "--clip_len",
        str(req.clip_len),
        # Rally frames are extracted at extract_fps, so training strides by 1;
        # recording it as sample_fps makes inference re-sample native video
        # to the same temporal density.
        "--sample_fps",
        str(req.extract_fps if rally else req.sample_fps),
        "--batch_size",
        str(req.batch_size),
        "--acc_grad_iter",
        str(1 if rally else req.acc_grad_iter),
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
        _audio_backend(req, recipe),
        *prepared.extra_args,
    ]
    if audio_dir is not None:
        cmd.extend(["--audio_dir", str(audio_dir)])
    if req.camera_view != "all":
        cmd.extend(["--camera_view", req.camera_view])
    if init_checkpoint is not None:
        cmd.extend(["--init_checkpoint", str(init_checkpoint)])
    if req.epoch_num_frames is not None:
        cmd.extend(["--epoch_num_frames", str(req.epoch_num_frames)])

    label_dir = prepared.label_dir
    if req.validation == "ratio":
        cmd.extend([
            "--label_dir", str(label_dir),
            "--val_ratio", str(req.val_ratio),
            "--split_seed", str(req.split_seed),
        ])
    elif req.validation == "manual":
        # train/ and val/ are symlink dirs materialized next to the flat
        # snapshot by materialize_holdout_split before training starts.
        cmd.extend([
            "--train_labels", str(label_dir.parent / "train"),
            "--val_labels", str(label_dir.parent / "val"),
        ])
    else:
        cmd.extend(["--train_labels", str(label_dir), "--val_labels", str(label_dir)])
    return cmd


def _export_package(
    *,
    run_dir: Path,
    package_dir: Path,
    req: FusionTrainRequest,
    recipe: Recipe,
    prepared: PreparedLabels,
    cmd: list[str],
) -> dict:
    return export_checkpoint_package(
        run_dir=run_dir,
        package_dir=package_dir,
        checkpoints_root=SPOT_CHECKPOINTS_DIR,
        package_type=SPOT_PACKAGE_TYPE,
        recipe=recipe.id,
        tasks=recipe.tasks,
        label_subdirs=prepared.label_subdirs,
        training={
            "recipe": recipe.id,
            "validation": req.validation,
            "dataset": prepared.dataset,
            "frame_dir": str(prepared.frame_dir),
            "camera_view": req.camera_view,
            "init_checkpoint": req.init_checkpoint or "",
            "label_summary": prepared.summary,
        },
        cmd=cmd,
        serveable_tasks=[t for t in recipe.tasks if TASKS[t].serveable],
    )


async def start_training_job(req: FusionTrainRequest) -> dict:
    if not SPOT_DIR.exists() or not SPOT_PYTHON.exists():
        raise HTTPException(503, f"SPOT is not available at {SPOT_DIR}")
    recipe = RECIPES[req.recipe]
    init_checkpoint = _resolve_init_checkpoint(req)
    name = resolve_run_name(req, recipe)
    save_dir = SPOT_DIR / "exp" / name
    checkpoint_dir = validate_checkpoint_dir(
        SPOT_CHECKPOINTS_DIR / name, root=SPOT_CHECKPOINTS_DIR
    )
    if save_dir.exists() or checkpoint_dir.exists():
        raise HTTPException(409, f"Run {name} already exists; run names are immutable")
    source = source_for(recipe)

    job = job_manager.create_job(
        JobType.SPOT_TRAIN,
        {
            "recipe": recipe.id,
            "tasks": list(recipe.tasks),
            "save_dir": str(save_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "init_checkpoint": str(init_checkpoint) if init_checkpoint else "",
            "gpu": req.gpu,
            "epochs": req.num_epochs,
            "feature_arch": req.feature_arch,
            "criterion": req.criterion,
            "validation": req.validation,
            "camera_view": req.camera_view,
        },
        name=f"Fusion Train · {recipe.name} ({name})",
    )

    async def run_job() -> None:
        exporter: PackageExporter | None = None
        try:
            await job_manager.update_job(
                job.id, status="running", message=f"Preparing {recipe.name} labels..."
            )
            loop = asyncio.get_running_loop()

            def frame_progress(done: int, total: int, message: str) -> None:
                progress = 0.02 + (0.16 * done / total if total else 0.0)
                loop.call_soon_threadsafe(
                    lambda progress=progress, message=message: asyncio.ensure_future(
                        job_manager.update_job(job.id, progress=progress, message=message)
                    )
                )

            save_dir.mkdir(parents=True, exist_ok=True)
            prepared = await asyncio.to_thread(
                source.prepare, req, recipe, save_dir=save_dir, progress=frame_progress
            )
            check_task_supervision(recipe, prepared)
            label_summary = dict(prepared.summary)
            if prepared.prediction_stems:
                label_summary["prediction_label_videos"] = sorted(prepared.prediction_stems)
            if req.validation == "manual":
                wanted = {label_stem(entry) for entry in req.validation_videos}
                leaked = prepared.prediction_stems & wanted
                if leaked:
                    raise RuntimeError(
                        "Validation video(s) carry prediction labels only — "
                        "they cannot validate: " + ", ".join(sorted(leaked))
                    )
                split = await asyncio.to_thread(
                    training.materialize_holdout_split,
                    prepared.label_dir,
                    wanted,
                    known_stems=prepared.all_stems,
                )
                label_summary.update(split)
            prepared.summary = label_summary
            await job_manager.update_job(
                job.id,
                progress=0.2,
                message="Training labels ready.",
                params={**job.params, "training_labels": label_summary},
            )

            audio_dir = None
            backend = _audio_backend(req, recipe)
            if backend != "none":
                audio_dir = ACTION_AUDIO_DIR / backend
                audio_dir.mkdir(parents=True, exist_ok=True)
                await job_manager.update_job(
                    job.id, message=f"Precomputing {backend} audio features..."
                )
                rc, last_line = await stream_subprocess(
                    job.id,
                    _audio_precompute_command(
                        backend, label_dir=prepared.label_dir, audio_dir=audio_dir
                    ),
                    cwd=SPOT_DIR,
                )
                if rc != 0:
                    raise RuntimeError(f"Audio precompute failed (rc={rc}): {last_line}")

            cmd = build_command(
                req, recipe, prepared,
                save_dir=save_dir, init_checkpoint=init_checkpoint, audio_dir=audio_dir,
            )
            await job_manager.update_job(
                job.id,
                params={**job.params, "command": cmd, "frame_dir": str(prepared.frame_dir)},
                message="Waiting for GPU...",
            )
            async with stop_vllm_for_job(job.id, when=req.stop_vllm):
                async with job_manager.gpu_lock:
                    await job_manager.update_job(
                        job.id, message=f"Starting SPOT {recipe.name} training..."
                    )
                    ctx = TrainProgress(epochs=req.num_epochs)
                    exporter = PackageExporter(
                        job.id,
                        save_dir,
                        lambda: _export_package(
                            run_dir=save_dir,
                            package_dir=checkpoint_dir,
                            req=req,
                            recipe=recipe,
                            prepared=prepared,
                            cmd=cmd,
                        ),
                    )
                    parsers, is_key_line = make_train_parsers(
                        ctx,
                        params_key=PROGRESS_KEY,
                        criterion=req.criterion,
                        headline_pattern=HEADLINE_PATTERNS[spotting_task(recipe.tasks)],
                        on_new_best=lambda: exporter.schedule(ctx.best_epoch, "new_best"),
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
                        log_path=save_dir / "terminal.log",
                    )
            if rc != 0:
                raise RuntimeError(last_line or f"SPOT training exited with code {rc}")
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
                message=f"{recipe.name} training complete: {checkpoint_dir}",
                params={**job.params, "checkpoint_package": checkpoint_summary},
            )
        except asyncio.CancelledError:
            await _export_after_stop(job.id, exporter, "cancelled")
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"{terminal_prefix(job)}Failed: {type(exc).__name__}: {exc}", flush=True)
            log.exception("SPOT training failed")
            await _export_after_stop(job.id, exporter, "failed")
            await fail_job_from_exc(job.id, exc)

    task = asyncio.create_task(run_job())
    job_manager.attach_task(job, task)
    return job.to_dict()


async def _export_after_stop(
    job_id: str, exporter: PackageExporter | None, reason: str
) -> None:
    """Keep whatever best epoch a stopped run reached."""
    if exporter is None:
        return
    try:
        summary = await exporter.export_once(
            expected_epoch=None, reason=reason, update_job=False
        )
    except Exception:  # noqa: BLE001
        log.exception("Failed to export checkpoint package after %s", reason)
        return
    if summary:
        job_obj = job_manager.get_job(job_id)
        await job_manager.update_job(
            job_id,
            params={**(job_obj.params if job_obj else {}), "checkpoint_package": summary},
        )
