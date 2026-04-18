"""TAD training pipeline router."""

import asyncio
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from yp_video.config import (
    ANNOTATIONS_DIR,
    FEATURES_DIR,
    PRE_ANNOTATIONS_DIR,
    PROJECT_ROOT,
    CUTS_DIR,
    VIDEOS_DIR,
    TAD_PKG_DIR,
    TAD_FEATURES_DIR,
    TAD_ANNOTATIONS_FILE,
    TAD_CHECKPOINTS_DIR,
    TAD_CONFIGS_DIR,
)
from yp_video.web.jobs import job_manager, JobStatus, make_progress_callback

router = APIRouter()


class ExtractFeaturesRequest(BaseModel):
    videos: list[str] | None = None
    batch_size: int = 32
    model: str = "base"


class ConvertAnnotationsRequest(BaseModel):
    train_ratio: float = 0.8
    videos: list[str] | None = None
    model: str = "base"


class TrainRequest(BaseModel):
    gpu: int = 0
    seed: int = 42
    resume: str | None = None
    model: str = "base"



@router.get("/status")
def get_status(model: str = "base"):
    """Get training pipeline status."""
    from yp_video.tad.extract_features import MODEL_CONFIGS

    cfg = MODEL_CONFIGS.get(model, MODEL_CONFIGS["base"])
    feat_dir = FEATURES_DIR / cfg.dir_suffix
    features_count = len(list(feat_dir.glob("*.npy"))) if feat_dir.exists() else 0

    # Per-model feature counts (for at-a-glance UI display)
    features_by_model = {
        name: len(list((FEATURES_DIR / c.dir_suffix).glob("*.npy")))
        if (FEATURES_DIR / c.dir_suffix).exists() else 0
        for name, c in MODEL_CONFIGS.items()
    }

    cuts_count = len(list(CUTS_DIR.glob("*.mp4"))) if CUTS_DIR.exists() else 0
    annotations_exist = TAD_ANNOTATIONS_FILE.exists()

    # List checkpoints (supports both old flat and new nested layout)
    checkpoints = []
    ckpt_base = TAD_CHECKPOINTS_DIR / "actionformer"
    if ckpt_base.exists():
        for f in ckpt_base.rglob("*.pth*"):
            checkpoints.append(str(f))

    # Find active training job
    active_train_job = None
    for j in job_manager.jobs.values():
        if j.type == "train" and j.status == JobStatus.RUNNING:
            active_train_job = j.to_dict()
            break

    return {
        "cuts_count": cuts_count,
        "features_count": features_count,
        "features_by_model": features_by_model,
        "annotations_exist": annotations_exist,
        "checkpoints": checkpoints,
        "gpu_available": True,
        "vllm_running": job_manager.vllm_using_gpu,
        "active_train_job": active_train_job,
    }


@router.get("/models")
def list_models() -> list[dict]:
    """List available V-JEPA 2.1 model sizes."""
    from yp_video.tad.extract_features import MODEL_CONFIGS

    return [
        {
            "name": name,
            "hub_name": cfg.hub_name,
            "feat_dim": cfg.feat_dim,
            "dir_suffix": cfg.dir_suffix,
            "features_count": len(list((FEATURES_DIR / cfg.dir_suffix).glob("*.npy")))
            if (FEATURES_DIR / cfg.dir_suffix).exists() else 0,
        }
        for name, cfg in MODEL_CONFIGS.items()
    ]


@router.post("/extract-features")
async def extract_features(req: ExtractFeaturesRequest):
    """Start feature extraction job."""
    from yp_video.tad.extract_features import MODEL_CONFIGS

    cfg = MODEL_CONFIGS.get(req.model, MODEL_CONFIGS["base"])
    output_dir = FEATURES_DIR / cfg.dir_suffix
    label = f"Feature extraction ({req.model})"

    job = job_manager.create_job("feature_extract", {"videos": req.videos, "model": req.model}, name=label)

    async def run_extraction():
        try:
            await job_manager.update_job(job.id, status="running", message="Waiting for GPU...")
            async with job_manager.gpu_lock:
                await job_manager.update_job(job.id, message=f"Loading V-JEPA 2.1 {req.model}...")

                from yp_video.tad.extract_features import process_directory
                import torch

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                loop = asyncio.get_event_loop()
                progress_cb = make_progress_callback(
                    job.id, loop,
                    message_template="Extracting features ({done}/{total} videos)",
                )
                await loop.run_in_executor(
                    None,
                    lambda: process_directory(
                        CUTS_DIR, output_dir, device,
                        videos=req.videos, batch_size=req.batch_size,
                        model_name=req.model,
                        on_progress=progress_cb,
                    ),
                )

                # V-JEPA model is cached at module level — clear before gpu_lock exits
                from yp_video.tad.extract_features import clear_model_cache
                clear_model_cache()

                count = len(list(output_dir.glob("*.npy")))
                await job_manager.update_job(
                    job.id, status="completed", progress=1.0,
                    message=f"Extracted {req.model} features for {count} videos",
                )
        except asyncio.CancelledError:
            await job_manager.update_job(job.id, status="cancelled")
        except Exception as e:
            await job_manager.update_job(job.id, status="failed", error=str(e))

    task = asyncio.create_task(run_extraction())
    job_manager.attach_task(job, task)

    return job.to_dict()


@router.post("/convert-annotations")
async def convert_annotations(req: ConvertAnnotationsRequest):
    """Convert JSONL annotations to ActionFormer format."""
    from yp_video.tad.convert_annotations import convert_annotations as do_convert
    from yp_video.tad.extract_features import MODEL_CONFIGS

    cfg = MODEL_CONFIGS.get(req.model, MODEL_CONFIGS["base"])
    feat_dir = FEATURES_DIR / cfg.dir_suffix

    # Prefer manual annotations over pre-annotations per video
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: do_convert(
            [ANNOTATIONS_DIR, PRE_ANNOTATIONS_DIR],
            feat_dir, TAD_ANNOTATIONS_FILE,
            req.train_ratio, videos=req.videos,
        ),
    )

    video_count = len(result.get("database", {})) if result else 0
    return {"ok": True, "video_count": video_count, "output": str(TAD_ANNOTATIONS_FILE)}


@router.post("/start")
async def start_training(req: TrainRequest):
    """Start TAD model training."""
    if not TAD_ANNOTATIONS_FILE.exists():
        raise HTTPException(400, "No annotations file found. Run convert-annotations first.")

    job = job_manager.create_job("train", {"gpu": req.gpu, "seed": req.seed, "model": req.model}, name=f"ActionFormer training ({req.model})")

    async def run_training():
        process = None
        try:
            await job_manager.update_job(job.id, status="running", message="Waiting for GPU...")
            async with job_manager.gpu_lock:
                await job_manager.update_job(job.id, message="Starting training...")

                import os
                from datetime import date

                config_path = TAD_CONFIGS_DIR / "volleyball_actionformer.yaml"

                from yp_video.tad.extract_features import MODEL_CONFIGS as _MC
                _mcfg = _MC.get(req.model, _MC["base"])

                today = date.today().strftime("%Y-%m%d")
                work_dir = TAD_CHECKPOINTS_DIR / "actionformer" / _mcfg.dir_suffix / today

                cmd = [
                    sys.executable, "-m", "yp_video.tad.train",
                    "--config", str(config_path.absolute()),
                    "--model", req.model,
                    "--seed", str(req.seed),
                    "--gpu", str(req.gpu),
                    "--work-dir", str(work_dir.absolute()),
                ]

                if req.resume:
                    cmd.extend(["--resume", req.resume])

                env = {**os.environ, "PYTHONUNBUFFERED": "1"}
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    cwd=str(PROJECT_ROOT),
                    env=env,
                )

                # Stream output and parse epoch progress
                import re
                import time as _time
                last_msg = ""
                max_epochs = 0
                current_progress = 0.0
                job_obj = job_manager.get_job(job.id)
                last_push = 0.0
                push_interval = 1.0  # send SSE update at most once per second

                while True:
                    line = await process.stdout.readline()
                    if not line:
                        break
                    text = line.decode().strip()
                    if not text:
                        continue
                    last_msg = text
                    job_obj.logs.append(text)

                    # Parse total epochs: "Start training ActionFormer for 605 epochs ..."
                    m = re.search(r"for (\d+) epochs", text)
                    if m:
                        max_epochs = int(m.group(1))

                    # Parse epoch progress: "[Train]: Epoch 42 finished"
                    m = re.search(r"\[Train\]: Epoch (\d+) finished", text)
                    if m and max_epochs > 0:
                        current_progress = min((int(m.group(1)) + 1) / max_epochs, 0.99)

                    # Throttle SSE updates; always push epoch/mAP lines immediately
                    now = _time.monotonic()
                    is_key_line = "[Train]: Epoch" in text and "finished" in text or "mAP" in text
                    if is_key_line or now - last_push >= push_interval:
                        last_push = now
                        await job_manager.update_job(
                            job.id, message=text, progress=current_progress,
                        )

                returncode = await process.wait()
                if returncode == 0:
                    # Auto-sync key files to R2
                    from yp_video.web.r2_client import sync_to_r2_nested
                    for fname in ("best.pth.tar", "config.txt", "train_log.jsonl"):
                        fpath = work_dir / fname
                        if fpath.exists():
                            sync_to_r2_nested(fpath, "tad-checkpoints", TAD_CHECKPOINTS_DIR)

                    await job_manager.update_job(
                        job.id, status="completed", progress=1.0,
                        message=f"Training complete. Output: {work_dir}",
                    )
                else:
                    await job_manager.update_job(
                        job.id, status="failed",
                        error=f"Training failed (exit code {returncode}): {last_msg}",
                    )
        except asyncio.CancelledError:
            if process and process.returncode is None:
                process.terminate()
            await job_manager.update_job(job.id, status="cancelled")
        except Exception as e:
            await job_manager.update_job(job.id, status="failed", error=str(e))

    task = asyncio.create_task(run_training())
    job_manager.attach_task(job, task)

    return job.to_dict()


@router.get("/checkpoints")
def list_checkpoints() -> list[dict]:
    """List available model checkpoints."""
    ckpt_base = TAD_CHECKPOINTS_DIR / "actionformer"
    checkpoints = []
    if ckpt_base.exists():
        for f in sorted(ckpt_base.rglob("best.pth*"), reverse=True):
            rel = f.relative_to(ckpt_base)
            checkpoints.append({
                "path": str(f),
                "name": str(rel),
                "size_mb": round(f.stat().st_size / 1024 / 1024, 1),
            })
    return checkpoints


@router.get("/performance")
def get_performance(model: str = "base"):
    """Return training performance log from the latest experiment."""
    import json as _json
    from yp_video.tad.extract_features import MODEL_CONFIGS

    ckpt_base = TAD_CHECKPOINTS_DIR / "actionformer"
    if not ckpt_base.exists():
        return {"entries": []}

    mcfg = MODEL_CONFIGS.get(model, MODEL_CONFIGS["base"])

    # Collect all log files — from new nested layout and old flat layout
    log_candidates: list[tuple[str, Path]] = []
    # New layout: actionformer/{model_dir}/{date}/train_log.*
    model_dir = ckpt_base / mcfg.dir_suffix
    if model_dir.exists():
        for d in sorted(model_dir.iterdir(), reverse=True):
            if d.is_dir():
                log_candidates.append((f"{mcfg.dir_suffix}/{d.name}", d))
    # Old flat layout: actionformer/{date}/train_log.*
    for d in sorted(ckpt_base.iterdir(), reverse=True):
        if d.is_dir() and not (d / mcfg.dir_suffix).exists() and d.name not in ("actionformer",):
            # Only include old dirs that match this model by checking config
            config_file = d / "config.txt"
            if config_file.exists():
                content = config_file.read_text()
                if f"/{mcfg.dir_suffix}/" in content or f"'input_dim': {mcfg.feat_dim}" in content:
                    log_candidates.append((d.name, d))

    for name, d in log_candidates:
        log_file = d / "train_log.jsonl"
        if not log_file.exists():
            log_file = d / "train_log.json"
        if log_file.exists():
            with open(log_file) as f:
                if log_file.suffix == ".jsonl":
                    entries = [_json.loads(line) for line in f if line.strip()]
                else:
                    entries = _json.load(f)
            return {"name": name, "entries": entries}

    return {"entries": []}
