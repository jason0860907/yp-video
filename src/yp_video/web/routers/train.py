"""TAD training pipeline router."""

import asyncio
import logging
import sys
import traceback
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

log = logging.getLogger(__name__)

from yp_video.config import (
    ANNOTATIONS_DIR,
    CUTS_DIRS,
    FEATURES_DIR,
    PRE_ANNOTATIONS_DIR,
    PROJECT_ROOT,
    VIDEOS_DIR,
    TAD_PKG_DIR,
    TAD_FEATURES_DIR,
    TAD_ANNOTATIONS_FILE,
    TAD_CHECKPOINTS_DIR,
    TAD_CONFIGS_DIR,
    iter_all_cuts,
)
from yp_video.web.jobs import job_manager, JobStatus, make_progress_callback
from yp_video.web.job_helpers import (
    ProgressParser,
    run_gpu_sync,
    stop_vllm_for_job,
    stream_subprocess,
)

router = APIRouter()


class ExtractFeaturesRequest(BaseModel):
    videos: list[str] | None = None
    batch_size: int = 32
    model: str = "base"
    stop_vllm: bool = False


class ConvertAnnotationsRequest(BaseModel):
    train_ratio: float = 0.8
    videos: list[str] | None = None
    model: str = "base"


class TrainRequest(BaseModel):
    gpu: int = 0
    seed: int = 42
    resume: str | None = None
    model: str = "base"
    # Optional overrides; when None we use the YAML config value
    lr: float | None = None
    epochs: int | None = None
    warmup_epochs: int | None = None
    schedule: str | None = None  # cosine | multistep | constant
    batch_size: int | None = None
    weight_decay: float | None = None
    balanced_sampler: bool = True
    sampler_alpha: float = 1.0



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

    cuts_count = sum(1 for _ in iter_all_cuts())
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


@router.get("/config-defaults")
def get_config_defaults():
    """Return YAML defaults so the UI placeholders/values stay in sync with the config.

    Without this, the Advanced panel hardcodes lr=5e-4, epochs=140, ... and silently
    drifts whenever the YAML changes. Read the source of truth instead.
    """
    import yaml

    cfg_path = TAD_CONFIGS_DIR / "volleyball_actionformer.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}
    opt = cfg.get("opt", {})
    loader = cfg.get("loader", {})
    return {
        "lr": opt.get("learning_rate"),
        "epochs": opt.get("epochs"),
        "warmup_epochs": opt.get("warmup_epochs"),
        "weight_decay": opt.get("weight_decay"),
        "schedule": opt.get("schedule_type"),
        "batch_size": loader.get("batch_size"),
        # Sampler alpha lives in train.py's argparse, not the YAML. Mirror its default
        # here so the UI shows the same value the CLI uses when no override is passed.
        "sampler_alpha": 0.5,
    }


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
            from yp_video.tad.extract_features import clear_model_cache, process_directory
            import torch

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            loop = asyncio.get_event_loop()
            progress_cb = make_progress_callback(
                job.id, loop,
                message_template="Extracting features ({done}/{total} videos)",
            )
            await job_manager.update_job(job.id, message=f"Loading V-JEPA 2.1 {req.model}...")

            await run_gpu_sync(
                job.id,
                lambda stop: process_directory(
                    CUTS_DIRS, output_dir, device,
                    videos=req.videos, batch_size=req.batch_size,
                    model_name=req.model,
                    on_progress=progress_cb,
                    should_stop=stop,
                ),
                stop_vllm=req.stop_vllm,
                on_cleanup=clear_model_cache,
            )

            count = len(list(output_dir.glob("*.npy")))
            await job_manager.update_job(
                job.id, status="completed", progress=1.0,
                message=f"Extracted {req.model} features for {count} videos",
            )
        except asyncio.CancelledError:
            await job_manager.update_job(job.id, status="cancelled")
        except Exception as e:
            tb = traceback.format_exc()
            print(f"\n[extract-features] Failed:\n{tb}", flush=True)
            log.error("Feature extraction failed:\n%s", tb)
            err_type = type(e).__name__
            err_msg = str(e) or "<no message>"
            job_obj = job_manager.get_job(job.id)
            if job_obj:
                job_obj.logs.append(f"{err_type}: {err_msg}")
                for line in tb.splitlines():
                    job_obj.logs.append(line)
            # Hard failure (e.g. CUDA OOM): drop the cached compiled module
            # so a retry doesn't reuse a possibly-bad allocation.
            try:
                from yp_video.tad.extract_features import clear_model_cache
                clear_model_cache()
            except Exception:
                pass
            await job_manager.update_job(
                job.id, status="failed",
                error=f"{err_type}: {err_msg}",
                message=f"Failed: {err_type}: {err_msg}",
            )

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
        try:
            await job_manager.update_job(job.id, status="running", message="Waiting for GPU...")
            async with job_manager.gpu_lock:
                await job_manager.update_job(job.id, message="Starting training...")

                import os
                from datetime import datetime

                config_path = TAD_CONFIGS_DIR / "volleyball_actionformer.yaml"

                from yp_video.tad.extract_features import MODEL_CONFIGS as _MC
                _mcfg = _MC.get(req.model, _MC["base"])

                stamp = datetime.now().strftime("%Y-%m%d-%H%M")
                work_dir = TAD_CHECKPOINTS_DIR / "actionformer" / _mcfg.dir_suffix / stamp

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
                if req.lr is not None:
                    cmd.extend(["--lr", str(req.lr)])
                if req.epochs is not None:
                    cmd.extend(["--epochs", str(req.epochs)])
                if req.warmup_epochs is not None:
                    cmd.extend(["--warmup-epochs", str(req.warmup_epochs)])
                if req.schedule:
                    cmd.extend(["--schedule", req.schedule])
                if req.batch_size is not None:
                    cmd.extend(["--batch-size", str(req.batch_size)])
                if req.weight_decay is not None:
                    cmd.extend(["--weight-decay", str(req.weight_decay)])
                if not req.balanced_sampler:
                    cmd.append("--no-balanced-sampler")
                cmd.extend(["--sampler-alpha", str(req.sampler_alpha)])

                # Closure-captured max_epochs lets the second parser convert
                # the printed epoch number into a 0..1 progress fraction.
                ctx = {"max_epochs": 0}
                def on_total(m):
                    ctx["max_epochs"] = int(m.group(1))
                    return None
                def on_epoch(m):
                    if ctx["max_epochs"] > 0:
                        return {"progress": min((int(m.group(1)) + 1) / ctx["max_epochs"], 0.99)}
                    return None

                rc, last_msg = await stream_subprocess(
                    job.id,
                    cmd,
                    cwd=PROJECT_ROOT,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                    parsers=[
                        ProgressParser(r"for (\d+) epochs", on_total),
                        ProgressParser(r"\[Train\]: Epoch (\d+) finished", on_epoch),
                    ],
                    is_key_line=lambda t: ("[Train]: Epoch" in t and "finished" in t) or "mAP" in t,
                )

                if rc == 0:
                    from yp_video.web.r2_client import sync_to_r2
                    for fname in ("best.pth.tar", "config.txt", "train_log.jsonl"):
                        fpath = work_dir / fname
                        if fpath.exists():
                            sync_to_r2(fpath, "tad-checkpoints", base_dir=TAD_CHECKPOINTS_DIR)
                    await job_manager.update_job(
                        job.id, status="completed", progress=1.0,
                        message=f"Training complete. Output: {work_dir}",
                    )
                else:
                    await job_manager.update_job(
                        job.id, status="failed",
                        error=f"Training failed (exit code {rc}): {last_msg}",
                    )
        except asyncio.CancelledError:
            await job_manager.update_job(job.id, status="cancelled")
        except Exception as e:
            await job_manager.update_job(job.id, status="failed", error=str(e))

    task = asyncio.create_task(run_training())
    job_manager.attach_task(job, task)

    return job.to_dict()


@router.get("/checkpoints")
def list_checkpoints(show_all: bool = False) -> list[dict]:
    """List available model checkpoints.

    Default: best.pth.tar + last (highest-numbered) epoch checkpoint per run dir.
    show_all=true: every *.pth.tar / *.pth file.
    """
    import re
    ckpt_base = TAD_CHECKPOINTS_DIR / "actionformer"
    if not ckpt_base.exists():
        return []

    # Group by parent directory so we can pick best + last per run.
    by_dir: dict[Path, list[Path]] = {}
    for f in ckpt_base.rglob("*.pth*"):
        if f.is_file():
            by_dir.setdefault(f.parent, []).append(f)

    epoch_re = re.compile(r"epoch_(\d+)\.pth")

    checkpoints = []
    for d, files in by_dir.items():
        best = next((f for f in files if f.name.startswith("best.pth")), None)
        epochs = sorted(
            [(int(m.group(1)), f) for f in files if (m := epoch_re.search(f.name))],
            key=lambda x: x[0],
        )
        last = epochs[-1][1] if epochs else None

        kept: list[tuple[Path, str]] = []
        if best is not None:
            kept.append((best, "best"))
        if last is not None and last != best:
            kept.append((last, "last"))
        if show_all:
            for _, f in epochs:
                if f != last:
                    kept.append((f, "epoch"))

        for f, kind in kept:
            rel = f.relative_to(ckpt_base)
            checkpoints.append({
                "path": str(f),
                "name": str(rel),
                "kind": kind,
                "size_mb": round(f.stat().st_size / 1024 / 1024, 1),
            })

    # Within each kind: newest run dir first (sort by name reverse).
    # Across kinds: best → last → epoch. Two passes leverage Python's stable sort.
    kind_order = {"best": 0, "last": 1, "epoch": 2}
    checkpoints.sort(key=lambda c: c["name"], reverse=True)
    checkpoints.sort(key=lambda c: kind_order.get(c["kind"], 3))
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
                    raw = [_json.loads(line) for line in f if line.strip()]
                else:
                    raw = _json.load(f)
            meta = next((e for e in raw if e.get("_meta")), None)
            entries = [e for e in raw if not e.get("_meta")]
            return {"name": name, "entries": entries, "meta": meta}

    return {"entries": []}
