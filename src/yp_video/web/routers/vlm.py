"""VLM (Qwen3.5-VL) fine-tune pipeline router.

Mirrors the TAD train router but for the simpler binary "rally / non_rally"
classification on 6-second windows. Two endpoints feed the UI:
- POST /vlm/build-manifest : derive (window, label) JSONL from cuts + annotations
- POST /vlm/start          : launch LoRA fine-tune as a managed job
"""

import asyncio
import json
import logging
import os
import re
import sys
import time as _time
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from yp_video.config import (
    ANNOTATIONS_DIR,
    PROJECT_ROOT,
    VLM_CHECKPOINTS_DIR,
    VLM_MANIFEST_FILE,
    iter_all_cuts,
)
from yp_video.web.jobs import job_manager

log = logging.getLogger(__name__)
router = APIRouter()


# ── Schemas ──────────────────────────────────────────────────────────────


class BuildManifestRequest(BaseModel):
    window: float = 6.0
    stride: float = 2.0
    iou_threshold: float = 0.5
    train_ratio: float = 0.8
    seed: int = 42


class TrainRequest(BaseModel):
    model: str = "Qwen/Qwen3.5-0.8B"
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation: int = 4
    lr: float = 1e-4
    warmup_ratio: float = 0.05
    n_frames: int = 8
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    gpu: int = 0
    seed: int = 42
    eval_samples: int = 256
    balanced_sampler: bool = True


# ── Endpoints ────────────────────────────────────────────────────────────


@router.get("/status")
def status() -> dict:
    """Surface what's required before training can start."""
    cuts_count = sum(1 for _ in iter_all_cuts())
    anno_count = sum(1 for _ in ANNOTATIONS_DIR.glob("*.jsonl")) if ANNOTATIONS_DIR.exists() else 0
    manifest_exists = VLM_MANIFEST_FILE.exists()
    n_windows = 0
    n_rally = 0
    if manifest_exists:
        # Cheap line count + class breakdown.
        for line in open(VLM_MANIFEST_FILE):
            if not line.strip():
                continue
            n_windows += 1
            if '"label": "rally"' in line:
                n_rally += 1

    # Detect an in-flight VLM training job.
    active = None
    for j in job_manager.list_jobs():
        if j["type"] == "vlm-train" and j["status"] in ("running", "pending"):
            active = {"id": j["id"], "progress": j["progress"], "message": j["message"]}
            break

    return {
        "cuts_count": cuts_count,
        "annotations_count": anno_count,
        "manifest_exists": manifest_exists,
        "manifest_path": str(VLM_MANIFEST_FILE),
        "n_windows": n_windows,
        "n_rally": n_rally,
        "n_non_rally": n_windows - n_rally,
        "active_train_job": active,
    }


@router.post("/build-manifest")
def build_manifest(req: BuildManifestRequest) -> dict:
    """Synchronously build the window-level manifest. Fast (a few seconds)."""
    from yp_video.vlm.build_manifest import build_manifest as do_build
    try:
        result = do_build(
            output_path=VLM_MANIFEST_FILE,
            window=req.window,
            stride=req.stride,
            iou_threshold=req.iou_threshold,
            train_ratio=req.train_ratio,
            seed=req.seed,
        )
    except Exception as e:
        log.exception("manifest build failed")
        raise HTTPException(500, f"Build failed: {e}") from e
    return {"ok": True, **result}


@router.get("/checkpoints")
def list_checkpoints(show_all: bool = False) -> list[dict]:
    """List trained LoRA adapters under VLM_CHECKPOINTS_DIR.

    Default: just the final adapter per run (`adapter_final/`).
    show_all=true: also list per-epoch checkpoints if they exist.
    """
    if not VLM_CHECKPOINTS_DIR.exists():
        return []
    out = []
    for run_dir in sorted(VLM_CHECKPOINTS_DIR.glob("*/*"), reverse=True):
        if not run_dir.is_dir():
            continue
        final = run_dir / "adapter_final"
        if final.exists():
            adapter = final / "adapter_model.safetensors"
            size_mb = round(adapter.stat().st_size / 1024 / 1024, 1) if adapter.exists() else 0
            out.append({
                "path": str(final),
                "name": str(final.relative_to(VLM_CHECKPOINTS_DIR)),
                "kind": "final",
                "size_mb": size_mb,
            })
        if show_all:
            for ep in sorted(run_dir.glob("checkpoint-*"), reverse=True):
                adapter = ep / "adapter_model.safetensors"
                if not adapter.exists():
                    continue
                out.append({
                    "path": str(ep),
                    "name": str(ep.relative_to(VLM_CHECKPOINTS_DIR)),
                    "kind": "epoch",
                    "size_mb": round(adapter.stat().st_size / 1024 / 1024, 1),
                })
    return out


@router.get("/performance")
def get_performance() -> dict:
    """Latest run's eval log."""
    if not VLM_CHECKPOINTS_DIR.exists():
        return {"entries": []}
    for run_dir in sorted(VLM_CHECKPOINTS_DIR.glob("*/*"), reverse=True):
        log_file = run_dir / "train_log.jsonl"
        if not log_file.exists():
            continue
        raw = [json.loads(l) for l in open(log_file) if l.strip()]
        meta = next((e for e in raw if e.get("_meta")), None)
        entries = [e for e in raw if not e.get("_meta")]
        return {
            "name": str(run_dir.relative_to(VLM_CHECKPOINTS_DIR)),
            "entries": entries,
            "meta": meta,
        }
    return {"entries": []}


@router.post("/start")
async def start_training(req: TrainRequest) -> dict:
    """Launch LoRA fine-tune as a tracked job."""
    if not VLM_MANIFEST_FILE.exists():
        raise HTTPException(400, "Manifest missing — POST /vlm/build-manifest first.")

    job = job_manager.create_job(
        "vlm-train",
        {"model": req.model, "epochs": req.epochs, "lr": req.lr,
         "batch_size": req.batch_size, "n_frames": req.n_frames},
        name=f"VLM fine-tune ({req.model.split('/')[-1]})",
    )

    async def run():
        process = None
        try:
            await job_manager.update_job(job.id, status="running",
                                         message="Waiting for GPU lock...")
            async with job_manager.gpu_lock:
                stamp = datetime.now().strftime("%Y-%m%d-%H%M")
                slug = req.model.split("/")[-1].lower()
                work_dir = VLM_CHECKPOINTS_DIR / slug / stamp
                work_dir.mkdir(parents=True, exist_ok=True)

                cmd = [
                    sys.executable, "-m", "yp_video.vlm.train",
                    "--model", req.model,
                    "--manifest", str(VLM_MANIFEST_FILE),
                    "--work-dir", str(work_dir),
                    "--epochs", str(req.epochs),
                    "--batch-size", str(req.batch_size),
                    "--gradient-accumulation", str(req.gradient_accumulation),
                    "--lr", str(req.lr),
                    "--warmup-ratio", str(req.warmup_ratio),
                    "--n-frames", str(req.n_frames),
                    "--lora-r", str(req.lora_r),
                    "--lora-alpha", str(req.lora_alpha),
                    "--lora-dropout", str(req.lora_dropout),
                    "--gpu", str(req.gpu),
                    "--seed", str(req.seed),
                    "--eval-samples", str(req.eval_samples),
                ]
                if not req.balanced_sampler:
                    cmd.append("--no-balanced-sampler")

                env = {**os.environ, "PYTHONUNBUFFERED": "1"}
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    cwd=str(PROJECT_ROOT),
                    env=env,
                )

                last_msg = ""
                progress = 0.0
                last_push = 0.0
                job_obj = job_manager.get_job(job.id)

                while True:
                    line = await process.stdout.readline()
                    if not line:
                        break
                    text = line.decode().rstrip()
                    if not text:
                        continue
                    last_msg = text
                    job_obj.logs.append(text)

                    # HF Trainer prints "{'loss': X, 'epoch': Y}" each logging_step
                    m = re.search(r"'epoch':\s*([\d.]+)", text)
                    if m:
                        ep = float(m.group(1))
                        progress = min(ep / max(req.epochs, 1), 0.99)

                    now = _time.monotonic()
                    is_key = ("[VLM Eval]" in text) or ("loss" in text.lower())
                    if is_key or now - last_push >= 1.0:
                        last_push = now
                        await job_manager.update_job(
                            job.id, message=text, progress=progress,
                        )

                rc = await process.wait()
                if rc == 0:
                    await job_manager.update_job(
                        job.id, status="completed", progress=1.0,
                        message=f"Done. Adapter: {work_dir / 'adapter_final'}",
                    )
                else:
                    await job_manager.update_job(
                        job.id, status="failed",
                        error=f"VLM training failed (exit {rc}): {last_msg}",
                    )
        except asyncio.CancelledError:
            if process and process.returncode is None:
                process.terminate()
            await job_manager.update_job(job.id, status="cancelled")
        except Exception as e:
            log.exception("VLM training crashed")
            await job_manager.update_job(job.id, status="failed", error=str(e))

    task = asyncio.create_task(run())
    job_manager.attach_task(job, task)
    return {"id": job.id, "status": "pending"}
