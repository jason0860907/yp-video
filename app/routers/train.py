"""TAD training pipeline router."""

import asyncio
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.jobs import job_manager, JobStatus

router = APIRouter()

VIDEOS_DIR = Path.home() / "videos"
CUTS_DIR = VIDEOS_DIR / "cuts"
PROJECT_ROOT = Path(__file__).parent.parent.parent
TAD_DIR = PROJECT_ROOT / "tad"
FEATURES_DIR = TAD_DIR / "data" / "features"
ANNOTATIONS_OUTPUT = TAD_DIR / "data" / "annotations" / "volleyball_anno.json"


class ExtractFeaturesRequest(BaseModel):
    videos: list[str] | None = None
    batch_size: int = 64


class ConvertAnnotationsRequest(BaseModel):
    source: str = "rally-annotations"  # or "rally-pre-annotations"
    train_ratio: float = 0.8


class TrainRequest(BaseModel):
    gpu: int = 0
    seed: int = 42
    resume: str | None = None


@router.get("/status")
def get_status():
    """Get training pipeline status."""
    features_count = len(list(FEATURES_DIR.glob("*.npy"))) if FEATURES_DIR.exists() else 0
    cuts_count = len(list(CUTS_DIR.glob("*.mp4"))) if CUTS_DIR.exists() else 0
    annotations_exist = ANNOTATIONS_OUTPUT.exists()

    # List checkpoints
    checkpoints = []
    ckpt_base = TAD_DIR / "checkpoints" / "actionformer"
    if ckpt_base.exists():
        for d in sorted(ckpt_base.iterdir(), reverse=True):
            if d.is_dir():
                for f in d.glob("*.pth"):
                    checkpoints.append(str(f.relative_to(PROJECT_ROOT)))

    return {
        "cuts_count": cuts_count,
        "features_count": features_count,
        "annotations_exist": annotations_exist,
        "checkpoints": checkpoints,
        "gpu_available": not job_manager.vllm_using_gpu,
    }


@router.post("/extract-features")
async def extract_features(req: ExtractFeaturesRequest):
    """Start feature extraction job."""
    if job_manager.vllm_using_gpu:
        raise HTTPException(400, "GPU is in use by vLLM. Stop vLLM first.")

    job = job_manager.create_job("feature_extract", {"videos": req.videos})

    async def run_extraction():
        try:
            await job_manager.update_job(job.id, status="running", message="Loading model...")

            from tad.extract_features import process_directory
            import torch

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: process_directory(
                    CUTS_DIR, FEATURES_DIR, device,
                    videos=req.videos, batch_size=req.batch_size,
                ),
            )

            count = len(list(FEATURES_DIR.glob("*.npy")))
            await job_manager.update_job(
                job.id, status="completed", progress=1.0,
                message=f"Extracted features for {count} videos",
            )
        except asyncio.CancelledError:
            await job_manager.update_job(job.id, status="cancelled")
        except Exception as e:
            await job_manager.update_job(job.id, status="failed", error=str(e))

    async with job_manager.gpu_lock:
        task = asyncio.create_task(run_extraction())
        job._task = task

    return job.to_dict()


@router.post("/convert-annotations")
async def convert_annotations(req: ConvertAnnotationsRequest):
    """Convert JSONL annotations to OpenTAD format."""
    from tad.convert_annotations import convert_annotations as do_convert

    annotations_dir = VIDEOS_DIR / req.source
    if not annotations_dir.exists():
        raise HTTPException(404, f"Annotations directory not found: {req.source}")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: do_convert(annotations_dir, FEATURES_DIR, ANNOTATIONS_OUTPUT, req.train_ratio),
    )

    video_count = len(result.get("database", {})) if result else 0
    return {"ok": True, "video_count": video_count, "output": str(ANNOTATIONS_OUTPUT)}


@router.post("/start")
async def start_training(req: TrainRequest):
    """Start TAD model training."""
    if job_manager.vllm_using_gpu:
        raise HTTPException(400, "GPU is in use by vLLM. Stop vLLM first.")

    if not ANNOTATIONS_OUTPUT.exists():
        raise HTTPException(400, "No annotations file found. Run convert-annotations first.")

    job = job_manager.create_job("train", {"gpu": req.gpu, "seed": req.seed})

    async def run_training():
        try:
            await job_manager.update_job(job.id, status="running", message="Starting training...")

            import os
            from datetime import date

            config_path = TAD_DIR / "configs" / "volleyball_actionformer.py"
            opentad_path = PROJECT_ROOT / "OpenTAD"
            train_script = opentad_path / "tools" / "train.py"

            today = date.today().strftime("%Y-%m%d")
            work_dir = TAD_DIR / "checkpoints" / "actionformer" / today

            cmd = [
                sys.executable, str(train_script),
                str(config_path.absolute()),
                "--seed", str(req.seed),
                "--cfg-options", f"work_dir={work_dir.absolute()}",
            ]

            if req.resume:
                cmd.extend(["--resume", req.resume])

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(req.gpu)
            env["PYTHONPATH"] = str(opentad_path)
            env["LOCAL_RANK"] = "0"
            env["WORLD_SIZE"] = "1"
            env["RANK"] = "0"
            env["MASTER_ADDR"] = "localhost"
            env["MASTER_PORT"] = "29500"

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
                env=env,
            )

            # Stream output for progress updates
            last_msg = ""
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                text = line.decode().strip()
                if text:
                    last_msg = text
                    # Try to parse epoch progress
                    if "Epoch" in text:
                        await job_manager.update_job(job.id, message=text)

            returncode = await process.wait()
            if returncode == 0:
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
    job._task = task

    return job.to_dict()


@router.get("/checkpoints")
def list_checkpoints() -> list[dict]:
    """List available model checkpoints."""
    ckpt_base = TAD_DIR / "checkpoints" / "actionformer"
    checkpoints = []
    if ckpt_base.exists():
        for d in sorted(ckpt_base.iterdir(), reverse=True):
            if d.is_dir():
                for f in sorted(d.glob("*.pth"), reverse=True):
                    checkpoints.append({
                        "path": str(f.relative_to(PROJECT_ROOT)),
                        "name": f"{d.name}/{f.name}",
                        "size_mb": round(f.stat().st_size / 1024 / 1024, 1),
                    })
    return checkpoints
