"""TAD inference router."""

import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from yp_video.config import (
    PROJECT_ROOT,
    OPENTAD_DIR,
    CUTS_DIR,
    PREDICTIONS_DIR,
    VIDEOS_DIR,
    TAD_PKG_DIR,
    TAD_CONFIGS_DIR,
    TAD_CHECKPOINTS_DIR,
)
from yp_video.web.jobs import job_manager

router = APIRouter()


class PredictRequest(BaseModel):
    video: str
    checkpoint: str
    threshold: float = 0.3
    device: str = "cuda"
    cut_rallies: bool = False


@router.get("/videos")
def list_videos() -> list[dict]:
    """List videos available for prediction."""
    if not CUTS_DIR.exists():
        return []
    results = []
    for f in sorted(CUTS_DIR.glob("*.mp4")):
        pred_path = PREDICTIONS_DIR / f"{f.stem}_annotations.jsonl"
        results.append({
            "name": f.name,
            "has_prediction": pred_path.exists(),
        })
    return results


@router.get("/results")
def list_results() -> list[str]:
    """List prediction result files."""
    if not PREDICTIONS_DIR.exists():
        return []
    return sorted(f.name for f in PREDICTIONS_DIR.glob("*.jsonl"))


@router.get("/results/{name}")
def get_result(name: str) -> dict:
    """Get prediction result contents."""
    import json
    path = PREDICTIONS_DIR / name
    if not path.exists():
        raise HTTPException(404, "Result not found")

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        return {"results": []}

    meta = json.loads(lines[0])
    meta.pop("_meta", None)

    results = []
    for line in lines[1:]:
        line = line.strip()
        if line:
            results.append(json.loads(line))

    meta["results"] = results
    return meta


@router.post("/start")
async def start_prediction(req: PredictRequest):
    """Start TAD prediction job."""
    if job_manager.vllm_using_gpu:
        raise HTTPException(400, "GPU is in use by vLLM. Stop vLLM first.")

    video_path = CUTS_DIR / req.video
    if not video_path.exists():
        raise HTTPException(404, f"Video not found: {req.video}")

    checkpoint_path = PROJECT_ROOT / req.checkpoint
    if not checkpoint_path.exists():
        raise HTTPException(404, f"Checkpoint not found: {req.checkpoint}")

    job = job_manager.create_job("infer", {
        "video": req.video,
        "checkpoint": req.checkpoint,
    }, name=req.video)

    async def run_prediction():
        try:
            await job_manager.update_job(job.id, status="running", message="Starting inference...")

            from yp_video.tad.infer import run_inference

            config_path = TAD_CONFIGS_DIR / "volleyball_actionformer.py"
            output_path = PREDICTIONS_DIR / f"{video_path.stem}_annotations.jsonl"

            cut_dir = None
            if req.cut_rallies:
                cut_dir = VIDEOS_DIR / "rally_clips" / video_path.stem

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: run_inference(
                    video_path, checkpoint_path, config_path,
                    output_path, req.device, req.threshold, cut_dir,
                ),
            )

            await job_manager.update_job(
                job.id, status="completed", progress=1.0,
                message=f"Inference complete: {output_path.name}",
            )
        except asyncio.CancelledError:
            await job_manager.update_job(job.id, status="cancelled")
        except Exception as e:
            await job_manager.update_job(job.id, status="failed", error=str(e))

    task = asyncio.create_task(run_prediction())
    job._task = task

    return job.to_dict()
