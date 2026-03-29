"""TAD inference router."""

import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from yp_video.config import (
    PROJECT_ROOT,
    CUTS_DIR,
    PREDICTIONS_DIR,
    VIDEOS_DIR,
    TAD_CONFIGS_DIR,
    TAD_CHECKPOINTS_DIR,
)
from yp_video.web.jobs import job_manager
from yp_video.web.r2_client import serve_video_or_r2_redirect, sync_to_r2, sync_directory_to_r2

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
    from yp_video.core.jsonl import read_jsonl

    path = PREDICTIONS_DIR / name
    if not path.exists():
        raise HTTPException(404, "Result not found")

    meta, records = read_jsonl(path)
    meta["results"] = records
    return meta


@router.get("/video/{video_name:path}")
def stream_video(video_name: str):
    """Serve a video file for playback."""
    from urllib.parse import unquote

    decoded = unquote(video_name)
    video_path = CUTS_DIR / decoded
    response = serve_video_or_r2_redirect(video_path, ("cuts",))
    if response:
        return response
    raise HTTPException(404, f"Video not found: {decoded}")


@router.post("/start")
async def start_prediction(req: PredictRequest):
    """Start TAD prediction job."""
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

            config_path = TAD_CONFIGS_DIR / "volleyball_actionformer.yaml"
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

            # Auto-sync to R2
            sync_to_r2(output_path, "tad-predictions")
            if cut_dir and cut_dir.exists():
                sync_directory_to_r2(cut_dir, f"rally_clips/{video_path.stem}", "*.mp4")

            await job_manager.update_job(
                job.id, status="completed", progress=1.0,
                message=f"Inference complete: {output_path.name}",
            )
        except asyncio.CancelledError:
            await job_manager.update_job(job.id, status="cancelled")
        except Exception as e:
            await job_manager.update_job(job.id, status="failed", error=str(e))

    task = asyncio.create_task(run_prediction())
    job_manager.attach_task(job, task)

    return job.to_dict()
