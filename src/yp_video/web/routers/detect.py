"""VLM rally detection router."""

import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from yp_video.config import CUTS_DIR, SEG_ANNOTATIONS_DIR, PRE_ANNOTATIONS_DIR
from yp_video.web.jobs import job_manager, JobStatus
from yp_video.web.vllm_manager import vllm_manager

router = APIRouter()


class DetectRequest(BaseModel):
    videos: list[str]
    batch_size: int = 32
    clip_duration: float = 6.0
    slide_interval: float = 3.0


class ConvertRequest(BaseModel):
    min_duration: float = 3.0
    min_score: float = 0.5


@router.get("/videos")
def list_videos() -> list[dict]:
    """List cut videos available for detection."""
    if not CUTS_DIR.exists():
        return []
    results = []
    for f in sorted(CUTS_DIR.glob("*.mp4")):
        # Check if detection already exists
        seg_path = SEG_ANNOTATIONS_DIR / f"{f.stem}.jsonl"
        results.append({
            "name": f.name,
            "has_detection": seg_path.exists(),
        })
    return results


@router.post("/start")
async def start_detection(req: DetectRequest):
    """Start VLM detection job."""
    if vllm_manager.status != "running":
        raise HTTPException(400, "vLLM server is not running. Start it first.")

    job = job_manager.create_job("vlm_detect", {
        "videos": req.videos,
        "batch_size": req.batch_size,
    })

    async def run_detection():
        try:
            await job_manager.update_job(job.id, status="running", message="Starting detection...")

            from yp_video.core.vlm_segment import process_video
            from yp_video.web.vllm_manager import vllm_manager

            total = len(req.videos)
            for i, video_name in enumerate(req.videos):
                video_path = str(CUTS_DIR / video_name)
                output_file = str(SEG_ANNOTATIONS_DIR / f"{Path(video_name).stem}.jsonl")

                await job_manager.update_job(
                    job.id,
                    progress=(i / total),
                    message=f"Processing {video_name} ({i+1}/{total})",
                )

                # Run in executor to not block event loop
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: process_video(
                        video_path=video_path,
                        server_url=vllm_manager.server_url,
                        model=vllm_manager.model,
                        clip_duration=req.clip_duration,
                        slide_interval=req.slide_interval,
                        output_file=output_file,
                        batch_size=req.batch_size,
                    ),
                )

            await job_manager.update_job(
                job.id, status="completed", progress=1.0,
                message=f"Completed detection for {total} videos",
            )
        except asyncio.CancelledError:
            await job_manager.update_job(job.id, status="cancelled", message="Cancelled")
        except Exception as e:
            await job_manager.update_job(job.id, status="failed", error=str(e))

    task = asyncio.create_task(run_detection())
    job._task = task

    return job.to_dict()


@router.post("/convert")
async def convert_to_rally(req: ConvertRequest):
    """Convert VLM detections to rally annotations."""
    from yp_video.tad.vlm_to_rally import convert_directory

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: convert_directory(
            SEG_ANNOTATIONS_DIR, PRE_ANNOTATIONS_DIR,
            req.min_duration, req.min_score,
        ),
    )

    # Count output files
    count = len(list(PRE_ANNOTATIONS_DIR.glob("*.jsonl"))) if PRE_ANNOTATIONS_DIR.exists() else 0
    return {"ok": True, "count": count}
