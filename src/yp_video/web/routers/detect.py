"""VLM rally detection router."""

import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from yp_video.config import CUTS_DIR, SEG_ANNOTATIONS_DIR, PRE_ANNOTATIONS_DIR
from yp_video.web.jobs import job_manager, JobStatus, make_progress_callback
from yp_video.web.r2_client import sync_to_r2, sync_directory_to_r2
from yp_video.web.vllm_manager import vllm_manager
from yp_video.config import load_vllm_env

router = APIRouter()

_default_max_seqs = int(load_vllm_env()["VLLM_MAX_NUM_SEQS"])


class DetectRequest(BaseModel):
    videos: list[str]
    batch_size: int = _default_max_seqs
    clip_duration: float = 6.0
    slide_interval: float = 2.0


class ConvertRequest(BaseModel):
    min_duration: float = 3.0
    min_score: float = 0.5




@router.post("/start")
async def start_detection(req: DetectRequest):
    """Start VLM detection jobs — one job per video."""
    await vllm_manager.sync_status()
    if vllm_manager.status != "running":
        raise HTTPException(400, "vLLM server is not running. Start it first.")

    # Create one job per video
    jobs = []
    for video_name in req.videos:
        job = job_manager.create_job("vlm_detect", {
            "video": video_name,
            "batch_size": req.batch_size,
        }, name=video_name)
        jobs.append(job)

    async def run_all():
        from yp_video.core.vlm_segment import process_video, build_clip_specs
        from yp_video.core.ffmpeg import get_video_duration
        from yp_video.web.vllm_manager import vllm_manager

        loop = asyncio.get_event_loop()

        for job in jobs:
            video_name = job.params["video"]
            video_path = str(CUTS_DIR / video_name)
            output_file = str(SEG_ANNOTATIONS_DIR / f"{Path(video_name).stem}.jsonl")

            try:
                await job_manager.update_job(job.id, status="running", message="Counting clips...")

                duration = await loop.run_in_executor(
                    None, lambda p=video_path: get_video_duration(p),
                )
                total_clips = len(build_clip_specs(duration, req.clip_duration, req.slide_interval))

                await job_manager.update_job(job.id, message=f"Processing {video_name}")

                progress_cb = make_progress_callback(
                    job.id, loop, "Processing clips ({done}/{total})",
                )
                max_concurrent = int(vllm_manager.config["VLLM_MAX_NUM_SEQS"])
                await loop.run_in_executor(
                    None,
                    lambda vp=video_path, of=output_file, cb=progress_cb, mc=max_concurrent, dur=duration: process_video(
                        video_path=vp,
                        server_url=vllm_manager.server_url,
                        model=vllm_manager.model,
                        clip_duration=req.clip_duration,
                        slide_interval=req.slide_interval,
                        output_file=of,
                        batch_size=req.batch_size,
                        max_concurrent=mc,
                        on_progress=cb,
                        total_duration=dur,
                    ),
                )

                # Auto-sync to R2
                sync_to_r2(Path(output_file), "seg-annotations")

                await job_manager.update_job(
                    job.id, status="completed", progress=1.0,
                    message="Detection complete",
                )
            except asyncio.CancelledError:
                await job_manager.update_job(job.id, status="cancelled", message="Cancelled")
                # Skip remaining jobs
                for remaining in jobs[jobs.index(job) + 1:]:
                    if remaining.status == JobStatus.PENDING:
                        await job_manager.update_job(remaining.id, status="cancelled", message="Cancelled")
                return
            except Exception as e:
                await job_manager.update_job(job.id, status="failed", error=str(e))

            # Allow vLLM to release GPU memory between videos
            await asyncio.sleep(2)

    task = asyncio.create_task(run_all())
    job_manager.attach_task(jobs, task)

    return [job.to_dict() for job in jobs]


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

    # Auto-sync to R2
    sync_directory_to_r2(PRE_ANNOTATIONS_DIR, "rally-pre-annotations")

    # Count output files
    count = len(list(PRE_ANNOTATIONS_DIR.glob("*.jsonl"))) if PRE_ANNOTATIONS_DIR.exists() else 0
    return {"ok": True, "count": count}


