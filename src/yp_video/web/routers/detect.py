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
    batch_size: int = 16
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
    """Start VLM detection jobs — one job per video."""
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
        from yp_video.core.vlm_segment import process_video, count_clips
        from yp_video.web.vllm_manager import vllm_manager

        loop = asyncio.get_event_loop()

        for job in jobs:
            video_name = job.params["video"]
            video_path = str(CUTS_DIR / video_name)
            output_file = str(SEG_ANNOTATIONS_DIR / f"{Path(video_name).stem}.jsonl")

            try:
                await job_manager.update_job(job.id, status="running", message="Counting clips...")

                total_clips = await loop.run_in_executor(
                    None,
                    lambda p=video_path: count_clips(p, req.clip_duration, req.slide_interval),
                )

                def make_callback(jid, total):
                    def on_progress(done, _total):
                        loop.call_soon_threadsafe(
                            lambda: asyncio.ensure_future(job_manager.update_job(
                                jid,
                                progress=done / total if total else 0,
                                message=f"Processing clips ({done}/{total})",
                            ))
                        )
                    return on_progress

                await job_manager.update_job(job.id, message=f"Processing {video_name}")

                max_concurrent = int(vllm_manager.config.get("VLLM_MAX_NUM_SEQS", "16"))
                await loop.run_in_executor(
                    None,
                    lambda vp=video_path, of=output_file, cb=make_callback(job.id, total_clips), mc=max_concurrent: process_video(
                        video_path=vp,
                        server_url=vllm_manager.server_url,
                        model=vllm_manager.model,
                        clip_duration=req.clip_duration,
                        slide_interval=req.slide_interval,
                        output_file=of,
                        batch_size=req.batch_size,
                        max_concurrent=mc,
                        on_progress=cb,
                    ),
                )

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
    # Attach task to all jobs so any can be cancelled
    for job in jobs:
        job._task = task

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

    # Count output files
    count = len(list(PRE_ANNOTATIONS_DIR.glob("*.jsonl"))) if PRE_ANNOTATIONS_DIR.exists() else 0
    return {"ok": True, "count": count}
