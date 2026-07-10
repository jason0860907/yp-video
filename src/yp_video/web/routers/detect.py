"""VLM rally detection router."""

import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from yp_video.config import (
    SEG_ANNOTATIONS_DIR,
    PRE_ANNOTATIONS_DIR,
    find_cut,
)
from yp_video.web.jobs import job_manager
from yp_video.web.job_helpers import (
    batch_message,
    batch_progress,
    finalize_batch_job,
    init_batch_items,
    update_batch_item,
)
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
    # Auto-convert-to-rally params applied per video after detection completes
    min_duration: float = 3.0
    min_score: float = 0.5


class ConvertRequest(BaseModel):
    min_duration: float = 3.0
    min_score: float = 0.5




@router.post("/start")
async def start_detection(req: DetectRequest):
    """Start a single VLM detection job that processes all selected videos."""
    await vllm_manager.sync_status()
    if vllm_manager.status != "running":
        raise HTTPException(400, "vLLM server is not running. Start it first.")

    total = len(req.videos)
    job = job_manager.create_job("vlm_detect", {
        "videos": req.videos,
        "batch_size": req.batch_size,
        "items": init_batch_items(req.videos),
    }, name=f"Rally Predict ({total} videos)")

    async def run_all():
        from yp_video.core.vlm_segment import process_video, build_clip_specs
        from yp_video.core.ffmpeg import get_video_duration
        from yp_video.web.vllm_manager import vllm_manager

        await job_manager.update_job(
            job.id, status="running", message="Waiting for GPU...",
        )

        async with job_manager.gpu_lock:
            loop = asyncio.get_event_loop()
            items = job.params["items"]
            failed = 0

            for i, video_name in enumerate(req.videos):
                resolved = find_cut(video_name)
                if resolved is None:
                    raise HTTPException(404, f"Cut video not found: {video_name}")
                video_path = str(resolved)
                output_file = str(SEG_ANNOTATIONS_DIR / f"{Path(video_name).stem}.jsonl")

                try:
                    await update_batch_item(
                        job.id, items, i, status="running", message="counting clips...",
                        overall_progress=batch_progress(i, 0.0, total),
                        overall_message=batch_message(i, total, video_name, "counting clips..."),
                    )

                    duration = await loop.run_in_executor(
                        None, lambda p=video_path: get_video_duration(p),
                    )

                    def progress_cb(done, total_clips, msg=None, *, index=i, name=video_name):
                        frac = done / total_clips if total_clips else 0.0
                        detail = msg if msg is not None else f"processing clips {int(done)}/{int(total_clips)}"
                        # Runs on an executor thread — schedule the item update
                        # onto the loop instead of awaiting it here.
                        loop.call_soon_threadsafe(
                            asyncio.ensure_future,
                            update_batch_item(
                                job.id, items, index, progress=frac, message=detail,
                                overall_progress=batch_progress(index, frac, total),
                                overall_message=batch_message(index, total, name, detail),
                            ),
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

                    sync_to_r2(Path(output_file), "seg-annotations")

                    # Auto convert-to-rally for this one video
                    from yp_video.core.vlm_to_rally import convert_vlm_to_rally
                    await update_batch_item(
                        job.id, items, i, message="converting to rally...",
                        overall_message=batch_message(i, total, video_name, "converting to rally..."),
                    )
                    rally_path = PRE_ANNOTATIONS_DIR / f"{Path(video_name).stem}_annotations.jsonl"
                    n_rallies = await loop.run_in_executor(
                        None,
                        lambda inp=Path(output_file), out=rally_path,
                               md=req.min_duration, ms=req.min_score: convert_vlm_to_rally(
                            inp, out, md, ms,
                        ),
                    )
                    sync_to_r2(rally_path, "rally-pre-annotations")
                    await update_batch_item(
                        job.id, items, i, status="completed", progress=1.0,
                        message=f"{n_rallies} rallies",
                        overall_progress=batch_progress(i, 1.0, total),
                        overall_message=batch_message(i, total, video_name, f"{n_rallies} rallies"),
                    )
                except asyncio.CancelledError:
                    await update_batch_item(job.id, items, i, status="cancelled", message="Cancelled")
                    await job_manager.update_job(job.id, status="cancelled", message="Cancelled")
                    return
                except Exception as e:
                    failed += 1
                    await update_batch_item(
                        job.id, items, i, status="failed", message=f"failed — {e}", error=str(e),
                        overall_message=batch_message(i, total, video_name, f"failed — {e}"),
                    )

                await asyncio.sleep(2)

            await finalize_batch_job(job.id, total, failed)

    task = asyncio.create_task(run_all())
    job_manager.attach_task([job], task)

    return job.to_dict()


@router.post("/convert")
async def convert_to_rally(req: ConvertRequest):
    """Convert VLM detections to rally annotations."""
    from yp_video.core.vlm_to_rally import convert_directory

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

