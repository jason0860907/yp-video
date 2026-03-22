"""Background jobs router."""

import asyncio
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.jobs import job_manager

router = APIRouter()


@router.get("")
def list_jobs():
    """List all jobs."""
    return job_manager.list_jobs()


@router.get("/active-count")
def active_count():
    """Get count of active jobs."""
    return {"count": job_manager.active_count()}


@router.get("/{job_id}")
def get_job(job_id: str):
    """Get job details."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job.to_dict()


@router.get("/{job_id}/events")
async def job_events(job_id: str):
    """SSE stream for job progress."""
    q = job_manager.subscribe(job_id)
    if q is None:
        raise HTTPException(404, "Job not found")

    async def event_generator():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=30.0)
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get("status") in ("completed", "failed", "cancelled"):
                        break
                except asyncio.TimeoutError:
                    yield "data: {}\n\n"
        finally:
            job_manager.unsubscribe(job_id, q)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a running job."""
    success = await job_manager.cancel_job(job_id)
    if not success:
        raise HTTPException(400, "Job cannot be cancelled")
    return {"ok": True}
