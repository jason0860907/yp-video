"""The dense 2XLarge person pass (person/dense.py): every rally frame's
people, the boxes Association Label picks an actor from.

Started by hand per video, never as a side effect of something else: it is
~4 GPU-minutes a video, and which videos are worth it is the operator's call.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import Field

from yp_video.core.rallies import load_rallies
from yp_video.person.dense import dense_path
from yp_video.web import person_dense_jobs
from yp_video.web.job_helpers import init_batch_items, spawn_batch_video_job
from yp_video.web.jobs import JobSummary, JobType, job_manager
from yp_video.web.r2_client import resolve_cut
from yp_video.web.schemas import StrictModel

router = APIRouter()


class DenseRequest(StrictModel):
    videos: list[str] = Field(min_length=1)


@router.post("/dense", response_model=JobSummary)
async def dense(req: DenseRequest) -> dict:
    if job_manager.active_job(JobType.PERSON_DENSE) is not None:
        raise HTTPException(409, "A dense 2XLarge pass is already running — wait for it to finish")
    paths: list[Path] = []
    for name in req.videos:
        path = resolve_cut(name)
        if path is None:
            raise HTTPException(404, f"Video not found: {name}")
        # The pass covers the rally spans and nothing else.
        if not load_rallies(path.stem):
            raise HTTPException(400, f"No rally spans for: {name}")
        if dense_path(path.stem).exists():
            raise HTTPException(400, f"Already has a dense pass: {name}")
        paths.append(path)

    job = job_manager.create_job(
        JobType.PERSON_DENSE,
        {"videos": [p.name for p in paths], "items": init_batch_items([p.name for p in paths])},
        name=f"Dense 2XLarge ({len(paths)} videos)",
    )
    spawn_batch_video_job(
        job,
        paths,
        work=person_dense_jobs.run,
        done_message=lambda c: f"{c['boxes']} people over {c['frames']} frames",
        start_message="detecting people on every rally frame...",
    )
    return job.to_dict()
