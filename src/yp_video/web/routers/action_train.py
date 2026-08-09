"""SPOT action-label training router.

HTTP surface only: the request models and job launcher live in
``yp_video.web.action_training``, the label-corpus rules in
``yp_video.action.training``.
"""

from __future__ import annotations

from fastapi import APIRouter

from yp_video.action import training
from yp_video.config import ACTION_CHECKPOINTS_DIR, SPOT_DIR, SPOT_PYTHON
from yp_video.contracts.action import ACTION_PACKAGE_TYPE
from yp_video.web.action_training import start_training_job
from yp_video.web.jobs import JobSummary, JobType, job_manager
from yp_video.web.r2_client import remote_cut_path
from yp_video.web.spot_runs import (
    SPOT_INIT_PACKAGE_TYPES,
    checkpoint_package_options,
    performance_payload,
)
from yp_video.web.train_requests import ActionTrainRequest

router = APIRouter()


@router.get("/status")
def status() -> dict:
    return {
        "spot_available": SPOT_DIR.exists() and SPOT_PYTHON.exists(),
        "spot_dir": str(SPOT_DIR),
        "spot_python": str(SPOT_PYTHON),
        "init_checkpoints": checkpoint_package_options(
            ACTION_CHECKPOINTS_DIR, package_types=SPOT_INIT_PACKAGE_TYPES
        ),
        "vnl_1_5": training.vnl_stats(),
        "action_annotations": training.annotation_stats(remote_cut_path),
        "action_checkpoints": training.checkpoint_stats(),
        "active_job": active.to_dict() if (active := job_manager.active_job(JobType.ACTION_TRAIN)) else None,
    }


@router.get("/performance")
def performance(run: str | None = None) -> dict:
    """Per-epoch validation metrics for an action-checkpoints run."""
    return performance_payload(
        ACTION_CHECKPOINTS_DIR, run, package_types=(ACTION_PACKAGE_TYPE,)
    )


@router.post("/start", response_model=JobSummary)
async def start(req: ActionTrainRequest) -> dict:
    return await start_training_job(req)
