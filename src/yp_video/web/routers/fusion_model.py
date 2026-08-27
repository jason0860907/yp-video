"""Fusion Train: every SPOT recipe behind one train surface.

A recipe is a task set from the contract registry (``RECIPES``); this router
serves the registry, the corpus stats each recipe draws from, the packages a
recipe can warm-start from, and launches ``spot_training.start_training_job``.
"""

from __future__ import annotations

from fastapi import APIRouter

from yp_video.action import rally as rally_spot
from yp_video.action import training
from yp_video.actor import labels as association_labels
from yp_video.config import SPOT_CHECKPOINTS_DIR, SPOT_DIR, SPOT_PYTHON, cut_kind_of
from yp_video.contracts.action import RECIPES, TASKS
from yp_video.web.jobs import JobSummary, JobType, job_manager
from yp_video.web.r2_client import resolve_cut
from yp_video.web.spot_runs import checkpoint_package_options, performance_payload
from yp_video.web.spot_training import start_training_job
from yp_video.web.train_requests import FusionTrainRequest

router = APIRouter()


@router.get("/status")
def status() -> dict:
    annotation_stats = training.annotation_stats(resolve_cut)
    reviewed = set(association_labels.labeled_stems())
    per_video = [
        {**row, "has_association_label": row["video"] in reviewed}
        for row in annotation_stats["per_video"]
    ]
    joint_videos = sum(1 for row in per_video if row["has_association_label"])
    rally_items, rally_missing = rally_spot.select_training_items(resolve_cut, 0)
    return {
        "recipes": [
            {
                "id": recipe.id,
                "name": recipe.name,
                "tasks": list(recipe.tasks),
                "description": recipe.description,
                "fields": list(recipe.fields),
                "defaults": dict(recipe.defaults),
                # What a package from this recipe can be loaded for on its own.
                "serveable_tasks": [t for t in recipe.tasks if TASKS[t].serveable],
            }
            for recipe in RECIPES.values()
        ],
        "task_labels": {name: spec.label for name, spec in TASKS.items()},
        "spot_available": SPOT_DIR.exists() and SPOT_PYTHON.exists(),
        # Per recipe: packages carrying every head it trains (a superset is
        # fine — unused heads are skipped on load; a missing one would leave
        # that head randomly initialized while looking like a fine-tune).
        "init_checkpoints": {
            recipe.id: checkpoint_package_options(SPOT_CHECKPOINTS_DIR, tasks=recipe.tasks)
            for recipe in RECIPES.values()
        },
        "action_annotations": {**annotation_stats, "per_video": per_video},
        "rally_annotations": {
            **rally_spot.rally_stats(),
            "with_video": len(rally_items),
            "missing_videos": len(rally_missing),
            "frame_caches": rally_spot.frame_cache_stats(),
            "per_video": [
                {"video": video.stem, "view": cut_kind_of(video)} for _ann, video in rally_items
            ],
        },
        "supervision": {
            "action_videos": len(per_video),
            "joint_videos": joint_videos,
            "action_only_videos": len(per_video) - joint_videos,
        },
        "active_job": active.to_dict() if (active := job_manager.active_job(JobType.SPOT_TRAIN)) else None,
    }


@router.get("/performance")
def performance(run: str | None = None) -> dict:
    return performance_payload(SPOT_CHECKPOINTS_DIR, run)


@router.post("/train", response_model=JobSummary)
async def train(req: FusionTrainRequest) -> dict:
    return await start_training_job(req)
