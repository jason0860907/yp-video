"""Multi-head SPOT model recipes exposed as one train/predict surface.

The registry is deliberately honest about capability.  Association + Action
already exists as the legacy joint actor head in ``yp_spot.train``.  Rally
uses a different dense-segment data contract and sampling rate, so recipes
that include it stay visible but unavailable until yp-spot gains a real
multi-task trainer; the web page must never imply that running two unrelated
trainers produced one fused checkpoint.
"""

from __future__ import annotations

import re
import time
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import Field

from yp_video.action import training
from yp_video.actor import labels as association_labels
from yp_video.actor import spot_associate
from yp_video.config import ACTION_CHECKPOINTS_DIR, SPOT_DIR, SPOT_PYTHON
from yp_video.contracts.action import FUSION_PACKAGE_TYPE
from yp_video.web.action_training import (
    AnnotationActionTrainRequest,
    TrainingFlavor,
    start_training_job,
)
from yp_video.web.jobs import JobSummary, JobType, job_manager
from yp_video.web.schemas import StrictModel
from yp_video.web.spot_runs import (
    SPOT_INIT_PACKAGE_TYPES,
    checkpoint_package_options,
    performance_payload,
)

router = APIRouter()

ASSOCIATION_ACTION = "association_action"
RALLY_ACTION = "rally_action"
ASSOCIATION_ACTION_RALLY = "association_action_rally"

RECIPES = (
    {
        "id": ASSOCIATION_ACTION,
        "name": "Association + Action",
        "tasks": ["association", "action"],
        "available": True,
        "trainable": True,
        "predict_outputs": ["association", "action"],
        "checkpoint_family": spot_associate.LEGACY_ACTOR_FORMAT,
        "description": (
            "One SPOT backbone with action, contact-location and actor heads."
        ),
        "blocked_on": None,
    },
    {
        "id": RALLY_ACTION,
        "name": "Rally + Action",
        "tasks": ["rally", "action"],
        "available": False,
        "trainable": False,
        "predict_outputs": [],
        "checkpoint_family": None,
        "description": (
            "A shared visual backbone with separate rally-segment and action-event "
            "heads."
        ),
        "blocked_on": (
            "yp-spot needs a multi-task data loader and separate rally/action "
            "temporal heads; the current trainers use different sampling rates "
            "and label contracts."
        ),
    },
    {
        "id": ASSOCIATION_ACTION_RALLY,
        "name": "Association + Action + Rally",
        "tasks": ["association", "action", "rally"],
        "available": False,
        "trainable": False,
        "predict_outputs": [],
        "checkpoint_family": None,
        "description": "The future three-head fusion recipe.",
        "blocked_on": (
            "Depends on the Rally + Action multi-task contract before the actor "
            "head can join the same model."
        ),
    },
)

FUSION_TRAINING = TrainingFlavor(
    job_type=JobType.FUSION_MODEL_TRAIN,
    job_name="Fusion Model Train",
    package_type=FUSION_PACKAGE_TYPE,
    progress_key="fusion_model_train_progress",
    subject="association + action",
    # Action Predict loads the headline (action-best) epoch; Association
    # Predict loads the actor head at ITS best epoch. `location` is absent
    # on purpose — it only ever rides along with action.
    serveable_tasks=("action", "actor"),
)


@router.get("/status")
def status() -> dict:
    annotation_stats = training.annotation_stats()
    reviewed = set(association_labels.labeled_stems())
    per_video = [
        {
            **row,
            "has_association_label": row["video"] in reviewed,
        }
        for row in annotation_stats["per_video"]
    ]
    joint_videos = sum(
        1 for row in per_video if row["has_association_label"]
    )
    action_annotations = {
        **annotation_stats,
        "per_video": per_video,
    }
    checkpoints = [
        row
        for row in spot_associate.list_association_checkpoints()
        if row["family"] == spot_associate.LEGACY_ACTOR_FORMAT
    ]
    return {
        "recipes": list(RECIPES),
        "checkpoints": checkpoints,
        "spot_available": SPOT_DIR.exists() and SPOT_PYTHON.exists(),
        # Only packages the SPOT trainer can warm-start from — the shared
        # checkpoints dir also holds independent association packages, whose
        # weights the shape-matching init would silently skip entirely.
        "init_checkpoints": checkpoint_package_options(
            ACTION_CHECKPOINTS_DIR, package_types=SPOT_INIT_PACKAGE_TYPES
        ),
        "action_annotations": action_annotations,
        "supervision": {
            "action_videos": len(per_video),
            "joint_videos": joint_videos,
            "action_only_videos": len(per_video) - joint_videos,
        },
        "active_job": active.to_dict() if (active := job_manager.active_job(JobType.FUSION_MODEL_TRAIN)) else None,
    }


@router.get("/performance")
def performance(run: str | None = None) -> dict:
    return performance_payload(
        ACTION_CHECKPOINTS_DIR, run, package_types=(FUSION_PACKAGE_TYPE,)
    )


class FusionTrainRequest(StrictModel):
    recipe: Literal[
        "association_action",
        "rally_action",
        "association_action_rally",
    ] = ASSOCIATION_ACTION
    run_name: str | None = None
    init_checkpoint: str | None = None
    validation_mode: Literal["manual", "ratio"] = "ratio"
    validation_videos: list[str] = Field(default_factory=list)
    dataset_scope: Literal["joint_only", "partial_labels"] = "joint_only"
    camera_view: Literal["all", "broadcast", "sideline"] = "all"
    audio_backend: Literal["logmel", "none"] = "logmel"
    feature_arch: str = "rny008_gsm"
    temporal_arch: str = "gru"
    clip_len: int = Field(default=64, ge=8, le=256)
    sample_fps: float = Field(default=30.0, ge=0, le=120)
    batch_size: int = Field(default=8, ge=1, le=64)
    num_epochs: int = Field(default=50, ge=1, le=1000)
    warm_up_epochs: int = Field(default=3, ge=0, le=100)
    learning_rate: float = Field(default=0.00003, gt=0)
    num_workers: int = Field(default=4, ge=0, le=32)
    criterion: Literal["map", "loss"] = "map"
    start_val_epoch: int = Field(default=0, ge=0)
    epoch_num_frames: int | None = Field(default=None, ge=1)
    val_ratio: float = Field(default=0.2, gt=0, lt=1)
    split_seed: int = 42
    gpu: int = Field(default=0, ge=0, le=7)
    stop_vllm: bool = False


@router.post("/train", response_model=JobSummary)
async def train(req: FusionTrainRequest) -> dict:
    if req.recipe != ASSOCIATION_ACTION:
        recipe = next(row for row in RECIPES if row["id"] == req.recipe)
        raise HTTPException(409, recipe["blocked_on"])
    if req.validation_mode == "manual" and not req.validation_videos:
        raise HTTPException(
            400,
            "Manual validation mode needs at least one validation video",
        )

    name = (
        req.run_name
        or f"yp_fusion_association_action_{time.strftime('%Y%m%d-%H%M%S')}"
    )
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name.startswith("."):
        raise HTTPException(
            400,
            "Run name may contain only letters, numbers, dot, underscore and dash",
        )

    action_request = AnnotationActionTrainRequest(
        source="action_annotations",
        dataset="yp_actions",
        save_dir=str(SPOT_DIR / "exp" / name),
        checkpoint_dir=str(ACTION_CHECKPOINTS_DIR / name),
        init_checkpoint=req.init_checkpoint,
        camera_view=req.camera_view,
        training_mode=(
            "holdout" if req.validation_mode == "manual" else "split"
        ),
        holdout_videos=req.validation_videos,
        audio_backend=req.audio_backend,
        feature_arch=req.feature_arch,
        temporal_arch=req.temporal_arch,
        clip_len=req.clip_len,
        sample_fps=req.sample_fps,
        batch_size=req.batch_size,
        num_epochs=req.num_epochs,
        warm_up_epochs=req.warm_up_epochs,
        learning_rate=req.learning_rate,
        num_workers=req.num_workers,
        criterion=req.criterion,
        start_val_epoch=req.start_val_epoch,
        epoch_num_frames=req.epoch_num_frames,
        val_ratio=req.val_ratio,
        split_seed=req.split_seed,
        predict_location=True,
        predict_actor=True,
        gpu=req.gpu,
        stop_vllm=req.stop_vllm,
    )
    label_items = None
    require_actor_targets = False
    if req.dataset_scope == "joint_only":
        reviewed = set(association_labels.labeled_stems())
        label_items = [
            item
            for item in training.label_items()
            if item[0].stem.removesuffix("_actions") in reviewed
        ]
        if not label_items:
            raise HTTPException(
                400,
                "Joint-only scope found no videos carrying both Action and "
                "Association labels",
            )
        require_actor_targets = True
    return await start_training_job(
        action_request,
        flavor=FUSION_TRAINING,
        label_items=label_items,
        require_actor_targets=require_actor_targets,
    )
