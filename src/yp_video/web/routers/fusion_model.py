"""Multi-head SPOT model recipes exposed as one train/predict surface.

The registry is deliberately honest about capability.  Association + Action
already exists as the fusion actor head in ``yp_spot.train``.  Rally
uses a different dense-segment data contract and sampling rate, so recipes
that include it stay visible but unavailable until yp-spot gains a real
multi-task trainer; the web page must never imply that running two unrelated
trainers produced one fused checkpoint.
"""

from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException

from yp_video.action import training
from yp_video.actor import labels as association_labels
from yp_video.actor import spot_associate
from yp_video.config import ACTION_CHECKPOINTS_DIR, SPOT_DIR, SPOT_PYTHON
from yp_video.contracts.action import FUSION_PACKAGE_TYPE
from yp_video.web.action_training import (
    TrainingFlavor,
    start_training_job,
)
from yp_video.web.jobs import JobSummary, JobType, job_manager
from yp_video.web.r2_client import remote_cut_path
from yp_video.web.spot_runs import (
    checkpoint_package_options,
    dedupe_run_name,
    performance_payload,
    spot_run_name,
)
from yp_video.web.train_requests import (
    AnnotationActionTrainRequest,
    FusionTrainRequest,
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
        "checkpoint_family": spot_associate.FUSION_ACTOR_FORMAT,
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
    annotation_stats = training.annotation_stats(remote_cut_path)
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
        if row["family"] == spot_associate.FUSION_ACTOR_FORMAT
    ]
    return {
        "recipes": list(RECIPES),
        "checkpoints": checkpoints,
        "spot_available": SPOT_DIR.exists() and SPOT_PYTHON.exists(),
        # Fusion packages only: action-only packages lack the actor head, so
        # warm-starting from one leaves that head randomly initialized while
        # looking like a fine-tune; independent association packages would
        # load zero tensors via the shape-matching init.
        "init_checkpoints": checkpoint_package_options(
            ACTION_CHECKPOINTS_DIR, package_types=(FUSION_PACKAGE_TYPE,)
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
    if req.batch_size % req.acc_grad_iter:
        raise HTTPException(
            400,
            f"Batch size ({req.batch_size}) must be divisible by grad "
            f"accumulation steps ({req.acc_grad_iter})",
        )

    name = req.run_name or dedupe_run_name(
        spot_run_name(
            view=req.camera_view, task="ass_act", feature_arch=req.feature_arch
        ),
        SPOT_DIR / "exp",
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
        include_predictions=req.include_predictions,
        audio_backend=req.audio_backend,
        feature_arch=req.feature_arch,
        temporal_arch=req.temporal_arch,
        clip_len=req.clip_len,
        sample_fps=req.sample_fps,
        batch_size=req.batch_size,
        acc_grad_iter=req.acc_grad_iter,
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
            for item in training.label_items(remote_cut_path)
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
