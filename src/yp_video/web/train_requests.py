"""The wire contract of every trainer surface, in one module.

These pydantic models are the single source of truth for train-request
parameters: defaults, bounds, enum options and field descriptions. They are
emitted as JSON schemas into ``contracts/`` (``make contract`` →
``yp_video.web.make_train_schemas``) and the frontend builds its config forms
from those schemas — so a bound or default changed here propagates to the UI
without touching TypeScript. Routers import their request model from here and
stay HTTP-thin.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from yp_video.web.schemas import StrictModel

#: Every backbone the yp-spot registry offers, times its temporal-shift
#: variants — mirrors yp_spot/model/backbones.py (BACKBONES × plain/_tsm/_gsm);
#: tests/test_train_request_schemas.py cross-checks against the checkout.
#: Best-first per base, GSM (the recommended shift) leading each group.
FeatureArch = Literal[
    "rny008_tv_gsm", "rny008_tv_tsm", "rny008_tv",
    "rny008_gsm", "rny008_tsm", "rny008",
    "rny002_gsm", "rny002_tsm", "rny002",
    "convnextt_dv3_gsm", "convnextt_dv3_tsm", "convnextt_dv3",
    "convnextt2_gsm", "convnextt2_tsm", "convnextt2",
    "convnextt_gsm", "convnextt_tsm", "convnextt",
    "rn50_gsm", "rn50_tsm", "rn50",
    "rn18_gsm", "rn18_tsm", "rn18",
]

#: Temporal heads yp_spot/model/e2e.py supports — the same five for every
#: SPOT-based trainer (rally, action, fusion).
TemporalArch = Literal["gru", "deeper_gru", "mingru", "mstcn", "asformer"]

CameraView = Literal["all", "broadcast", "sideline"]

#: "map" keeps the epoch with the best validation mAP; "loss" the lowest loss.
Criterion = Literal["map", "loss"]


RecipeId = Literal[
    "rally",
    "rally_winner",
    "action",
    "association_action",
    "action_rally_winner",
]

#: One of the two label families a recipe draws from; which request fields
#: apply follows from it (see ``Recipe.fields`` in the contract).
Validation = Literal["ratio", "manual", "none"]


class FusionTrainRequest(StrictModel):
    """One SPOT training run: a recipe (which heads) plus how to train it.

    Fields outside the chosen recipe's ``fields`` (contract ``RECIPES``) are
    accepted and ignored — the form sends every field with its default — so
    a rally run carries ``audio_backend`` harmlessly and an action run
    ``video_limit``.
    """

    recipe: RecipeId = Field(
        default="association_action",
        description="Which task heads share the one checkpoint.",
    )
    run_name: str | None = Field(
        default=None,
        description="Run name; empty picks {date}_{view}_{recipe}_{model}.",
    )
    init_checkpoint: str | None = Field(
        default=None,
        description=(
            "Empty trains from scratch; a package carrying every head of this "
            "recipe fine-tunes from it (extra heads are skipped on load)."
        ),
    )
    camera_view: CameraView = Field(
        default="all",
        description="Train on all views together or restrict to one camera view.",
    )
    validation: Validation = Field(
        default="ratio",
        description=(
            "ratio holds out a seeded val_ratio of the videos; manual validates "
            "on validation_videos; none validates on the training set (a "
            "final fit, no model selection)."
        ),
    )
    validation_videos: list[str] = Field(
        default_factory=list,
        description="manual: the videos (stems) held out as the validation set.",
    )
    val_ratio: float = Field(
        default=0.2,
        gt=0,
        lt=1,
        description="ratio: fraction of videos held out as the validation split.",
    )
    split_seed: int = Field(
        default=42,
        description="Seed for the train/val video shuffle — keep it fixed to compare runs.",
    )
    dataset_scope: Literal["joint_only", "partial_labels"] = Field(
        default="joint_only",
        description=(
            "association_action: joint_only trains on videos with both action "
            "and association labels; partial_labels also uses action-only "
            "videos (their actor head sees no supervision)."
        ),
    )
    include_predictions: bool = Field(
        default=False,
        description=(
            "Action recipes: also train on videos that only have SPOT "
            "pre-annotations (no human labels). Needs manual validation so "
            "pseudo-labeled videos never validate, and partial_labels scope."
        ),
    )
    # ── Rally recipes ────────────────────────────────────────────
    video_limit: int = Field(
        default=100,
        ge=0,
        description=(
            "Rally recipes: 0 = every annotated video. A positive limit takes "
            "a seeded-shuffle subset (stable across runs) so a quick "
            "experiment doesn't extract hundreds of hours of frames first."
        ),
    )
    sample_fps: float = Field(
        default=30.0,
        ge=0,
        le=120,
        description="Per-video frame stride targeting this sampling rate. 0 = every frame.",
    )
    action_sample_fps: float = Field(
        default=30.0,
        gt=0,
        le=120,
        description="Mixed recipe: action batch sampling rate.",
    )
    rally_sample_fps: float = Field(
        default=5.0,
        gt=0,
        le=120,
        description="Mixed recipe: rally batch sampling rate.",
    )
    winner_sample_fps: float = Field(
        default=5.0,
        gt=0,
        le=120,
        description="Mixed recipe: winner batch sampling rate.",
    )
    # ── Action recipes ───────────────────────────────────────────
    acc_grad_iter: int = Field(
        default=1,
        ge=1,
        le=64,
        description=(
            "Action recipes: split each optimizer step into N micro-batches "
            "(batch_size must divide evenly)."
        ),
    )
    audio_backend: Literal["logmel", "none"] = Field(
        default="logmel",
        description=(
            "Action recipes: logmel adds late-fusion audio (precomputed before "
            "training); none trains a pure-visual model. Must match at "
            "inference. Rally recipes are always visual-only."
        ),
    )
    # ── Common ───────────────────────────────────────────────────
    feature_arch: FeatureArch = Field(
        default="rny008_tv_gsm", description="Visual backbone."
    )
    temporal_arch: TemporalArch = Field(
        default="gru", description="Temporal head over the frame features."
    )
    clip_len: int = Field(
        default=64, ge=8, le=256, description="Frames per training clip."
    )
    batch_size: int = Field(default=8, ge=1, le=64)
    num_epochs: int = Field(default=50, ge=1, le=1000)
    warm_up_epochs: int = Field(
        default=3, ge=0, le=100, description="Linear LR warm-up epochs."
    )
    learning_rate: float = Field(default=0.00003, gt=0)
    num_workers: int = Field(
        default=8,
        ge=0,
        le=32,
        description="Actual training DataLoader worker processes; validation uses at most 4.",
    )
    criterion: Criterion = Field(
        default="map", description="Which epoch counts as best: top validation mAP or lowest loss."
    )
    start_val_epoch: int = Field(
        default=0,
        ge=0,
        description="Skip validation before this epoch to save time early in a long run.",
    )
    epoch_num_frames: int | None = Field(
        default=None,
        ge=1,
        description="Frames sampled per epoch; empty uses the trainer's default budget.",
    )
    gpu: int = Field(default=0, ge=0, le=7, description="CUDA device index.")
    stop_vllm: bool = Field(
        default=False,
        description="Stop the vLLM server first to free its GPU memory for training.",
    )

    @model_validator(mode="after")
    def _consistent(self) -> "FusionTrainRequest":
        if self.validation == "manual" and not self.validation_videos:
            raise ValueError("manual validation needs at least one validation video")
        if self.include_predictions:
            if self.validation != "manual":
                raise ValueError(
                    "include_predictions requires validation='manual' so "
                    "validation stays human-labeled"
                )
            if self.recipe == "association_action" and self.dataset_scope != "partial_labels":
                raise ValueError(
                    "include_predictions requires dataset_scope='partial_labels'; "
                    "prediction-only videos carry no association labels"
                )
        if self.batch_size % self.acc_grad_iter:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be divisible by "
                f"acc_grad_iter ({self.acc_grad_iter})"
            )
        return self


class AssociationTrainRequest(StrictModel):
    """An independent event-level association experiment."""

    train_videos: list[str] = Field(
        min_length=1, description="Videos whose association labels train the model."
    )
    val_videos: list[str] = Field(
        min_length=1, description="Videos held out as the validation set."
    )
    run_name: str | None = Field(
        default=None, description="Run name; empty picks a timestamped one."
    )
    init_checkpoint: str | None = Field(
        default=None,
        description="Empty trains from scratch; an association run's checkpoint fine-tunes from it.",
    )
    gpu: int = Field(default=0, ge=0, le=7, description="CUDA device index.")
    num_epochs: int = Field(default=40, ge=1, le=1000)
    batch_size: int = Field(default=8, ge=1, le=64)
    learning_rate: float = Field(
        default=0.0003, gt=0, description="LR for the association head."
    )
    warm_up_epochs: int = Field(
        default=3, ge=0, le=100, description="Linear LR warm-up epochs."
    )
    #: Same registry as FeatureArch but bare bases only — single crops have
    #: no temporal axis, so the trainer takes no shift variants.
    backbone: Literal[
        "rny002",
        "rny008",
        "rny008_tv",
        "convnextt_dv3",
        "convnextt2",
        "convnextt",
        "rn50",
        "rn18",
    ] = Field(default="rny002", description="Visual backbone over the player crops.")
    backbone_learning_rate: float = Field(
        default=0.00003,
        gt=0,
        description="Separate, smaller LR for the pretrained backbone.",
    )
    crop_dim: int = Field(
        default=224, ge=64, le=512, description="Player-crop side length in pixels."
    )
    num_workers: int = Field(
        default=4, ge=0, le=32, description="DataLoader worker processes."
    )
    stop_vllm: bool = Field(
        default=False,
        description="Stop the vLLM server first to free its GPU memory for training.",
    )

    @model_validator(mode="after")
    def distinct_splits(self):
        train = set(self.train_videos)
        validation = set(self.val_videos)
        overlap = sorted(train & validation)
        if overlap:
            raise ValueError(
                "Train and validation videos must be disjoint: "
                + ", ".join(overlap)
            )
        return self


class ReidExportRequest(StrictModel):
    name: str | None = Field(
        default=None, description="Dataset name; empty picks a timestamped one."
    )
    split_mode: Literal["auto", "session", "crops", "all_train"] = Field(
        default="auto",
        description=(
            "How labeled events split into train/test: auto picks per corpus "
            "size, session holds out whole sessions, crops splits within "
            "sessions, all_train keeps everything for training."
        ),
    )
    test_ratio: float = Field(
        default=0.25,
        gt=0,
        lt=1,
        description="Fraction of identities/crops held out as the test split.",
    )
    seed: int = Field(default=42, description="Seed for the split shuffle.")
    masked: bool = Field(
        default=False,
        description="Reference the background-suppressed crops the masked embedders saw.",
    )
    overwrite: bool = Field(
        default=False, description="Replace an existing dataset of the same name."
    )


class ReidTrainRequest(StrictModel):
    dataset: str = Field(description="Exported dataset under reid/datasets to train on.")
    run_name: str | None = Field(
        default=None, description="Run name; empty picks a timestamped one."
    )
    epochs: int = Field(default=4, ge=1)
    batch_size: int = Field(default=16, ge=2)
    lr: float = Field(default=4e-5, gt=0)
    init_checkpoint: str | None = Field(
        default=None,
        description="Checkpoint package ref to fine-tune from (reid/checkpoints/<run>).",
    )
    overwrite: bool = Field(
        default=False, description="Replace an existing run of the same name."
    )
