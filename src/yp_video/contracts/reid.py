"""Contract for the data exchanged with the yp-reid model package.

yp-video is the *producer* of training data and the *consumer* of model
artifacts: it exports labeled-crop datasets (Contract A) that yp-reid trains
on, reads back the checkpoint packages (Contract B) that training produces,
and spawns yp-reid across a subprocess boundary to embed crops (Contract C).
yp-reid lives in a separate repo + venv, so the two cannot share Python at
runtime.

This module is the single authoritative definition on the yp-video side;
yp-reid mirrors the same constants in ``yp_reid/contract.py``. The two copies
are kept honest by a version handshake: yp-video exports
``REID_CONTRACT_VERSION`` through the ``YP_REID_CONTRACT_VERSION`` env var
when it spawns yp-reid, and the consumer fails loud if its compiled-in
version differs. Bump the version whenever any layout below changes — and
update both sides.

── Contract A: training dataset (yp-video writes, yp-reid reads) ──
A dataset is a directory under ``videos/reid/datasets/<name>/`` holding
exactly two files: ``manifest.json`` (DatasetManifest) and ``samples.jsonl``
(one DatasetSample per line, no header). Crops are referenced by paths
relative to ``manifest.crops_root`` — no symlinks, no copies — so a dataset
directory is two small files and the crop store stays the single source of
image bytes.

── Contract B: checkpoint package (yp-reid writes, yp-video reads) ──
A checkpoint package is a directory under ``videos/reid/checkpoints/<run>/``
holding ``manifest.json`` (CheckpointManifest), the state-dict checkpoint it
names, and training byproducts (metrics.jsonl, config.json, terminal.log).
The manifest fully describes how to rebuild the model (architecture +
preprocessing), so loading never depends on CLI flags matching the weights.

── Contract C: embedding I/O (yp-video spawns yp-reid) ──
``python -m yp_reid.embed --checkpoint <package> --crops-list <txt> --out
<npy>`` reads a UTF-8 text file with one absolute crop path per line and
writes a float32 ``(N, dim)`` L2-normalized matrix whose row i corresponds to
line i. N always equals the line count: unreadable images become NaN rows
(matching the NaN-means-no-crop convention of the embedding store), warned on
stderr, exit code 0.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Bump on ANY breaking change to the dataset layout, checkpoint package
# layout, or embed CLI below.
REID_CONTRACT_VERSION = "1.0.0"

# Env var carrying REID_CONTRACT_VERSION from producer to consumer.
REID_CONTRACT_VERSION_ENV = "YP_REID_CONTRACT_VERSION"

# ── Contract A: training dataset ──────────────────────────────────
DATASET_TYPE = "yp-reid-dataset"
DATASET_MANIFEST_NAME = "manifest.json"
DATASET_SAMPLES_NAME = "samples.jsonl"

SPLIT_TRAIN = "train"
SPLIT_TEST = "test"
ROLE_QUERY = "query"
ROLE_GALLERY = "gallery"

# Positive-pair sampling needs a second crop of the same player.
MIN_TRAIN_CROPS_PER_PLAYER = 2


class DatasetSample(BaseModel):
    """One labeled crop — the unit of a ``samples.jsonl`` line."""

    model_config = {"extra": "forbid"}

    id: str = Field(description="Event id, unique within the dataset")
    path: str = Field(description="Crop path relative to manifest.crops_root, e.g. 'crops-masked/<stem>/<event>.jpg'")
    pid: int = Field(ge=0, description="Player index; key into manifest.players")
    split: Literal["train", "test"]
    role: Literal["query", "gallery"] | None = Field(
        default=None, description="Retrieval role within the test split; null for train samples"
    )
    group: str = Field(description="Session group id the source video belonged to at export time")
    fold: int = Field(ge=0, description="Group ordinal at export time; reserved for k-fold, unused by training")


class DatasetConfig(BaseModel):
    """The export parameters that produced a dataset."""

    model_config = {"extra": "forbid"}

    split_mode: str = Field(description="Resolved mode: session | crops | all_train")
    requested_mode: str = Field(description="Mode as requested, e.g. 'auto'")
    test_ratio: float = Field(gt=0, lt=1)
    seed: int
    masked: bool = Field(description="Whether samples reference background-suppressed crops (crops-masked/)")
    test_groups: list[str] = Field(default_factory=list, description="Groups held out entirely (session mode)")


class PlayerEntry(BaseModel):
    """One identity in the dataset, keyed by stringified pid in the manifest."""

    model_config = {"extra": "forbid"}

    name: str
    group: str
    n_train: int = Field(ge=0)
    n_test: int = Field(ge=0)


class DatasetManifest(BaseModel):
    """The ``manifest.json`` of a dataset directory."""

    model_config = {"extra": "forbid"}

    type: Literal["yp-reid-dataset"] = DATASET_TYPE
    contract_version: str = REID_CONTRACT_VERSION
    created_at: float = Field(description="Unix timestamp of the export")
    name: str
    crops_root: str = Field(description="Absolute path sample paths are relative to (videos/reid); consumer may re-anchor")
    config: DatasetConfig
    players: dict[str, PlayerEntry] = Field(description="Stringified pid → identity")
    groups: dict[str, list[str]] = Field(description="Group id → member video stems, snapshotted (ids drift across relabels)")
    counts: dict[str, int] = Field(description="n_samples / n_players / n_train / n_test / n_query / n_gallery / n_dropped")
    dropped: dict[str, list[str]] = Field(default_factory=dict, description="Drop reason → dropped ids")


# Invariants both sides enforce (violation raises):
#   - every train pid has >= MIN_TRAIN_CROPS_PER_PLAYER samples
#   - every test pid has >= 1 query and >= 1 gallery sample
#   - every sample's pid exists in players; every path resolves to a file
#   - type and contract_version match exactly

# ── Contract B: checkpoint package ────────────────────────────────
CHECKPOINT_TYPE = "yp-reid-checkpoint"
CHECKPOINT_MANIFEST_NAME = "manifest.json"
CHECKPOINT_BEST = "checkpoint_best.pt"


class Preprocessing(BaseModel):
    """Everything between an image file and the model's input tensor."""

    model_config = {"extra": "forbid"}

    resize: Literal["rect_pad"] = Field(description="Aspect-preserving resize, centered zero padding")
    image_size: list[int] = Field(min_length=2, max_length=2, description="[height, width]")
    padding_value: int = 0
    interpolation: str = Field(description="cv2 interpolation name, e.g. 'linear_exact'")
    mean: list[float] = Field(min_length=3, max_length=3)
    std: list[float] = Field(min_length=3, max_length=3)
    color_order: Literal["rgb"] = "rgb"


class ModelSpec(BaseModel):
    """How to rebuild the network a checkpoint's state dict fits.

    The manifest is the sole authority — loading never trusts CLI flags to
    match the weights.
    """

    model_config = {"extra": "forbid"}

    architecture: str = Field(description="'open_clip:<model-name>', e.g. 'open_clip:ViT-L-14-quickgelu'")
    pretrained: str | None = Field(default=None, description="open_clip pretrained tag; null when the state dict covers everything")
    remove_proj: bool = Field(description="Drop visual.proj → raw visual width output")
    embedding_dim: int = Field(gt=0)
    preprocessing: Preprocessing


class CheckpointManifest(BaseModel):
    """The ``manifest.json`` of a checkpoint package."""

    model_config = {"extra": "forbid"}

    type: Literal["yp-reid-checkpoint"] = CHECKPOINT_TYPE
    version: int = 1
    contract_version: str = REID_CONTRACT_VERSION
    created_at: str = Field(description="ISO timestamp")
    run_name: str
    source: Literal["trained", "imported"]
    checkpoint: str = Field(description="State-dict filename within the package, e.g. 'checkpoint_best.pt'")
    model: ModelSpec
    dataset: dict | None = Field(default=None, description="Contract A manifest snapshot (name/path/config/counts); null when imported")
    training: dict | None = Field(default=None, description="Training hyperparameters; null when imported")
    best: dict | None = Field(default=None, description="{'epoch': int, 'metric': str, 'value': float}")
    metrics: dict | None = Field(default=None, description="Test metrics of the packaged epoch (m_ap/rank1/rank5)")
    command: list[str] = Field(default_factory=list, description="argv that produced the package")
    files: list[str] = Field(default_factory=list)
    note: str | None = None


# ── Contract C: progress protocol (yp-reid stdout → yp-video) ─────
# yp-reid emits one line per progress tick:
#   ``REID_PROGRESS {"phase":"embed","done":..,"total":..}``
#   ``REID_PROGRESS {"phase":"train","epoch":..,"epochs":..,"step":..,"steps":..,"loss":..}``
#   ``REID_PROGRESS {"phase":"eval","epoch":..,"m_ap":..,"rank1":..,"rank5":..}``
#   ``REID_PROGRESS {"phase":"best","epoch":..,"value":..}``
# The producer parses these defensively; only the prefix is a hard contract.
REID_PROGRESS_PREFIX = "REID_PROGRESS "
