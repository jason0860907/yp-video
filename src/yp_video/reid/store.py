"""Where ReID data lives on disk — paths, IO and record-level policy.

The lowest layer of the reid package: pipeline (orchestration), identity
(matching), tracking (tracklets) and the web router all depend on this
module, never on each other, for where files live and which records count.

Layout under videos/reid/ — annotations/ is the hand-made part, the rest is
recomputable derived data:
    annotations/<stem>_players.json  assignments + actor fixes
    embeddings/<stem>_reid.jsonl     per-event extraction records
    embeddings/<stem>.<model>.npy    float32 (n_records, dim) embedding
                                     matrix, row i ↔ record i, NaN = none
    crops/<stem>/<event>.jpg         actor crops (display box)
    tracks/<stem>_tracks.jsonl       per-rally ByteTrack tracklets

Embeddings are a pure numeric matrix, so they live as npy sidecars, not JSON:
records stay small enough to serve raw, matrices load in milliseconds, and a
one-row update (actor fix) never rewrites the record file. Which weights a
model name stands for is code, not data — see embedder.EMBEDDER_WEIGHTS.
"""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from yp_video.config import (
    ACTION_ANNOTATIONS_DIR,
    ACTION_PRE_ANNOTATIONS_DIR,
    REID_ANNOTATIONS_DIR,
    REID_DIR,
)

EMBEDDINGS_DIR = REID_DIR / "embeddings"
CROPS_DIR = REID_DIR / "crops"
TRACKS_DIR = REID_DIR / "tracks"

# Action labels with nobody to re-identify: "score" marks where the ball
# lands, not a person. Applied at extraction AND at read time, so old
# extractions that predate the rule stay filtered too.
SKIP_LABELS = frozenset({"score"})


def reid_path(stem: str) -> Path:
    return EMBEDDINGS_DIR / f"{stem}_reid.jsonl"


def embedding_path(stem: str, model: str) -> Path:
    return EMBEDDINGS_DIR / f"{stem}.{model}.npy"


def embedded_models(stem: str) -> list[str]:
    """Models that have an embedding matrix for this video."""
    return sorted(p.suffixes[-2].lstrip(".") for p in EMBEDDINGS_DIR.glob(f"{stem}.*.npy"))


def load_embedding_matrix(stem: str, model: str) -> np.ndarray:
    """The (n_records, dim) matrix for one model; NaN rows = no embedding."""
    path = embedding_path(stem, model)
    if not path.exists():
        raise FileNotFoundError(
            f"No {model} embeddings for {stem} — run extraction or backfill embeddings"
        )
    return np.load(path)


def save_embedding_matrix(stem: str, model: str, matrix: np.ndarray) -> None:
    """Atomic replace, mirroring jsonl.atomic_write: readers see old or new."""
    path = embedding_path(stem, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp", delete=False) as f:
        try:
            np.save(f, matrix.astype(np.float32, copy=False))
        except BaseException:
            os.unlink(f.name)
            raise
    os.replace(f.name, path)


def crop_dir(stem: str) -> Path:
    return CROPS_DIR / stem


def players_path(stem: str) -> Path:
    return REID_ANNOTATIONS_DIR / f"{stem}_players.json"


def tracks_path(stem: str) -> Path:
    return TRACKS_DIR / f"{stem}_tracks.jsonl"


def action_annotation_path(stem: str) -> Path | None:
    """Manual action annotations win over pre-annotations."""
    for directory in (ACTION_ANNOTATIONS_DIR, ACTION_PRE_ANNOTATIONS_DIR):
        path = directory / f"{stem}_actions.jsonl"
        if path.exists():
            return path
    return None
