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
    tracks/<stem>_masks.npz          packed per-frame instance masks, one
                                     entry per tracklet ("rally:track")

Embeddings are a pure numeric matrix, so they live as npy sidecars, not JSON:
records stay small enough to serve raw, matrices load in milliseconds, and a
one-row update (actor fix) never rewrites the record file. Which weights a
model name stands for is answered by embedder.weights_id() at runtime.
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
from yp_video.core.cache import StatCache

EMBEDDINGS_DIR = REID_DIR / "embeddings"
CROPS_DIR = REID_DIR / "crops"
# What the masked embedders actually saw (background-suppressed variants of
# crops/, same filenames) — persisted so the UI can show them. Regenerated on
# every masked embed run; recomputable like everything outside annotations/.
MASKED_CROPS_DIR = REID_DIR / "crops-masked"
TRACKS_DIR = REID_DIR / "tracks"

# Action labels with nobody to re-identify: "score" marks where the ball
# lands, not a person. Applied at extraction AND at read time, so old
# extractions that predate the rule stay filtered too.
SKIP_LABELS = frozenset({"score"})


def reid_path(stem: str) -> Path:
    return EMBEDDINGS_DIR / f"{stem}_reid.jsonl"


def embedding_path(stem: str, model: str) -> Path:
    return EMBEDDINGS_DIR / f"{stem}.{model}.npy"


# One dir scan serves every embedded_models call (the video list asks per
# cut); any matrix create/delete churns the directory entry via temp+rename,
# so the dir's own stat is a correct invalidation key.
_models_cache: StatCache = StatCache()


def embedded_models(stem: str) -> list[str]:
    """Models that have an embedding matrix for this video."""
    if not EMBEDDINGS_DIR.exists():
        return []
    return _models_map().get(stem, [])


def _models_map() -> dict[str, list[str]]:
    def scan() -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for p in EMBEDDINGS_DIR.glob("*.npy"):
            stem, _, model = p.name[: -len(".npy")].rpartition(".")
            if stem:
                out.setdefault(stem, []).append(model)
        return {stem: sorted(models) for stem, models in out.items()}

    return _models_cache.get("map", [EMBEDDINGS_DIR], scan)


def require_embedding_path(stem: str, model: str) -> Path:
    """The matrix path, or an actionable FileNotFoundError when it's absent."""
    path = embedding_path(stem, model)
    if not path.exists():
        raise FileNotFoundError(
            f"No {model} embeddings for {stem} — run extraction or backfill embeddings"
        )
    return path


def load_embedding_matrix(stem: str, model: str) -> np.ndarray:
    """The (n_records, dim) matrix for one model; NaN rows = no embedding."""
    return np.load(require_embedding_path(stem, model))


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


def masked_crop_dir(stem: str) -> Path:
    return MASKED_CROPS_DIR / stem


def players_path(stem: str) -> Path:
    return REID_ANNOTATIONS_DIR / f"{stem}_players.json"


def tracks_masks_path(stem: str) -> Path:
    return TRACKS_DIR / f"{stem}_masks.npz"


def save_track_masks(stem: str, mask_hw: tuple[int, int], masks: dict[str, np.ndarray]) -> None:
    """Per-tracklet packed instance masks, atomically replaced.

    ``masks`` maps ``"{rally_id}:{track_id}"`` to a (n_frames, H*W/8) uint8
    packbits array, rows aligned with the tracklet's frames in the tracks
    jsonl; ``mask_hw`` rides along as ``_shape`` so readers can unpack.
    """
    path = tracks_masks_path(stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp", delete=False) as f:
        try:
            np.savez_compressed(f, _shape=np.array(mask_hw), **masks)
        except BaseException:
            os.unlink(f.name)
            raise
    os.replace(f.name, path)


def load_track_masks(stem: str, rally_id: int, track_id: int) -> np.ndarray:
    """One tracklet's masks as (n_frames, H, W) bool, aligned with its frames."""
    path = tracks_masks_path(stem)
    if not path.exists():
        raise FileNotFoundError(f"No track masks for {stem} — re-run tracking")
    with np.load(path) as z:
        h, w = (int(v) for v in z["_shape"])
        packed = z[f"{rally_id}:{track_id}"]
    return np.unpackbits(packed, axis=1)[:, : h * w].reshape(-1, h, w).astype(bool)


def tracks_path(stem: str) -> Path:
    return TRACKS_DIR / f"{stem}_tracks.jsonl"


def action_annotation_path(stem: str) -> Path | None:
    """Manual action annotations win over pre-annotations."""
    for directory in (ACTION_ANNOTATIONS_DIR, ACTION_PRE_ANNOTATIONS_DIR):
        path = directory / f"{stem}_actions.jsonl"
        if path.exists():
            return path
    return None
