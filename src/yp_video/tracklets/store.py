"""Where tracklets live on disk.

    tracks/<stem>_tracks.jsonl   one record per tracklet:
                                 {rally_id, track_id, frames[], boxes[], scores[]}
                                 — the three arrays share an index
    tracks/<stem>_masks.npz      packed per-frame instance masks, one entry
                                 per tracklet keyed "{rally_id}:{track_id}"

``track_id`` restarts at 1 in every rally (one tracker per rally), so the
identity of a tracklet is the PAIR — the composite key is what every
consumer must carry, never the bare track_id.

A leaf: paths and IO only, nothing here imports a domain package.
"""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from yp_video.config import REID_DIR

TRACKS_DIR = REID_DIR / "tracks"


def track_key(rally_id: int, track_id: int) -> str:
    """The canonical tracklet identity, as used for mask entries and labels."""
    return f"{rally_id}:{track_id}"


def tracks_path(stem: str) -> Path:
    return TRACKS_DIR / f"{stem}_tracks.jsonl"


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
        packed = z[track_key(rally_id, track_id)]
    return np.unpackbits(packed, axis=1)[:, : h * w].reshape(-1, h, w).astype(bool)
