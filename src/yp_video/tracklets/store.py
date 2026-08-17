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

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from yp_video.config import TRACKS_DIR
from yp_video.core.cache import StatCache
from yp_video.core.jsonl import atomic_binary, read_jsonl_cached
from yp_video.tracklets.geometry import TrackletIndex


def track_key(rally_id: int, track_id: int) -> str:
    """The canonical tracklet identity, as used for mask entries and labels."""
    return f"{rally_id}:{track_id}"


def tracks_path(stem: str) -> Path:
    return TRACKS_DIR / f"{stem}_tracks.jsonl"


def tracks_masks_path(stem: str) -> Path:
    return TRACKS_DIR / f"{stem}_masks.npz"


# One index per video, rebuilt when the jsonl changes. Shared — read-only,
# like everything StatCache hands out.
_index_cache: StatCache = StatCache()


def tracklet_index(stem: str) -> TrackletIndex:
    """This video's tracklets, indexed by frame and identity.

    The single accessor every consumer reads tracklets through, so building
    the index is paid once per video instead of once per event — and so the
    four modules that each used to scan the raw list ask one object instead.
    """
    path = tracks_path(stem)
    return _index_cache.get(
        stem, [path], lambda: TrackletIndex(read_jsonl_cached(path)[1])
    )


def span_detections_path(stem: str) -> Path:
    return TRACKS_DIR / f"{stem}_span_detections.npz"


def save_span_detections(stem: str, detector: str, detections: dict[int, np.ndarray]) -> None:
    """Raw per-event-frame detections the dense pass saw, atomically replaced.

    ``detections`` maps a native frame index to an (n, 5) float32 array of
    ``x0, y0, x1, y1, score`` rows in frame pixels, captured BEFORE the
    tracker touched them — the full candidate set, flicker included. The
    detector name rides along so a reader can refuse a cache produced by a
    different model.
    """
    path = span_detections_path(stem)
    with atomic_binary(path) as f:
        np.savez_compressed(
            f,
            _detector=np.array(detector),
            **{str(frame): array for frame, array in detections.items()},
        )


def load_span_detections(stem: str, detector: str) -> dict[int, np.ndarray]:
    """The saved span detections, or {} when absent or from another detector."""
    path = span_detections_path(stem)
    if not path.exists():
        return {}
    with np.load(path) as data:
        if "_detector" not in data.files or str(data["_detector"]) != detector:
            return {}
        return {
            int(key): data[key] for key in data.files if not key.startswith("_")
        }


def save_track_masks(stem: str, mask_hw: tuple[int, int], masks: dict[str, np.ndarray]) -> None:
    """Per-tracklet packed instance masks, atomically replaced.

    ``masks`` maps ``"{rally_id}:{track_id}"`` to a (n_frames, H*W/8) uint8
    packbits array, rows aligned with the tracklet's frames in the tracks
    jsonl; ``mask_hw`` rides along as ``_shape`` so readers can unpack.
    """
    path = tracks_masks_path(stem)
    with atomic_binary(path) as f:
        np.savez_compressed(f, _shape=np.array(mask_hw), **masks)


def load_track_masks(stem: str, rally_id: int, track_id: int) -> np.ndarray:
    """One tracklet's masks as (n_frames, H, W) bool, aligned with its frames."""
    path = tracks_masks_path(stem)
    if not path.exists():
        raise FileNotFoundError(f"No track masks for {stem} — re-run tracking")
    with np.load(path) as z:
        h, w = (int(v) for v in z["_shape"])
        packed = z[track_key(rally_id, track_id)]
    return _unpack(packed, h, w)


def _unpack(packed: np.ndarray, h: int, w: int) -> np.ndarray:
    return np.unpackbits(packed, axis=1)[:, : h * w].reshape(-1, h, w).astype(bool)


class TrackMasks(Mapping):
    """Every tracklet's silhouettes for one video, unpacked on first use.

    A whole video's masks decompress to ~100 MB of bool, and a consumer that
    scores events touches only the tracklets alive near one — so this stays a
    lazy view over the open archive rather than a dict comprehension. Missing
    keys read as ``None`` (tracked before masks existed, or a tracklet the
    segmenter never produced) so callers branch on data, not on exceptions.
    """

    def __init__(self, path: Path):
        self._archive = np.load(path)
        self._h, self._w = (int(v) for v in self._archive["_shape"])
        self._keys = tuple(k for k in self._archive.files if k != "_shape")
        self._cache: dict[str, np.ndarray] = {}

    def __getitem__(self, key: str) -> np.ndarray | None:
        if key not in self._cache:
            if key not in self._keys:
                return None
            self._cache[key] = _unpack(self._archive[key], self._h, self._w)
        return self._cache[key]

    def __iter__(self):
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def close(self) -> None:
        self._archive.close()
        self._cache.clear()

    def __enter__(self) -> "TrackMasks":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def open_track_masks(stem: str) -> TrackMasks | None:
    """This video's silhouettes, or None when it was tracked without them."""
    path = tracks_masks_path(stem)
    return TrackMasks(path) if path.exists() else None
