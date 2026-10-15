"""Frame cache helpers for SPOT action prediction and training.

yp-spot's ``yp_spot.jpeg_frames`` is the one producer of the cache; this
module checks a cache against the contract and asks yp-spot to build it
when one is missing or stale.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path

from yp_video.config import ACTION_FRAMES_DIR, SPOT_DIR, SPOT_FRAME_CACHE_MODULE, SPOT_PYTHON
from yp_video.contracts.action import (
    FRAME_CACHE_FORMAT,
    FRAME_CACHE_METADATA,
    FRAME_LONGEST_EDGE,
    frame_filename,
)

_EXTRACT_SEMAPHORE = threading.Semaphore(2)
_CACHE_LOCKS: dict[str, threading.Lock] = {}
_CACHE_LOCKS_GUARD = threading.Lock()


class ActionFrameCacheError(RuntimeError):
    """Raised when an action frame cache cannot be created."""


def action_frame_dir(video_path: Path, *, cache_root: Path = ACTION_FRAMES_DIR) -> Path:
    return cache_root / video_path.stem


def ensure_action_frame_cache(
    video_path: Path,
    *,
    cache_root: Path = ACTION_FRAMES_DIR,
    expected_frames: int | None = None,
) -> dict:
    """Ensure ``<cache_root>/<video_stem>/000000.jpg…`` exists and is current.

    Every native frame is kept — the one cache serves action and rally
    training alike, each striding it to its own ``sample_fps``.
    """

    video_path = Path(video_path)
    output_dir = action_frame_dir(video_path, cache_root=cache_root)
    with _cache_lock(output_dir):
        cached = inspect_action_frame_cache(
            video_path, cache_root=cache_root, expected_frames=expected_frames
        )
        if cached["ready"]:
            return {**cached, "created": False}

        # Only a (re)build needs the source bytes — a ready cache serves videos
        # whose cut now lives only in R2.
        if not video_path.exists():
            raise ActionFrameCacheError(
                f"Frame cache for {video_path.stem} is missing or stale and the source "
                f"video is not on local disk ({video_path}); download that cut to rebuild it"
            )
        # Holding the lock, a leftover partial can only be a killed build.
        for stale in (output_dir, cache_root / f".{video_path.stem}.partial"):
            if stale.exists():
                shutil.rmtree(stale)
        cache_root.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(SPOT_PYTHON), "-m", SPOT_FRAME_CACHE_MODULE,
            "--video", video_path.stem,
            "--source", str(video_path),
            "--output", str(cache_root),
            "--longest-edge", str(FRAME_LONGEST_EDGE),
        ]
        with _EXTRACT_SEMAPHORE:
            result = subprocess.run(cmd, cwd=SPOT_DIR, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()[-500:]
            raise ActionFrameCacheError(
                f"Failed to extract action frames for {video_path.name}: {detail}"
            )
        built = inspect_action_frame_cache(
            video_path, cache_root=cache_root, expected_frames=expected_frames
        )
        if not built["ready"]:
            raise ActionFrameCacheError(
                f"Frame cache for {video_path.name} was built but does not match the "
                f"contract ({built['frame_count']} frames, expected {expected_frames})"
            )
        return {**built, "created": True}


def _cache_lock(output_dir: Path) -> threading.Lock:
    key = str(output_dir)
    with _CACHE_LOCKS_GUARD:
        lock = _CACHE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _CACHE_LOCKS[key] = lock
        return lock


def inspect_action_frame_cache(
    video_path: Path,
    *,
    cache_root: Path = ACTION_FRAMES_DIR,
    expected_frames: int | None = None,
) -> dict:
    video_path = Path(video_path)
    output_dir = action_frame_dir(video_path, cache_root=cache_root)
    metadata = _read_metadata(output_dir)
    frame_count = int(metadata.get("frames") or 0) if metadata else 0
    ready = (
        metadata is not None
        and metadata.get("format") == FRAME_CACHE_FORMAT
        and metadata.get("longest_edge") == FRAME_LONGEST_EDGE
        and frame_count > 0
        and (output_dir / frame_filename(0)).exists()
        and (output_dir / frame_filename(frame_count - 1)).exists()
    )
    # Size is the cut's identity across copies: the bytes live in R2 and are
    # materialized per job, so an mtime would describe the download.
    if ready and video_path.exists():
        ready = metadata.get("source_size") == video_path.stat().st_size
    if ready and expected_frames:
        # ffprobe metadata can be off by one for some cut files. Treat that as
        # usable, but rebuild larger mismatches.
        ready = abs(frame_count - expected_frames) <= 1

    return {
        "ready": ready,
        "path": str(output_dir),
        "frame_count": frame_count,
        "expected_frames": expected_frames,
    }


def ensure_action_frame_caches(
    items: list[tuple[Path, int | None]],
    *,
    cache_root: Path = ACTION_FRAMES_DIR,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict:
    created = 0
    reused = 0
    total_frames = 0
    total = len(items)

    for idx, (video_path, expected_frames) in enumerate(items, start=1):
        if progress:
            progress(idx - 1, total, f"Preparing frames: {video_path.name}")
        info = ensure_action_frame_cache(
            video_path,
            cache_root=cache_root,
            expected_frames=expected_frames,
        )
        created += int(info["created"])
        reused += int(not info["created"])
        total_frames += info["frame_count"]

    if progress:
        progress(total, total, f"Frame cache ready for {total} video(s)")

    return {
        "videos": total,
        "created": created,
        "reused": reused,
        "frames": total_frames,
        "root": str(cache_root),
    }


def _read_metadata(directory: Path) -> dict | None:
    path = directory / FRAME_CACHE_METADATA
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None
