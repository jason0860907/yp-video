"""Frame cache helpers for SPOT action prediction and training."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

from yp_video.config import ACTION_FRAMES_DIR
from yp_video.contracts.action import (
    FRAME_FFMPEG_PATTERN,
    FRAME_HEIGHT,
    frame_filename,
)


FRAME_PATTERN = FRAME_FFMPEG_PATTERN
META_NAME = ".frame-cache.json"
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
    height: int = FRAME_HEIGHT,
    overwrite: bool = False,
) -> dict:
    """Ensure a 0-based JPEG frame cache exists for ``video_path``.

    The output layout matches ``yp_spot.train``:
    ``<cache_root>/<video_stem>/000000.jpg``.
    """

    video_path = Path(video_path)
    if not video_path.exists():
        raise ActionFrameCacheError(f"Video not found: {video_path}")

    output_dir = action_frame_dir(video_path, cache_root=cache_root)
    with _cache_lock(output_dir):
        return _ensure_action_frame_cache_locked(
            video_path,
            cache_root=cache_root,
            output_dir=output_dir,
            expected_frames=expected_frames,
            height=height,
            overwrite=overwrite,
        )


def _ensure_action_frame_cache_locked(
    video_path: Path,
    *,
    cache_root: Path,
    output_dir: Path,
    expected_frames: int | None,
    height: int,
    overwrite: bool,
) -> dict:
    if not overwrite:
        cached = inspect_action_frame_cache(
            video_path,
            cache_root=cache_root,
            expected_frames=expected_frames,
            height=height,
        )
        if cached["ready"]:
            return {**cached, "created": False}

    cache_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = cache_root / f".{video_path.stem}.frames.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-vf",
        f"scale=-2:{height}",
        "-q:v",
        "3",
        "-vsync",
        "0",
        "-start_number",
        "0",
        str(tmp_dir / FRAME_PATTERN),
    ]

    started = time.monotonic()
    with _EXTRACT_SEMAPHORE:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        detail = (result.stderr or result.stdout or "").strip()[:500]
        raise ActionFrameCacheError(
            f"Failed to extract action frames for {video_path.name}: {detail}"
        )

    frame_count = _count_frames(tmp_dir)
    if frame_count <= 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise ActionFrameCacheError(f"No frames extracted from {video_path}")

    source_stat = video_path.stat()
    metadata = {
        "source": str(video_path),
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "height": height,
        "frame_count": frame_count,
        "expected_frames": expected_frames,
        "created_at": time.time(),
        "extract_seconds": round(time.monotonic() - started, 3),
    }
    (tmp_dir / META_NAME).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    tmp_dir.replace(output_dir)

    return {
        "ready": True,
        "created": True,
        "path": str(output_dir),
        "frame_count": frame_count,
        "expected_frames": expected_frames,
        "height": height,
    }


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
    height: int = FRAME_HEIGHT,
) -> dict:
    output_dir = action_frame_dir(Path(video_path), cache_root=cache_root)
    metadata = _read_metadata(output_dir)
    frame_count = int(metadata.get("frame_count") or 0) if metadata else 0
    if frame_count <= 0:
        frame_count = _count_frames(output_dir)
    first = output_dir / frame_filename(0)
    last = output_dir / frame_filename(frame_count - 1) if frame_count else None
    ready = (
        output_dir.is_dir()
        and first.exists()
        and frame_count > 0
        and (last.exists() if last is not None else False)
    )

    source_stat = Path(video_path).stat() if Path(video_path).exists() else None
    if ready and metadata and source_stat:
        ready = (
            metadata.get("source_size") == source_stat.st_size
            and metadata.get("source_mtime_ns") == source_stat.st_mtime_ns
            and metadata.get("height") == height
            and metadata.get("frame_count") == frame_count
        )

    if ready and expected_frames:
        # ffprobe metadata can be off by one for some cut files. Treat that as
        # usable, but rebuild larger mismatches.
        ready = abs(frame_count - expected_frames) <= 1

    return {
        "ready": ready,
        "path": str(output_dir),
        "frame_count": frame_count,
        "expected_frames": expected_frames,
        "height": height,
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
        created += int(bool(info.get("created")))
        reused += int(not info.get("created"))
        total_frames += int(info.get("frame_count") or 0)

    if progress:
        progress(total, total, f"Frame cache ready for {total} video(s)")

    return {
        "videos": total,
        "created": created,
        "reused": reused,
        "frames": total_frames,
        "root": str(cache_root),
    }


def _count_frames(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for path in directory.glob("*.jpg") if path.name[:6].isdigit())


def _read_metadata(directory: Path) -> dict | None:
    path = directory / META_NAME
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None
