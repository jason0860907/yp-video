"""Exact native-frame images for the detection editor, independent of actions."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
from fastapi import HTTPException

from yp_video.config import ACTION_FRAMES_DIR
from yp_video.contracts.action import frame_filename
from yp_video.web.action_waveform import video_metadata
from yp_video.web.r2_client import cut_media_source


def native_cache(video: Path) -> tuple[Path, dict] | None:
    directory = ACTION_FRAMES_DIR / video.stem
    path = directory / "metadata.json"
    if not path.exists():
        return None
    meta = json.loads(path.read_text())
    if meta.get("sampling") != "native" or not meta.get("frames"):
        return None
    if video.exists() and meta.get("source_size") != video.stat().st_size:
        return None
    return directory, meta


def metadata(video: Path) -> dict:
    cache = native_cache(video)
    if cache:
        directory, meta = cache
        pts = np.load(directory / "pts.npy", allow_pickle=False)
        if len(pts) != meta["frames"]:
            raise HTTPException(409, "Native frame cache timestamps do not match images")
        # Native cache PTS use the stream time base.
        time_base = meta["time_base"][0] / meta["time_base"][1]
        fps = (len(pts) - 1) / (float(pts[-1] - pts[0]) * time_base) if len(pts) > 1 else 30.0
        return {"num_frames": len(pts), "fps": fps}
    return video_metadata(video)


def frame_image(video: Path, frame: int, *, original: bool = False) -> bytes:
    cache = native_cache(video)
    if cache and not original:
        image = cache[0] / frame_filename(frame)
        if not image.exists():
            raise HTTPException(409, "Native frame cache incomplete; rebuild it before labeling")
        return image.read_bytes()
    source = cut_media_source(video)
    if source is None:
        raise HTTPException(404, "Video not found")
    # select counts decoded frames, avoiding keyframe/time-seek rounding.
    try:
        result = subprocess.run([
            "ffmpeg", "-v", "error", "-threads", "2", "-i", source,
            "-map", "0:v:0", "-vf", f"select=eq(n\\,{frame})",
            "-frames:v", "1", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
        ], capture_output=True, timeout=120, check=False)
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(504, "Frame decode timed out") from exc
    if result.returncode or not result.stdout:
        raise HTTPException(502, "Could not decode the requested native frame")
    return result.stdout
