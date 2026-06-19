"""FFmpeg utilities for video processing."""

import asyncio
import json
import math
import subprocess
from fractions import Fraction
from pathlib import Path


class FFmpegError(Exception):
    """Base exception for FFmpeg operations."""

    pass


class FFmpegTimeoutError(FFmpegError):
    """Raised when FFmpeg operation times out."""

    def __init__(self, output_path: str, timeout: int):
        self.output_path = output_path
        self.timeout = timeout
        super().__init__(f"FFmpeg timeout after {timeout}s for {output_path}")


# 10 minutes timeout for FFmpeg operations
FFMPEG_TIMEOUT = 600


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def parse_rate(rate: str | None) -> float:
    """Parse an ffprobe rate field (``"30000/1001"`` or ``"30"``) to fps.

    Returns 0.0 for missing/unparseable values so callers can ``or`` through
    fallbacks.
    """
    if not rate or rate == "0/0":
        return 0.0
    try:
        return float(Fraction(str(rate)))
    except (ValueError, ZeroDivisionError):
        return 0.0


def parse_optional_float(value: object) -> float | None:
    """Best-effort float parse; ``None`` when missing or unparseable."""
    if value in (None, "", "N/A"):
        return None
    if not isinstance(value, (int, float, str)):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def probe_video_metadata(path: Path | str) -> dict:
    """Probe the first video stream via ffprobe.

    Returns ``{fps, duration, num_frames, start_time}``. ``num_frames`` falls
    back to ``round(duration * fps)`` when the container stores no frame count.

    Raises:
        FFmpegError: ffprobe failed or returned invalid JSON.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,r_frame_rate,nb_frames,duration,start_time",
        "-of", "json",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise FFmpegError(f"ffprobe failed: {result.stderr[:200]}")
    try:
        stream = (json.loads(result.stdout).get("streams") or [{}])[0]
    except json.JSONDecodeError as exc:
        raise FFmpegError("ffprobe returned invalid JSON") from exc

    fps = parse_rate(stream.get("avg_frame_rate")) or parse_rate(stream.get("r_frame_rate")) or 30.0
    duration = float(stream.get("duration") or 0)
    num_frames = int(stream.get("nb_frames") or round(duration * fps))
    return {
        "fps": fps,
        "duration": duration,
        "num_frames": num_frames,
        "start_time": parse_optional_float(stream.get("start_time")) or 0.0,
    }


def extract_clip(video_path: str, start_time: float, duration: float, output_path: str) -> bool:
    """Extract a clip from video using ffmpeg.

    Args:
        video_path: Source video path
        start_time: Start time in seconds
        duration: Clip duration in seconds
        output_path: Output file path

    Returns:
        True if extraction succeeded, False otherwise

    Raises:
        FFmpegTimeoutError: If FFmpeg operation times out
        FFmpegError: If FFmpeg returns non-zero exit code
    """
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-i", video_path,
        "-t", str(duration),
        "-c:v", "copy",
        "-an",  # No audio
        output_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=FFMPEG_TIMEOUT)
        if result.returncode != 0:
            raise FFmpegError(f"FFmpeg failed with code {result.returncode}: {result.stderr.decode()[:200]}")
        return True
    except subprocess.TimeoutExpired as e:
        raise FFmpegTimeoutError(output_path, FFMPEG_TIMEOUT) from e


async def export_segment(source: Path | str, start: float, end: float, output: Path | str, *, copy: bool = False) -> bool:
    """Export a single video segment. Does not block the event loop.

    Args:
        source: Source video path
        start: Start time in seconds
        end: End time in seconds
        output: Output file path
        copy: If True, use stream copy (fast, not frame-accurate).
              If False (default), re-encode with libx264 (slower, frame-accurate).

    Raises:
        FFmpegTimeoutError: If FFmpeg operation times out
        FFmpegError: If FFmpeg returns non-zero exit code
    """
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(source),
        "-t", str(end - start),
    ]
    if copy:
        cmd += ["-c:v", "copy", "-c:a", "copy"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-c:a", "aac"]
    cmd += ["-movflags", "+faststart", str(output)]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=FFMPEG_TIMEOUT)
    except asyncio.TimeoutError as e:
        proc.kill()
        await proc.wait()
        raise FFmpegTimeoutError(str(output), FFMPEG_TIMEOUT) from e
    if proc.returncode != 0:
        raise FFmpegError(f"FFmpeg failed with code {proc.returncode}: {stderr.decode()[:200]}")
    return True
