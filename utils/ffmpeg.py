"""FFmpeg utilities for video processing."""

import subprocess
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


def export_segment(source: Path | str, start: float, end: float, output: Path | str, *, copy: bool = False) -> bool:
    """Export a single video segment.

    Args:
        source: Source video path
        start: Start time in seconds
        end: End time in seconds
        output: Output file path
        copy: If True, use stream copy (fast, not frame-accurate).
              If False (default), re-encode with libx264 (slower, frame-accurate).

    Returns:
        True if export succeeded, False otherwise

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
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=FFMPEG_TIMEOUT)
        if result.returncode != 0:
            raise FFmpegError(f"FFmpeg failed with code {result.returncode}: {result.stderr.decode()[:200]}")
        return True
    except subprocess.TimeoutExpired as e:
        raise FFmpegTimeoutError(str(output), FFMPEG_TIMEOUT) from e
