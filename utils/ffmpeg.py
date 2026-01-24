"""FFmpeg utilities for video processing."""

import subprocess
from pathlib import Path

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
    """
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-i", video_path,
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-an",  # No audio
        output_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=FFMPEG_TIMEOUT)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"FFmpeg timeout after {FFMPEG_TIMEOUT}s for {output_path}")
        return False


def export_segment(source: Path | str, start: float, end: float, output: Path | str) -> bool:
    """Export a single segment with re-encoding for frame-accurate cuts.

    Args:
        source: Source video path
        start: Start time in seconds
        end: End time in seconds
        output: Output file path

    Returns:
        True if export succeeded, False otherwise
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(source),
        "-ss", str(start),
        "-to", str(end),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-movflags", "+faststart",
        str(output)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=FFMPEG_TIMEOUT)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"FFmpeg timeout after {FFMPEG_TIMEOUT}s for {output}")
        return False
