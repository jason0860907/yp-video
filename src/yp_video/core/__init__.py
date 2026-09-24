"""Core domain logic: video processing."""

from .ffmpeg import (
    FFmpegError,
    FFmpegTimeoutError,
    export_segment,
    get_video_duration,
)

__all__ = ["FFmpegError", "FFmpegTimeoutError", "export_segment", "get_video_duration"]
