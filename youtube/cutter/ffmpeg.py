"""FFmpeg utilities - redirects to utils.ffmpeg for backward compatibility."""

from utils.ffmpeg import (
    FFmpegError,
    FFmpegTimeoutError,
    export_segment,
    extract_clip,
    get_video_duration,
)

__all__ = [
    "FFmpegError",
    "FFmpegTimeoutError",
    "export_segment",
    "extract_clip",
    "get_video_duration",
]
