"""FFmpeg utilities - redirects to utils.ffmpeg for backward compatibility."""

from utils.ffmpeg import export_segment, get_video_duration, extract_clip

__all__ = ["export_segment", "get_video_duration", "extract_clip"]
