"""cv2-based video duration probe.

For container-precise duration in seconds prefer
``yp_video.core.ffmpeg.get_video_duration`` (uses ffprobe). The cv2 path here
trades accuracy for skipping a subprocess in tight loops.
"""

from __future__ import annotations

from pathlib import Path


def get_video_duration_cv2(video_path: Path | str) -> float:
    """Estimate duration via cv2's frame count / fps. Returns 0.0 on error.

    Faster than ffprobe in tight loops but reports header-declared duration,
    which may diverge from the actual track on poorly-muxed files. Use
    ``yp_video.core.ffmpeg.get_video_duration`` when you need accuracy.
    """
    try:
        import cv2  # type: ignore
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return frame_count / fps if fps > 0 else 0.0
    except Exception:
        return 0.0
