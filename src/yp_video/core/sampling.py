"""Frame ↔ time and FPS utilities used across TAD/VLM pipelines.

Centralizes math that previously appeared in extract_features, convert_annotations,
output_converter, vlm_to_rally, and infer with subtly different fallbacks.

For container-precise duration in seconds prefer
``yp_video.core.ffmpeg.get_video_duration`` (uses ffprobe). The cv2 path here
trades accuracy for skipping a subprocess in tight loops.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_FPS = 30.0


def get_fps(video_path: Path | str) -> float:
    """Return container-reported fps, with decord → cv2 → ``DEFAULT_FPS`` fallbacks.

    decord is preferred when available because it returns the average fps
    rather than the (sometimes wrong) header value cv2 reads. Falls back to
    cv2, then to a 30 Hz default for files that fail to open.
    """
    try:
        from decord import VideoReader, cpu  # type: ignore
        return float(VideoReader(str(video_path), ctx=cpu(0)).get_avg_fps())
    except Exception:
        pass
    try:
        import cv2  # type: ignore
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return float(fps) if fps and fps > 0 else DEFAULT_FPS
    except Exception:
        return DEFAULT_FPS


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


def frame_to_time(frame_idx: float, fps: float) -> float:
    """Convert a frame index to seconds. Returns 0.0 when fps <= 0."""
    return frame_idx / fps if fps > 0 else 0.0


def time_to_frame(seconds: float, fps: float) -> int:
    """Convert seconds to a (rounded) integer frame index. Returns 0 when fps <= 0."""
    return int(round(seconds * fps)) if fps > 0 else 0
