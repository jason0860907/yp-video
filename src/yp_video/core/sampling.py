"""Frame ↔ time and FPS utilities used across the video pipelines.

Centralizes math that previously appeared in several modules with subtly
different fallbacks.

For container-precise duration in seconds prefer
``yp_video.core.ffmpeg.get_video_duration`` (uses ffprobe). The cv2 path here
trades accuracy for skipping a subprocess in tight loops.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_FPS = 30.0


def get_fps(video_path: Path | str) -> float:
    """Return container-reported fps, with decord → cv2 → ``DEFAULT_FPS`` fallbacks.

    decord is preferred when available because it returns the average fps
    rather than the (sometimes wrong) header value cv2 reads. Falls back to
    cv2, then to a 30 Hz default for files that fail to open.

    decord/cv2 simply being absent is expected and stays quiet, but a probe
    that *errors* (corrupt/unreadable file) is logged before we fall back —
    a silently-wrong fps scales every frame↔time conversion downstream.
    """
    try:
        from decord import VideoReader, cpu  # type: ignore
    except ImportError:
        pass
    else:
        try:
            return float(VideoReader(str(video_path), ctx=cpu(0)).get_avg_fps())
        except Exception as exc:  # noqa: BLE001 — decord raises bare RuntimeError on bad files
            logger.warning("decord failed to read fps from %s: %s; trying cv2", video_path, exc)

    try:
        import cv2  # type: ignore
    except ImportError:
        logger.warning("neither decord nor cv2 available; assuming %.1f fps for %s", DEFAULT_FPS, video_path)
        return DEFAULT_FPS

    try:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
    except Exception as exc:  # noqa: BLE001
        logger.warning("cv2 failed to read fps from %s: %s; assuming %.1f fps", video_path, exc, DEFAULT_FPS)
        return DEFAULT_FPS

    if fps and fps > 0:
        return float(fps)
    logger.warning("cv2 reported non-positive fps for %s; assuming %.1f fps", video_path, DEFAULT_FPS)
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
