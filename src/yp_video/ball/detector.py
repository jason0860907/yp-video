"""The fine-tuned ball detector over frame spans (GPU).

One RF-DETR Medium at a ball-sized input, fine-tuned (ball/train.py) on the
pseudo-labels (ball/pseudo.py). Two callers: the pseudo-label pass, which
detects every contact-to-contact flight of the hand-marked videos, and the
analysis-time track (ball/track.py), which detects the flights between the
analysis's predicted touches.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from yp_video.config import BALL_CHECKPOINTS_DIR
from yp_video.core.progress import ProgressFn
from yp_video.person.seg_batch import BatchSegDetector, span_frames

VARIANT = "RFDETRMedium"
# The ball is ~15-30 px in a 1080p sideline frame: COCO sports-ball recall at
# contacts is 70% at the default 576 and 82% at 1024 (2XLarge at 1008: 77%).
RESOLUTION = 1024
BATCH_SIZE = 8
# Round 2: fine-tuned on the round-1 (COCO sports ball) labels; contact-frame
# top-1 hit 57% -> 95% on held-out videos.
WEIGHTS = BALL_CHECKPOINTS_DIR / "20261014_medium1024" / "checkpoint_best_ema.pth"
# The fine-tuned head's only class.
CLASS_ID = 0
# The head spreads ~40 boxes a frame below 0.1 and a curve fit then finds
# clutter arcs; at 0.1 (~4 a frame) more flights fit (two held-out videos:
# 224 -> 247 and 231 -> 240 of 252 / 247 vs round 1). An occluded ball at a
# contact can still score ~0.1.
SCORE_FLOOR = 0.1
# Detect at about this rate: neighbouring frames are near-duplicates.
HZ = 15.0

_detector = BatchSegDetector(VARIANT, BATCH_SIZE, resolution=RESOLUTION, weights=str(WEIGHTS), class_id=CLASS_ID)


class Paused(Exception):
    """``should_pause`` asked to stop; the detector's VRAM is already freed."""


@dataclass(frozen=True)
class Detections:
    fps: float
    #: (width, height) in pixels.
    frame_size: tuple[int, int]
    stride: int
    #: cv2 frame indices, ascending within each span.
    frames: list[int]
    #: Per frame, (N, 5): xyxy normalized to the frame, then score.
    boxes: list[np.ndarray]


def release() -> None:
    _detector.release()


def video_fps(video_path: Path) -> float:
    """cv2's frame rate — the one the frame indices here count in."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()
    if not fps > 0:
        raise ValueError(f"Cannot read the frame rate of {video_path}")
    return fps


def detect(
    video_path: Path,
    spans: list[tuple[int, int]],
    *,
    should_pause: Callable[[], bool] = lambda: False,
    on_progress: ProgressFn | None = None,
) -> Detections:
    """Every box above SCORE_FLOOR on the frames of ``spans`` (inclusive
    [first, last] frame pairs), sampled at about HZ."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not fps > 0 or width <= 0 or height <= 0:
        cap.release()
        raise ValueError(f"Invalid video geometry: {video_path}")
    stride = max(1, round(fps / HZ))
    total = sum((b - a) // stride + 1 for a, b in spans)

    _detector.ensure()
    res = _detector.resolution
    frames: list[int] = []
    boxes: list[np.ndarray] = []
    try:
        indexed = [(i, a, b) for i, (a, b) in enumerate(spans)]
        with span_frames(cap, indexed, stride=stride, resolution=res, name=f"ball-decode-{video_path.stem}") as items:
            pending: list = []
            exhausted = False
            while not exhausted or pending:
                if should_pause():
                    _detector.release()
                    raise Paused(video_path.stem)
                while not exhausted and len(pending) < BATCH_SIZE:
                    item = next(items, None)
                    if item is None:
                        exhausted = True
                        break
                    pending.append(item)
                if not pending:
                    break
                for item, (xyxy, confidence) in zip(
                    pending, _detector.predict_boxes([p.tensor for p in pending], SCORE_FLOOR)
                ):
                    frames.append(item.frame)
                    boxes.append(np.hstack([(xyxy / res).clip(0, 1), confidence[:, None]]).astype(np.float32))
                pending = []
                if on_progress:
                    on_progress(len(frames), total, "ball")
    finally:
        cap.release()
    return Detections(fps=fps, frame_size=(width, height), stride=stride, frames=frames, boxes=boxes)
