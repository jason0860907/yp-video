"""Every rally frame's people from the strongest RF-DETR Seg: the boxes actor
labels are picked from and the actor head's candidates train on.

A tracker's boxes only hold detections that joined a tracklet — the occluded
and far-side players it drops are exactly the ones a label may need. This
pass keeps every detection of the largest seg model above a low floor, with
its score, so the cut is chosen by the reader rather than baked in here.

One ``<stem>_dense.npz`` per video in PERSON_DENSE_DIR: ``frames`` (int32,
cv2 frame indices — the tracks' convention — of the detected padded
rally-span frames), ``times`` (float64, each frame's presentation time in seconds, so a
reader can align to any other frame index), ``counts`` (int32, boxes on that
frame), ``boxes`` (float16, xyxy normalized to the frame), ``scores``
(float16) and ``meta`` (JSON: model, floor, fps, stride, frame size, rally
fingerprint). A frame with zero boxes is a real answer.

A source above DENSE_MAX_FPS is detected every ``stride``-th frame: its
neighbours are 17 ms apart, so a label writer gives a skipped frame its
neighbour's boxes instead of paying for both.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from yp_video.config import PERSON_DENSE_DIR
from yp_video.core.frames import open_video
from yp_video.core.jsonl import atomic_binary
from yp_video.core.progress import ProgressFn
from yp_video.core.rallies import RALLY_PAD_S, load_rallies, rally_fingerprint
from yp_video.person.seg_batch import BatchSegDetector, span_frames

DENSE_VARIANT = "RFDETRSeg2XLarge"
# Batch 4 at res 768: ~52 ms/frame and ~7.6 GB reserved on the 4090 (batch 8
# is only 12% faster for twice the VRAM, which the production worker needs).
DENSE_BATCH_SIZE = 4
# Low enough to keep the occluded players; the label writer picks the cut.
DENSE_SCORE_FLOOR = 0.1
# The actor labels' cut, calibrated 2026-10-08 against the human-reviewed
# person frames.
PERSON_LABEL_MIN_SCORE = 0.4
DENSE_SUFFIX = "_dense.npz"
# Detect at about this rate: a 59.94/60 fps source every 2nd frame.
DENSE_MAX_FPS = 30.0

_detector = BatchSegDetector(DENSE_VARIANT, DENSE_BATCH_SIZE)


class DensePass:
    """One video's dense pass, read frame by frame. The pass indexes frames
    the way cv2 counts them, and the frame cache keeps every native frame in
    decode order, so the indices agree.

    ``at`` answers for any native frame without expanding the whole video —
    a request needs a handful of frames. A
    source the pass strode (above 30 fps) gives each skipped frame the boxes
    of the detected frame before it — 17 ms apart — but only inside a span:
    the next detected frame must be exactly one stride on."""

    def __init__(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as data:
            self.meta: dict = json.loads(str(data["meta"]))
            self.frames = data["frames"]
            counts = data["counts"]
            self._boxes, self._scores = data["boxes"].astype(np.float32), data["scores"]
        self.fps = float(self.meta["fps"])
        self.stride = int(self.meta["stride"])
        self._ends = np.cumsum(counts)
        self._starts = self._ends - counts

    def rows(self, frames: np.ndarray) -> np.ndarray:
        """The detected row each frame takes its boxes from; -1 = not covered."""
        frames = np.asarray(frames)
        n = len(self.frames)
        if not n:
            return np.full(frames.shape, -1)
        i = np.searchsorted(self.frames, frames, side="right") - 1
        start = self.frames[np.maximum(i, 0)]
        following = self.frames[np.minimum(i + 1, n - 1)]
        strode = (frames - start < self.stride) & (i + 1 < n) & (following == start + self.stride)
        return np.where((i >= 0) & ((frames == start) | strode), i, -1)

    def row_boxes(self, row: int, min_score: float) -> tuple[list[list[float]], list[float]]:
        """One detected row's normalized boxes and scores at or above ``min_score``."""
        start, end = self._starts[row], self._ends[row]
        kept = self._scores[start:end] >= min_score
        return self._boxes[start:end][kept].tolist(), self._scores[start:end][kept].astype(float).tolist()

    def at(self, frame: int, min_score: float) -> tuple[list[list[float]], list[float]] | None:
        """``frame``'s boxes and scores, or None when the pass did not cover it."""
        row = int(self.rows(np.array([frame]))[0])
        return None if row < 0 else self.row_boxes(row, min_score)

    def people(self, frames, min_score: float = DENSE_SCORE_FLOOR) -> dict[int, list[list[float]]]:
        """``{frame: normalized boxes}`` over ``frames``, covered frames only."""
        frames = np.asarray(list(frames))
        return {
            frame: self.row_boxes(row, min_score)[0]
            for frame, row in zip(frames.tolist(), self.rows(frames).tolist())
            if row >= 0
        }


class DensePaused(Exception):
    """``should_pause`` asked to stop; the detector's VRAM is already freed."""


def dense_path(stem: str) -> Path:
    return PERSON_DENSE_DIR / f"{stem}{DENSE_SUFFIX}"


def release_detector() -> None:
    _detector.release()


def detect_dense(
    video_path: Path,
    *,
    should_pause: Callable[[], bool],
    on_progress: ProgressFn | None = None,
) -> dict:
    """Detect people on every frame of every rally span of one video, each
    span padded by RALLY_PAD_S on both sides.

    ``should_pause`` is polled between batches; when it returns True the
    detector is released and DensePaused raised — nothing is written, so the
    video simply runs again later.
    """

    import cv2

    stem = video_path.stem
    rallies = load_rallies(stem)
    if not rallies:
        raise ValueError(f"No rally spans for {stem}")
    cap, geometry = open_video(video_path)
    fps, width, height = geometry.fps, geometry.width, geometry.height
    stride = max(1, round(fps / DENSE_MAX_FPS))
    last_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    # The same padded spans tracking scans (tracklets/tracking.py).
    spans = [
        (
            r["rally_id"],
            max(0, int(round((r["start"] - RALLY_PAD_S) * fps))),
            min(last_frame, int(round((r["end"] + RALLY_PAD_S) * fps))),
        )
        for r in rallies
    ]
    total = sum((f1 - f0) // stride + 1 for _, f0, f1 in spans)

    _detector.ensure()
    res = _detector.resolution
    scale = np.array([res, res, res, res], dtype=np.float32)
    frames: list[int] = []
    times: list[float] = []
    counts: list[int] = []
    boxes: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    try:
        with span_frames(cap, spans, stride=stride, resolution=res, name=f"dense-decode-{stem}") as items:
            pending: list = []
            exhausted = False
            while not exhausted or pending:
                if should_pause():
                    _detector.release()
                    raise DensePaused(stem)
                while not exhausted and len(pending) < DENSE_BATCH_SIZE:
                    item = next(items, None)
                    if item is None:
                        exhausted = True
                        break
                    pending.append(item)
                if not pending:
                    break
                # Boxes only: the masks rfdetr's predict() post-processes are ~90% of its time.
                for (_, frame_idx, seconds, _), (xyxy, confidence) in zip(
                    pending, _detector.predict_boxes([p.tensor for p in pending], DENSE_SCORE_FLOOR)
                ):
                    frames.append(frame_idx)
                    times.append(seconds)
                    counts.append(len(xyxy))
                    boxes.append((xyxy / scale).clip(0, 1).astype(np.float16))
                    scores.append(confidence.astype(np.float16))
                if on_progress:
                    on_progress(len(frames), total, f"frame {len(frames)}/{total}")
                pending = []
    finally:
        cap.release()

    meta = {
        "video": stem,
        "model": DENSE_VARIANT,
        "score_floor": DENSE_SCORE_FLOOR,
        "fps": fps,
        "stride": stride,
        "frame_size": [width, height],
        "rallies": {"count": len(spans), "fingerprint": rally_fingerprint(stem)},
        "created_at": time.time(),
    }
    out = dense_path(stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    with atomic_binary(out) as f:
        np.savez_compressed(
            f,
            frames=np.asarray(frames, dtype=np.int32),
            times=np.asarray(times, dtype=np.float64),
            counts=np.asarray(counts, dtype=np.int32),
            boxes=np.concatenate(boxes) if boxes else np.zeros((0, 4), np.float16),
            scores=np.concatenate(scores) if scores else np.zeros((0,), np.float16),
            meta=np.array(json.dumps(meta)),
        )
    return {"frames": len(frames), "boxes": int(sum(counts)), "rallies": len(spans)}
