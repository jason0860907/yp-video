"""The RF-DETR Seg 2XLarge dense pass (person/dense.py), read for the actor
labels and box style.

The pass indexes frames the way cv2 counts them, and the frame cache keeps
every native frame in decode order, so the indices agree.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from yp_video.person.dense import DENSE_SCORE_FLOOR

#: The dense pass keeps boxes from 0.1 so the cut is chosen here. 0.4 was
#: calibrated 2026-10-08 against the human-reviewed person frames.
PERSON_LABEL_MIN_SCORE = 0.4


class DensePass:
    """One video's dense pass (person/dense.py), read frame by frame.

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
        self.stride = int(self.meta.get("stride") or 1)
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
