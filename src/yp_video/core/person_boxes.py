"""Fusion's sampled person boxes, shared by tracking and event detection.

The archive preserves native frame indices and normalized xyxy boxes. An
empty sampled frame means no people; a missing sampled frame is an error.
These are whole-video artifacts, so rally edits never require re-detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from yp_video.config import TRACKS_DIR
from yp_video.core.jsonl import atomic_binary

DETECTOR_NAME = "spot-person"


def person_boxes_path(stem: str) -> Path:
    return TRACKS_DIR / f"{stem}_persons.npz"


def checkpoint_identity(checkpoint: Path) -> str:
    stat = checkpoint.stat()
    return f"{checkpoint.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"


def person_boxes_current(stem: str, checkpoint: Path) -> bool:
    path = person_boxes_path(stem)
    if not path.exists():
        return False
    with np.load(path, allow_pickle=False) as data:
        return str(data["checkpoint"]) == checkpoint_identity(checkpoint)


def save_person_boxes(source: Path, target: Path, checkpoint: Path) -> None:
    """Validate and retain subprocess output before its temporary dir closes."""
    people = PersonBoxes.load(source)
    with atomic_binary(target) as out:
        np.savez_compressed(
            out, frames=people.frames, counts=people.counts, boxes=people.boxes, scores=people.scores,
            stride=people.stride, num_frames=people.num_frames,
            checkpoint=np.array(checkpoint_identity(checkpoint)),
        )


@dataclass(frozen=True)
class PersonBoxes:
    frames: np.ndarray
    counts: np.ndarray
    boxes: np.ndarray
    scores: np.ndarray
    stride: int
    num_frames: int
    offsets: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.stride < 1 or self.num_frames < 1:
            raise ValueError("Person boxes require positive stride and num_frames")
        if not np.array_equal(self.frames, np.arange(0, self.num_frames, self.stride)):
            raise ValueError("Person boxes must cover every sampled frame of the whole video")
        if self.counts.shape != self.frames.shape or np.any(self.counts < 0):
            raise ValueError("Person counts must have one nonnegative count per sampled frame")
        if not np.issubdtype(self.counts.dtype, np.integer):
            raise ValueError("Person counts must be integers")
        total = int(self.counts.sum())
        if self.boxes.shape != (total, 4) or self.scores.shape != (total,):
            raise ValueError("Person box/score lengths do not match frame counts")
        if (
            not np.isfinite(self.boxes).all() or not np.isfinite(self.scores).all()
            or np.any((self.boxes < 0) | (self.boxes > 1))
            or np.any((self.scores < 0) | (self.scores > 1))
            or np.any(self.boxes[:, 2:] < self.boxes[:, :2])
        ):
            raise ValueError("Person boxes must be normalized xyxy with finite probabilities")
        object.__setattr__(self, "offsets", np.concatenate(([0], np.cumsum(self.counts))))

    @classmethod
    def load(cls, path: Path) -> PersonBoxes:
        with np.load(path, allow_pickle=False) as data:
            return cls(
                frames=data["frames"], counts=data["counts"],
                boxes=data["boxes"], scores=data["scores"],
                stride=int(data["stride"]), num_frames=int(data["num_frames"]),
            )

    def pixels(self, index: int, width: int, height: int) -> np.ndarray:
        start, end = self.offsets[index:index + 2]
        xyxy = self.boxes[start:end] * np.array([width, height, width, height])
        rows = np.column_stack((xyxy, self.scores[start:end])).astype(np.float32)
        return rows[(rows[:, 2] > rows[:, 0]) & (rows[:, 3] > rows[:, 1])]

    def for_frame(self, frame: int, width: int, height: int) -> np.ndarray:
        """Nearest sampled frame (earlier on ties), including a final partial step.

        At 60 fps a 30 Hz stream has stride 2. Keep its actual sampling
        cadence; never interpret its boxes as a dense 60 Hz detector.
        """
        if not 0 <= frame < self.num_frames:
            raise ValueError(f"Event frame {frame} is outside person output (0..{self.num_frames - 1})")
        index = min((frame + (self.stride - 1) // 2) // self.stride, len(self.frames) - 1)
        return self.pixels(index, width, height)
