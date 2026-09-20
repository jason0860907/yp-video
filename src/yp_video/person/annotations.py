"""Human person boxes. Only explicitly reviewed frames are ground truth.

All recognizable people (including sidelines), visible extent only. Boxes
are normalized xyxy in native decoded-frame order, never action-only time.
"""
from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from yp_video.config import PERSON_ANNOTATIONS_DIR
from yp_video.core.jsonl import atomic_write

_lock = RLock()
Coordinate = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Box = tuple[Coordinate, Coordinate, Coordinate, Coordinate]


class FrameAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: int = Field(default=0, ge=0)
    state: Literal["draft", "reviewed"] = "draft"
    boxes: list[Box] = Field(default_factory=list, max_length=1000)

    @field_validator("boxes")
    @classmethod
    def nonempty_boxes(cls, boxes: list[Box]) -> list[Box]:
        if any(x2 <= x1 or y2 <= y1 for x1, y1, x2, y2 in boxes):
            raise ValueError("Boxes must have positive width and height")
        return boxes


class Annotations(BaseModel):
    model_config = ConfigDict(extra="forbid")
    video: str
    num_frames: int = Field(gt=0)
    policy: Literal["all-visible-people"] = "all-visible-people"
    frames: dict[int, FrameAnnotation] = Field(default_factory=dict)


def annotation_path(stem: str) -> Path:
    if Path(stem).name != stem or stem in ("", ".", ".."):
        raise ValueError("Invalid video stem")
    return PERSON_ANNOTATIONS_DIR / f"{stem}_persons.json"


def load(stem: str) -> Annotations | None:
    path = annotation_path(stem)
    return Annotations.model_validate_json(path.read_text()) if path.exists() else None


class RevisionConflict(ValueError):
    pass


def save(stem: str, num_frames: int, frame: int, annotation: FrameAnnotation) -> FrameAnnotation:
    if not 0 <= frame < num_frames:
        raise ValueError("Frame outside video")
    with _lock:
        data = load(stem) or Annotations(video=stem, num_frames=num_frames)
        if data.num_frames != num_frames:
            raise RevisionConflict("Video frame count changed; check source before labeling")
        previous = data.frames.get(frame, FrameAnnotation())
        if previous.revision != annotation.revision:
            raise RevisionConflict("This frame was edited elsewhere. Reload before saving")
        saved = annotation.model_copy(update={"revision": previous.revision + 1})
        data.frames[frame] = saved
        with atomic_write(annotation_path(stem)) as out:
            json.dump(data.model_dump(mode="json"), out, ensure_ascii=False)
        return saved


def apply_annotations(stem: str, num_frames: int, per_frame: dict[int, list]) -> int:
    """Overlay reviewed truth; exclude saved drafts even from pseudo supervision."""
    data = load(stem)
    if data is None:
        return 0
    if data.num_frames != num_frames or any(not 0 <= f < num_frames for f in data.frames):
        raise ValueError(f"Person annotation frame count mismatch: {stem}")
    reviewed = 0
    for frame, annotation in data.frames.items():
        if annotation.state == "reviewed":
            per_frame[frame] = annotation.boxes
            reviewed += 1
        else:
            per_frame.pop(frame, None)
    return reviewed
