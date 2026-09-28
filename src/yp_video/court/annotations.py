"""Human court calibrations: one file per video, the landmarks a user marked.

Only the marks are stored — the homography is derived (geometry.fit), so a
better solver never leaves stale matrices behind.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from yp_video.config import COURT_ANNOTATIONS_DIR
from yp_video.core.jsonl import atomic_write
from yp_video.court.geometry import Landmark

#: How far past the frame edge a mark may sit, as a fraction of the frame:
#: a corner the camera cut off is still marked, at the crossing of the guide
#: lines drawn along the paint that is visible.
OUTSIDE_FRAME = 0.25
Coordinate = Annotated[float, Field(ge=-OUTSIDE_FRAME, le=1 + OUTSIDE_FRAME, allow_inf_nan=False)]


class Calibration(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: int = 1
    #: Height of the net's top band, metres — what the net-top marks stand at.
    net_height_m: float = Field(default=2.43, ge=1.5, le=3.0)
    #: The video's frame (width, height) in pixels: the camera solve needs
    #: the aspect ratio the normalized marks were taken in.
    frame_size: tuple[int, int] | None = None
    #: Landmark → where it sits in the frame (normalized x, y; may lie
    #: OUTSIDE_FRAME past the edge).
    points: dict[Landmark, tuple[Coordinate, Coordinate]] = Field(default_factory=dict)


def annotation_path(stem: str) -> Path:
    if Path(stem).name != stem or stem in ("", ".", ".."):
        raise ValueError("Invalid video stem")
    return COURT_ANNOTATIONS_DIR / f"{stem}_court.json"


def load(stem: str) -> Calibration | None:
    path = annotation_path(stem)
    return Calibration.model_validate_json(path.read_text()) if path.exists() else None


def save(stem: str, calibration: Calibration) -> None:
    with atomic_write(annotation_path(stem)) as f:
        f.write(calibration.model_dump_json(indent=1))
