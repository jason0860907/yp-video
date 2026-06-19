"""Contract for the action-spotting data exchanged with the yp-spot model.

yp-video is the *producer*: it writes ``*_actions.jsonl`` label files and extracts
the JPEG frame caches that yp-spot trains and runs inference on. yp-spot is the
*consumer*, living in a separate repo + venv and reached across a subprocess
boundary, so the two cannot share Python at runtime.

This module is therefore the single authoritative definition on the producer
side. ``contracts/action_label.schema.json`` is generated from the models here
(via ``make_schema.py``), and yp-spot mirrors the same constants in
``yp_spot/contract.py``. The two copies are kept honest by a version handshake:
yp-video exports ``ACTION_CONTRACT_VERSION`` through the
``YP_ACTION_CONTRACT_VERSION`` env var when it spawns yp-spot, and the consumer
fails loud if its compiled-in version differs. Bump the version whenever the
field layout, frame layout, or label set below changes — and update both sides.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

# Bump on ANY breaking change to the label record, frame layout, or label set.
ACTION_CONTRACT_VERSION = "1.0.0"

# Env var carrying ACTION_CONTRACT_VERSION from producer to consumer.
ACTION_CONTRACT_VERSION_ENV = "YP_ACTION_CONTRACT_VERSION"

# ── Frame cache layout ────────────────────────────────────────────
# Frames are extracted as 0-based, zero-padded JPEGs under
# ``<cache_root>/<video_stem>/000000.jpg``, scaled to FRAME_HEIGHT (aspect
# ratio preserved). Producer writes with ffmpeg (FRAME_FFMPEG_PATTERN);
# consumer reads with str.format (FRAME_PY_PATTERN).
FRAME_HEIGHT = 224
FRAME_FILENAME_DIGITS = 6
FRAME_FFMPEG_PATTERN = "%06d.jpg"
FRAME_PY_PATTERN = "{:06d}.jpg"
FRAME_GLOB = "*.jpg"


def frame_filename(index: int) -> str:
    """Return the cache filename for a 0-based frame index."""
    return FRAME_PY_PATTERN.format(index)


# ── Label files ───────────────────────────────────────────────────
# Per-video label files are JSONL with a ``_meta`` header line followed by one
# record per video (see yp_video.core.jsonl).
LABEL_FILE_SUFFIX = "_actions.jsonl"
LABEL_FILE_GLOB = "*_actions.jsonl"
DEFAULT_FPS = 30.0


class ActionLabel(str, Enum):
    serve = "serve"
    receive = "receive"
    set = "set"
    spike = "spike"
    block = "block"
    score = "score"


# Canonical labels: ordered tuple for UI/display, frozenset for membership.
ACTION_LABELS_ORDERED = tuple(label.value for label in ActionLabel)
ACTION_LABELS = frozenset(ACTION_LABELS_ORDERED)


class ActionEvent(BaseModel):
    """A single spotted action at one frame, with a normalized court location."""

    model_config = {"extra": "forbid"}

    frame: int = Field(ge=0, description="0-based frame index into the frame cache")
    label: str = Field(description="One of ACTION_LABELS")
    xy: list[float] = Field(
        min_length=2,
        max_length=2,
        description="Normalized [x, y] court location, each in [0, 1]",
    )
    visible: bool = Field(default=True, description="Whether the action is visible on screen")


class ActionLabelRecord(BaseModel):
    """One video's worth of action labels — the unit of a ``*_actions.jsonl`` row."""

    # Tolerate _meta-derived extras the trainer may carry through.
    model_config = {"extra": "allow"}

    video: str = Field(description="Video stem; matches the frame-cache directory name")
    num_frames: int = Field(ge=0, description="Total frames in the cache for this video")
    fps: float = Field(default=DEFAULT_FPS, gt=0)
    events: list[ActionEvent] = Field(default_factory=list)


# ── Progress protocol (yp-spot stdout → yp-video) ─────────────────
# yp-spot emits one line per progress tick:
#   ``SPOT_PROGRESS {"phase":"inference","clips_done":..,"clips_total":..,
#                     "end_frame":..,"total_frames":..,"batch_done":..,
#                     "batch_total":..,"video":..,"video_basename":..}``
# The producer parses these defensively (web/routers/action_annotate.py); only
# the prefix is a hard contract.
SPOT_PROGRESS_PREFIX = "SPOT_PROGRESS "
