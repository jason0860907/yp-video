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
ACTION_CONTRACT_VERSION = "1.2.0"

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


class SegmentLabelEvent(BaseModel):
    """A label covering a contiguous frame span (e.g. one rally), inclusive.

    Segment label files (rally training) reuse the ``ActionLabelRecord`` layout
    with these events instead of point actions; yp-spot fills every frame of the
    span with the class during training and evaluates with segment mAP.
    """

    model_config = {"extra": "forbid"}

    frame: int = Field(ge=0, description="0-based first frame of the span")
    end_frame: int = Field(ge=0, description="0-based last frame of the span, inclusive")
    label: str = Field(description="Segment class, e.g. 'rally'")


# ── Actor candidates (who performed each action) ──────────────────
# A SEPARATE file from the action labels, and deliberately so. The action
# labels are read by every spotting run over every video; actor supervision
# exists for a handful of videos and carries ~11 boxes per event, so folding it
# in would inflate the file every run reads with data almost none of them use.
ACTOR_FILE_SUFFIX = "_actor_candidates.jsonl"
ACTOR_FILE_GLOB = "*_actor_candidates.jsonl"
#: Sub-directory of a training run's label snapshot.
ACTOR_LABEL_SUBDIR = "actor-candidates"

#: Frame offsets, relative to the event, at which each candidate's box is
#: exported. The model samples its visual features at the candidate's OWN box
#: at each offset, so this window is what lets it see a player move — the
#: approach, the jump, the swing — rather than a single frozen pose.
#: +/-16 frames is ~0.53 s at 30 fps, which covers a spiker's last stride and
#: contact; every 4th frame keeps the token count sane.
ACTOR_WINDOW_RADIUS = 16
ACTOR_WINDOW_STRIDE = 4
ACTOR_WINDOW_OFFSETS = tuple(
    range(-ACTOR_WINDOW_RADIUS, ACTOR_WINDOW_RADIUS + 1, ACTOR_WINDOW_STRIDE)
)


# ── Checkpoint packages ───────────────────────────────────────────
# The manifest ``type`` each trainer stamps on its exported package. Every
# reader — init-checkpoint pickers, family detection, predict surfaces —
# matches on these strings, so they live here rather than at each export
# site, where a rename would silently break every matcher.
ACTION_PACKAGE_TYPE = "yp-video-action-checkpoint"
FUSION_PACKAGE_TYPE = "actor-association-spot"
ASSOCIATION_PACKAGE_TYPE = "yp-video-association-checkpoint"
RALLY_PACKAGE_TYPE = "yp-video-rally-spot-checkpoint"


class ActorTargetKind(str, Enum):
    """Which of the three answers an event carries.

    ``occluded`` is a human's verdict — they watched the event and could not
    see who performed it. It does NOT mean the court was empty: those events
    carry a median of ten other tracked players.

    ``untracked`` is the opposite situation and is derived, not declared: a
    human did name the actor, but no candidate on the event frame is them,
    because tracking dropped them. Both abstain downstream; they must not
    train identically, or a tracking failure teaches the model to answer
    "nobody could be seen" and it keeps answering that once tracking improves.
    """

    TRACK = "track"
    OCCLUDED = "occluded"
    UNTRACKED = "untracked"


class ActorCandidate(BaseModel):
    """One tracklet's path through the window around an event."""

    model_config = {"extra": "forbid"}

    track: str = Field(description='Tracklet identity, "<rally_id>:<track_id>"')
    boxes: list[list[float] | None] = Field(
        description=(
            "Normalized [x0, y0, x1, y1] at each ACTOR_WINDOW_OFFSETS position, "
            "aligned with it; null where tracking has no box for that frame. "
            "The absence is itself supervision — it says this player was not "
            "being tracked then."
        ),
    )


class ActorCandidateEvent(BaseModel):
    """The candidate set for one action event, and which one acted."""

    model_config = {"extra": "forbid"}

    id: str = Field(description="Extraction event id; joins to the action label")
    frame: int = Field(ge=0, description="0-based frame index into the frame cache")
    candidates: list[ActorCandidate] = Field(
        default_factory=list,
        description=(
            "Tracklets with a box on the EVENT frame, in a stable order. "
            "Membership is decided at offset 0; the other offsets only add "
            "history for a player already established as present."
        ),
    )
    target_kind: ActorTargetKind
    target: int | None = Field(
        default=None,
        ge=0,
        description="Index into candidates; set only when target_kind is 'track'",
    )


class ActorCandidateRecord(BaseModel):
    """One video's worth of actor supervision — a ``*_actor_candidates.jsonl``."""

    model_config = {"extra": "allow"}

    video: str = Field(description="Video stem; matches the action label record")
    events: list[ActorCandidateEvent] = Field(default_factory=list)


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
