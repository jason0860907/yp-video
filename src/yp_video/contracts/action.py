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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

# Bump on ANY breaking change to the label record, frame layout, or label set.
ACTION_CONTRACT_VERSION = "2.0.0"

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


def event_id(event: Mapping) -> str:
    """The stable id every stage joins an action event on.

    Raw label records carry only a frame, so the id is derived — ``f<frame>``
    — and stages that do carry an explicit ``id`` keep it. Deriving it in one
    place matters: extraction records, actor candidates and the ReID exporter
    all key on this string, and a stage that spelled it differently (or let it
    fall through to ``str(None)``) would silently join against nothing.
    """
    explicit = event.get("id")
    if explicit:
        return str(explicit)
    return f"f{int(event['frame'])}"


# ── Label files ───────────────────────────────────────────────────
# Per-video label files are JSONL with a ``_meta`` header line followed by one
# record per video (see yp_video.core.jsonl).
DEFAULT_FPS = 30.0


# ── Tasks ─────────────────────────────────────────────────────────
# Every head a SPOT training run can learn, declared once. Both repos derive
# from this table: yp-video builds the label snapshot (one ``labels/<subdir>``
# per distinct ``label_subdir``), the ``--tasks`` argument and the package
# manifest from it; yp-spot builds heads, the loss sum, per-task metrics and
# inference from the same names. A checkpoint's ``config.json["tasks"]`` and
# ``manifest.json["tasks"]`` carry the list, so a predict surface asks "does
# this package serve task X" instead of matching on a package type string.
@dataclass(frozen=True)
class TaskSpec:
    name: str
    #: UI label.
    label: str
    #: ``segment``/``point`` is THE classification head of a run (rally spans
    #: vs. action frames — exactly one per run); ``aux`` heads ride on it.
    kind: Literal["segment", "point", "aux"]
    #: ``labels/<subdir>`` in a run and its package; the file glob inside it.
    label_subdir: str
    label_glob: str
    #: Label-event keys that must be present for this head to be supervised.
    #: Empty for actor: its supervision is a sidecar file, not an event key.
    event_fields: tuple[str, ...]
    #: Heads this one cannot exist without (the model wires them together).
    requires: tuple[str, ...]
    #: Validation metric that picks this task's best epoch (``criterion=map``).
    primary_metric: str
    #: Whether a predict surface can load this head on its own — only then
    #: does the package carry its own best-epoch weights file.
    serveable: bool
    loss_weight: float = 1.0


TASKS: dict[str, TaskSpec] = {
    spec.name: spec
    for spec in (
        TaskSpec(
            "rally", "Rally", "segment", "rally-annotations", "*_rally.jsonl",
            ("frame", "end_frame"), (), "segment_mAP", True,
        ),
        TaskSpec(
            "winner", "Winner", "aux", "rally-annotations", "*_rally.jsonl",
            ("winner",), ("rally",), "winner_top1", True,
        ),
        TaskSpec(
            "action", "Action", "point", "action-annotations", "*_actions.jsonl",
            ("frame", "label"), (), "harmonic_mAP", True,
        ),
        TaskSpec(
            "location", "Location", "aux", "action-annotations", "*_actions.jsonl",
            ("xy",), ("action",), "spatial_mAP", False,
        ),
        TaskSpec(
            "actor", "Actor", "aux", "actor-candidates", "*_actor_candidates.jsonl",
            (), ("action", "location"), "player_top1", True,
        ),
    )
}

SPOTTING_KINDS = ("segment", "point")


def spotting_task(tasks: Sequence[str]) -> str:
    """The one classification task of a run."""
    (name,) = [t for t in tasks if TASKS[t].kind in SPOTTING_KINDS]
    return name


def label_subdirs(tasks: Sequence[str]) -> tuple[str, ...]:
    """Distinct ``labels/<subdir>`` a run with these tasks snapshots, in task order."""
    return tuple(dict.fromkeys(TASKS[t].label_subdir for t in tasks))


def validate_tasks(tasks: Sequence[str]) -> tuple[str, ...]:
    """Fail loud on an unknown name, a missing dependency, or 0/2+ spotting tasks.

    The single-spotting-task rule is what keeps rally and action out of one
    run until the trainer has a second classification head and a second
    frame source; the registry itself can already express that recipe.
    """
    names = tuple(tasks)
    unknown = [t for t in names if t not in TASKS]
    if unknown:
        raise ValueError(f"Unknown task(s) {unknown}; known: {sorted(TASKS)}")
    for t in names:
        missing = [r for r in TASKS[t].requires if r not in names]
        if missing:
            raise ValueError(f"Task {t!r} requires {missing}")
    spotting = [t for t in names if TASKS[t].kind in SPOTTING_KINDS]
    if len(spotting) != 1:
        raise ValueError(
            f"A run needs exactly one segment/point task, got {spotting or 'none'}"
        )
    return names


@dataclass(frozen=True)
class Recipe:
    """A named task set the Fusion Train page offers."""

    id: str
    name: str
    tasks: tuple[str, ...]
    description: str
    #: Request fields the UI shows for this recipe (on top of the common ones).
    fields: tuple[str, ...]
    #: Request defaults the form resets to when this recipe is picked.
    defaults: Mapping[str, object]


_RALLY_FIELDS = ("sample_fps", "video_limit")
_RALLY_DEFAULTS = {
    "batch_size": 8, "acc_grad_iter": 1, "num_epochs": 30,
    "warm_up_epochs": 2, "learning_rate": 3e-4, "audio_backend": "none",
    "sample_fps": 5.0,
}
_ACTION_FIELDS = ("sample_fps", "acc_grad_iter", "audio_backend", "include_predictions")
_ACTION_DEFAULTS = {
    "batch_size": 32, "acc_grad_iter": 4, "num_epochs": 100,
    "warm_up_epochs": 3, "learning_rate": 1e-5, "audio_backend": "logmel",
}
_FUSION_DEFAULTS = {
    "batch_size": 8, "acc_grad_iter": 1, "num_epochs": 50,
    "warm_up_epochs": 3, "learning_rate": 3e-5, "audio_backend": "logmel",
}

RECIPES: dict[str, Recipe] = {
    recipe.id: recipe
    for recipe in (
        Recipe(
            "rally", "Rally", ("rally",),
            "Rally on/off segments from the rally annotations.",
            _RALLY_FIELDS, _RALLY_DEFAULTS,
        ),
        Recipe(
            "rally_winner", "Rally + Winner", ("rally", "winner"),
            "Rally segments plus which court side won each rally.",
            _RALLY_FIELDS, _RALLY_DEFAULTS,
        ),
        Recipe(
            "action", "Action", ("action", "location"),
            "Touch spotting with the contact-point location head.",
            _ACTION_FIELDS, _ACTION_DEFAULTS,
        ),
        Recipe(
            "association_action", "Association + Action",
            ("action", "location", "actor"),
            "Touch spotting plus which player acted, from the actor-candidate sidecar.",
            _ACTION_FIELDS + ("dataset_scope",), _FUSION_DEFAULTS,
        ),
    )
}

for _recipe in RECIPES.values():
    validate_tasks(_recipe.tasks)
del _recipe

# Derived so there is one truth for the file layout.
LABEL_FILE_GLOB = TASKS["action"].label_glob
LABEL_FILE_SUFFIX = LABEL_FILE_GLOB.removeprefix("*")
RALLY_LABEL_FILE_GLOB = TASKS["rally"].label_glob
RALLY_LABEL_FILE_SUFFIX = RALLY_LABEL_FILE_GLOB.removeprefix("*")


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


class CourtSide(str, Enum):
    """Where a court side sits in camera-frame terms.

    The value space of the ``winner`` task: which side of the frame the team
    that WON the rally was playing on. Sideline footage uses left/right,
    broadcast/baseline footage near/far — one 4-class vocabulary so a single
    head serves both camera setups. The winning side, not where the ball
    landed: an out ball lands on the loser's side.
    """

    left = "left"
    right = "right"
    near = "near"
    far = "far"


# Index order matters to the model: 0/1 are horizontal mirrors of each other
# (training flips them together with the frames), 2/3 are flip-invariant.
COURT_SIDES_ORDERED = tuple(side.value for side in CourtSide)
COURT_SIDES = frozenset(COURT_SIDES_ORDERED)

#: Seconds at the END of a rally that carry the winner supervision, and the
#: window inference aggregates over. The outcome is only visible around the
#: final play; frames earlier in the rally cannot know who will win.
WINNER_TAIL_S = 5.0


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
    winner: CourtSide | None = Field(
        default=None,
        description=(
            "Court side the rally winner was playing on (camera-frame). "
            "None = unannotated; the model ignores the span's winner "
            "supervision."
        ),
    )


# ── Actor candidates (who performed each action) ──────────────────
# A SEPARATE file from the action labels, and deliberately so. The action
# labels are read by every spotting run over every video; actor supervision
# exists for a handful of videos and carries ~11 boxes per event, so folding it
# in would inflate the file every run reads with data almost none of them use.
ACTOR_FILE_GLOB = TASKS["actor"].label_glob
ACTOR_FILE_SUFFIX = ACTOR_FILE_GLOB.removeprefix("*")
#: Sub-directory of a training run's label snapshot.
ACTOR_LABEL_SUBDIR = TASKS["actor"].label_subdir

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
# The manifest ``type`` a trainer stamps on its exported package. A SPOT
# package (any recipe) is one type; WHICH heads it carries is
# ``manifest["tasks"]``, and every reader — init-checkpoint pickers, predict
# surfaces — asks for the task it needs. The independent association trainer
# exports a different model class, hence its own type.
SPOT_PACKAGE_TYPE = "yp-video-spot-checkpoint"
ASSOCIATION_PACKAGE_TYPE = "yp-video-association-checkpoint"


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

# yp-spot may ALSO stream partial foreground events as inference runs, so the
# consumer can surface results progressively instead of waiting for the final
# ``predictions.json``. One line per inference batch (native frame numbers):
#   ``SPOT_PARTIAL {"cumulative":<bool>,"events":[...]}``
# Dense (rally) runs stream deltas — that batch's newly-settled per-frame
# events, ``{"frame","score"}`` plus ``winner_probs`` on winner-head checkpoints —
# with ``cumulative=false``: the reader accumulates them. Postprocessed
# (action) runs stream the postprocessed events of the whole settled prefix,
# ``{"label","frame","score"}`` plus ``xy``/``visible`` when predicted, with
# ``cumulative=true``: each line REPLACES all previous ones (NMS is only
# stable when re-run over the full prefix). Optional and additive — a yp-spot
# build that never emits it degrades to the all-at-once behaviour. Only the
# prefix is a hard contract.
SPOT_PARTIAL_PREFIX = "SPOT_PARTIAL "
