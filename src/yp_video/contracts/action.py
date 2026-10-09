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
fails loud if its compiled-in version differs. Bump the version on a breaking
change to the field layout, frame layout or label set — and update both sides.
Adding an optional field is not one: a reader that predates it never sees it,
and checkpoint packages stamped with the version stay loadable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

# Bump on ANY breaking change to the label record, frame layout, or label set.
ACTION_CONTRACT_VERSION = "3.1.0"

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
    place matters: extraction records, actor labels and the ReID exporter
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
    #: ``segment``/``point`` is a classification head (rally spans vs. action
    #: frames); mixed-FPS runs may carry both. ``aux`` heads ride on one.
    kind: Literal["segment", "point", "aux"]
    #: ``labels/<subdir>`` in a run and its package; the file glob inside it.
    label_subdir: str
    label_glob: str
    #: Label-event keys that must be present for this head to be supervised.
    #: Empty for a task whose supervision is a sidecar file, not an event key.
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
        # L1 on normalized [0, 1] coordinates hands the backbone a
        # gradient bounded by 0.25 per event; the class cross-entropies
        # reach 1. DETR's L1 box weight is 5 for the same reason. The weight covers the xy L1 only;
        # the visibility BCE that rides on this task is unbounded and trains
        # at weight 1 (yp_spot VIS_LOSS_WEIGHT).
        TaskSpec(
            "location", "Location", "aux", "action-annotations", "*_actions.jsonl",
            ("xy",), ("action",), "spatial_mAP", False, loss_weight=5.0,
        ),
        # Which court side the touching player stood on, camera-frame like
        # winner. Labels are mostly derived by the touch-order rules from
        # the previous rally's winner (action/attributes.py), a stored
        # ``side`` overriding; touches the rules cannot place stay unsupervised.
        TaskSpec(
            "side", "Side", "aux", "action-annotations", "*_actions.jsonl",
            ("side",), ("action",), "side_top1", False,
        ),
        # Whether the touching player was off the floor. Spikes and blocks
        # default to airborne and receives to grounded; a stored ``jump``
        # overrides, and serves/sets supervise only when one is stored.
        TaskSpec(
            "jump", "Jump", "aux", "action-annotations", "*_actions.jsonl",
            ("jump",), ("action",), "jump_balanced_accuracy", False,
        ),
        # Where the people are, per frame: the model's own boxes, distilled
        # from RF-DETR Seg 2XLarge (actor/person_labels.py writes the sidecar
        # from the dense pass). Rides the rally stream — rally spans are where
        # the pass ran — so every covered video supervises it, action labels
        # or not.
        TaskSpec(
            "person", "Person", "aux", "person-boxes", "*_person.npz",
            (), ("rally",), "person_ap50", False,
        ),
        # Who touched the ball, per action event: the person/action head
        # scoring the event's candidate boxes (the SPOT pass picks among the
        # person head's own; advanced identify among GTA-refined tracklets).
        # Supervised by the association annotations through
        # yp_spot.person_action.joint, never by Fusion Train.
        TaskSpec(
            "actor", "Actor", "aux", "actor-annotations", "*_actors.json",
            (), ("action", "location", "person"), "actor_hit", False,
        ),
    )
}

SPOTTING_KINDS = ("segment", "point")


def spotting_tasks(tasks: Sequence[str]) -> tuple[str, ...]:
    """Classification tasks carried by a run, in task order."""
    return tuple(t for t in tasks if TASKS[t].kind in SPOTTING_KINDS)


def spotting_task(tasks: Sequence[str]) -> str:
    """The classification task of a single-stream recipe."""
    spotting = spotting_tasks(tasks)
    if len(spotting) != 1:
        raise ValueError(f"Expected one spotting task, got {list(spotting)}")
    (name,) = spotting
    return name


def label_subdirs(tasks: Sequence[str]) -> tuple[str, ...]:
    """Distinct ``labels/<subdir>`` a run with these tasks snapshots, in task order."""
    return tuple(dict.fromkeys(TASKS[t].label_subdir for t in tasks))


def validate_tasks(tasks: Sequence[str]) -> tuple[str, ...]:
    """Fail loud on an unknown name, a missing dependency, or no spotting task."""
    names = tuple(tasks)
    unknown = [t for t in names if t not in TASKS]
    if unknown:
        raise ValueError(f"Unknown task(s) {unknown}; known: {sorted(TASKS)}")
    for t in names:
        missing = [r for r in TASKS[t].requires if r not in names]
        if missing:
            raise ValueError(f"Task {t!r} requires {missing}")
    if not spotting_tasks(names):
        raise ValueError("A run needs at least one segment/point task")
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


_RALLY_FIELDS = ("sample_fps", "acc_grad_iter", "video_limit")
# Every recipe defaults to the same base learning rate (3e-5, matching the
# yp-spot CLI default); per-task overrides start equal to it so the form
# shows one number everywhere until the operator deliberately diverges.
# Every recipe trains at an effective batch of 64 as 32 micro-batches of 2
# clips: at the 224x398 model input (yp_spot DEFAULT_INPUT_SIZE) a 4-clip
# step no longer fits the 24 GB card for either backbone, while 2 clips
# leaves headroom for the five-head runs.
_RALLY_DEFAULTS = {
    "batch_size": 64, "acc_grad_iter": 32, "num_epochs": 30,
    "warm_up_epochs": 2, "learning_rate": 3e-5, "audio_backend": "none",
    "sample_fps": 5.0,
}
_ACTION_FIELDS = (
    "sample_fps",
    "acc_grad_iter",
    "audio_backend",
    "action_dilate_len",
    "include_predictions",
)
_ACTION_DEFAULTS = {
    "batch_size": 64, "acc_grad_iter": 32, "num_epochs": 100,
    "warm_up_epochs": 3, "learning_rate": 3e-5, "audio_backend": "logmel",
    "action_dilate_len": 0,
}
_MULTI_FPS_FIELDS = (
    "action_sample_fps",
    "rally_sample_fps",
    "winner_sample_fps",
    "video_limit",
    "audio_backend",
    "action_learning_rate",
    "rally_learning_rate",
    "winner_learning_rate",
    "action_stream_weight",
    "rally_stream_weight",
    "winner_stream_weight",
    "action_fg_upsample",
    "action_dilate_len",
    "acc_grad_iter",
)
_MULTI_FPS_DEFAULTS = {
    "batch_size": 64,
    "feature_arch": "convnextt_dv3_gsm",
    "num_workers": 4,
    "acc_grad_iter": 32,
    "num_epochs": 50,
    "warm_up_epochs": 3,
    "learning_rate": 3e-5,
    "audio_backend": "logmel",
    "action_learning_rate": 3e-5,
    "rally_learning_rate": 3e-5,
    "winner_learning_rate": 3e-5,
    "action_stream_weight": 1,
    "rally_stream_weight": 1,
    "winner_stream_weight": 1,
    "action_fg_upsample": None,
    "action_dilate_len": 0,
    "action_sample_fps": 30.0,
    "rally_sample_fps": 5.0,
    "winner_sample_fps": 5.0,
    "video_limit": 0,
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
            "action", "Action", ("action", "location", "side", "jump"),
            "Touch spotting with the contact-point, actor side and jump heads.",
            _ACTION_FIELDS, _ACTION_DEFAULTS,
        ),
        Recipe(
            "action_rally_winner",
            "Action + Rally + Winner + Person",
            ("action", "location", "side", "jump", "rally", "winner", "person"),
            "One backbone; Action uses audio and geometry supervision plus the actor's "
            "side and jump, Rally/Winner are visual-only, Person distils the tracker's "
            "boxes on the rally stream.",
            _MULTI_FPS_FIELDS,
            _MULTI_FPS_DEFAULTS,
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

    The value space of the ``winner`` task — which side of the frame the team
    that WON the rally was playing on — and of the ``side`` task, which side
    the player making a touch stood on. Broadcast footage uses left/right,
    sideline (amateur) footage near/far — one 4-class vocabulary so a single
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
    score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Model confidence; present on machine pre-annotations only",
    )
    side: CourtSide | None = Field(
        default=None,
        description=(
            "Court side the touching player stood on (camera-frame). "
            "None = no stored value; training derives one where the rules can."
        ),
    )
    jump: bool | None = Field(
        default=None,
        description=(
            "Whether the touching player was off the floor. None = no stored "
            "value; training falls back to the label's default."
        ),
    )


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


# ── Checkpoint packages ───────────────────────────────────────────
# The manifest ``type`` a trainer stamps on its exported package. A SPOT
# package (any recipe) is one type; WHICH heads it carries is
# ``manifest["tasks"]``, and every reader — init-checkpoint pickers, predict
# surfaces — asks for the task it needs.
SPOT_PACKAGE_TYPE = "yp-video-spot-checkpoint"


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
#   ``SPOT_PARTIAL {"task":<head>,"cumulative":<bool>,"events":[...]}``
# ``task`` names the head the events belong to — one run may carry several,
# and the reader keeps one cumulative list per task; heads are never merged.
# Dense (rally) runs stream deltas — that batch's newly-settled per-frame
# events, ``{"frame","score"}`` plus ``winner_probs`` on winner-head checkpoints —
# with ``cumulative=false``: the reader accumulates them. Postprocessed
# (action) runs stream the postprocessed events of the whole settled prefix,
# ``{"label","frame","score","xy","visible"}`` (plus side / jump when predicted), with
# ``cumulative=true``: each line REPLACES all previous ones (NMS is only
# stable when re-run over the full prefix). Optional and additive — a yp-spot
# build that never emits it degrades to the all-at-once behaviour. Only the
# prefix is a hard contract.
SPOT_PARTIAL_PREFIX = "SPOT_PARTIAL "
