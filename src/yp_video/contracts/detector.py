"""Pydantic models that define the rally-detection wire contract.

Edited by hand; ``detector.schema.json`` is generated from here. Keep field
names in snake_case — they go on the wire as-is and the iOS client decodes
them directly.
"""

from enum import Enum

from pydantic import BaseModel, Field


class CameraAngle(str, Enum):
    phone_sideline = "phoneSideline"
    fixed_venue = "fixedVenue"
    broadcast = "broadcast"


class VideoQuality(str, Enum):
    p360 = "360p"
    p480 = "480p"
    p720 = "720p"
    p1080 = "1080p"


class ErrorCode(str, Enum):
    """Failure categories the backend reports.

    A new code may be added at any time; clients must treat unknown codes as a
    generic failure rather than crashing.
    """

    invalid_input = "INVALID_INPUT"
    download_failed = "DOWNLOAD_FAILED"
    extraction_failed = "EXTRACTION_FAILED"
    model_inference_error = "MODEL_INFERENCE_ERROR"
    upload_failed = "UPLOAD_FAILED"
    internal_error = "INTERNAL_ERROR"


class ErrorPayload(BaseModel):
    code: ErrorCode
    message: str
    retryable: bool = Field(
        description=(
            "Whether the caller can retry this exact request. False for "
            "permanent failures (invalid input, model crash); true for "
            "transient failures (network, ffmpeg flake, R2 hiccup, GPU OOM)."
        )
    )


class ErrorResult(BaseModel):
    """Payload returned when detection fails. Mutually exclusive with SuccessResult."""

    error: ErrorPayload


class Rally(BaseModel):
    index: int = Field(ge=1, description="1-based rally number within the match")
    set: int = Field(ge=1, description="Set number this rally belongs to")
    start: float = Field(ge=0, description="Seconds from video start")
    end: float = Field(ge=0, description="Seconds from video start")
    score: int = Field(ge=0, le=100, description="Highlight score 0-100")


class SegmentEvent(BaseModel):
    """One touch inside an action's build-up (receive / set / spike, …).

    Mirrors ``yp_video.action.segments._public``: a label plus a timestamp,
    with the source frame and normalized court location carried when known.
    Distinct from ``contracts.action.ActionEvent`` (the frame-indexed label
    record yp-spot emits) — this is the seconds-based, app-facing shape.
    """

    label: str = Field(description="Touch label, e.g. receive / set / spike")
    time: float = Field(ge=0, description="Seconds from video start")
    frame: int | None = Field(default=None, ge=0, description="Source frame index, when known")
    xy: list[float] | None = Field(
        default=None, min_length=2, max_length=2,
        description="Normalized [x, y] court location, each in [0, 1]",
    )


class RallyBounds(BaseModel):
    """The rally an action sits in: its 1-based timeline index and span."""

    index: int = Field(ge=1, description="1-based rally number (matches Rally.index)")
    start: float = Field(ge=0, description="Seconds from video start")
    end: float = Field(ge=0, description="Seconds from video start")


class ActionSegment(BaseModel):
    """A per-action highlight segment, anchored on a spike.

    Mode-agnostic: it carries the *structure* around the anchor (build-up
    chain, the rally it sits in, the next action) and leaves the clip-window
    choice to the client. ``player_id`` / ``outcome`` / ``team`` are reserved
    for a future re-id + scoring pass and are null until then, so adding those
    features never breaks this decode.
    """

    action: str = Field(description="Anchor action label (currently always 'spike')")
    anchor: SegmentEvent = Field(description="The anchor action itself")
    chain: list[SegmentEvent] = Field(description="Build-up touches, ending on the anchor")
    rally: RallyBounds | None = Field(default=None, description="Rally bounds, null if unmatched")
    next: SegmentEvent | None = Field(default=None, description="First action after the anchor")
    player_id: str | None = Field(default=None, description="Reserved: filled by a future re-id pass")
    outcome: str | None = Field(default=None, description="Reserved: kill / error / blocked (future scoring pass)")
    team: str | None = Field(default=None, description="Reserved: owning team (future pass)")


class DetectorInput(BaseModel):
    """Detection request posted by the iOS client."""

    model_config = {"extra": "forbid"}

    video_url: str = Field(
        min_length=1, description="HTTP(S) or YouTube URL of the video to analyze"
    )
    camera_angle: CameraAngle = Field(
        default=CameraAngle.phone_sideline,
        description="Camera position; affects model preprocessing.",
    )
    quality: VideoQuality | None = Field(
        default=None,
        description="Optional download quality. Only honoured for YouTube URLs.",
    )
    locale: str | None = Field(
        default=None,
        description="BCP-47 language tag echoed back so the iOS UI can match.",
    )


class SuccessResult(BaseModel):
    """Payload returned when detection succeeds."""

    total_duration: float = Field(ge=0, description="Source video length in seconds")
    rallies: list[Rally]
    action_segments: list[ActionSegment] = Field(
        default_factory=list,
        description=(
            "Per-action (spike) highlight segments for the Action scope. "
            "Additive and may be empty — absent/empty when SPOT action spotting "
            "did not run. The client projects each clip window on demand and "
            "decodes an empty list when the field is missing, so older workers "
            "stay compatible."
        ),
    )
    score_segments: list[ActionSegment] = Field(
        default_factory=list,
        description=(
            "Point-decided (score) segments for the Score scope: each anchors "
            "on a score event and its `chain` is the deciding 接舉打 build-up. "
            "Same shape as `action_segments`; `outcome` (won/lost) stays null "
            "until a scoring pass fills it. Additive and may be empty."
        ),
    )
    action_events: list[SegmentEvent] = Field(
        default_factory=list,
        description=(
            "Flat, time-sorted list of every spotted event (serve / receive / "
            "set / spike / block / score) with seconds-based `time`. Powers the "
            "rally-wide touch timeline, which needs the full action set rather "
            "than just a spike's build-up. Additive and may be empty."
        ),
    )
    locale_echo: str | None = Field(
        default=None, description="Echoes back DetectorInput.locale if provided"
    )
    video_url: str | None = Field(
        default=None,
        description=(
            "Playable URL the iOS client should hand to AVPlayer. Echoes the "
            "input URL for direct HTTP sources; points at R2 for YouTube "
            "imports; null if the worker had to skip the upload."
        ),
    )
