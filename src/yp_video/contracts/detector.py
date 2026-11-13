"""Pydantic models that define the rally-detection wire contract.

Edited by hand; ``detector.schema.json`` is generated from here. Keep field
names in snake_case — they go on the wire as-is and the iOS client decodes
them directly.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class CameraAngle(str, Enum):
    """The App's upload angles (D1 matches.angle, config.USER_ANGLES)."""

    baseline = "baseline"
    corner = "corner"
    high = "high"
    center = "center"


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
    winner: str | None = Field(
        default=None,
        description=(
            "Court side the rally's winner played on, camera-frame: "
            "left/right (broadcast footage) or near/far (sideline). Null when "
            "the checkpoint has no winner head."
        ),
    )


class SegmentEvent(BaseModel):
    """One spotted touch (serve / receive / set / spike / block / score).

    Mirrors ``yp_video.action.segments._public``: a stable event id, label,
    timestamp and source frame, with the ball point and whether it was seen.
    Distinct from ``contracts.action.ActionEvent`` (the frame-indexed label
    record yp-spot emits) — this is the seconds-based, app-facing shape.

    ``id`` is the join key player identification points back at
    (``IdentifyUnit.event_ids``), so it is also what a clip, a per-touch
    player override and a per-player stat are all keyed by.
    """

    id: str = Field(description="Stable extraction id, normally f<source frame>")
    label: str = Field(description="Touch label, e.g. receive / set / spike")
    time: float = Field(ge=0, description="Seconds from video start")
    frame: int = Field(ge=0, description="Source frame index")
    xy: list[float] = Field(
        min_length=2, max_length=2,
        description="Normalized [x, y] ball point at contact in the frame, each in [0, 1]",
    )
    visible: bool = Field(
        description="Whether the ball point was seen; false marks an estimated (off-screen or occluded) point",
    )
    score: float = Field(ge=0, le=1, description="The action head's confidence in this event")


class Attack(BaseModel):
    """組織進攻: one spike plus whatever 接舉 build-up was spotted before it.

    Derived by ``yp_video.action.rules.attacks`` — the rule lives there, not
    in any client."""

    rally_index: int = Field(ge=1, description="Rally the attack happened in")
    event_indices: list[int] = Field(
        min_length=1,
        description=(
            "Positions in `action_events`, in time order: [receive?, set?, "
            "spike]. Missing receive / set are simply absent; the spike is "
            "always last. Positions, not ids: two touches on one frame share "
            "an id."
        ),
    )


class RallyOutcome(BaseModel):
    """How a rally's point was decided. Derived by
    ``yp_video.action.rules.rally_outcomes``."""

    rally_index: int = Field(ge=1)
    serve_point: bool | None = Field(
        description=(
            "The serve decided the point (ace or service error): touches but "
            "no set / spike / block. Null when the rally holds no touch."
        ),
    )
    score_event_index: int | None = Field(
        description=(
            "Position in `action_events` of the rally's final spotted "
            "whistle; null when none was spotted."
        ),
    )
    deciding_event_indices: list[int] = Field(
        description=(
            "Positions in `action_events` of the play that decided the point: "
            "the last attack before the whistle, else the last touch. Empty "
            "when the rally holds none."
        ),
    )
    serving_side: Literal["left", "right", "near", "far"] | None = Field(
        description=(
            "Camera-frame court side the rally was served from (see "
            "`serving_side_from`). Null when nothing names it."
        ),
    )
    serving_side_from: Literal["serve", "previous_winner"] | None = Field(
        description=(
            "What named `serving_side`: `serve` — where the rally's opening "
            "serve was struck; `previous_winner` — the match's serves do not "
            "split by side, so the side that won the set's previous rally "
            "serves. Null exactly when `serving_side` is null."
        ),
    )

    @model_validator(mode="after")
    def side_and_source_agree(self):
        if (self.serving_side is None) != (self.serving_side_from is None):
            raise ValueError("serving_side and serving_side_from are null together")
        return self


class DetectorInput(BaseModel):
    """Detection request posted by the iOS client."""

    model_config = {"extra": "forbid"}

    video_url: str = Field(
        min_length=1, description="HTTP(S) or YouTube URL of the video to analyze"
    )
    camera_angle: CameraAngle = Field(description="Where the camera stood, as picked on upload.")
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
    action_events: list[SegmentEvent] = Field(
        default_factory=list,
        description=(
            "Flat, time-sorted list of every spotted event (serve / receive / "
            "set / spike / block / score) with seconds-based `time`. This is "
            "the entire action payload: the client projects clips, touch "
            "timelines and per-player stats from it, and joins player "
            "identification to it by event `id`. Empty when SPOT action "
            "spotting did not run (a rally-only result)."
        ),
    )
    action_candidates: list[SegmentEvent] = Field(
        description=(
            "Time-sorted events the action head spotted below the operating "
            "threshold — every `action_events` event scores at least 0.15, "
            "these score less. Display-only: rules and player identification "
            "never see them, so no position or id elsewhere points at one. "
            "Developer builds use them to lower a video's threshold."
        ),
    )
    attacks: list[Attack] = Field(
        description="Every 組織進攻 in the match, timeline order.",
    )
    rally_outcomes: list[RallyOutcome] = Field(
        description="One per rally, timeline order.",
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
