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
