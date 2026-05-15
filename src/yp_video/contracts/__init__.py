"""Wire contract between the iOS app and any rally-detection backend.

Transport-agnostic on purpose: the same models are used by the self-hosted
4090 worker today and could be reused by any future transport. The JSON
Schema in ``detector.schema.json`` is generated from these models via
``make_schema.py`` and is the source both the backend and the iOS client
codegen against.
"""

from .detector import (
    CameraAngle,
    DetectorInput,
    ErrorCode,
    ErrorPayload,
    ErrorResult,
    Rally,
    SuccessResult,
    VideoQuality,
)
from .exceptions import (
    DetectorError,
    DownloadError,
    ExtractionError,
    InvalidInputError,
    ModelInferenceError,
    UploadError,
)

__all__ = [
    "CameraAngle",
    "DetectorInput",
    "ErrorCode",
    "ErrorPayload",
    "ErrorResult",
    "Rally",
    "SuccessResult",
    "VideoQuality",
    "DetectorError",
    "DownloadError",
    "ExtractionError",
    "InvalidInputError",
    "ModelInferenceError",
    "UploadError",
]
