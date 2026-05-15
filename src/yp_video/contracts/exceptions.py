"""Typed exceptions that carry a wire ErrorCode and a retryable flag.

Inference code raises these; the worker translates a caught DetectorError
straight into an ErrorPayload without re-classifying.
"""

from .detector import ErrorCode, ErrorPayload


class DetectorError(Exception):
    """Base class for failures with a wire-reportable code.

    Each subclass binds a default ``code`` and ``retryable``; callers may
    override ``retryable`` per-instance (e.g. a GPU OOM during inference is
    retryable even though a generic inference crash is not).
    """

    code: ErrorCode = ErrorCode.internal_error
    retryable: bool = False

    def __init__(self, message: str, *, retryable: bool | None = None) -> None:
        super().__init__(message)
        self.message = message
        if retryable is not None:
            self.retryable = retryable

    def to_payload(self) -> ErrorPayload:
        return ErrorPayload(
            code=self.code, message=self.message, retryable=self.retryable
        )


class InvalidInputError(DetectorError):
    code = ErrorCode.invalid_input
    retryable = False


class DownloadError(DetectorError):
    code = ErrorCode.download_failed
    retryable = True


class ExtractionError(DetectorError):
    code = ErrorCode.extraction_failed
    retryable = True


class ModelInferenceError(DetectorError):
    code = ErrorCode.model_inference_error
    retryable = False


class UploadError(DetectorError):
    code = ErrorCode.upload_failed
    retryable = True
