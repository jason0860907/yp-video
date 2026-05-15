"""Generate ``contracts/detector.schema.json`` from the Pydantic models.

Run after editing ``detector.py``::

    python -m yp_video.contracts.make_schema

The emitted schema is the source of truth for both the backend wire format
and the iOS Swift codegen. Do not edit the JSON by hand.
"""

import json
from pathlib import Path

from pydantic.json_schema import GenerateJsonSchema, models_json_schema

from .detector import (
    DetectorInput,
    ErrorPayload,
    ErrorResult,
    Rally,
    SuccessResult,
)

SCHEMA_VERSION = "2.0.0"

# Only BaseModel classes — the referenced enums (CameraAngle, VideoQuality,
# ErrorCode) land in $defs automatically.
_MODELS = [
    ErrorPayload,
    ErrorResult,
    Rally,
    DetectorInput,
    SuccessResult,
]


def build_schema() -> dict:
    _, schema = models_json_schema(
        [(m, "validation") for m in _MODELS],
        ref_template="#/$defs/{model}",
        schema_generator=GenerateJsonSchema,
    )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "VolleyIQ Rally Detector Wire Contract",
        "version": SCHEMA_VERSION,
        "description": (
            "Source of truth for the request/response shape between the iOS "
            "app and any rally-detection backend. Generated from "
            "yp_video/contracts/detector.py via make_schema.py — do not edit "
            "by hand."
        ),
        "$defs": schema["$defs"],
    }


def _output_path() -> Path:
    # src/yp_video/contracts/make_schema.py -> repo root is parents[3]
    return Path(__file__).resolve().parents[3] / "contracts" / "detector.schema.json"


def main() -> None:
    out = _output_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build_schema(), indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
