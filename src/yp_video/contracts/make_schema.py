"""Generate the JSON Schema contracts from the Pydantic models.

Run after editing ``detector.py`` or ``action.py``::

    python -m yp_video.contracts.make_schema

The emitted schemas are the source of truth:
  - ``contracts/detector.schema.json`` — iOS app ⇄ rally-detection backend.
  - ``contracts/action_label.schema.json`` — yp-video ⇄ yp-spot action labels.
Do not edit the JSON by hand.
"""

import json
from dataclasses import asdict
from pathlib import Path

from pydantic.json_schema import GenerateJsonSchema, models_json_schema

from .action import (
    ACTION_CONTRACT_VERSION,
    FRAME_FFMPEG_PATTERN,
    FRAME_GLOB,
    FRAME_HEIGHT,
    FRAME_PY_PATTERN,
    LABEL_FILE_GLOB,
    RECIPES,
    TASKS,
    ActionEvent,
    ActionLabelRecord,
    SegmentLabelEvent,
)
from .detector import (
    DetectorInput,
    ErrorPayload,
    ErrorResult,
    Rally,
    SuccessResult,
)

SCHEMA_VERSION = "1.0.0"

# Only BaseModel classes — the referenced enums (CameraAngle, VideoQuality,
# ErrorCode) and the nested SegmentEvent land in $defs automatically once a
# listed model references them.
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


def build_action_schema() -> dict:
    _, schema = models_json_schema(
        [
            (ActionLabelRecord, "validation"),
            (ActionEvent, "validation"),
            (SegmentLabelEvent, "validation"),
        ],
        ref_template="#/$defs/{model}",
        schema_generator=GenerateJsonSchema,
    )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "YP Action-Spotting Label Contract",
        "version": ACTION_CONTRACT_VERSION,
        "description": (
            "Source of truth for the *_actions.jsonl label records and frame "
            "cache layout exchanged between yp-video (producer) and yp-spot "
            "(consumer). Generated from yp_video/contracts/action.py via "
            "make_schema.py — do not edit by hand. yp-spot mirrors these "
            "constants in yp_spot/contract.py under the same version."
        ),
        "x-frame-layout": {
            "height": FRAME_HEIGHT,
            "ffmpeg_pattern": FRAME_FFMPEG_PATTERN,
            "py_pattern": FRAME_PY_PATTERN,
            "glob": FRAME_GLOB,
            "zero_based": True,
        },
        "x-label-file-glob": LABEL_FILE_GLOB,
        # The task registry, so the consumer's mirror and the frontend can be
        # checked against the producer's table.
        "x-tasks": {name: asdict(spec) for name, spec in TASKS.items()},
        "x-recipes": {
            recipe_id: {**asdict(recipe), "defaults": dict(recipe.defaults)}
            for recipe_id, recipe in RECIPES.items()
        },
        "$defs": schema["$defs"],
    }


def _repo_root() -> Path:
    # src/yp_video/contracts/make_schema.py -> repo root is parents[3]
    return Path(__file__).resolve().parents[3]


def main() -> None:
    contracts_dir = _repo_root() / "contracts"
    contracts_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("detector.schema.json", build_schema()),
        ("action_label.schema.json", build_action_schema()),
    ):
        out = contracts_dir / name
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
