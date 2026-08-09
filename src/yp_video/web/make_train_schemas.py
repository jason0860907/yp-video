"""Emit the train request models as JSON Schemas into ``contracts/``.

Run after editing ``train_requests.py``::

    python -m yp_video.web.make_train_schemas

One model per file, the model itself as the schema root, so
``npm run gen:types`` (json2ts) produces a named interface per request and
the frontend forms can read defaults, bounds, enum options and field
descriptions straight from the schema. Do not edit the JSON by hand.

Only ``AnnotationActionTrainRequest`` is emitted for action training — the
UI never builds a VNL request.
"""

import json
from pathlib import Path

from yp_video.web.train_requests import (
    AnnotationActionTrainRequest,
    AssociationTrainRequest,
    FusionTrainRequest,
    RallyTrainRequest,
    ReidExportRequest,
    ReidTrainRequest,
)

_SCHEMAS = {
    "spot_train_request.schema.json": RallyTrainRequest,
    "action_train_request.schema.json": AnnotationActionTrainRequest,
    "fusion_train_request.schema.json": FusionTrainRequest,
    "association_train_request.schema.json": AssociationTrainRequest,
    "reid_export_request.schema.json": ReidExportRequest,
    "reid_train_request.schema.json": ReidTrainRequest,
}


def main() -> None:
    contracts_dir = Path(__file__).resolve().parents[3] / "contracts"
    contracts_dir.mkdir(parents=True, exist_ok=True)
    for name, model in _SCHEMAS.items():
        payload = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$comment": (
                "Generated from yp_video/web/train_requests.py via "
                "make_train_schemas.py — do not edit by hand."
            ),
            **model.model_json_schema(ref_template="#/$defs/{model}"),
        }
        out = contracts_dir / name
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
