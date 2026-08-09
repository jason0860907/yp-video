"""The train request contracts hold the invariant the frontend forms rely on.

The web UI builds each config form from contracts/*_request.schema.json:
`buildDefaults` in lib/schemaForm.ts turns a schema into the initial form
values (default ?? const ?? [] for arrays ?? null when nullable; anything
else must come from a page seed), and submit posts that object verbatim.
These tests mirror that construction in Python and require the result to
validate against the model — if a field ever needs a value the schema can't
supply, this fails before the UI does.
"""

from __future__ import annotations

import unittest

from pydantic import BaseModel, ValidationError

from yp_video.web.make_train_schemas import _SCHEMAS
from yp_video.web.train_requests import (
    AnnotationActionTrainRequest,
    AssociationTrainRequest,
    RallyTrainRequest,
    ReidTrainRequest,
)

#: Page-provided seeds for fields with neither default nor const, keyed the
#: way the pages seed them (see useSchemaForm call sites).
SEEDS: dict[type[BaseModel], dict[str, object]] = {
    ReidTrainRequest: {"dataset": "scratch_dataset"},
}


def build_defaults(model: type[BaseModel], seed: dict[str, object] | None = None) -> dict:
    """Python mirror of lib/schemaForm.ts buildDefaults."""
    schema = model.model_json_schema()
    values: dict[str, object] = {}
    for name, prop in schema["properties"].items():
        merged = dict(prop)
        nullable = False
        if "anyOf" in prop:
            branches = [b for b in prop["anyOf"] if b.get("type") != "null"]
            nullable = len(branches) < len(prop["anyOf"])
            merged = {**branches[0], **prop}
        if seed and name in seed:
            values[name] = seed[name]
        elif "default" in merged:
            values[name] = merged["default"]
        elif "const" in merged:
            values[name] = merged["const"]
        elif merged.get("type") == "array":
            values[name] = []
        elif nullable:
            values[name] = None
        else:
            raise AssertionError(f"{model.__name__}.{name} needs a seed value")
    return values


class DefaultsAreValidRequests(unittest.TestCase):
    def test_every_emitted_schema_round_trips_its_defaults(self) -> None:
        for filename, model in _SCHEMAS.items():
            with self.subTest(schema=filename):
                seed = SEEDS.get(model)
                if model is AssociationTrainRequest:
                    # min_length=1 lists are the page's job; the form starts
                    # empty and fills them from the Done split at submit.
                    seed = {
                        "train_videos": ["train_video"],
                        "val_videos": ["val_video"],
                    }
                payload = build_defaults(model, seed)
                model.model_validate(payload)

    def test_action_holdout_submission_shape(self) -> None:
        payload = build_defaults(
            AnnotationActionTrainRequest,
            {"training_mode": "holdout"},
        )
        payload["holdout_videos"] = ["match_actions.jsonl"]
        req = AnnotationActionTrainRequest.model_validate(payload)
        self.assertEqual(req.source, "action_annotations")
        self.assertEqual(req.dataset, "yp_actions")

    def test_out_of_range_field_is_the_only_error(self) -> None:
        payload = build_defaults(RallyTrainRequest)
        payload["gpu"] = -1
        with self.assertRaises(ValidationError) as ctx:
            RallyTrainRequest.model_validate(payload)
        errors = ctx.exception.errors()
        self.assertEqual([e["loc"] for e in errors], [("gpu",)])

    def test_unknown_field_is_rejected(self) -> None:
        payload = build_defaults(RallyTrainRequest)
        payload["btach_size"] = 8
        with self.assertRaises(ValidationError):
            RallyTrainRequest.model_validate(payload)


if __name__ == "__main__":
    unittest.main()
