"""Pseudo-label training: pre-annotations join the corpus, never validation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from yp_video.action import training
from yp_video.core.jsonl import write_jsonl
from yp_video.web.train_requests import FusionTrainRequest


def _write_label(path: Path, stem: str) -> None:
    write_jsonl(
        path,
        {"video": stem, "num_frames": 100, "fps": 30},
        [{"id": "event", "frame": 10, "label": "spike"}],
    )


class LabelItemsWithPredictionsTests(unittest.TestCase):
    def test_predictions_fill_only_videos_without_human_labels(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            annotations = root / "annotations"
            pre_annotations = root / "pre-annotations"
            annotations.mkdir()
            pre_annotations.mkdir()

            _write_label(annotations / "human_actions.jsonl", "human")
            # Same video also has a stale prediction: the human file must win.
            _write_label(pre_annotations / "human_actions.jsonl", "human")
            _write_label(pre_annotations / "pseudo_actions.jsonl", "pseudo")

            cuts = {name: root / name for name in ("human.mp4", "pseudo.mp4")}
            with (
                patch.object(training, "ACTION_ANNOTATIONS_DIR", annotations),
                patch.object(training, "ACTION_PRE_ANNOTATIONS_DIR", pre_annotations),
                patch.object(training, "find_cut", cuts.get),
            ):
                default_items = training.label_items()
                items = training.label_items(include_predictions=True)
                pseudo_stems = training.prediction_label_stems(items)
                default_stems = training.prediction_label_stems(default_items)

            self.assertEqual(
                [path for path, _video in default_items],
                [annotations / "human_actions.jsonl"],
            )
            self.assertEqual(
                [path for path, _video in items],
                [
                    annotations / "human_actions.jsonl",
                    pre_annotations / "pseudo_actions.jsonl",
                ],
            )
            self.assertEqual(pseudo_stems, {"pseudo"})
            self.assertEqual(default_stems, set())


class RequestValidationTests(unittest.TestCase):
    def test_predictions_require_manual_validation(self) -> None:
        with self.assertRaises(ValidationError):
            FusionTrainRequest(
                recipe="action", validation="ratio", include_predictions=True
            )
        request = FusionTrainRequest(
            recipe="action",
            validation="manual",
            validation_videos=["match"],
            include_predictions=True,
        )
        self.assertTrue(request.include_predictions)

    def test_fusion_predictions_also_need_partial_scope(self) -> None:
        with self.assertRaises(ValidationError):
            FusionTrainRequest(
                dataset_scope="joint_only",
                validation="manual",
                validation_videos=["match"],
                include_predictions=True,
            )
        request = FusionTrainRequest(
            dataset_scope="partial_labels",
            validation="manual",
            validation_videos=["match"],
            include_predictions=True,
        )
        self.assertTrue(request.include_predictions)

    def test_manual_validation_needs_videos(self) -> None:
        with self.assertRaises(ValidationError):
            FusionTrainRequest(validation="manual", validation_videos=[])


if __name__ == "__main__":
    unittest.main()
