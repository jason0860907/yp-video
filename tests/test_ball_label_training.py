"""By default the training snapshot keeps SPOT on its six classes: ground and
antenna train as score, net not at all. ``separate_ball_labels`` keeps all nine."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from yp_video.action import training_labels
from yp_video.contracts.action import RECIPES
from yp_video.core.jsonl import write_jsonl
from yp_video.web import spot_training
from yp_video.web.label_sources import PreparedLabels
from yp_video.web.train_requests import FusionTrainRequest


def _written_labels(**options) -> list[tuple[str, str]]:
    with tempfile.TemporaryDirectory() as raw_dir:
        root = Path(raw_dir)
        label = root / "match_actions.jsonl"
        video = root / "match.mp4"
        video.touch()
        write_jsonl(
            label,
            {"video": "match", "num_frames": 300, "fps": 30},
            [
                {"id": "a", "frame": 30, "label": "serve"},
                {"id": "b", "frame": 40, "label": "net"},
                {"id": "c", "frame": 60, "label": "ground"},
                {"id": "d", "frame": 90, "label": "score"},
                {"id": "e", "frame": 120, "label": "antenna"},
            ],
        )
        with (
            patch.object(training_labels, "inspect_action_frame_cache", return_value={"frame_count": 300}),
            patch.object(training_labels, "cut_kind_of", return_value="sideline"),
            patch.object(training_labels, "load_rallies", return_value=[]),
            patch.object(training_labels, "rally_match_span", return_value=None),
        ):
            training_labels.prepare_action_training_labels(
                items=[(label, video)], frame_dir=root / "frames", save_dir=root / "run", **options
            )
        written = next((root / "run" / "labels").rglob("match_actions.jsonl"))
        rows = [json.loads(line) for line in written.read_text().splitlines()[1:]]
        return [(r["id"], r["label"]) for r in rows]


class BallLabelTrainingTests(unittest.TestCase):
    def test_ground_trains_as_score_and_net_is_dropped(self) -> None:
        self.assertEqual(
            _written_labels(),
            [("a", "serve"), ("c", "score"), ("d", "score"), ("e", "score")],
        )

    def test_separate_ball_labels_keeps_every_label(self) -> None:
        self.assertEqual(
            _written_labels(separate_ball_labels=True),
            [("a", "serve"), ("b", "net"), ("c", "ground"), ("d", "score"), ("e", "antenna")],
        )

    def test_only_recipes_with_action_take_the_option(self) -> None:
        req = FusionTrainRequest(
            recipe="action_rally_winner_actor", validation="ratio", separate_ball_labels=True
        )
        self.assertTrue(req.separate_ball_labels)
        self.assertFalse(FusionTrainRequest(recipe="action", validation="ratio").separate_ball_labels)
        with self.assertRaises(ValueError):
            FusionTrainRequest(recipe="rally_winner", validation="ratio", separate_ball_labels=True)

    def test_separate_ball_labels_score_the_training_folding(self) -> None:
        recipe = RECIPES["action_actor"]
        prepared = PreparedLabels(
            label_dirs={"action": Path("/a")},
            label_subdirs=(), frame_dir=Path("/frames"), dataset="yp_action",
        )

        def eval_label_as(**fields) -> list[str]:
            req = FusionTrainRequest(recipe=recipe.id, validation="ratio", **fields)
            cmd = spot_training.build_command(
                req, recipe, prepared, save_dir=Path("/run"), init_checkpoint=None, audio_dir=None,
            )
            return [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "--eval_label_as"]

        self.assertEqual(
            eval_label_as(separate_ball_labels=True),
            [f"{label}={scored}" for label, scored in training_labels.TRAINED_AS.items()],
        )
        self.assertEqual(eval_label_as(), [])


if __name__ == "__main__":
    unittest.main()
