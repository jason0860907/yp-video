"""The training snapshot keeps SPOT on its six classes while the corpus is
re-reviewed for the ball events: ground trains as score, net not at all."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from yp_video.action import training_labels
from yp_video.core.jsonl import write_jsonl


class BallLabelTrainingTests(unittest.TestCase):
    def test_ground_trains_as_score_and_net_is_dropped(self) -> None:
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
                ],
            )
            with (
                patch.object(training_labels, "inspect_action_frame_cache", return_value={"frame_count": 300}),
                patch.object(training_labels, "cut_kind_of", return_value="sideline"),
                patch.object(training_labels, "load_rallies", return_value=[]),
                patch.object(training_labels, "rally_match_span", return_value=None),
            ):
                training_labels.prepare_action_training_labels(
                    items=[(label, video)], frame_dir=root / "frames", save_dir=root / "run"
                )
            written = next((root / "run" / "labels").rglob("match_actions.jsonl"))
            rows = [json.loads(line) for line in written.read_text().splitlines()[1:]]
            self.assertEqual([(r["id"], r["label"]) for r in rows], [("a", "serve"), ("c", "score"), ("d", "score")])


if __name__ == "__main__":
    unittest.main()
