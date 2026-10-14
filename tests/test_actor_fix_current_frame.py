"""An actor fix works on the action's current frame, not extraction's copy."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from yp_video.actor.labels import ActorLabel, ActorVerdict
from yp_video.core.jsonl import read_jsonl, write_jsonl
from yp_video.extraction import pipeline, store


class CurrentFrameTests(unittest.TestCase):
    def test_a_fix_after_an_action_moved_refreshes_the_stale_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            records, actions = Path(tmp) / "records.jsonl", Path(tmp) / "actions.jsonl"
            # Extraction ran when the event sat on frame 1902; review moved it to 1901.
            write_jsonl(records, {"frame_size": [1920, 1080]},
                        [{"id": "e1", "frame": 1902, "label": "receive", "box": [0, 0, 10, 10],
                          "resolution": "auto"}])
            write_jsonl(actions, {}, [{"id": "e1", "frame": 1901, "label": "receive"}])
            with (
                patch.object(pipeline, "records_path", return_value=records),
                patch.object(store, "action_annotation_path", return_value=actions),
            ):
                record = pipeline._apply_actor_fix(
                    "v", "unused.mp4", "e1", ActorLabel(ActorVerdict.OCCLUDED), []
                )
            self.assertEqual(record["frame"], 1901)
            self.assertEqual(read_jsonl(records)[1][0]["frame"], 1901)


if __name__ == "__main__":
    unittest.main()
