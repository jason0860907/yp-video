"""The person-box sidecar: every rally-span frame, the tracker's boxes on
it normalized to the frame, and nothing outside the spans."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from yp_video.actor.person_labels import _frame_boxes


class FrameBoxTests(unittest.TestCase):
    def test_spans_pick_the_frames_and_boxes_normalize(self) -> None:
        data = SimpleNamespace(
            meta={"frame_size": [1000, 500]},
            records=[
                {"rally_id": 1, "track_id": 1, "frames": [10, 11, 12, 40], "boxes": [[100, 50, 200, 250]] * 4},
                {"rally_id": 1, "track_id": 2, "frames": [11], "boxes": [[600, 0, 700, 100]]},
            ],
        )
        # One rally 1.0–1.2 s at 10 fps → frames 10..12; frame 40 is outside.
        frames, counts, boxes = _frame_boxes(
            data, ann_path_rows=[{"start": 1.0, "end": 1.2}], fps=10.0, num_frames=100
        )
        self.assertEqual(frames.tolist(), [10, 11, 12])
        self.assertEqual(counts.tolist(), [1, 2, 1])
        self.assertEqual(boxes.shape, (4, 4))
        np.testing.assert_allclose(boxes[0].astype(np.float32), [0.1, 0.1, 0.2, 0.5], atol=1e-3)
        np.testing.assert_allclose(boxes[2].astype(np.float32), [0.6, 0.0, 0.7, 0.2], atol=1e-3)

    def test_empty_span_frames_are_kept_as_nobody(self) -> None:
        data = SimpleNamespace(meta={"frame_size": [100, 100]}, records=[])
        frames, counts, boxes = _frame_boxes(
            data, ann_path_rows=[{"start": 0.0, "end": 0.2}], fps=10.0, num_frames=100
        )
        self.assertEqual(frames.tolist(), [0, 1, 2])
        self.assertEqual(counts.tolist(), [0, 0, 0])
        self.assertEqual(boxes.shape, (0, 4))


if __name__ == "__main__":
    unittest.main()
