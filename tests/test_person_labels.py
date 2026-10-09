"""The person-box sidecar from the 2XLarge dense pass: every covered frame,
its boxes above the label cut, and strode frames filled from their neighbour."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from yp_video.actor.person_labels import dense_frame_boxes


def write_dense(path: Path, *, frames, counts, boxes, scores, stride) -> None:
    np.savez_compressed(
        path,
        frames=np.asarray(frames, np.int32),
        counts=np.asarray(counts, np.int32),
        boxes=np.asarray(boxes, np.float16).reshape(-1, 4),
        scores=np.asarray(scores, np.float16),
        meta=np.array(json.dumps({"fps": 60.0, "stride": stride})),
    )


class DenseFrameBoxTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "game_dense.npz"

    def tearDown(self) -> None:
        self.dir.cleanup()

    def test_the_cut_drops_low_scores_and_keeps_empty_frames(self) -> None:
        write_dense(self.path, frames=[10, 11], counts=[2, 1],
                    boxes=[[.1, .1, .2, .5], [.6, 0, .7, .2], [.3, .3, .4, .6]],
                    scores=[.9, .2, .3], stride=1)
        fps, labels = dense_frame_boxes(self.path, min_score=.4)
        self.assertEqual(fps, 60.0)
        self.assertEqual(sorted(labels), [10, 11])
        np.testing.assert_allclose(labels[10], [[.1, .1, .2, .5]], atol=1e-3)
        self.assertEqual(labels[11], [])

    def test_a_strode_frame_takes_the_previous_boxes_only_inside_a_span(self) -> None:
        # Span one: 10, 12, 14; span two starts at 40.
        write_dense(self.path, frames=[10, 12, 14, 40], counts=[1, 0, 1, 1],
                    boxes=[[.1, .1, .2, .2], [.3, .3, .4, .4], [.5, .5, .6, .6]],
                    scores=[.9, .9, .9], stride=2)
        _, labels = dense_frame_boxes(self.path, min_score=.4)
        self.assertEqual(sorted(labels), [10, 11, 12, 13, 14, 40])
        self.assertEqual(labels[11], labels[10])
        self.assertEqual(labels[13], [])


if __name__ == "__main__":
    unittest.main()
