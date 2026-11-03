"""The 2XLarge dense pass read frame by frame: every covered frame, its boxes
above the cut, and strode frames filled from their neighbour."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from yp_video.person.dense import DensePass


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
        dense = DensePass(self.path)
        self.assertEqual(dense.fps, 60.0)
        boxes, scores = dense.at(10, min_score=.4)
        np.testing.assert_allclose(boxes, [[.1, .1, .2, .5]], atol=1e-3)
        self.assertEqual(len(scores), 1)
        self.assertEqual(dense.at(11, min_score=.4), ([], []))
        self.assertIsNone(dense.at(12, min_score=.4))

    def test_a_strode_frame_takes_the_previous_boxes_only_inside_a_span(self) -> None:
        # Span one: 10, 12, 14; span two starts at 40.
        write_dense(self.path, frames=[10, 12, 14, 40], counts=[1, 0, 1, 1],
                    boxes=[[.1, .1, .2, .2], [.3, .3, .4, .4], [.5, .5, .6, .6]],
                    scores=[.9, .9, .9], stride=2)
        dense = DensePass(self.path)
        covered = dense.people(range(9, 42), min_score=.4)
        self.assertEqual(sorted(covered), [10, 11, 12, 13, 14, 40])
        self.assertEqual(covered[11], covered[10])
        self.assertEqual(covered[13], [])


if __name__ == "__main__":
    unittest.main()
