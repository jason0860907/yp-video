"""core/frames.read_frames returns exactly what one seek per frame returns."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from yp_video.core import frames


def _video(path: Path, count: int) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    rng = np.random.default_rng(0)
    for _ in range(count):
        writer.write(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8))
    writer.release()


def _seek(path: Path, frame: int):
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, image = capture.read()
    capture.release()
    return image if ok else None


class ReadFramesTest(unittest.TestCase):
    def test_matches_a_seek_per_frame_across_runs_gaps_and_the_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v.mp4"
            _video(path, 400)
            # Unsorted with a duplicate, neighbours, a gap past GRAB_GAP,
            # more than one run, and a frame past the end.
            wanted = [5, 3, 3, 4, 40, 41, 200, 399, 420] + list(range(60, 100, 3))
            got = list(frames.read_frames(path, wanted))
            self.assertEqual([f for f, _ in got], sorted(set(wanted)))
            for frame, image in got:
                expected = _seek(path, frame)
                if expected is None:
                    self.assertIsNone(image, frame)
                else:
                    self.assertTrue(np.array_equal(image, expected), frame)

    def test_nothing_requested_reads_nothing(self) -> None:
        self.assertEqual(list(frames.read_frames(Path("missing.mp4"), [])), [])


if __name__ == "__main__":
    unittest.main()
