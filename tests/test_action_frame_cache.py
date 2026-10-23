"""action/frames: a cut that lives only in R2 is fetched for a cache build
and never for a cache that is already ready."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

from yp_video.action import frames


def _video(path: Path, count: int) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    rng = np.random.default_rng(0)
    for _ in range(count):
        writer.write(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8))
    writer.release()


class R2OnlyCutTest(unittest.TestCase):
    def test_build_fetches_the_cut_once_and_a_ready_cache_does_not(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            remote = tmp / "r2" / "match.mp4"
            remote.parent.mkdir()
            _video(remote, 12)
            cut = tmp / "cuts" / "match.mp4"  # canonical path, bytes absent
            cut.parent.mkdir()
            fetched = []

            @contextmanager
            def fetch(path: Path):
                fetched.append(path.name)
                shutil.copy2(remote, path)
                try:
                    yield path
                finally:
                    path.unlink()

            cache = tmp / "frames"
            built = frames.ensure_action_frame_cache(cut, fetch=fetch, cache_root=cache)
            again = frames.ensure_action_frame_cache(cut, fetch=fetch, cache_root=cache)

            self.assertTrue(built["created"])
            self.assertEqual(built["frame_count"], 12)
            self.assertFalse(again["created"])
            self.assertEqual(fetched, ["match.mp4"])
            self.assertFalse(cut.exists())


if __name__ == "__main__":
    unittest.main()
