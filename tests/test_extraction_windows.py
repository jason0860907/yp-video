"""Tracklet-window embeddings: which views an event averages, and rows that
stay aligned with the records."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from yp_video.extraction import windows


class WindowPickTests(unittest.TestCase):
    def test_spreads_at_most_window_crops_over_the_window(self):
        frames = np.arange(0, 200, 2)  # stride-2 tracklet
        picks = windows.window_picks(frames, 100)
        chosen = frames[picks]
        self.assertEqual(len(picks), windows.WINDOW_CROPS)
        self.assertTrue((np.abs(chosen - 100) <= windows.WINDOW_FRAMES).all())
        self.assertEqual(chosen.min(), 70)
        self.assertEqual(chosen.max(), 130)

    def test_short_window_keeps_every_frame_it_has(self):
        frames = np.array([96, 98, 100])
        self.assertEqual(windows.window_picks(frames, 100), [0, 1, 2])

    def test_tracklet_outside_the_window_falls_to_its_nearest_frame(self):
        frames = np.array([10, 20, 300])
        self.assertEqual(windows.window_picks(frames, 200), [2])


class WindowBoxTests(unittest.TestCase):
    def test_views_per_record(self):
        records = [
            {"id": "linked", "frame": 100, "crop": "a.jpg", "box": [0, 0, 1, 1]},
            {"id": "lone", "frame": 50, "crop": "b.jpg", "box": [0, 0, 9, 9], "actor_box": [1, 2, 3, 4]},
            {"id": "nobody", "frame": 70, "box": [0, 0, 1, 1]},
        ]
        tracks = {"1:3": {"frames": [98, 100, 102], "boxes": [[0, 0, 5, 5], [1, 1, 6, 6], [2, 2, 7, 7]]}}
        views = windows.window_boxes(records, {"linked": "1:3"}, tracks)

        self.assertEqual([f for f, _ in views[0]], [98, 100, 102])
        self.assertEqual(views[1], [(50, (1, 2, 3, 4))])
        self.assertEqual(views[2], [])


class _Embedder:
    def embed_paths(self, paths, *, on_progress=None, **_):
        # Each crop's vector is its own one-hot, so a mean is checkable.
        out = np.zeros((len(paths), 4), dtype=np.float32)
        for i, path in enumerate(paths):
            out[i, int(Path(path).stem)] = 1.0
        return out


class EmbedTests(unittest.TestCase):
    def test_rows_align_with_records_and_average_their_views(self):
        records = [
            {"id": "a", "frame": 100, "crop": "a.jpg", "box": [0, 0, 1, 1]},
            {"id": "skip", "frame": 120, "box": [0, 0, 1, 1]},
            {"id": "b", "frame": 200, "crop": "b.jpg", "box": [0, 0, 1, 1]},
        ]
        tracks = [{"rally_id": 1, "track_id": 1, "frames": [99, 101], "boxes": [[0, 0, 5, 5], [0, 0, 6, 6]]}]
        crops = {(99, (0, 0, 5, 5)): Path("0.jpg"), (101, (0, 0, 6, 6)): Path("1.jpg"),
                 (200, (0, 0, 1, 1)): Path("2.jpg")}
        saved = {}
        with (
            patch.object(windows, "read_jsonl", return_value=({}, records)),
            patch.object(windows, "tracklet_data", return_value=type("T", (), {"records": tracks})()),
            patch.object(windows, "track_keys", return_value={"a": "1:1"}),
            patch.object(windows, "_cut_masked", return_value=crops),
            patch.object(windows, "build_embedders", return_value={windows.BASE_EMBEDDER: _Embedder()}),
            patch.object(windows, "save_embedding_matrix", side_effect=lambda s, m, x: saved.update(m=x)),
            patch.object(windows, "clear_embedding_refreshes"),
        ):
            result = windows.embed_tracklet_windows("v", Path("v.mp4"))

        matrix = saved["m"]
        self.assertEqual(result, {"models": [windows.WINDOWED_EMBEDDER], "crops": 3})
        self.assertEqual(matrix.shape, (3, 4))
        np.testing.assert_allclose(matrix[0], [2 ** -0.5, 2 ** -0.5, 0, 0], rtol=1e-5)
        self.assertTrue(np.isnan(matrix[1]).all())
        np.testing.assert_allclose(matrix[2], [0, 0, 1, 0])


if __name__ == "__main__":
    unittest.main()
