"""resolve_track: which stored detection IS the picked tracklet's player."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from yp_video.extraction import links
from yp_video.tracklets.geometry import TrackRef

# A player standing behind a teammate: their own tight box, and the
# teammate's larger box in front that encloses everything visible of them.
BEHIND = [100.0, 40.0, 140.0, 160.0]
IN_FRONT = [80.0, 20.0, 160.0, 220.0]


class _Index:
    def __init__(self, tracklet: dict):
        self._tracklet = tracklet

    def tracklet(self, _ref):
        return self._tracklet


def _resolve(detections: list[dict]):
    tracklet = {"frames": [50], "boxes": [BEHIND]}
    mask = np.ones((96, 48), dtype=bool)  # the whole tracklet box is them
    record = {"frame": 50, "detections": detections}
    with tempfile.TemporaryDirectory() as tmp:
        tracks = Path(tmp) / "v_tracks.jsonl"
        tracks.write_text("")
        with (
            patch.object(links, "tracks_path", return_value=tracks),
            patch.object(links, "tracklet_index", return_value=_Index(tracklet)),
            patch.object(links, "_mask_at", return_value=mask),
        ):
            return links.resolve_track("v", record, TrackRef(1, 3))


class ResolveTrackTests(unittest.TestCase):
    def test_the_occluder_in_front_never_stands_in_for_the_player_behind(self):
        pick = _resolve([
            {"box": IN_FRONT, "score": 0.95},
            {"box": BEHIND, "score": 0.40},
        ])
        self.assertEqual(pick.box, tuple(BEHIND))
        self.assertTrue(pick.snap)

    def test_only_the_occluder_detected_keeps_the_track_box_and_vetoes_snapping(self):
        pick = _resolve([{"box": IN_FRONT, "score": 0.95}])
        self.assertEqual(pick.box, tuple(BEHIND))
        self.assertFalse(pick.snap)

    def test_among_duplicates_of_the_player_the_confident_one_wins(self):
        near = [101.0, 41.0, 141.0, 158.0]
        pick = _resolve([
            {"box": BEHIND, "score": 0.30},
            {"box": near, "score": 0.90},
        ])
        self.assertEqual(pick.box, tuple(near))


if __name__ == "__main__":
    unittest.main()
