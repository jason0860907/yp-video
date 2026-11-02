"""Identify's perception: dense RF-DETR Seg + ByteTrack + GTA at its stride,
detection from that pass's sidecar, the actor head among the tracklets — and
never a SPOT pass."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from yp_video.action import spot_pass
from yp_video.ball import track as ball_track
from yp_video.core.frames import VideoGeometry
from yp_video.extraction import court_guides, identify, pipeline
from yp_video.tracklets import tracking


class _StopAfterDetection(Exception):
    pass


class IdentifyPerceptionTests(unittest.TestCase):
    def test_tracks_with_gta_and_detects_from_its_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint_best.pt"
            checkpoint.write_bytes(b"")
            events = [{"frame": 120}, {"frame": 300}]
            with (
                patch.object(pipeline, "load_events", return_value=events),
                patch.object(tracking, "track_video", autospec=True) as track_video,
                patch.object(identify, "_touch_frames", return_value=[120, 300, 330]),
                patch.object(ball_track, "track_video", return_value={}) as track_ball,
                patch.object(spot_pass, "run_spot_pass") as run_spot_pass,
                patch.object(pipeline, "detect_video", side_effect=_StopAfterDetection) as detect_video,
                self.assertRaises(_StopAfterDetection),
            ):
                identify.identify_players(Path(tmp) / "match.mp4", fusion_checkpoint=checkpoint)

        track_video.assert_called_once()
        self.assertEqual(track_video.call_args.kwargs["stride"], identify.TRACKING_STRIDE)
        self.assertEqual(track_video.call_args.kwargs["event_frames"], {120, 300})
        detect_video.assert_called_once()
        run_spot_pass.assert_not_called()
        # The ball's flights end at every touch, the score at 330 included.
        self.assertEqual(track_ball.call_args.args[1], [120, 300, 330])

    def test_picks_actors_among_tracklets(self):
        from yp_video.actor import candidates, person_action

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint_best.pt"
            checkpoint.write_bytes(b"")
            with (
                patch.object(pipeline, "load_events", return_value=[{"frame": 120}]),
                patch.object(tracking, "track_video"),
                patch.object(identify, "_touch_frames", return_value=[120]),
                patch.object(ball_track, "track_video", return_value={}),
                patch.object(pipeline, "detect_video"),
                patch.object(court_guides, "court_guides", return_value=[]),
                patch.object(candidates, "track_paths", return_value={"1:1": {120: [0, 0, 9, 9]}}),
                patch.object(identify, "video_geometry", return_value=VideoGeometry(30.0, 1920, 1080)),
                patch.object(person_action, "build_policy", side_effect=_StopAfterDetection) as build,
                self.assertRaises(_StopAfterDetection),
            ):
                identify.identify_players(Path(tmp) / "match.mp4", fusion_checkpoint=checkpoint)

        self.assertEqual(build.call_args.args[1], checkpoint)
        self.assertEqual(build.call_args.kwargs["tracks"], {"1:1": {120: [0, 0, 9, 9]}})


if __name__ == "__main__":
    unittest.main()
