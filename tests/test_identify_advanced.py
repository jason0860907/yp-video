"""Advanced identify swaps perception: dense RF-DETR Seg + McByte++ instead
of fusion person boxes + ByteTrack, and never runs the SPOT pass."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from yp_video.action import spot_pass
from yp_video.extraction import identify, pipeline
from yp_video.tracklets import fusion, tracking


class _StopAfterDetection(Exception):
    pass


class AdvancedIdentifyTests(unittest.TestCase):
    def _run(self, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint_best.pt"
            checkpoint.write_bytes(b"")
            events = [{"frame": 120}, {"frame": 300}]
            with (
                patch.object(pipeline, "load_events", return_value=events),
                patch.object(tracking, "track_video", autospec=True) as track_video,
                patch.object(fusion, "track_person_boxes", autospec=True) as track_person_boxes,
                patch.object(spot_pass, "run_spot_pass") as run_spot_pass,
                patch.object(pipeline, "detect_video", side_effect=_StopAfterDetection) as detect_video,
                self.assertRaises(_StopAfterDetection),
            ):
                identify.identify_players(Path(tmp) / "match.mp4", fusion_checkpoint=checkpoint, **kwargs)
            return track_video, track_person_boxes, run_spot_pass, detect_video

    def test_advanced_tracks_with_mcbyte_and_detects_from_its_sidecar(self):
        track_video, track_person_boxes, run_spot_pass, detect_video = self._run(advanced=True)

        track_video.assert_called_once()
        self.assertEqual(track_video.call_args.kwargs["tracker"], "mcbyte")
        self.assertEqual(track_video.call_args.kwargs["stride"], identify.ADVANCED_STRIDE)
        self.assertEqual(track_video.call_args.kwargs["event_frames"], {120, 300})
        self.assertIs(track_video.call_args.kwargs["moving_camera"], False)
        self.assertIsNone(detect_video.call_args.kwargs.get("person_boxes"))
        track_person_boxes.assert_not_called()
        run_spot_pass.assert_not_called()

    def test_standard_keeps_fusion_boxes_and_bytetrack(self):
        track_video, track_person_boxes, run_spot_pass, detect_video = self._run(embedder="clip-reident")

        track_video.assert_not_called()
        run_spot_pass.assert_called_once()
        self.assertIsNotNone(run_spot_pass.call_args.kwargs["actor_output"])
        track_person_boxes.assert_called_once()
        self.assertIs(track_person_boxes.call_args.kwargs["moving_camera"], False)
        self.assertIsNotNone(detect_video.call_args.kwargs["person_boxes"])

    def _associate(self, **kwargs):
        """Run up to the actor policy and return (tracklet call kwargs, SPOT-pick call args)."""
        from yp_video.actor import candidates, person_action
        from yp_video.core import actor_picks

        class _Capture:
            def get(self, prop):
                return 1920 if prop == 3 else 1080

            def release(self):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint_best.pt"
            checkpoint.write_bytes(b"")
            with (
                patch.object(pipeline, "load_events", return_value=[{"frame": 120}]),
                patch.object(tracking, "track_video"),
                patch.object(fusion, "track_person_boxes"),
                patch.object(spot_pass, "run_spot_pass"),
                patch.object(pipeline, "detect_video"),
                patch.object(candidates, "track_paths", return_value={"1:1": {120: [0, 0, 9, 9]}}),
                patch("cv2.VideoCapture", return_value=_Capture()),
                patch.object(actor_picks, "load_actor_picks", return_value={"picks": 1}),
                patch.object(person_action, "build_policy", side_effect=_StopAfterDetection) as build,
                patch.object(person_action, "policy_from_spot_picks", side_effect=_StopAfterDetection) as spot,
                self.assertRaises(_StopAfterDetection),
            ):
                identify.identify_players(Path(tmp) / "match.mp4", fusion_checkpoint=checkpoint, **kwargs)
            return build, spot, checkpoint

    def test_advanced_picks_actors_among_tracklets(self):
        build, spot, checkpoint = self._associate(advanced=True)
        spot.assert_not_called()
        self.assertEqual(build.call_args.args[1], checkpoint)
        self.assertEqual(build.call_args.kwargs["tracks"], {"1:1": {120: [0, 0, 9, 9]}})

    def test_standard_reads_the_spot_pass_picks(self):
        build, spot, _ = self._associate(embedder="clip-reident")
        build.assert_not_called()
        self.assertEqual(spot.call_args.args[1], {"picks": 1})

    def test_standard_takes_both_spot_outputs_or_neither(self):
        with patch.object(pipeline, "load_events", return_value=[{"frame": 1}]), self.assertRaises(ValueError):
            identify.identify_players(
                Path("/tmp/m.mp4"), embedder="clip-reident", fusion_checkpoint=Path("/tmp/c.pt"),
                person_boxes=Path("/tmp/persons.npz"),
            )

    def test_mode_decides_the_embedder(self):
        for kwargs in ({"advanced": True, "embedder": "clip-reident"}, {}):
            with self.assertRaises(ValueError):
                identify.identify_players(Path("/tmp/m.mp4"), fusion_checkpoint=Path("/tmp/c.pt"), **kwargs)

    def test_advanced_refuses_fusion_boxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint_best.pt"
            checkpoint.write_bytes(b"")
            with (
                patch.object(pipeline, "load_events", return_value=[{"frame": 1}]),
                self.assertRaises(ValueError),
            ):
                identify.identify_players(
                    Path(tmp) / "match.mp4", fusion_checkpoint=checkpoint,
                    person_boxes=Path(tmp) / "persons.npz", advanced=True,
                )


if __name__ == "__main__":
    unittest.main()
