"""The tracklets near an action event: the candidate set the joint
person/action head is asked to choose from."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from yp_video.actor import candidates as actor_candidates
from yp_video.core.jsonl import write_jsonl

STEM = "match"
FRAME_SIZE = [1000, 500]
EVENT_FRAME = 100


class CandidateSetTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        root = Path(self._dir.name)
        self.records = root / "match.jsonl"
        self.tracks = root / "match_tracks.jsonl"
        write_jsonl(self.records, {"frame_size": FRAME_SIZE}, [])
        write_jsonl(
            self.tracks,
            {"frame_size": FRAME_SIZE, "stride": 1},
            [
                # Two players on the event frame, one in a stride gap there
                # (boxes 2 frames either side), one gone 10 frames before it.
                {
                    "rally_id": 2,
                    "track_id": 7,
                    "frames": [98, 100, 102],
                    "boxes": [[100, 50, 200, 250]] * 3,
                    "scores": [0.9] * 3,
                },
                {
                    "rally_id": 2,
                    "track_id": 3,
                    "frames": [100],
                    "boxes": [[600, 50, 700, 250]],
                    "scores": [0.9],
                },
                {
                    "rally_id": 2,
                    "track_id": 9,
                    "frames": [98, 102],
                    "boxes": [[300, 50, 400, 250]] * 2,
                    "scores": [0.9] * 2,
                },
                {
                    "rally_id": 2,
                    "track_id": 11,
                    "frames": [80, 90],
                    "boxes": [[800, 50, 900, 250]] * 2,
                    "scores": [0.9] * 2,
                },
            ],
        )
        self._patches = [
            patch.object(actor_candidates, "records_path", return_value=self.records),
            patch.object(actor_candidates, "tracks_path", return_value=self.tracks),
        ]
        for item in self._patches:
            item.start()

    def tearDown(self) -> None:
        for item in self._patches:
            item.stop()
        self._dir.cleanup()

    def test_the_candidate_set_is_who_was_tracked_around_the_event_frame(self) -> None:
        """Within ±3 frames, so a stride-2 gap on the event frame (2:9) keeps
        its player; no wider, so one who left 10 frames earlier (2:11) is not
        asked about."""
        paths = actor_candidates.track_paths(STEM)
        self.assertEqual(
            actor_candidates.candidates_on(paths, EVENT_FRAME), ["2:3", "2:7", "2:9"]
        )

    def test_the_frame_size_comes_from_the_detection_records(self) -> None:
        self.assertEqual(actor_candidates.frame_size(STEM), (1000, 500))


class ContractTests(unittest.TestCase):
    def test_the_two_repos_agree_on_the_contract_version(self) -> None:
        """The handshake is exact-match and only fails at subprocess spawn
        time, i.e. minutes into a training job. Catch it here instead."""
        from yp_video.config import SPOT_DIR
        from yp_video.contracts.action import ACTION_CONTRACT_VERSION

        mirror = SPOT_DIR / "yp_spot" / "contract.py"
        if not mirror.exists():
            self.skipTest("yp-spot checkout not present")
        for line in mirror.read_text(encoding="utf-8").splitlines():
            if line.startswith("CONTRACT_VERSION"):
                self.assertEqual(
                    line.split("=")[1].strip().strip('"'), ACTION_CONTRACT_VERSION
                )
                return
        self.fail("yp-spot contract.py declares no CONTRACT_VERSION")


if __name__ == "__main__":
    unittest.main()
