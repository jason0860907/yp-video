"""Exporting "who acted" as a CHOICE among tracked players.

The unit of supervision is the candidate set plus an index into it, because
that is what the model is asked to produce. The tests are mostly about which
of the three target kinds a given verdict lands in — getting that wrong is
silent, and the failure mode is teaching the model that a tracking failure
means nobody acted.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from yp_video.actor import candidates as actor_labels
from yp_video.actor import labels as verdict_labels
from yp_video.actor.labels import ActorLabel, ActorVerdict
from yp_video.contracts.action import ACTOR_WINDOW_OFFSETS
from yp_video.core.cache import StatCache
from yp_video.core.jsonl import write_jsonl
from yp_video.tracklets.geometry import TrackRef

STEM = "match"
FRAME_SIZE = [1000, 500]
EVENT_FRAME = 100


class ActorCandidateExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        root = Path(self._dir.name)
        self.records = root / "match.jsonl"
        self.tracks = root / "match_tracks.jsonl"
        self.verdicts = root / "match_actors.json"

        write_jsonl(self.records, {"frame_size": FRAME_SIZE}, [])
        write_jsonl(
            self.tracks,
            {"frame_size": FRAME_SIZE, "stride": 1},
            [
                # Two players on the event frame, plus one that has a gap there.
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
            ],
        )
        self._patches = [
            patch.object(actor_labels, "records_path", return_value=self.records),
            patch.object(actor_labels, "tracks_path", return_value=self.tracks),
            patch.object(verdict_labels, "actors_path", return_value=self.verdicts),
            patch.object(verdict_labels._store, "_cache", StatCache()),
        ]
        for item in self._patches:
            item.start()

    def tearDown(self) -> None:
        for item in self._patches:
            item.stop()
        self._dir.cleanup()

    def _build(self, event_id: str, label: ActorLabel | None) -> dict:
        if label is not None:
            verdict_labels.save(STEM, event_id, label)
        rows, self.tally = actor_labels.build(
            STEM, [{"id": event_id, "frame": EVENT_FRAME}]
        )
        return rows[0] if rows else {}

    def test_the_candidate_set_is_who_was_tracked_on_the_event_frame(self) -> None:
        """Only the event frame. A window would re-admit a player who had
        already left before the ball was touched."""
        row = self._build("a", ActorLabel(ActorVerdict.MANUAL, track=TrackRef(2, 7)))
        self.assertEqual([c["track"] for c in row["candidates"]], ["2:3", "2:7"])

    def test_a_tracklet_verdict_points_at_its_candidate(self) -> None:
        row = self._build("a", ActorLabel(ActorVerdict.MANUAL, track=TrackRef(2, 7)))
        self.assertEqual(row["target_kind"], "track")
        self.assertEqual(row["candidates"][row["target"]]["track"], "2:7")

    def test_an_occluded_verdict_is_its_own_answer(self) -> None:
        """Occluded does NOT mean an empty court: these events carry a median
        of ten other tracked players. It means the one who acted is not among
        them."""
        row = self._build("a", ActorLabel(ActorVerdict.OCCLUDED))
        self.assertEqual(row["target_kind"], "occluded")
        self.assertNotIn("target", row)

    def test_a_tracklet_absent_from_the_event_frame_is_untracked(self) -> None:
        """Track 2:9 exists either side but has no box ON the event frame. The
        answer is genuinely not in the candidate set, and calling that
        'occluded' would train the model to read a tracking gap as a player it
        could not see."""
        row = self._build("a", ActorLabel(ActorVerdict.MANUAL, track=TrackRef(2, 9)))
        self.assertEqual(row["target_kind"], "untracked")
        self.assertNotIn("target", row)

    def test_a_box_verdict_resolves_to_the_tracklet_it_sits_on(self) -> None:
        """A verdict naming a person by box alone — a legacy hand-drawn pick,
        or a confirm snapshot of the rule's box — resolves by the same overlap
        rule production and evaluation use, so the exporter answers the
        tracklet question rather than dropping real supervision."""
        row = self._build(
            "a", ActorLabel(ActorVerdict.MANUAL, box=(105.0, 55.0, 205.0, 255.0))
        )
        self.assertEqual(row["target_kind"], "track")
        self.assertEqual(row["candidates"][row["target"]]["track"], "2:7")

    def test_a_box_on_nobody_is_dropped_not_reinterpreted(self) -> None:
        """No tracklet sits under this box. Calling it 'untracked' would teach
        the model that the label FORMAT is a visual condition, and there is
        nothing in the frame to learn that from."""
        row = self._build(
            "b", ActorLabel(ActorVerdict.MANUAL, box=(10.0, 10.0, 40.0, 60.0))
        )
        self.assertEqual(row, {})
        self.assertEqual(self.tally["unresolved_box"], 1)

    def test_an_unreviewed_event_produces_no_row(self) -> None:
        """This file carries supervision. An event's absence from it is what
        'nobody has looked at this yet' means."""
        verdict_labels.save("match", "b", ActorLabel(ActorVerdict.OCCLUDED))
        rows, tally = actor_labels.build(
            STEM, [{"id": "unreviewed", "frame": EVENT_FRAME}]
        )
        self.assertEqual(rows, [])
        self.assertEqual(tally["unlabelled"], 1)

    def test_each_candidate_carries_its_path_through_the_window(self) -> None:
        """One box per window offset, aligned with it, so the model sees the
        player MOVE rather than a frozen pose. A null is not a hole to be
        filled: it says this player was not being tracked then."""
        row = self._build("a", ActorLabel(ActorVerdict.MANUAL, track=TrackRef(2, 7)))
        for candidate in row["candidates"]:
            self.assertEqual(len(candidate["boxes"]), len(ACTOR_WINDOW_OFFSETS))
            for box in candidate["boxes"]:
                if box is None:
                    continue
                self.assertTrue(all(0.0 <= v <= 1.0 for v in box), box)
                self.assertLess(box[0], box[2])
                self.assertLess(box[1], box[3])
        at_event = ACTOR_WINDOW_OFFSETS.index(0)
        walker = next(c for c in row["candidates"] if c["track"] == "2:7")
        self.assertEqual(walker["boxes"][at_event], [0.1, 0.1, 0.2, 0.5])
        # Tracked only at 98/100/102, so every other offset is a gap.
        self.assertEqual(sum(b is not None for b in walker["boxes"]), 1)


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

    def test_a_track_target_must_index_a_candidate(self) -> None:
        from yp_video.contracts.action import ActorCandidateEvent

        event = ActorCandidateEvent(
            id="a",
            frame=1,
            candidates=[
                {"track": "1:1", "boxes": [[0.1, 0.1, 0.2, 0.5]] * len(ACTOR_WINDOW_OFFSETS)}
            ],
            target_kind="track",
            target=0,
        )
        self.assertEqual(event.target, 0)

        for kind in ("occluded", "untracked"):
            abstention = ActorCandidateEvent(
                id="a", frame=1, candidates=[], target_kind=kind
            )
            self.assertIsNone(abstention.target)


if __name__ == "__main__":
    unittest.main()
