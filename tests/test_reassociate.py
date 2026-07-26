"""Association Predict: what it may rewrite, and what it must not touch."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from yp_video.actor import labels as actor_labels
from yp_video.actor.labels import ActorLabel, ActorVerdict
from yp_video.actor.policy import ActorPick, EventContext, RulePolicy
from yp_video.core.cache import StatCache
from yp_video.core.jsonl import read_jsonl, write_jsonl
from yp_video.extraction import reassociate
from yp_video.tracklets.geometry import TrackRef


def _detection(box, score=0.9):
    return {"box": list(box), "score": score}


class _StubPolicy:
    """Answers with whatever the test hands it, keyed by event id."""

    name = "stub"
    needs_tracklets = False

    def __init__(self, picks: dict[str, ActorPick], frames: dict[int, str]):
        self._picks = picks
        self._frames = frames

    def decide(self, context: EventContext) -> ActorPick:
        return self._picks.get(self._frames[context.frame], ActorPick())


class ReassociationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        root = Path(self._dir.name)
        self.records = root / "match.jsonl"
        self.crops = root / "crops"
        self.crops.mkdir()
        self.labels = root / "match_actors.json"

        self.meta = {"frame_size": [1920, 1080], "events": 2}
        self.rows = [
            {
                "id": "human",
                "frame": 100,
                "xy": [0.5, 0.5],
                "resolution": "manual",
                "box": [900, 500, 990, 700],
                "actor_box": [910, 510, 980, 690],
                "crop": "human.jpg",
                "crop_schema": reassociate.CROP_SCHEMA_VERSION,
                "score": 0.8,
                "detections": [_detection([910, 510, 980, 690])],
            },
            {
                "id": "auto",
                "frame": 200,
                "xy": [0.25, 0.5],
                "resolution": "auto",
                "box": [400, 500, 520, 700],
                "actor_box": [410, 510, 510, 690],
                "crop": "auto.jpg",
                "crop_schema": reassociate.CROP_SCHEMA_VERSION,
                "score": 0.7,
                "detections": [
                    _detection([410, 510, 510, 690]),
                    _detection([100, 100, 200, 300], score=0.6),
                ],
            },
        ]
        write_jsonl(self.records, self.meta, self.rows)
        (self.crops / "human.jpg").write_bytes(b"human")
        (self.crops / "auto.jpg").write_bytes(b"auto")

        self._patches = [
            patch.object(reassociate, "records_path", return_value=self.records),
            patch.object(reassociate, "crop_dir", return_value=self.crops),
            patch.object(
                reassociate, "masked_crop_dir", return_value=self.crops / "masked"
            ),
            patch.object(actor_labels, "actors_path", return_value=self.labels),
            patch.object(actor_labels, "_cache", StatCache()),
        ]
        for item in self._patches:
            item.start()
        # The human's verdict on event "human".
        actor_labels.save(
            "match", "human", ActorLabel(ActorVerdict.MANUAL, box=(910, 510, 980, 690))
        )

    def tearDown(self) -> None:
        for item in self._patches:
            item.stop()
        self._dir.cleanup()

    def _run(self, policy) -> dict:
        return reassociate.reassociate_video(Path("/nonexistent/match.mp4"), policy)

    def test_a_human_verdict_is_never_re_decided(self) -> None:
        """The labeled event must come out byte-identical — no new diagnostic,
        no bumped revision, no re-crop. A policy that touched it would silently
        undo somebody's work."""
        before = read_jsonl(self.records)[1][0]

        counts = self._run(
            _StubPolicy(
                # A pick that WOULD move the human event, if it were consulted.
                {"human": ActorPick(box=(0, 0, 50, 50)), "auto": ActorPick()},
                {100: "human", 200: "auto"},
            )
        )

        after = read_jsonl(self.records)[1][0]
        self.assertEqual(counts["labeled"], 1)
        self.assertEqual(before, after)
        self.assertEqual((self.crops / "human.jpg").read_bytes(), b"human")

    def test_a_crop_from_the_retired_geometry_is_not_materialized(self) -> None:
        """A saved verdict stays authoritative but its old pixels are rebuilt."""
        record = dict(self.rows[0])
        record.pop("crop_schema")
        label = ActorLabel(
            ActorVerdict.MANUAL,
            box=(910, 510, 980, 690),
        )
        self.assertFalse(reassociate._is_materialized(record, label))
        record["crop_schema"] = reassociate.CROP_SCHEMA_VERSION
        self.assertTrue(reassociate._is_materialized(record, label))

    def test_an_unchanged_pick_costs_no_re_crop(self) -> None:
        """Re-running the same policy is idempotent: the crop file on disk is
        the one that was already there."""
        counts = self._run(
            _StubPolicy(
                {"auto": ActorPick(box=(410, 510, 510, 690))},
                {100: "human", 200: "auto"},
            )
        )

        self.assertEqual(counts["changed"], 0)
        self.assertEqual(counts["unchanged"], 1)
        self.assertEqual((self.crops / "auto.jpg").read_bytes(), b"auto")
        self.assertEqual(read_jsonl(self.records)[1][1]["crop"], "auto.jpg")

    def test_abstention_clears_the_event_rather_than_keeping_a_stale_pick(
        self,
    ) -> None:
        counts = self._run(
            _StubPolicy({}, {100: "human", 200: "auto"})  # abstains everywhere
        )

        record = read_jsonl(self.records)[1][1]
        self.assertEqual(counts["abstained"], 1)
        self.assertIsNone(record["box"])
        self.assertIsNone(record["crop"])
        self.assertEqual(record["resolution"], "unresolved")

    def test_a_tracklet_pick_that_resolves_nowhere_is_not_forced(self) -> None:
        """No tracklets on disk → resolve_track finds nothing. The event must
        end unresolved, not cropped from the tracklet's own box."""
        with patch("yp_video.extraction.cropping.resolve_track", return_value=None):
            counts = self._run(
                _StubPolicy(
                    {"auto": ActorPick(track=TrackRef(3, 7))},
                    {100: "human", 200: "auto"},
                )
            )

        self.assertEqual(counts["unresolvable"], 1)
        self.assertIsNone(read_jsonl(self.records)[1][1]["box"])

    def test_a_fully_reviewed_video_is_not_even_rewritten(self) -> None:
        """The records mtime is what marks every embedding stale, so a run
        that decides nothing must not touch the file — otherwise re-running on
        a finished video orders a full re-embed to produce identical vectors."""
        actor_labels.save(
            "match", "auto", ActorLabel(ActorVerdict.CONFIRMED_AUTO, box=(1, 2, 3, 4))
        )
        before = self.records.stat().st_mtime_ns

        counts = self._run(
            _StubPolicy(
                {"auto": ActorPick(box=(0, 0, 50, 50))}, {100: "human", 200: "auto"}
            )
        )

        self.assertEqual(counts["labeled"], 2)
        self.assertEqual(self.records.stat().st_mtime_ns, before)

    def test_the_policy_name_is_recorded_in_the_header(self) -> None:
        self._run(_StubPolicy({}, {100: "human", 200: "auto"}))
        self.assertEqual(read_jsonl(self.records)[0]["association_policy"], "stub")

    def test_progress_speaks_the_shared_worker_contract(self) -> None:
        """``core.progress.ProgressFn`` — (done, total, message). This module
        used to report (message, fraction) instead, which every batch job
        wired in happily and then died on at the first callback."""
        calls: list[tuple] = []
        reassociate.reassociate_video(
            Path("/nonexistent/match.mp4"),
            _StubPolicy(
                {"auto": ActorPick(box=(100, 100, 200, 300))},
                {100: "human", 200: "auto"},
            ),
            on_progress=lambda *args: calls.append(args),
        )

        self.assertTrue(calls)
        for done, total, message in calls:
            self.assertIsInstance(done, int)
            self.assertIsInstance(total, int)
            self.assertIsInstance(message, str)
            self.assertLessEqual(done, total)


class RulePolicyContractTests(unittest.TestCase):
    def test_an_unattributable_event_gets_no_pick(self) -> None:
        """Two states, neither a missing value: no contact point at all, and a
        point on an INVISIBLE event — where the nearest player is provably not
        the actor. Extraction has always refused both, and a policy that
        answered anyway would invent picks on re-association."""
        detections = [_detection([0, 0, 40, 100])]
        for contact, visible in (((20.0, 10.0), False), (None, True), (None, False)):
            with self.subTest(contact=contact, visible=visible):
                pick = RulePolicy().decide(
                    EventContext(
                        frame=1,
                        contact=contact,
                        visible=visible,
                        detections=detections,
                    )
                )
                self.assertFalse(pick.decided)

    def test_the_rule_answers_with_a_box_and_never_a_tracklet(self) -> None:
        pick = RulePolicy().decide(
            EventContext(
                frame=1,
                contact=(20.0, 10.0),
                visible=True,
                detections=[_detection([0, 0, 40, 100])],
            )
        )
        self.assertIsNone(pick.track)
        self.assertEqual(pick.box, (0.0, 0.0, 40.0, 100.0))
        self.assertEqual(pick.diagnostic["version"], "rule-based")


if __name__ == "__main__":
    unittest.main()
