"""Scoring a policy on the events a human already ruled on.

The rule answers with a box and yp-spot's head answers with a tracklet, so the
evaluator has to put them in the same terms before comparing — and it must
score them on the question that was asked, not on the aggregate, which is
dominated by events the rule already gets right.
"""

from __future__ import annotations

import unittest

from yp_video.actor.evaluate import _PolicyScore, as_track, is_hard
from yp_video.actor.labels import ActorLabel, ActorVerdict
from yp_video.actor.policy import ActorPick, EventContext, SpotActorPolicy
from yp_video.actor.review import ReviewedEvent
from yp_video.actor.spot_predictions import SpotAnswer
from yp_video.tracklets.geometry import TrackRef

FRAME = 100


def _tracklet(rally, track, box, frames=(FRAME,)):
    return {
        "rally_id": rally,
        "track_id": track,
        "frames": list(frames),
        "boxes": [list(box) for _ in frames],
        "scores": [0.9 for _ in frames],
    }


def _event(label, tracklets=(), contact=(150.0, 150.0), event_id="a"):
    return ReviewedEvent(
        stem="match",
        event_id=event_id,
        record={"id": event_id, "frame": FRAME},
        label=label,
        context=EventContext(
            frame=FRAME,
            contact=contact,
            visible=True,
            event_id=event_id,
            tracklets=list(tracklets),
        ),
    )


class BoxToTrackTests(unittest.TestCase):
    def test_a_box_answer_is_resolved_to_the_tracklet_it_lands_on(self) -> None:
        """The rule answers with a box and the pipeline resolves it downstream,
        so scoring the box as 'no answer' would score the plumbing."""
        event = _event(
            ActorLabel(ActorVerdict.MANUAL, track=TrackRef(1, 1)),
            [_tracklet(1, 1, (100, 100, 200, 300)), _tracklet(1, 2, (600, 100, 700, 300))],
        )
        pick = ActorPick(box=(105.0, 105.0, 205.0, 305.0))
        self.assertEqual(as_track(pick, event), TrackRef(1, 1))

    def test_a_box_landing_on_nobody_resolves_to_nothing(self) -> None:
        event = _event(
            ActorLabel(ActorVerdict.MANUAL, track=TrackRef(1, 1)),
            [_tracklet(1, 1, (100, 100, 200, 300))],
        )
        self.assertIsNone(as_track(ActorPick(box=(900.0, 900.0, 950.0, 980.0)), event))

    def test_a_tracklet_answer_passes_through(self) -> None:
        event = _event(ActorLabel(ActorVerdict.MANUAL, track=TrackRef(1, 1)))
        pick = ActorPick(track=TrackRef(4, 2))
        self.assertEqual(as_track(pick, event), TrackRef(4, 2))


class SliceTests(unittest.TestCase):
    def test_hard_counts_tracklets_not_detections(self) -> None:
        """One box per person. The raw detection list counts the same player
        two or three times and would call almost everything hard."""
        overlapping = [
            _tracklet(1, 1, (100, 100, 200, 300)),
            _tracklet(1, 2, (120, 100, 220, 300)),
        ]
        self.assertTrue(is_hard(_event(ActorLabel(ActorVerdict.MANUAL), overlapping)))
        alone = [_tracklet(1, 1, (100, 100, 200, 300))]
        self.assertFalse(is_hard(_event(ActorLabel(ActorVerdict.MANUAL), alone)))


class ScoringTests(unittest.TestCase):
    def test_an_abstention_on_a_visible_actor_counts_as_wrong(self) -> None:
        """Coverage and accuracy are reported separately, but a policy that
        declines to answer has not got the event right."""
        score = _PolicyScore()
        score.add(_event(ActorLabel(ActorVerdict.MANUAL, track=TrackRef(1, 1))), None)
        payload = score.payload()
        self.assertEqual(payload["top1_accuracy"], 0.0)
        self.assertEqual(payload["auto_coverage"], 0.0)
        self.assertIsNone(payload["selective_accuracy"])

    def test_an_occluded_event_is_scored_on_abstention_alone(self) -> None:
        score = _PolicyScore()
        score.add(_event(ActorLabel(ActorVerdict.OCCLUDED)), None)
        score.add(_event(ActorLabel(ActorVerdict.OCCLUDED)), TrackRef(1, 1))
        payload = score.payload()
        self.assertEqual(payload["occluded"], 2)
        self.assertEqual(payload["occluded_rejection_rate"], 0.5)
        self.assertEqual(payload["positive"], 0)

    def test_a_legacy_box_verdict_does_not_dilute_the_rate(self) -> None:
        """It names a person but no tracklet, so it is not answerable in these
        terms; counting it as a miss would punish every policy for the label
        format."""
        score = _PolicyScore()
        score.add(_event(ActorLabel(ActorVerdict.MANUAL, box=(1.0, 2.0, 3.0, 4.0))), None)
        payload = score.payload()
        self.assertEqual(payload["unscorable"], 1)
        self.assertEqual(payload["positive"], 0)
        self.assertIsNone(payload["top1_accuracy"])


class SpotPolicyTests(unittest.TestCase):
    def test_it_answers_with_the_tracklet_the_head_chose(self) -> None:
        policy = SpotActorPolicy({"a": SpotAnswer(TrackRef(2, 7), 0.9, "track")})
        pick = policy.decide(_event(ActorLabel(ActorVerdict.MANUAL)).context)
        self.assertEqual(pick.track, TrackRef(2, 7))
        self.assertEqual(pick.diagnostic["kind"], "track")

    def test_occluded_and_untracked_both_abstain_but_say_which(self) -> None:
        """They abstain identically and mean different things: `untracked`
        says go fix tracking, not go relabel."""
        for kind in ("occluded", "untracked"):
            policy = SpotActorPolicy({"a": SpotAnswer(None, 0.8, kind)})
            pick = policy.decide(_event(ActorLabel(ActorVerdict.MANUAL)).context)
            self.assertFalse(pick.decided)
            self.assertEqual(pick.diagnostic["kind"], kind)

    def test_an_event_the_head_never_saw_gets_no_answer(self) -> None:
        policy = SpotActorPolicy({})
        self.assertFalse(
            policy.decide(_event(ActorLabel(ActorVerdict.MANUAL)).context).decided
        )

    def test_an_unattributable_event_is_refused_like_every_policy(self) -> None:
        policy = SpotActorPolicy({"a": SpotAnswer(TrackRef(2, 7), 0.9, "track")})
        context = EventContext(frame=FRAME, contact=None, visible=True, event_id="a")
        self.assertFalse(policy.decide(context).decided)


if __name__ == "__main__":
    unittest.main()
