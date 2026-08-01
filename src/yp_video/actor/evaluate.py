"""Score actor-association policies against the verdicts humans already gave.

Every policy answers with a TRACKLET, so every policy is scored on the same
question: of the players tracking found, did you name the one the human
named. There was once a second, box-shaped evaluation here for a box-shaped
ranker; it went with the ranker, and with it the trap of comparing two
numbers that answered different questions.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING

from yp_video.actor import labels as actor_labels
from yp_video.actor.labels import ActorVerdict
from yp_video.actor.metrics import association_rates, ratio
from yp_video.actor.review import iter_reviewed

if TYPE_CHECKING:
    from yp_video.actor.policy import ActorPolicy


class _PolicyScore:
    """Counts for one policy on one slice of the reviewed events."""

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()

    def add(self, event, track) -> None:
        count = self.counts
        count["reviewed"] += 1
        decided = track is not None
        if event.is_occluded:
            count["occluded"] += 1
            count["occluded_rejected"] += int(not decided)
            return
        if event.truth is None:
            # A verdict naming no tracklet directly — a legacy box pick or a
            # confirm snapshot. Scoring these would need the same geometric
            # resolution training now applies (dataset.build_track_dataset);
            # until truth resolves the same way, they must not dilute the rate.
            count["unscorable"] += 1
            return
        count["positive"] += 1
        if decided:
            count["decided"] += 1
            count["correct"] += int(track == event.truth)

    def payload(self) -> dict:
        count = self.counts
        return {
            "reviewed": count["reviewed"],
            "positive": count["positive"],
            "occluded": count["occluded"],
            "unscorable": count["unscorable"],
            # Of the events with a knowable answer, how many did it get right
            # — abstentions included, because an abstention is a wrong answer
            # when somebody visibly acted.
            "top1_accuracy": ratio(count, "correct", "positive"),
            **association_rates(count),
        }


def as_track(pick, event):
    """The tracklet a pick names, resolving a BOX answer if that is what it is.

    The rule answers with a box and the pipeline resolves it to a tracklet
    downstream, so scoring the box as "no answer" would score the plumbing
    rather than the policy. The resolution is tracklets.geometry's — the same
    one production uses on a pick and training uses on a labelled box, so a
    policy cannot be scored against a stricter reading of "who was named"
    than it will face in service.
    """
    if pick.track is not None or pick.box is None:
        return pick.track
    if event.context.tracks is None:
        return None
    return event.context.tracks.at_box(event.context.frame, pick.box)


def _contains_contact(box, contact) -> bool:
    from yp_video.actor.ranking import X_PAD_FRAC, Y_ABOVE_FRAC

    x, y = contact
    x0, y0, x1, y1 = (float(v) for v in box)
    width, height = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    return (
        x0 - X_PAD_FRAC * width <= x <= x1 + X_PAD_FRAC * width
        and y0 - Y_ABOVE_FRAC * height <= y <= y1
    )


def is_hard(event) -> bool:
    """More than one tracklet box contains the contact point.

    Pinned here so every arm is scored on the same definition. Note this
    counts TRACKLETS, one box per person; the raw detection list counts the
    same player two or three times and would give a quite different number
    for the same idea.
    """
    if event.context.contact is None or event.context.tracks is None:
        return False
    hits = sum(
        _contains_contact(box, event.context.contact)
        for _ref, box in event.context.tracks.at(event.context.frame)
    )
    return hits > 1


#: The slices a result has to be read on. The aggregate is dominated by events
#: the rule already got right, so a policy can move it without touching a
#: single case the rule fails — which is the only thing worth moving.
SLICES = {
    "all": lambda event: True,
    "hard": is_hard,
    "manual": lambda event: event.label.verdict is ActorVerdict.MANUAL,
}


def evaluate_policies(
    builders: Mapping[str, Callable[[str], "ActorPolicy | None"]],
    stems: Sequence[str] | None = None,
) -> dict:
    """Score each named policy on the reviewed events, sliced three ways.

    Takes a BUILDER per policy rather than a policy, because not every policy
    is the same object for every video: the yp-spot head's answers arrive as
    one file per video, and merging them into a single lookup would make the
    evaluator depend on event ids never colliding across videos. A builder
    returning None means "this policy has nothing to say about this video" —
    scored as absent rather than as an abstention, which is a different claim.
    """
    scores = {
        name: {slice_name: _PolicyScore() for slice_name in SLICES}
        for name in builders
    }
    selected = list(stems) if stems is not None else actor_labels.labeled_stems()
    for stem in selected:
        policies = {
            name: build(stem) for name, build in builders.items()
        }
        if not any(policy is not None for policy in policies.values()):
            continue
        for event in iter_reviewed([stem]):
            member = {name: test(event) for name, test in SLICES.items()}
            for name, policy in policies.items():
                if policy is None:
                    continue
                track = as_track(policy.decide(event.context), event)
                for slice_name, inside in member.items():
                    if inside:
                        scores[name][slice_name].add(event, track)
    return {
        name: {
            slice_name: score.payload() for slice_name, score in by_slice.items()
        }
        for name, by_slice in scores.items()
    }
