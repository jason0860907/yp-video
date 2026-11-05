"""Volleyball rules over a result's touch timeline — the one place they live.

Input is what a detector result already carries: rallies (``index`` /
``start`` / ``end``) and the flat ``action_events`` (``id`` / ``label`` /
``time``). Output is small and semantic — touch groupings and per-rally
classifications — shipped in the result as ``attacks`` and
``rally_outcomes``. Touches are named by their position in ``action_events``:
event ids are ``f<frame>``, and two touches spotted on one frame share one.
Consumers (iOS, App Review) read these instead of re-deriving them; clip
windows and playback framing stay with the client.

A touch belongs to the rally whose ``start <= time <= end`` (no tolerance).
Touches outside every rally belong to no attack and no outcome.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

from yp_video.contracts.action import ACTION_LABELS, ACTOR_LABELS, END_LABELS

# Position in the 接→舉→打 build-up. A serve is the opponent's ball into play
# and a block answers *their* attack, so neither is part of one.
STAGE = {"receive": 0, "set": 1, "spike": 2}
# Any of these means the rally went past the serve exchange.
RALLY_PLAY = {"set", "spike", "block"}

Event = Mapping[str, Any]

#: The other team's half, per camera-frame court side.
OPPOSITE_SIDE = {"left": "right", "right": "left", "near": "far", "far": "near"}
#: Touches one team makes in a row before the ball must cross.
TEAM_TOUCHES = 3
#: The two camera axes: their sides and the ``xy`` component running along each.
SIDE_AXES = {"lr": ("left", "right", 0), "nf": ("far", "near", 1)}
AXIS_OF = {side: axis for axis, (a, b, _) in SIDE_AXES.items() for side in (a, b)}
#: Share of a video's anchoring serves that must sit on their anchor's half
#: for its serve positions to name the serving side. The anchors are the
#: predicted winners, so this measures their errors too: on 473 labelled
#: videos with 10% of winners flipped, 0.7 names 93% of serves at 96.5%
#: (0.8: 80% at 97.5%).
SERVING_MIN_AGREEMENT = 0.7


def kind(event: Event) -> str:
    """The event's contract label; anything else is a broken result."""
    label = event["label"]
    if label not in ACTION_LABELS:
        raise ValueError(f"Not an action label: {label!r}")
    return label


def _build_up(events: Sequence[Event], before: Sequence[int], anchor: int) -> list[int]:
    """接→舉 leading up to ``anchor``: walking down the stages below the
    anchor's own, take the latest earlier touch at each, each one before the
    last picked. A missing stage (the model skipped a receive or a set) just
    yields a shorter chain."""
    chain: list[int] = []
    cutoff = events[anchor]["time"]
    for stage in reversed(range(STAGE.get(kind(events[anchor]), 0))):
        found = next(
            (
                i
                for i in reversed(before)
                if STAGE.get(kind(events[i])) == stage and events[i]["time"] < cutoff
            ),
            None,
        )
        if found is not None:
            chain.append(found)
            cutoff = events[found]["time"]
    return list(reversed(chain))


def _by_rally(
    rallies: Sequence[Mapping[str, Any]], events: Sequence[Event]
) -> list[tuple[int, list[int]]]:
    """``(rally_index, positions)`` per rally in timeline order; positions
    index ``events`` and are sorted by time."""
    ordered = sorted(range(len(events)), key=lambda i: events[i]["time"])
    return [
        (
            int(r["index"]),
            [
                i
                for i in ordered
                if float(r["start"]) <= events[i]["time"] <= float(r["end"])
            ],
        )
        for r in sorted(rallies, key=lambda r: float(r["start"]))
    ]


def _rally_attacks(events: Sequence[Event], touches: Sequence[int]) -> list[list[int]]:
    """組織進攻 (Attack): one per spike, with whatever 接舉 build-up the model
    spotted since the previous spike — a touch never feeds two attacks."""
    attacks: list[list[int]] = []
    window_start = 0
    for offset, position in enumerate(touches):
        if kind(events[position]) != "spike":
            continue
        before = touches[window_start:offset]
        attacks.append(_build_up(events, before, position) + [position])
        window_start = offset + 1
    return attacks


def attacks(
    rallies: Sequence[Mapping[str, Any]], events: Sequence[Event]
) -> list[dict]:
    """Every attack in the match, timeline order."""
    return [
        {"rally_index": index, "event_indices": attack}
        for index, touches in _by_rally(rallies, events)
        for attack in _rally_attacks(events, touches)
    ]


class ServeSplit(NamedTuple):
    """Where one video's serves divide its two sides along the camera axis."""

    coord: int
    middle: float
    larger: str
    smaller: str
    #: Share of the anchoring serves on their anchor's half.
    agreement: float

    def side(self, xy: Sequence[float]) -> str:
        return self.larger if float(xy[self.coord]) > self.middle else self.smaller


def serve_split(anchored: Sequence[tuple[str, Sequence[float]]]) -> ServeSplit | None:
    """The line between the two sides' serve positions, learned from serves
    whose side is known otherwise — ``(side, serve xy)``, all on one axis.

    Which side's serves sit at the larger image coordinate is learned rather
    than assumed: a near server's toss can be struck higher in the frame than
    a far one's. None unless both sides served."""
    if not anchored:
        return None
    low, high, coord = SIDE_AXES[AXIS_OF[anchored[0][0]]]
    by_side = {side: [float(xy[coord]) for s, xy in anchored if s == side] for side in (low, high)}
    if not by_side[low] or not by_side[high]:
        return None
    low_median, high_median = (statistics.median(by_side[s]) for s in (low, high))
    middle = (low_median + high_median) / 2
    larger, smaller = (high, low) if high_median > low_median else (low, high)
    agrees = [
        (float(xy[coord]) > middle) == (side == larger) for side, xy in anchored
    ]
    return ServeSplit(coord, middle, larger, smaller, sum(agrees) / len(agrees))


def _opening_serve(events: Sequence[Event], touches: Sequence[int]) -> int | None:
    """The rally's first player touch when it is a serve, else None."""
    plays = [i for i in touches if kind(events[i]) in ACTOR_LABELS]
    return plays[0] if plays and kind(events[plays[0]]) == "serve" else None


def serving_sides(
    rallies: Sequence[Mapping[str, Any]], events: Sequence[Event]
) -> dict[int, str | None]:
    """The camera-frame side each rally was served from, by rally index.

    Read off where the serve was struck: the team that won a rally serves the
    next one in the set, so the predicted winners anchor which half of the
    frame each side serves from (``serve_split``), and each rally's own
    serve then names its side — one wrong winner moves no rally but its own
    anchor. An off-screen serve counts by its entry point, which comes in
    over the server's half. None where the rally opens on no serve, and for
    the whole video when its serves do not split cleanly (no winners, one
    side's serves only, or under SERVING_MIN_AGREEMENT on their anchors).
    """
    groups = _by_rally(rallies, events)
    serves = {index: _opening_serve(events, touches) for index, touches in groups}
    ordered = sorted(rallies, key=lambda r: float(r["start"]))
    winners = [r.get("winner") for r in ordered if r.get("winner") in AXIS_OF]
    split = None
    if winners:
        axes = [AXIS_OF[w] for w in winners]
        axis = max(sorted(set(axes)), key=axes.count)
        anchored = [
            (previous["winner"], events[serve]["xy"])
            for previous, rally in zip(ordered, ordered[1:])
            if previous["set"] == rally["set"]
            and AXIS_OF.get(previous.get("winner")) == axis
            and (serve := serves[int(rally["index"])]) is not None
        ]
        split = serve_split(anchored)
    if split is None or split.agreement < SERVING_MIN_AGREEMENT:
        return {index: None for index, _ in groups}
    return {
        index: None if serve is None else split.side(events[serve]["xy"])
        for index, serve in serves.items()
    }


def rally_outcomes(
    rallies: Sequence[Mapping[str, Any]], events: Sequence[Event]
) -> list[dict]:
    """One outcome per rally, timeline order.

    - ``serve_point``: the serve decided the point (ace or service error) —
      the rally holds touches but no set / spike / block. A shanked receive
      still counts. Null when the rally holds no touch at all.
    - ``score_event_index``: the rally's final whistle; duplicate whistles a
      beat apart collapse to the last one. Null when none was spotted.
    - ``deciding_event_indices``: the play that decided the point — the last
      attack, or with no spike the last non-whistle touch.
    - ``serving_side``: the camera-frame side that served (``serving_sides``).
    """
    served_from = serving_sides(rallies, events)
    out = []
    for index, touches in _by_rally(rallies, events):
        kinds = {kind(events[i]) for i in touches}
        whistles = [i for i in touches if kind(events[i]) in END_LABELS]
        # Only what happened before the final whistle can have decided it.
        cutoff = events[whistles[-1]]["time"] if whistles else float("inf")
        rally_attacks = [
            a for a in _rally_attacks(events, touches) if events[a[-1]]["time"] < cutoff
        ]
        plays = [
            i for i in touches if kind(events[i]) in ACTOR_LABELS and events[i]["time"] < cutoff
        ]
        out.append(
            {
                "rally_index": index,
                "serve_point": (not (kinds & RALLY_PLAY)) if touches else None,
                "score_event_index": whistles[-1] if whistles else None,
                "deciding_event_indices": rally_attacks[-1] if rally_attacks else plays[-1:],
                "serving_side": served_from[index],
            }
        )
    return out


def touch_sides(events: Sequence[Event], serve_side: str) -> list[str | None]:
    """Which court side each touch of ONE rally was made on, given the side
    that served; ``events`` are that rally's touches in time order.

    The ball crosses after a serve and after a spike; receive / set / spike
    of one possession share a side; a block stands opposite the spike it
    answers. Where the order stops determining sides the walk stops and the
    rest stay None: a block (the ball may come back to either side), a
    receive in mid-possession (a free ball came over — or a second pass), a
    fourth touch. Ball events (net, ground, antenna, score) name no side. A rally not
    opening with its serve yields all None.
    """
    sides: list[str | None] = [None] * len(events)
    touches = [i for i, event in enumerate(events) if kind(event) in ACTOR_LABELS]
    if not touches or kind(events[touches[0]]) != "serve":
        return sides
    side, touched, crossing, previous = serve_side, 0, False, None
    for i in touches:
        current = kind(events[i])
        if current == "serve":
            if previous is not None:
                break
            sides[i] = side
            crossing = True
        elif current == "block":
            if previous != "spike":
                break
            sides[i] = OPPOSITE_SIDE[side]
            break
        elif current in STAGE:
            if crossing:
                side, touched, crossing = OPPOSITE_SIDE[side], 0, False
            elif current == "receive" and touched:
                break
            touched += 1
            if touched > TEAM_TOUCHES:
                break
            sides[i] = side
            crossing = current == "spike"
        else:
            break
        previous = current
    return sides
