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

from collections.abc import Mapping, Sequence
from typing import Any

KINDS = {
    **dict.fromkeys(["發", "發球", "serve"], "serve"),
    **dict.fromkeys(["接", "receive", "reception", "dig", "pass"], "receive"),
    **dict.fromkeys(["舉", "set", "setting"], "set"),
    **dict.fromkeys(["打", "扣", "扣球", "spike", "attack", "hit"], "spike"),
    **dict.fromkeys(["攔", "攔網", "block"], "block"),
    **dict.fromkeys(["分", "得分", "score", "point"], "score"),
}
# Position in the 接→舉→打 build-up. A serve is the opponent's ball into play
# and a block answers *their* attack, so neither is part of one.
STAGE = {"receive": 0, "set": 1, "spike": 2}
# Any of these means the rally went past the serve exchange.
RALLY_PLAY = {"set", "spike", "block"}

Event = Mapping[str, Any]


def kind(event: Event) -> str:
    return KINDS.get(str(event["label"]).lower(), "other")


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
    """
    out = []
    for index, touches in _by_rally(rallies, events):
        kinds = {kind(events[i]) for i in touches}
        whistles = [i for i in touches if kind(events[i]) == "score"]
        # Only what happened before the final whistle can have decided it.
        cutoff = events[whistles[-1]]["time"] if whistles else float("inf")
        rally_attacks = [
            a for a in _rally_attacks(events, touches) if events[a[-1]]["time"] < cutoff
        ]
        plays = [
            i for i in touches if kind(events[i]) != "score" and events[i]["time"] < cutoff
        ]
        out.append(
            {
                "rally_index": index,
                "serve_point": (not (kinds & RALLY_PLAY)) if touches else None,
                "score_event_index": whistles[-1] if whistles else None,
                "deciding_event_indices": rally_attacks[-1] if rally_attacks else plays[-1:],
            }
        )
    return out
