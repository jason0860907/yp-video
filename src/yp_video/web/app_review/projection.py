"""App-shaped projection of a customer result: a port of the iOS clip builder.

Volleyball rules (attacks, how a point was decided) are not derived here: they
arrive in the result as ``attacks`` / ``rally_outcomes``, computed once by
``yp_video.action.rules``. What is left is framing clips around them and
applying the customer's corrections, which the App does on-device. This module
mirrors ``Match+Actions.swift`` (``TouchIndex``, ``actionClips``,
``scoreClips``) and ``TouchContext.window``; the shared fixture
``VolleyIQTests/Fixtures/clip_windows.json`` holds both sides to one answer.
"""

from collections import Counter
from dataclasses import dataclass

from yp_video.action.rules import KINDS

from .models import Bundle, Event, Rally, Window

PAD = 1.0


def kind(event: Event) -> str:
    return KINDS.get(event.label.lower(), "other")


def clip_key(event: Event) -> str:
    return f"{event.time:.3f}"


def player_numbers(bundle: Bundle) -> dict[str, int]:
    correction = bundle.corrections
    if not correction or not correction.player_identification:
        return {}
    pi = correction.player_identification
    players = {}
    identification = bundle.identification
    if identification and pi.result_id == identification.job_id:
        for unit in identification.units:
            if unit.key not in pi.removed_units and unit.key in pi.unit_roster:
                players.update(dict.fromkeys(unit.events, pi.unit_roster[unit.key]))
    players.update(pi.event_overrides)
    return {event: number for event, number in players.items() if number is not None}


@dataclass(frozen=True)
class Context:
    """``TouchContext``: one anchor plus what frames a clip around it."""

    anchor: Event
    chain: list[Event]
    prev: Event | None
    next: Event | None
    rally: Rally | None
    key: str

    @property
    def markable(self) -> Event:
        if kind(self.anchor) != "score":
            return self.anchor
        touches = [e for e in self.chain if kind(e) != "score"]
        return touches[-1] if touches else self.prev or self.anchor

    def window(self, mode: Window) -> tuple[float, float]:
        if mode == "whole_rally" and self.rally:
            start = max(0, self.rally.start)
            return start, max(start, self.rally.end)
        anchor = self.anchor.time
        chain_start = self.chain[0].time if self.chain else anchor
        if kind(self.anchor) == "score":
            start = max(0, chain_start - PAD)
            return start, max(start, anchor + PAD)
        result = (
            self.next.time if self.next else self.rally.end if self.rally else anchor
        )
        start = max(0, (chain_start if mode == "full_play" else anchor) - PAD)
        return start, max(start, result + PAD)


class TouchIndex:
    """Events sliced by rally with the backend's rules re-addressed onto them."""

    def __init__(self, events: list[Event], rallies: list[Rally], bundle: Bundle):
        # Sort positions, not values: the rules address events by position.
        order = sorted(range(len(events)), key=lambda i: (events[i].time, i))
        position_of = {shipped: p for p, shipped in enumerate(order)}
        self.all = [events[i] for i in order]
        ordered = sorted(rallies, key=lambda r: r.start)
        # Containment is start <= t <= end with no tolerance, as the backend.
        self.rally_of: list[Rally | None] = [None] * len(self.all)
        self.ranges: dict[int, range] = {}
        cursor = 0
        for position, event in enumerate(self.all):
            while cursor < len(ordered) and event.time > ordered[cursor].end:
                cursor += 1
            if cursor == len(ordered) or event.time < ordered[cursor].start:
                continue
            rally = ordered[cursor]
            self.rally_of[position] = rally
            first = self.ranges.get(rally.index, range(position, position)).start
            self.ranges[rally.index] = range(first, position + 1)
        self.attack_of: dict[int, list[int]] = {}
        for attack in bundle.result.attacks:
            members = [position_of[i] for i in attack.event_indices]
            for position in members:
                self.attack_of[position] = members
        self.outcomes: dict[int, tuple[int | None, list[int]]] = {}
        for outcome in bundle.result.rally_outcomes:
            self.outcomes.setdefault(
                outcome.rally_index,
                (
                    None
                    if outcome.score_event_index is None
                    else position_of[outcome.score_event_index],
                    [position_of[i] for i in outcome.deciding_event_indices],
                ),
            )

    def events_in(self, rally_index: int) -> list[Event]:
        return [self.all[p] for p in self.ranges.get(rally_index, [])]

    def whistle(self, rally_index: int) -> int | None:
        position = self.outcomes.get(rally_index, (None, []))[0]
        if position is None:
            return None
        rally = self.rally_of[position]
        return position if rally and rally.index == rally_index else None

    def inferred_loss_reason(self, rally_index: int, removed: set[str]) -> str | None:
        kinds = [
            kind(e)
            for e in self.events_in(rally_index)
            if kind(e) != "score" and clip_key(e) not in removed
        ]
        return {("serve",): "serve_error", ("serve", "receive"): "receive_error"}.get(
            tuple(kinds)
        )

    def chain(self, position: int) -> list[Event]:
        attack = self.attack_of.get(position)
        if attack is None:
            return [self.all[position]]
        return [self.all[p] for p in attack[: attack.index(position) + 1]]

    def context(self, position: int) -> Context:
        anchor = self.all[position]
        rally = self.rally_of[position]
        scope = self.ranges[rally.index] if rally else range(len(self.all))
        scoped = [self.all[p] for p in scope]
        return Context(
            anchor=anchor,
            chain=self.chain(position),
            prev=next((e for e in reversed(scoped) if e.time < anchor.time), None),
            next=next((e for e in scoped if e.time > anchor.time), None),
            rally=rally,
            key=clip_key(anchor),
        )

    def framing(
        self, position: int, mode: Window
    ) -> tuple[tuple[float, float], list[Event]]:
        """Playback bounds and timeline events; 組織進攻 runs a receive or a
        set on through its attack's spike."""
        original = self.context(position)
        terminal = (
            self.attack_of.get(position, [position])[-1]
            if mode == "full_play"
            else position
        )
        framed = self.context(terminal)
        if mode == "to_next":
            events = [original.anchor]
        elif mode == "full_play":
            events = framed.chain
        else:
            events = (
                self.events_in(original.rally.index)
                if original.rally
                else [original.anchor]
            )
        return framed.window(mode), events

    def ending(self, anchor: Event, key: str, rally: Rally | None) -> Context:
        deciding = (
            [self.all[p] for p in self.outcomes.get(rally.index, (None, []))[1]]
            if rally
            else []
        )
        return Context(
            anchor=anchor,
            chain=deciding,
            prev=deciding[-1] if deciding else None,
            next=None,
            rally=rally,
            key=key,
        )


def court_scores(rallies: list[Rally], winner) -> tuple[dict[int, dict], list[dict]]:
    """``CourtScoreTimeline``: running camera-side points per set."""
    by_rally, totals = {}, []
    for set_number in sorted({r.set for r in rallies}):
        ordered = sorted(
            (r for r in rallies if r.set == set_number),
            key=lambda r: (r.start, r.index),
        )
        observed = {winner(r) for r in ordered} - {None}
        opposite = {"left": "right", "right": "left", "near": "far", "far": "near"}
        sides = [s for s in opposite if s in observed or opposite[s] in observed]
        points = dict.fromkeys(sides, 0)
        unknown = 0
        for rally in ordered:
            if side := winner(rally):
                points[side] += 1
            else:
                unknown += 1
            by_rally[rally.index] = {
                "set": set_number,
                "points": dict(points),
                "unknown": unknown,
            }
        totals.append({"set": set_number, "points": points, "unknown": unknown})
    return by_rally, totals


def project(
    bundle: Bundle, *, mode: Window = "full_play", corrected: bool = True
) -> dict:
    correction = bundle.corrections if corrected else None
    source_rallies = bundle.result.rallies
    if corrected and bundle.library_rallies is not None:
        source_rallies = [r for r in bundle.library_rallies if r.deleted_at is None]
    rallies = sorted(
        (
            r
            for r in source_rallies
            if not correction or r.index not in correction.deleted_rally_indices
        ),
        key=lambda r: (r.start, r.index),
    )
    overrides = correction.rally_winner_overrides if correction else {}

    def winner(rally: Rally | None):
        return rally and (overrides.get(rally.index) or rally.winner)

    roster = {r.number: r.model_dump() for r in correction.roster} if correction else {}
    players = player_numbers(bundle) if corrected else {}
    annotations = (
        {c.key: c for c in correction.actions + correction.scores} if correction else {}
    )
    removed_actions = (
        {c.key for c in correction.actions if c.removed} if correction else set()
    )
    removed_scores = (
        {c.key for c in correction.scores if c.removed} if correction else set()
    )
    index = TouchIndex(bundle.result.action_events, rallies, bundle)
    by_rally, totals = court_scores(rallies, winner)

    def player(e: Event):
        number = players.get(e.id)
        return roster.get(number) if number is not None else None

    def touch(e: Event) -> dict:
        return {"kind": kind(e), "time": e.time, "event_id": e.id, "player": player(e)}

    def clip(
        context: Context,
        number: int,
        window: tuple[float, float],
        timeline: list[Event],
        *,
        inferred: str | None = None,
        score: bool = False,
    ) -> dict:
        ann = annotations.get(context.key)
        trimmed = ann is not None and ann.trim_start is not None
        start, end = (ann.trim_start, ann.trim_end) if trimmed else window
        rally = context.rally
        loss_reason = (ann.loss_reason if ann else None) or inferred
        return {
            "key": context.key,
            "kind": kind(context.anchor),
            "index": number,
            "anchor_time": context.anchor.time,
            "event_id": context.markable.id,
            "rally_index": rally.index if rally else None,
            "set": rally.set if rally else 1,
            "start": start,
            "end": end,
            "default_window": list(window),
            "player": player(context.markable),
            "loss_reason": loss_reason if score else None,
            "loss_reason_inferred": bool(
                score and not (ann and ann.loss_reason) and inferred
            ),
            "winner": winner(rally) if score else None,
            "court_score": by_rally.get(rally.index) if score and rally else None,
            "tag_ids": ann.tag_ids if ann else [],
            "touches": [
                touch(e)
                for e in sorted(
                    index.all if trimmed else timeline, key=lambda e: e.time
                )
                if start <= e.time <= end
            ],
        }

    action_contexts = [
        (p, index.context(p)) for p, e in enumerate(index.all) if kind(e) != "score"
    ]
    actions = [
        clip(context, number, *index.framing(position, mode))
        for number, (position, context) in enumerate(
            (item for item in action_contexts if item[1].key not in removed_actions),
            1,
        )
    ]

    def rally_ending(rally: Rally) -> Context:
        whistle = index.whistle(rally.index)
        if whistle is not None:
            return index.ending(index.all[whistle], clip_key(index.all[whistle]), rally)
        uuid = bundle.rally_ids.get(rally.index)
        key = f"rally-{uuid}" if uuid else f"unmapped-rally-{rally.index}"
        return index.ending(
            Event(id=key, label="score", time=rally.end, frame=0), key, rally
        )

    score_contexts = [rally_ending(r) for r in rallies] + [
        index.ending(e, clip_key(e), None)
        for p, e in enumerate(index.all)
        if kind(e) == "score" and index.rally_of[p] is None
    ]
    score_contexts.sort(key=lambda c: c.anchor.time)
    visible_scores = [c for c in score_contexts if c.key not in removed_scores]
    scores = [
        clip(
            context,
            context.rally.index if context.rally else number,
            context.window("whole_rally" if not context.chain else "full_play"),
            context.chain,
            inferred=context.rally
            and index.inferred_loss_reason(context.rally.index, removed_actions),
            score=True,
        )
        for number, context in enumerate(visible_scores, 1)
    ]

    warnings = []
    if bundle.library_rallies is None:
        warnings.append(
            "未提供 Library Rally：使用模型回合界線；App 裁切及無哨聲 Score 的修正無法還原。"
        )
    if correction and correction.player_identification:
        pi = correction.player_identification
        if pi.unit_roster and (
            not bundle.identification or pi.result_id != bundle.identification.job_id
        ):
            warnings.append(
                "人物辨識結果未提供或 result_id 不符；只套用單次觸球指派，未套用整組人物配對。"
            )
    if correction:
        known = {c.key for _, c in action_contexts} | {c.key for c in score_contexts}
        unresolved = [
            c.key for c in correction.actions + correction.scores if c.key not in known
        ]
        if unresolved:
            warnings.append(
                "無法對應的修正（需要相同分析版本／Rally UUID）："
                + ", ".join(unresolved)
            )
    return {
        "rallies": [
            {
                **r.model_dump(),
                "key": str(r.index),
                "kind": "rally",
                "rally_index": r.index,
                "winner": winner(r),
                "touches": [touch(e) for e in index.events_in(r.index)],
            }
            for r in rallies
        ],
        "actions": actions,
        "scores": scores,
        "roster": list(roster.values()),
        "court_totals": totals,
        "warnings": warnings,
        "loss_reasons": dict(
            Counter(c["loss_reason"] for c in scores if c["loss_reason"])
        ),
    }
