"""App-shaped projection of a customer result, mirroring iOS Match+Actions.

Volleyball rules (attacks, how a point was decided) are not derived here: they
arrive in the result as ``attacks`` / ``rally_outcomes``, computed once by
``yp_video.action.rules``. This module only frames clips around them and
applies the customer's corrections, as the App does.
"""

from collections import Counter

from yp_video.action.rules import KINDS

from .models import Bundle, Event, Rally, Window


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
    return players


def project(
    bundle: Bundle, *, mode: Window = "full_play", corrected: bool = True
) -> dict:
    correction = bundle.corrections if corrected else None
    shipped = bundle.result.action_events
    events = sorted(shipped, key=lambda e: e.time)
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
    roster = {r.number: r.model_dump() for r in correction.roster} if correction else {}
    players = player_numbers(bundle) if corrected else {}
    annotations = (
        {c.key: c for c in correction.actions + correction.scores} if correction else {}
    )
    hidden_actions = (
        {c.key for c in correction.actions if c.removed} if correction else set()
    )
    hidden_scores = (
        {c.key for c in correction.scores if c.removed} if correction else set()
    )
    memberships = []
    cursor = 0
    for event in events:
        while cursor < len(rallies) and event.time > rallies[cursor].end:
            cursor += 1
        memberships.append(
            rallies[cursor]
            if cursor < len(rallies) and event.time >= rallies[cursor].start
            else None
        )
    # Event → its attack as ordered touches, keyed by identity: the rules name
    # touches by position because two touches on one frame share an id.
    attack_of: dict[int, list[Event]] = {}
    for attack in bundle.result.attacks:
        touches = [shipped[i] for i in attack.event_indices]
        for touch in touches:
            attack_of[id(touch)] = touches
    outcomes = {o.rally_index: o for o in bundle.result.rally_outcomes}
    warnings = []
    if bundle.result.user_id != "local" and bundle.library_rallies is None:
        warnings.append(
            "未提供 Library Rally 快照：使用模型回合界線；App 自由裁切及無哨聲 Score 的 UUID 修正尚無法還原。"
        )
    if bundle.corrections and bundle.corrections.player_identification:
        pi = bundle.corrections.player_identification
        if pi.unit_roster and (
            not bundle.identification or pi.result_id != bundle.identification.job_id
        ):
            warnings.append(
                "人物辨識結果未提供或 result_id 不符；只套用單次觸球指派，未套用整組人物配對。"
            )

    def player(e: Event):
        number = players.get(e.id)
        return roster.get(number) if number is not None else None

    def make(
        anchor: Event,
        chain: list[Event],
        rally: Rally | None,
        key: str,
        next_event: Event | None = None,
        *,
        score: bool = False,
    ) -> dict:
        effective_mode = (
            ("whole_rally" if not chain else "full_play") if score else mode
        )
        start_of_chain = chain[0].time if chain else anchor.time
        if effective_mode == "whole_rally" and rally:
            start, end = max(0, rally.start), max(0, rally.start, rally.end)
        elif score:
            start = max(0, start_of_chain - 1)
            end = max(start, anchor.time + 1)
        else:
            start = max(
                0,
                (start_of_chain if effective_mode == "full_play" else anchor.time) - 1,
            )
            end = max(
                start,
                (next_event.time if next_event else rally.end if rally else anchor.time)
                + 1,
            )
        original = [start, end]
        ann = annotations.get(key)
        if ann and ann.trim_start is not None and ann.trim_end is not None:
            start, end = ann.trim_start, ann.trim_end
        markable = chain[-1] if score and chain else anchor
        return {
            "key": key,
            "kind": kind(anchor),
            "anchor_time": anchor.time,
            "event_id": markable.id,
            "rally_index": rally.index if rally else None,
            "start": start,
            "end": end,
            "default_window": original,
            "player": player(markable),
            "result": ann.result if ann and score else None,
            "loss_reason": ann.loss_reason if ann and score else None,
            "winner": rally.winner if rally and score else None,
            "touches": [
                {"kind": kind(e), "time": e.time, "event_id": e.id, "player": player(e)}
                for e in chain
                if start - 0.01 <= e.time <= end + 0.01
            ],
        }

    actions = []
    standalone = []
    for position, (event, rally) in enumerate(zip(events, memberships)):
        if kind(event) == "score":
            if not rally:
                standalone.append((event, None))
            continue
        attack = attack_of.get(id(event), [event])
        chain = attack[: next(k for k, t in enumerate(attack) if t is event) + 1]
        after = next(
            (
                e
                for e, m in zip(events[position + 1 :], memberships[position + 1 :])
                if (rally is None or m == rally) and e.time > event.time
            ),
            None,
        )
        if clip_key(event) not in hidden_actions:
            actions.append(make(event, chain, rally, clip_key(event), after))

    scores = []
    def whistle(rally: Rally) -> Event | None:
        outcome = outcomes.get(rally.index)
        if outcome is None or outcome.score_event_index is None:
            return None
        return shipped[outcome.score_event_index]

    endings = [(whistle(r), r) for r in rallies] + standalone
    for event, rally in endings:
        key = (
            clip_key(event)
            if event
            else (
                f"rally-{bundle.rally_ids[rally.index]}"
                if rally.index in bundle.rally_ids
                else f"unmapped-rally-{rally.index}"
            )
        )
        if event is None:
            event = Event(id=key, label="score", time=rally.end, frame=0)
        outcome = outcomes.get(rally.index) if rally else None
        chain = (
            [shipped[i] for i in outcome.deciding_event_indices] if outcome else []
        )
        if key not in hidden_scores:
            scores.append(make(event, chain, rally, key, score=True))
    scores.sort(key=lambda c: c["anchor_time"])

    court = {}
    totals = {}
    for r in sorted(rallies, key=lambda r: (r.set, r.start, r.index)):
        tally = totals.setdefault(
            r.set, {"left": 0, "right": 0, "near": 0, "far": 0, "unknown": 0}
        )
        tally[r.winner or "unknown"] += 1
        court[r.index] = dict(tally)
    for c in scores:
        c["court_score"] = court.get(c["rally_index"])
    # Hidden keys still need real identity checks: do not claim unknown keys
    # resolved just because a correction happens to mark them hidden.
    if correction:
        base = project(bundle, mode=mode, corrected=False)
        known = {c["key"] for c in base["actions"] + base["scores"]}
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
                "touches": [
                    {
                        "kind": kind(e),
                        "time": e.time,
                        "event_id": e.id,
                        "player": player(e),
                    }
                    for e in events
                    if r.start - 0.01 <= e.time <= r.end + 0.01
                ],
            }
            for r in rallies
        ],
        "actions": actions,
        "scores": scores,
        "roster": list(roster.values()),
        "court_totals": totals,
        "warnings": warnings,
        "loss_reasons": dict(
            Counter(
                c["loss_reason"] or "未分類" for c in scores if c["result"] == "loss"
            )
        ),
    }
