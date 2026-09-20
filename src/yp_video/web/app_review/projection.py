"""Pure port of iOS TouchIndex/TouchContext, Match+Actions and CourtScoreTimeline.

Keep parity cases in tests/test_app_review_projection.py when changing these
rules. Source: VolleyIQ/Models/{ActionEvent,Match+Actions,CourtScore}.swift.
"""

from collections import Counter

from .models import Bundle, Event, Rally, Window

KINDS = {
    **dict.fromkeys(["發", "發球", "serve"], "serve"),
    **dict.fromkeys(["接", "receive", "reception", "dig", "pass"], "receive"),
    **dict.fromkeys(["舉", "set", "setting"], "set"),
    **dict.fromkeys(["打", "扣", "扣球", "spike", "attack", "hit"], "spike"),
    **dict.fromkeys(["攔", "攔網", "block"], "block"),
    **dict.fromkeys(["分", "得分", "score", "point"], "score"),
}
STAGE = {"receive": 0, "set": 1, "spike": 2}


def kind(event: Event) -> str:
    return KINDS.get(event.label.lower(), "other")


def clip_key(event: Event) -> str:
    return f"{event.time:.3f}"


def build_up(before: list[Event], anchor: Event) -> list[Event]:
    chain = []
    cutoff = anchor.time
    for stage in reversed(range(STAGE.get(kind(anchor), 0))):
        found = next(
            (
                e
                for e in reversed(before)
                if STAGE.get(kind(e)) == stage and e.time < cutoff
            ),
            None,
        )
        if found:
            chain.append(found)
            cutoff = found.time
    return list(reversed(chain))


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
    events = sorted(bundle.result.action_events, key=lambda e: e.time)
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
    by_rally = {
        r.index: [e for e, m in zip(events, memberships) if m == r] for r in rallies
    }
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
    last_scores = {}
    standalone = []
    for event, rally in zip(events, memberships):
        scope = by_rally[rally.index] if rally else events
        if kind(event) == "score":
            if rally:
                # Swift keeps the first on equal timestamps.
                if (
                    rally.index not in last_scores
                    or last_scores[rally.index].time < event.time
                ):
                    last_scores[rally.index] = event
            else:
                standalone.append((event, None))
            continue
        before = [e for e in scope if e.time < event.time]
        after = next((e for e in scope if e.time > event.time), None)
        if clip_key(event) not in hidden_actions:
            actions.append(
                make(
                    event,
                    build_up(before, event) + [event],
                    rally,
                    clip_key(event),
                    after,
                )
            )

    scores = []
    endings = [(last_scores.get(r.index), r) for r in rallies] + standalone
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
        scope = by_rally[rally.index] if rally else events
        before = [e for e in scope if e.time < event.time and kind(e) != "score"]
        deciding = next(
            (e for e in reversed(before) if kind(e) == "spike"),
            before[-1] if before else None,
        )
        chain = (build_up(before, deciding) + [deciding]) if deciding else []
        # A non-spike ending has only its deciding touch, no build-up.
        if deciding and kind(deciding) != "spike":
            chain = [deciding]
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
