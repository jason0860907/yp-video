"""Derive per-action highlight segments from SPOT action events.

Mode-agnostic by design. For each anchor action (default: spike) this emits the
*structure* around it — its build-up chain (receive→set→spike), the rally it
sits in, the previous action, and the next action — and leaves the choice of
clip window to the CLIENT (iOS `ActionSegment.window`, the single projector).
That lets each video, or even each segment, pick a different window (whole rally
/ build-up / full play / to-next) **without re-running inference**: switching
mode is a pure re-projection over data already emitted.

Future enrichment (player_id, outcome, team) rides on the same records as
nullable fields, so a later re-id or scoring pass only *fills values* and never
forces a schema change on the app or DB.

Schema of one record::

    {
      "action": "spike",
      "anchor": {"label": "spike", "time": 13.7, "frame": 410, "xy": [..]},
      "chain":  [ {receive..}, {set..}, {spike..} ],   # build-up, ends on anchor
      "rally":  {"index": 7, "start": 11.0, "end": 18.2} | None,
      "next":   {label, time, frame, xy} | None,        # first action after anchor
      "player_id": None,   # ← filled by a future re-id pass
      "outcome":   None,   # ← kill / error / blocked (future scoring pass)
      "team":      None,
    }
"""

from __future__ import annotations

from collections.abc import Sequence

# ── Window modes — the consumer contract ─────────────────────────────
# These mode NAMES are mirrored by the iOS client (SpikeClipWindow). The
# name→[start,end] projection math lives only in the client (single source of
# truth); backends emit mode-agnostic structure and never project here.
WINDOW_WHOLE_RALLY = "whole_rally"   # 整個 rally
WINDOW_BUILD_UP = "build_up"         # 接舉打: receive→set→spike
WINDOW_FULL_PLAY = "full_play"       # 接舉打 → 結果
WINDOW_TO_NEXT = "to_next"           # 扣球 → 下一個動作

WINDOW_MODES = (WINDOW_WHOLE_RALLY, WINDOW_BUILD_UP, WINDOW_FULL_PLAY, WINDOW_TO_NEXT)
DEFAULT_MODE = WINDOW_FULL_PLAY


def _event_time(ev: dict, fps: float) -> float:
    """Event timestamp in seconds (prefer an explicit ``time`` field)."""
    if ev.get("time") is not None:
        return float(ev["time"])
    return float(ev.get("frame", 0)) / fps if fps > 0 else 0.0


def _public(ev: dict | None) -> dict | None:
    """Project a normalized event to the public, serializable shape."""
    if ev is None:
        return None
    out: dict = {"label": ev.get("label"), "time": round(ev["_t"], 2)}
    if ev.get("frame") is not None:
        out["frame"] = ev["frame"]
    if ev.get("xy") is not None:
        out["xy"] = ev["xy"]
    return out


def _build_up_chain(before: list[dict]) -> list[dict]:
    """receive→set leading up to a spike, from same-rally events before it.

    Best-effort: take the most recent set, then the most recent receive before
    that set. Either may be absent (degrades to a shorter chain).
    """
    chain: list[dict] = []
    last_set = next((e for e in reversed(before) if e.get("label") == "set"), None)
    cutoff = last_set["_t"] if last_set is not None else None
    last_receive = next(
        (e for e in reversed(before)
         if e.get("label") == "receive" and (cutoff is None or e["_t"] < cutoff)),
        None,
    )
    if last_receive is not None:
        chain.append(_public(last_receive))
    if last_set is not None:
        chain.append(_public(last_set))
    return chain


def build_action_segments(
    events: Sequence[dict],
    rallies: Sequence[dict],
    *,
    fps: float,
    anchor: str = "spike",
) -> list[dict]:
    """Build mode-agnostic segment records anchored on each ``anchor`` action.

    Args:
        events: SPOT action events (``{frame|time, label, xy, ...}``).
        rallies: rally dicts carrying a ``segment`` ``[start, end]`` in seconds
            (the worker's TAD detections work directly). Indexed 1-based in
            timeline order to match the public rally numbering.
        fps: frame rate used to convert event frames to seconds.
        anchor: action label to anchor on (default ``"spike"`` — parametric so
            ``"block"`` / ``"serve"`` highlight reels come for free later).

    Returns:
        One record per anchor event; see the module docstring for the schema.
    """
    if fps <= 0:
        fps = 30.0

    evs = sorted(
        ({**e, "_t": _event_time(e, fps)} for e in events),
        key=lambda e: e["_t"],
    )
    bounds = sorted(
        (
            (float(r["segment"][0]), float(r["segment"][1]))
            for r in rallies
            if r.get("segment")
        ),
        key=lambda b: b[0],
    )

    def _rally_of(t: float) -> tuple[int, float, float] | None:
        # 1-based index in timeline order, matching _detections_to_rallies.
        for i, (s, e) in enumerate(bounds, start=1):
            if s <= t <= e:
                return i, s, e
        return None

    segments: list[dict] = []
    for ev in evs:
        if ev.get("label") != anchor:
            continue
        t = ev["_t"]
        r = _rally_of(t)
        if r is not None:
            r_idx, r_start, r_end = r
            scope = [e for e in evs if r_start <= e["_t"] <= r_end]
        else:
            r_idx = r_start = r_end = None
            scope = evs  # no rally bounds — fall back to the global timeline

        before = [e for e in scope if e["_t"] < t]
        chain = _build_up_chain(before)
        chain.append(_public(ev))
        prv = before[-1] if before else None
        nxt = next((e for e in scope if e["_t"] > t), None)

        segments.append({
            "action": anchor,
            "anchor": _public(ev),
            "chain": chain,
            "rally": (
                {"index": r_idx, "start": r_start, "end": r_end}
                if r is not None else None
            ),
            "prev": _public(prv) if prv is not None else None,
            "next": _public(nxt) if nxt is not None else None,
            # Future enrichment — null now; a re-id / scoring pass fills these.
            "player_id": None,
            "outcome": None,
            "team": None,
        })
    return segments


def build_score_segments(
    events: Sequence[dict],
    rallies: Sequence[dict],
    *,
    fps: float,
    anchor: str = "score",
) -> list[dict]:
    """Build segment records anchored on each ``score`` (point-decided) event.

    Unlike :func:`build_action_segments` (anchored on the spike), this anchors on
    the moment the point ends and reaches *back* to the attack that decided it:
    the build-up chain is the 接舉打 (receive→set→spike) of the last spike before
    the score. The consumer frames the clip as ``[chain start … score + pad]``
    (see the app's score-clip projection) so a Score item shows "what lost/won
    the point" plus a beat after the whistle.

    One segment per rally: a rally is one point, so duplicate ``score`` whistles
    within it collapse to a single anchor (the last whistle). Scores outside any
    rally are kept individually.

    These records carry the same nullable ``player_id`` / ``outcome`` fields, so
    a future scoring/re-id pass enriches them in place. ``outcome`` here will
    eventually mean won/lost from our side; today the user marks it by hand.

    Args:
        events: SPOT action events (``{frame|time, label, xy, ...}``).
        rallies: rally dicts carrying a ``segment`` ``[start, end]`` in seconds.
        fps: frame rate used to convert event frames to seconds.
        anchor: action label to anchor on (default ``"score"``).

    Returns:
        One record per rally that contains a score (plus one per rally-less
        score); same schema as ``build_action_segments``, but ``action ==
        anchor`` and ``chain`` ends on the deciding spike (the anchor is the
        score, which sits *after* the chain).
    """
    if fps <= 0:
        fps = 30.0

    evs = sorted(
        ({**e, "_t": _event_time(e, fps)} for e in events),
        key=lambda e: e["_t"],
    )
    bounds = sorted(
        (
            (float(r["segment"][0]), float(r["segment"][1]))
            for r in rallies
            if r.get("segment")
        ),
        key=lambda b: b[0],
    )

    def _rally_of(t: float) -> tuple[int, float, float] | None:
        for i, (s, e) in enumerate(bounds, start=1):
            if s <= t <= e:
                return i, s, e
        return None

    # A rally is one point, so it yields one Score segment. Real footage often
    # carries duplicate `score` whistles a beat apart; collapse each rally's
    # scores to a single anchor — the LAST one, the final whistle being the
    # point's hard end. Scores outside every rally have no group to collapse
    # into, so each stands alone.
    anchor_by_rally: dict[int, dict] = {}
    standalone: list[dict] = []
    for ev in evs:
        if ev.get("label") != anchor:
            continue
        r = _rally_of(ev["_t"])
        if r is None:
            standalone.append(ev)
            continue
        kept = anchor_by_rally.get(r[0])
        if kept is None or ev["_t"] > kept["_t"]:
            anchor_by_rally[r[0]] = ev
    anchors = sorted([*anchor_by_rally.values(), *standalone], key=lambda e: e["_t"])

    segments: list[dict] = []
    for ev in anchors:
        t = ev["_t"]
        r = _rally_of(t)
        if r is not None:
            r_idx, r_start, r_end = r
            scope = [e for e in evs if r_start <= e["_t"] <= r_end]
        else:
            r_idx = r_start = r_end = None
            scope = [e for e in evs if e["_t"] <= t]  # no rally → everything up to the score

        # Reach back to the attack that decided the point: the LAST spike before
        # the score, not merely the touch right before the whistle (which may be
        # a defensive dig, or a duplicate-whistle artefact). Frame its full 接舉扣
        # build-up. With no spike — a serve ace, a block, an opponent error —
        # fall back to the last real touch; there's no attack to show. Other
        # `score` events are skipped so `prev` names the deciding touch.
        before = [e for e in scope if e["_t"] < t and e.get("label") != anchor]
        prev = next(
            (e for e in reversed(before) if e.get("label") == "spike"),
            before[-1] if before else None,
        )
        if prev is not None and prev.get("label") == "spike":
            chain = _build_up_chain([e for e in before if e["_t"] < prev["_t"]])
            chain.append(_public(prev))
        elif prev is not None:
            chain = [_public(prev)]
        else:
            chain = []

        segments.append({
            "action": anchor,
            "anchor": _public(ev),
            "chain": chain,
            "rally": (
                {"index": r_idx, "start": r_start, "end": r_end}
                if r is not None else None
            ),
            "prev": _public(prev) if prev is not None else None,
            "next": None,
            "player_id": None,
            "outcome": None,
            "team": None,
        })
    return segments


def event_timeline(events: Sequence[dict], *, fps: float) -> list[dict]:
    """Flat ``[{label, time}]`` of every spotted event, seconds-based and time
    sorted.

    Unlike the segment builders (which only surface a spike's 接舉打 build-up),
    this carries *all* labels — serve / receive / set / spike / block / score —
    so the app can draw a rally-wide touch timeline with the full action set.
    """
    if fps <= 0:
        fps = 30.0
    out = [
        {"label": e.get("label"), "time": round(_event_time(e, fps), 2)}
        for e in events
    ]
    return sorted(out, key=lambda x: x["time"])


# NOTE: window projection ([start, end] per mode) is NOT done here. The mode
# names above (WINDOW_*) are the contract the iOS client mirrors
# (SpikeClipWindow), but the projection MATH lives only in the client
# (ActionSegment.window) so there is a single source of truth. Backends only
# emit the mode-agnostic structure (anchor / chain / prev / next / rally); the
# client frames the clip. (An earlier `project_window` here duplicated that math
# and was never called in serving — removed to avoid drift.)
