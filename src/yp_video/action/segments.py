"""Derive per-action highlight segments from SPOT action events.

Mode-agnostic by design. For each anchor action (default: spike) this emits the
*structure* around it — its build-up chain (receive→set→spike), the rally it
sits in, and the next action — and leaves the choice of clip window to the
consumer via :func:`project_window`. That lets each video, or even each segment,
pick a different window (whole rally / build-up / full play / to-next) **without
re-running inference**: switching mode is a pure re-projection over data already
emitted.

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
# project_window() below is the single source of truth for how each maps to
# [start, end]; the iOS app mirrors these names.
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

        chain = _build_up_chain([e for e in scope if e["_t"] < t])
        chain.append(_public(ev))
        nxt = next((e for e in scope if e["_t"] > t), None)

        segments.append({
            "action": anchor,
            "anchor": _public(ev),
            "chain": chain,
            "rally": (
                {"index": r_idx, "start": r_start, "end": r_end}
                if r is not None else None
            ),
            "next": _public(nxt) if nxt is not None else None,
            # Future enrichment — null now; a re-id / scoring pass fills these.
            "player_id": None,
            "outcome": None,
            "team": None,
        })
    return segments


def project_window(
    segment: dict,
    mode: str = DEFAULT_MODE,
    *,
    pre_pad: float = 1.0,
    post_pad: float = 1.0,
) -> list[float]:
    """Compute ``[start, end]`` seconds for a segment under a window ``mode``.

    Single source of truth for the mode→window mapping; the iOS app mirrors
    this. Degrades gracefully when the data a mode needs is absent (e.g. no
    rally bounds → falls back to a chain/anchor window).
    """
    anchor_t = float(segment["anchor"]["time"])
    chain = segment.get("chain") or [segment["anchor"]]
    chain_start = float(chain[0]["time"])
    rally = segment.get("rally")
    nxt = segment.get("next")
    result_t = float(
        nxt["time"] if nxt is not None
        else (rally["end"] if rally else anchor_t)
    )

    if mode == WINDOW_WHOLE_RALLY and rally:
        return [float(rally["start"]), float(rally["end"])]
    if mode == WINDOW_BUILD_UP:
        return [max(0.0, chain_start - pre_pad), anchor_t + post_pad]
    if mode == WINDOW_TO_NEXT:
        return [max(0.0, anchor_t - pre_pad), result_t + post_pad]
    # WINDOW_FULL_PLAY (default) — and the whole_rally fallback when no rally.
    return [max(0.0, chain_start - pre_pad), result_t + post_pad]
