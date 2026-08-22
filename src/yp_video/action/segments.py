"""SPOT action events → the two shapes the rest of the system consumes.

``event_timeline`` is the flat, public, seconds-based event list shipped in the
detector result. It carries *every* label, and each event's ``id`` is the stable
join key that player identification (``IdentifyUnit.event_ids``) points back at.

``pad_and_merge_spans`` turns a rally pass's output into the scan spans action
inference runs over.

Deliberately NOT here: per-action highlight segments. A clip is
``anchor + chain + prev + next + rally``, and every one of those is derivable
from this timeline plus the rally bounds the client already holds — so the
client derives it (iOS ``TouchContext``) and frames the window itself. Shipping
a pre-expanded segment per spike duplicated the same events several times over
and, because it only ever anchored on ``spike``, made a set or a receive
impossible to address as a clip at all.
"""

from __future__ import annotations

from collections.abc import Sequence


def _event_time(ev: dict, fps: float) -> float:
    """Event timestamp in seconds (prefer an explicit ``time`` field)."""
    if ev.get("time") is not None:
        return float(ev["time"])
    return float(ev.get("frame", 0)) / fps if fps > 0 else 0.0


def _public(ev: dict | None) -> dict | None:
    """Project a normalized event to the public, serializable shape."""
    if ev is None:
        return None
    frame = ev.get("frame")
    if frame is None:
        raise ValueError("public action events require a source frame")
    out: dict = {
        "id": str(ev.get("id") or f"f{int(frame)}"),
        "label": ev.get("label"),
        "time": round(ev["_t"], 2),
        "frame": int(frame),
    }
    if ev.get("xy") is not None:
        out["xy"] = ev["xy"]
    return out


def _rally_bounds(rallies: Sequence[dict]) -> list[tuple[float, float]]:
    """``(start, end)`` seconds per rally, timeline-sorted."""
    return sorted(
        (
            (float(r["start"]), float(r["end"]))
            for r in rallies
            if r.get("start") is not None and r.get("end") is not None
        ),
        key=lambda b: b[0],
    )


def pad_and_merge_spans(
    rallies: Sequence[dict],
    *,
    pad_s: float,
    duration_s: float,
) -> list[tuple[float, float]]:
    """Rally segments → padded, merged ``(start, end)`` scan spans in seconds.

    Each rally grows by ``pad_s`` on both sides (so the serve run-up and the
    point's tail stay in view), clamps to ``[0, duration_s]``, and overlapping
    spans merge. This is what turns a rally pass's output into the ``segments``
    argument of action inference.
    """
    spans: list[list[float]] = []
    for start, end in _rally_bounds(rallies):
        lo = max(0.0, start - pad_s)
        hi = min(duration_s, end + pad_s)
        if hi <= lo:
            continue
        if spans and lo <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], hi)
        else:
            spans.append([lo, hi])
    return [(lo, hi) for lo, hi in spans]


def event_timeline(events: Sequence[dict], *, fps: float) -> list[dict]:
    """Flat ``[{id, label, time, frame, xy?}]`` of every spotted event,
    seconds-based and time sorted.

    Carries *all* labels — serve / receive / set / spike / block / score. This
    is the whole action payload of a result: the client projects clips, touch
    timelines and per-player stats from it, and player identification joins to
    it by event ``id``.
    """
    if fps <= 0:
        fps = 30.0
    out = []
    for event in events:
        item = _public({**event, "_t": _event_time(event, fps)})
        if item is not None:
            out.append(item)
    return sorted(out, key=lambda x: x["time"])
