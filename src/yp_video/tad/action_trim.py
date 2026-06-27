"""Optional rally refinement for TAD predict: trim each TAD first-pass rally to
its true action boundaries using SPOT action predictions.

A volleyball rally truly starts on a *serve* and ends on a *score*. TAD tends to
mis-place both ends, so for each TAD segment we:

  - take the *first* serve in ``[start - max_gap_s, end]`` and move the start to
    ``serve - serve_pad`` (1 s of lead-in by default), and
  - take the *last* score in ``[start, end + max_gap_s]`` and move the end to
    ``score + score_pad`` (1 s of tail by default).

``max_gap_s`` lets the anchoring serve/score sit slightly *outside* the TAD
segment: the true serve/score is often detected just before/after the imperfect
TAD boundary, so a small window (3 s) recovers it. The cap matters because when
the event was genuinely not detected, the nearest serve/score belongs to a
neighbouring rally tens of seconds away — beyond the cap we treat it as missing
and keep the original TAD boundary.

Design note: this module is intentionally **independent** of the core TAD
predict flow. It only consumes SPOT action predictions already written to disk
(``~/videos/action-pre-annotations/{stem}_actions.jsonl``) and operates on plain
detection dicts. ``infer.run_inference`` calls it lazily and only when
``--trim-with-actions`` is set, so this whole refinement can be swapped for a
different strategy or deleted without touching the TAD pipeline.
"""
from __future__ import annotations

from pathlib import Path

from yp_video.config import ACTION_PRE_ANNOTATIONS_DIR as ACTION_PRE_DIR
from yp_video.core.jsonl import read_jsonl


def default_action_path(video_stem: str) -> Path:
    """Conventional location of a video's SPOT action predictions."""
    return ACTION_PRE_DIR / f"{video_stem}_actions.jsonl"


def load_action_events(path: Path) -> tuple[dict, list[dict]]:
    """Return ``(meta, events)`` from a SPOT action pre-annotation JSONL.

    The first line is the ``_meta`` header (fps, num_frames, ...); the remaining
    lines are per-frame action events, returned sorted by frame.
    """
    meta, events = read_jsonl(path)
    events.sort(key=lambda e: e.get("frame", 0))
    return meta, events


def _event_time(event: dict, fps: float) -> float:
    """Event timestamp in seconds (prefer the explicit ``time`` field)."""
    if event.get("time") is not None:
        return float(event["time"])
    return float(event.get("frame", 0)) / fps


def trim_detections(
    detections: list[dict],
    events: list[dict],
    fps: float,
    *,
    serve_pad: float = 1.0,
    score_pad: float = 1.0,
    max_gap_s: float = 3.0,
    min_rally_s: float = 1.0,
) -> list[dict]:
    """Trim each detection's ``segment`` [start, end] (seconds) to serve/score.

    The anchoring serve is taken from ``[start - max_gap_s, end]`` and the score
    from ``[start, end + max_gap_s]``, so an event detected just outside the TAD
    boundary still anchors the rally; ``max_gap_s=0`` restricts the search to
    strictly inside the segment. Detections without a serve or score within range
    keep the corresponding TAD boundary. A ``trim_via`` field records how each end
    was set:
      - "trimmed"  : both serve and score found within range
      - "no_serve" : no serve found -> original TAD start kept
      - "no_score" : no score found -> original TAD end kept
      - "none"     : neither found -> segment passed through unchanged
    Detections shorter than ``min_rally_s`` after trimming are dropped.
    """
    if fps <= 0:
        fps = 30.0
    serves = sorted(_event_time(e, fps) for e in events if e.get("label") == "serve")
    scores = sorted(_event_time(e, fps) for e in events if e.get("label") == "score")

    out: list[dict] = []
    for det in detections:
        seg = det.get("segment", [0.0, 0.0])
        s0, e0 = float(seg[0]), float(seg[1])
        srv = next((t for t in serves if s0 - max_gap_s <= t <= e0), None)
        scr = next((t for t in reversed(scores) if s0 <= t <= e0 + max_gap_s), None)

        if srv is not None and scr is not None:
            via = "trimmed"
        elif srv is None and scr is None:
            via = "none"
        elif srv is None:
            via = "no_serve"
        else:
            via = "no_score"

        start = max(0.0, srv - serve_pad) if srv is not None else s0
        end = scr + score_pad if scr is not None else e0
        if end - start < min_rally_s:
            continue

        refined = dict(det)
        refined["segment"] = [round(start, 2), round(end, 2)]
        refined["trim_via"] = via
        out.append(refined)
    return out


def refine_detections(
    detections: list[dict],
    video_stem: str,
    *,
    action_path: Path | None = None,
    serve_pad: float = 1.0,
    score_pad: float = 1.0,
    max_gap_s: float = 3.0,
    min_rally_s: float = 1.0,
    on_message=None,
) -> list[dict]:
    """Refine TAD detections using a video's SPOT action predictions on disk.

    If the action prediction file is missing, the detections are returned
    unchanged (with a warning) so the TAD flow degrades gracefully rather than
    failing. Returns a new list; the input is not mutated.
    """
    def _msg(text: str):
        print(text)
        if on_message:
            on_message(text)

    path = Path(action_path) if action_path else default_action_path(video_stem)
    if not path.exists():
        _msg(f"  [trim] action predictions not found ({path}); leaving rallies untrimmed")
        return detections

    meta, events = load_action_events(path)
    fps = float(meta.get("fps") or 30.0)
    trimmed = trim_detections(
        detections, events, fps,
        serve_pad=serve_pad, score_pad=score_pad,
        max_gap_s=max_gap_s, min_rally_s=min_rally_s,
    )
    n_full = sum(1 for d in trimmed if d.get("trim_via") == "trimmed")
    _msg(f"  [trim] {len(detections)} TAD -> {len(trimmed)} trimmed "
         f"({n_full} fully refined) using {path.name}")
    return trimmed
