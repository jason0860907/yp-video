"""Optional rally refinement for TAD predict: trim each TAD first-pass rally to
its true action boundaries using SPOT action predictions.

A volleyball rally truly starts on a *serve* and ends on a *score*. TAD tends to
over-shoot both ends, so for each TAD segment we:

  - find the *first* serve inside the segment and move the start to
    ``serve - serve_pad`` (1 s of lead-in by default), and
  - find the *last* score inside the segment and move the end to
    ``score + score_pad`` (1 s of tail by default).

Design note: this module is intentionally **independent** of the core TAD
predict flow. It only consumes SPOT action predictions already written to disk
(``~/videos/action-pre-annotations/{stem}_actions.jsonl``) and operates on plain
detection dicts. ``infer.run_inference`` calls it lazily and only when
``--trim-with-actions`` is set, so this whole refinement can be swapped for a
different strategy or deleted without touching the TAD pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path

ACTION_PRE_DIR = Path.home() / "videos" / "action-pre-annotations"


def default_action_path(video_stem: str) -> Path:
    """Conventional location of a video's SPOT action predictions."""
    return ACTION_PRE_DIR / f"{video_stem}_actions.jsonl"


def load_action_events(path: Path) -> tuple[dict, list[dict]]:
    """Return ``(meta, events)`` from a SPOT action pre-annotation JSONL.

    The first ``{"_meta": true, ...}`` line is the metadata header (fps,
    num_frames, ...); the remaining lines are per-frame action events.
    """
    meta: dict = {}
    events: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if i == 0 and (obj.get("_meta") or ("frame" not in obj and "label" not in obj)):
                meta = obj
                continue
            events.append(obj)
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
    min_rally_s: float = 1.0,
) -> list[dict]:
    """Trim each detection's ``segment`` [start, end] (seconds) to serve/score.

    Detections without a serve or score inside their span keep the corresponding
    TAD boundary. A ``trim_via`` field records how each end was set:
      - "trimmed"  : both serve and score found inside the segment
      - "no_serve" : no serve inside -> original TAD start kept
      - "no_score" : no score inside -> original TAD end kept
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
        srv = next((t for t in serves if s0 <= t <= e0), None)
        scr = next((t for t in reversed(scores) if s0 <= t <= e0), None)

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
        serve_pad=serve_pad, score_pad=score_pad, min_rally_s=min_rally_s,
    )
    n_full = sum(1 for d in trimmed if d.get("trim_via") == "trimmed")
    _msg(f"  [trim] {len(detections)} TAD -> {len(trimmed)} trimmed "
         f"({n_full} fully refined) using {path.name}")
    return trimmed
