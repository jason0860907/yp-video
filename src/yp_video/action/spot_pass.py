"""One SPOT pass over a video: rally spans and/or action events from one
decode, with progressive delivery of both.

Router-free; shared by the web Inference page (``web.fusion_inference``) and
the selfhost GPU worker, so the merge → pad → cut chain that turns yp-spot's
per-frame output into rallies and in-rally actions exists exactly once.

The rally head needs every frame, so a pass that includes it decodes the
whole video and the action events are cut to the padded rally spans
afterwards. A pass that runs the action head alone (re-scoring actions
against rallies already known) hands those spans to yp-spot up front, so
dead time is never decoded.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from yp_video.action import prelabel
from yp_video.action.predict import SpotInferenceError, run_spot_inference
from yp_video.action.rally import events_to_rally_segments
from yp_video.action.segments import filter_events_to_spans, pad_and_merge_spans
from yp_video.core.ffmpeg import FFmpegError, probe_video_metadata
from yp_video.core.jsonl import write_jsonl

SPOT_TASKS = ("rally", "action")


@dataclass(frozen=True)
class RallyOptions:
    """How per-frame rally scores merge into spans (``events_to_rally_segments``)."""

    min_score: float
    max_gap_s: float
    min_duration_s: float


@dataclass(frozen=True)
class SpotOptions:
    batch_size: int
    #: ffmpeg decode threads; 0 lets ffmpeg pick.
    num_workers: int
    clip_len: int
    use_amp: bool = True


@dataclass(frozen=True)
class SpotPassResult:
    fps: float
    num_frames: int
    duration_s: float
    checkpoint: Path
    action_min_score: float
    #: Merged ``{start, end, label, score[, winner, winner_score]}`` in
    #: seconds, timeline order; None when the pass had no rally head.
    rallies: list[dict] | None
    #: yp-spot action records (native frame numbers), events already cut to
    #: the padded rally spans; None when the pass had no action head.
    actions: list[dict] | None


def run_spot_pass(
    source: str | Path,
    *,
    checkpoint: Path,
    tasks: Sequence[str],
    rally: RallyOptions,
    spot: SpotOptions,
    rally_pad_s: float,
    action_min_score: float = 0.0,
    rallies: Sequence[dict] | None = None,
    on_progress: Callable[[float], None] | None = None,
    on_rallies: Callable[[list[dict]], None] | None = None,
    on_action_events: Callable[[list[dict]], None] | None = None,
) -> SpotPassResult:
    """Run the heads in ``tasks`` (a non-empty subset of ``rally``/``action``)
    over ``source`` in one yp-spot subprocess.

    ``rallies`` (``{start, end}`` dicts in seconds) is the span source for an
    action-only pass; it cannot be combined with the rally head, which
    produces its own. Without it an action-only pass scans the whole video.

    Progressive delivery, both driven by yp-spot's per-head partial stream:
      * ``on_rallies(segments)`` — the settled merged rallies so far, the
        last one held back (inference has not passed its end, so its bounds
        would still move and it could merge with the next); their
        chronological indexing therefore never renumbers.
      * ``on_action_events(events)`` — the cumulative action events so far,
        cut to the rally spans known at that moment and normalized exactly
        like the final annotation (label whitelist, ``min_score``, frame
        clamp) plus ``time`` in seconds.

    Raises ``SpotInferenceError`` when the video cannot be probed or yp-spot
    fails, ``ValueError`` on an invalid task set.
    """
    tasks = tuple(tasks)
    if not tasks or any(task not in SPOT_TASKS for task in tasks):
        raise ValueError(f"tasks must be a non-empty subset of {SPOT_TASKS}, got {tasks}")
    if rallies is not None and "rally" in tasks:
        raise ValueError("rallies can only seed an action-only pass; the rally head produces its own")

    try:
        meta = probe_video_metadata(source)
    except FFmpegError as exc:
        raise SpotInferenceError(str(exc)) from exc
    fps = float(meta["fps"])
    num_frames = int(meta["num_frames"])
    duration_s = float(meta["duration"])

    def merge(events: list[dict]) -> list[dict]:
        return events_to_rally_segments(
            events,
            native_fps=fps,
            min_score=rally.min_score,
            max_gap_s=rally.max_gap_s,
            min_duration_s=rally.min_duration_s,
        )

    def spans_of(segments: Sequence[dict] | None) -> list[tuple[float, float]] | None:
        if segments is None:
            return None
        return pad_and_merge_spans(segments, pad_s=rally_pad_s, duration_s=duration_s)

    fixed_spans = spans_of(rallies) if "rally" not in tasks else None
    rally_events: list[dict] = []

    def _on_events(task: str, events: list[dict]) -> None:
        if task == "rally":
            rally_events[:] = events
            if on_action_events is None and on_rallies is None:
                return
            segments = merge(events)
            if on_rallies is not None and len(segments) > 1:
                on_rallies(segments[:-1])
        elif task == "action" and on_action_events is not None:
            # In a joint pass the spans include the still-open rally, so its
            # touches are shown rather than cut until it closes.
            spans = fixed_spans if "rally" not in tasks else spans_of(merge(rally_events))
            if spans is not None:
                events = filter_events_to_spans([{"events": events}], spans, fps=fps)[0]["events"]
            normalized = []
            for event in events:
                item = prelabel.normalize_event(event, num_frames=num_frames, min_score=action_min_score)
                if item is not None:
                    item["time"] = item["frame"] / fps if fps > 0 else 0.0
                    normalized.append(item)
            on_action_events(normalized)

    predictions = run_spot_inference(
        source,
        checkpoint=checkpoint,
        tasks=tasks,
        batch_size=spot.batch_size,
        num_workers=spot.num_workers,
        clip_len=spot.clip_len,
        use_amp=spot.use_amp,
        segments=fixed_spans,
        on_progress=on_progress,
        on_events=_on_events if (on_rallies or on_action_events) else None,
    )

    merged = None
    if "rally" in tasks:
        records = predictions["rally"]
        merged = merge((records[0].get("events") or []) if records else [])
    actions = None
    if "action" in tasks:
        actions = predictions["action"]
        # An action-only pass was already decoded inside its spans; a joint
        # pass scanned everything and is cut here.
        if "rally" in tasks:
            actions = filter_events_to_spans(actions, spans_of(merged), fps=fps)
    return SpotPassResult(
        fps=fps,
        num_frames=num_frames,
        duration_s=duration_s,
        checkpoint=checkpoint,
        action_min_score=action_min_score,
        rallies=merged,
        actions=actions,
    )


def write_actions_jsonl(result: SpotPassResult, output_path: Path, *, video_path: Path) -> dict:
    """Write the pass's actions as a ``*_actions.jsonl`` (``_meta`` line with
    ``fps``/``num_frames`` + one event per line) and return the annotation
    dict it came from, so callers need not read the file back."""
    if result.actions is None:
        raise ValueError("this pass ran no action head")
    data = prelabel.predictions_to_annotation(
        result.actions,
        video_path=video_path,
        metadata={"fps": result.fps, "num_frames": result.num_frames},
        checkpoint_path=result.checkpoint,
        min_score=result.action_min_score,
    )
    meta = {k: v for k, v in data.items() if k != "events"}
    write_jsonl(output_path, meta, data.get("events", []))
    return data
