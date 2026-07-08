"""SPOT rally (segment) training and prediction helpers.

Rally annotations (start/end seconds per rally, ``rally-annotations/``) train
the same yp-spot model the action workflow uses, but as *dense segments*: every
frame between start and end is the "rally" class.

Native-fps frame caches are far too large for full matches (~1 TB for the
current library), so rally training extracts frames at a reduced rate and
writes the labels in that reduced-fps frame space — frame ``i`` of the cache is
the video at ``i / extract_fps`` seconds, and the label records carry
``fps = extract_fps``. yp-spot trains on them unchanged; at inference its
``sample_fps`` handling re-samples the native video to the same temporal
density and reports native frame numbers back.
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Callable
from pathlib import Path

from yp_video.config import (
    ANNOTATIONS_DIR,
    RALLY_SPOT_CHECKPOINTS_DIR,
    RALLY_SPOT_FRAMES_DIR,
    cut_kind_of,
    find_cut,
)
from yp_video.action import prelabel
from yp_video.action.frames import inspect_action_frame_cache
from yp_video.action.predict import SpotInferenceError, run_spot_inference
from yp_video.core.ffmpeg import FFmpegError, probe_video_metadata
from yp_video.core.jsonl import read_jsonl, write_jsonl

log = logging.getLogger(__name__)

RALLY_LABEL = "rally"
RALLY_LABEL_FILE_SUFFIX = "_rally.jsonl"

# Deterministic subset selection: the same limit always picks the same videos,
# so their frame caches are reused across runs.
_SUBSET_SEED = 42


def frame_cache_root(extract_fps: float) -> Path:
    return RALLY_SPOT_FRAMES_DIR / f"fps{extract_fps:g}"


def annotation_files() -> list[Path]:
    return sorted(ANNOTATIONS_DIR.glob("*_annotations.jsonl"))


def rally_stats() -> dict:
    """Library-wide rally annotation totals for the training page."""
    videos = 0
    rallies = 0
    rally_seconds = 0.0
    total_seconds = 0.0
    for path in annotation_files():
        try:
            meta, rows = read_jsonl(path)
        except (OSError, json.JSONDecodeError):
            continue
        videos += 1
        total_seconds += float(meta.get("duration") or 0)
        for row in rows:
            try:
                span = float(row["end"]) - float(row["start"])
            except (KeyError, TypeError, ValueError):
                continue
            if span > 0:
                rallies += 1
                rally_seconds += span
    return {
        "label_dir": str(ANNOTATIONS_DIR),
        "videos": videos,
        "rallies": rallies,
        "rally_hours": rally_seconds / 3600,
        "total_hours": total_seconds / 3600,
    }


def select_training_items(limit: int = 0) -> tuple[list[tuple[Path, Path]], list[str]]:
    """Pick ``(annotation_file, video_path)`` pairs for a training run.

    ``limit`` = 0 uses every annotated video with a resolvable cut; a positive
    limit takes a seeded-shuffle subset, so growing the limit only *adds*
    freshly extracted caches. Annotations whose cut is missing locally are
    skipped and reported, not fatal — the library syncs incrementally.
    """
    items: list[tuple[Path, Path]] = []
    missing: list[str] = []
    for path in annotation_files():
        stem = path.name.removesuffix("_annotations.jsonl")
        video_path = find_cut(f"{stem}.mp4")
        if video_path is None:
            missing.append(f"{stem}.mp4")
            continue
        items.append((path, video_path))

    if missing:
        log.warning(
            "rally-spot: %d annotation(s) without a local cut (e.g. %s)",
            len(missing), ", ".join(missing[:3]),
        )
    if limit and limit < len(items):
        random.Random(_SUBSET_SEED).shuffle(items)
        items = sorted(items[:limit])
    return items, missing


def write_training_labels(
    items: list[tuple[Path, Path]],
    *,
    cache_root: Path,
    extract_fps: float,
    label_dir: Path,
) -> dict:
    """Write per-video SPOT segment labels in the reduced-fps frame space.

    Each rally ``[start, end]`` in seconds becomes one event
    ``{"frame", "end_frame", "label": "rally"}`` clamped to the extracted frame
    count. The frame caches must already exist (``ensure_action_frame_caches``
    with the same root/fps runs first).
    """
    label_dir.mkdir(parents=True, exist_ok=True)
    for stale in label_dir.glob(f"*{RALLY_LABEL_FILE_SUFFIX}"):
        stale.unlink()

    videos = 0
    rallies = 0
    total_frames = 0
    rally_frames = 0
    for ann_path, video_path in items:
        meta, rows = read_jsonl(ann_path)
        cache = inspect_action_frame_cache(
            video_path, cache_root=cache_root, fps=extract_fps
        )
        num_frames = int(cache.get("frame_count") or 0)
        if not cache.get("ready") or num_frames <= 0:
            raise RuntimeError(f"Missing rally frame cache for {video_path.stem}")

        events = []
        for row in rows:
            try:
                start = float(row["start"])
                end = float(row["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if end <= start:
                continue
            first = max(0, min(int(round(start * extract_fps)), num_frames - 1))
            last = max(first, min(int(round(end * extract_fps)), num_frames - 1))
            events.append({
                "frame": first,
                "end_frame": last,
                "label": str(row.get("label") or RALLY_LABEL),
            })
            rally_frames += last - first + 1

        stem = video_path.stem
        write_jsonl(
            label_dir / f"{stem}{RALLY_LABEL_FILE_SUFFIX}",
            {
                "video": stem,
                "fps": extract_fps,
                "num_frames": num_frames,
                "camera_view": cut_kind_of(video_path),
                "source_annotation": ann_path.name,
                "source_duration": meta.get("duration"),
            },
            events,
        )
        videos += 1
        rallies += len(events)
        total_frames += num_frames

    if videos == 0:
        raise RuntimeError(f"No rally annotations produced any labels in {ANNOTATIONS_DIR}")

    return {
        "label_dir": str(label_dir),
        "source_label_dir": str(ANNOTATIONS_DIR),
        "extract_fps": extract_fps,
        "videos": videos,
        "rallies": rallies,
        "frames": total_frames,
        "rally_frames": rally_frames,
    }


def events_to_rally_segments(
    events: list[dict],
    *,
    native_fps: float,
    min_score: float,
    max_gap_s: float,
    min_duration_s: float,
) -> list[dict]:
    """Merge per-frame rally predictions into ``{start, end, label, score}``.

    ``events`` are yp-spot inference events in native frame numbers (one per
    sampled frame the model called foreground). Frames closer than
    ``max_gap_s`` join one segment — this both bridges the sampling stride and
    heals brief mid-rally flickers; segments shorter than ``min_duration_s``
    are dropped as noise. ``score`` is the mean per-frame confidence of the
    frames that formed the segment.
    """
    if native_fps <= 0:
        raise ValueError(f"native_fps must be positive, got {native_fps}")

    ticks = sorted(
        (event["frame"] / native_fps, float(event.get("score", 1.0)))
        for event in events
        if float(event.get("score", 1.0)) >= min_score
    )

    segments: list[dict] = []
    for t, score in ticks:
        if segments and t - segments[-1]["end"] <= max_gap_s:
            segments[-1]["end"] = t
            segments[-1]["scores"].append(score)
        else:
            segments.append({"start": t, "end": t, "scores": [score]})

    return [
        {
            "start": round(s["start"], 2),
            "end": round(s["end"], 2),
            "label": RALLY_LABEL,
            "score": round(sum(s["scores"]) / len(s["scores"]), 4),
        }
        for s in segments
        if s["end"] - s["start"] >= min_duration_s
    ]


def predict_rally_segments(
    video_path: Path,
    *,
    checkpoint_path: str | Path,
    min_score: float = 0.5,
    max_gap_s: float = 2.0,
    min_duration_s: float = 4.0,
    batch_size: int = 8,
    num_workers: int = 4,
    clip_len: int = 64,
    use_amp: bool = True,
    on_message: Callable[[str], None] | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> list[dict]:
    """Run SPOT rally inference on one video and return merged rally segments.

    Router-free entry point shared by the web dashboard flow and the selfhost
    GPU worker, symmetric to ``yp_video.action.predict.predict_actions_to_jsonl``.

    ``checkpoint_path`` must live under ``rally-spot-checkpoints`` and may name
    either a run directory (its ``checkpoint_best.pt`` is used) or a ``.pt``
    file directly. Returns ``{start, end, label, score}`` dicts in seconds,
    timeline order.

    Raises:
        SpotInferenceError: yp-spot is unavailable, the checkpoint does not
            resolve, the video cannot be probed, or inference failed.
    """
    def _msg(text: str) -> None:
        if on_message:
            on_message(text)

    checkpoint = Path(checkpoint_path).expanduser()
    if checkpoint.is_dir():
        checkpoint = checkpoint / "checkpoint_best.pt"
    try:
        checkpoint = prelabel.resolve_checkpoint(
            checkpoint, root=RALLY_SPOT_CHECKPOINTS_DIR
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SpotInferenceError(f"Rally checkpoint unavailable: {exc}") from exc

    _msg("Reading video metadata...")
    try:
        metadata = probe_video_metadata(video_path)
    except FFmpegError as exc:
        raise SpotInferenceError(str(exc)) from exc

    _msg("Running SPOT rally inference...")
    # postprocess=False: the dense segment model needs every per-frame event;
    # score filtering and NMS would shred contiguous runs.
    predictions = run_spot_inference(
        video_path,
        checkpoint=checkpoint,
        batch_size=batch_size,
        num_workers=num_workers,
        clip_len=clip_len,
        use_amp=use_amp,
        postprocess=False,
        on_progress=on_progress,
    )
    events = (predictions[0].get("events") or []) if predictions else []
    segments = events_to_rally_segments(
        events,
        native_fps=float(metadata["fps"]),
        min_score=min_score,
        max_gap_s=max_gap_s,
        min_duration_s=min_duration_s,
    )
    _msg(f"Merged {len(events)} rally frames into {len(segments)} rallies")
    return segments
