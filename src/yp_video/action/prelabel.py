"""Integration helpers for the local yp-spot action spotting model."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

from yp_video.config import (
    SPOT_CHECKPOINTS_DIR,
    SPOT_DIR,
    SPOT_INFERENCE_MODULE,
    SPOT_PACKAGE_DIR,
    SPOT_PYTHON,
)
from yp_video.contracts.action import ACTION_LABELS, SPOT_PACKAGE_TYPE
from yp_video.core.checkpoints import checkpoint_ref, is_under, resolve_ref

_BEST_CHECKPOINT = "checkpoint_best.pt"


def spot_available() -> bool:
    return (
        SPOT_DIR.exists()
        and SPOT_PYTHON.exists()
        and (SPOT_PACKAGE_DIR / "inference.py").exists()
    )


def list_checkpoints(
    root: Path = SPOT_CHECKPOINTS_DIR,
    *,
    task: str | None = None,
    package_type: str = SPOT_PACKAGE_TYPE,
) -> list[dict]:
    """Checkpoint packages under ``root``, one row per package.

    ``task`` keeps only packages whose manifest serves it, and points the row
    at that task's own best-epoch weights (``best_per_task[task].file``) —
    a fusion run's action-best and actor-best epochs rarely coincide, and
    serving a task its selection-criterion epoch quietly hands it a
    compromised head. ``package_type`` selects the family (the independent
    association trainer has its own).
    """
    checkpoints = []
    for run_dir in _iter_package_dirs(root):
        manifest = _load_json(run_dir / "manifest.json")
        if manifest.get("type") != package_type:
            continue
        tasks = list(manifest.get("tasks") or [])
        if task is not None and task not in tasks:
            continue
        pick = ((manifest.get("best_per_task") or {}).get(task) or {}) if task else {}
        path = run_dir / (pick.get("file") or _BEST_CHECKPOINT)
        if not path.is_file():
            continue
        best = manifest.get("best") if isinstance(manifest.get("best"), dict) else {}
        stat = path.stat()
        checkpoints.append({
            "path": checkpoint_ref(path),
            "name": f"{run_dir.name}/{path.name}",
            "experiment": run_dir.name,
            "epoch": int(pick.get("epoch", best.get("epoch", -1))),
            "is_best": True,
            "best_metric": pick.get("metric", best.get("metric")),
            "best_value": pick.get("value", best.get("value")),
            "mtime": stat.st_mtime,
            "size_mb": stat.st_size / (1024 * 1024),
            "source": root.name,
            "tasks": tasks,
            "recipe": manifest.get("recipe"),
        })
    checkpoints.sort(key=lambda c: (c["mtime"], c["epoch"]), reverse=True)
    return checkpoints


def default_checkpoint(
    root: Path = SPOT_CHECKPOINTS_DIR, *, task: str | None = None
) -> Path | None:
    checkpoints = list_checkpoints(root, task=task)
    if not checkpoints:
        return None
    return resolve_checkpoint(checkpoints[0]["path"], root=root)


def resolve_checkpoint(
    value: str | Path | None, root: Path = SPOT_CHECKPOINTS_DIR, *, task: str | None = None
) -> Path:
    if value:
        path = resolve_checkpoint_path(value, root=root)
    else:
        path = default_checkpoint(root, task=task)
        if path is None:
            raise FileNotFoundError(f"No SPOT checkpoint found under {root}")

    resolved = path.resolve()
    if not is_under(resolved, root):
        raise ValueError(f"SPOT checkpoint must live under {root}")
    if not resolved.exists():
        raise FileNotFoundError(f"SPOT checkpoint not found: {resolved}")
    if resolved.suffix != ".pt":
        raise ValueError("SPOT checkpoint must be a .pt file")
    return resolved


def _iter_package_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(d for d in root.iterdir() if d.is_dir() and (d / "manifest.json").is_file())


def _load_json(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def resolve_checkpoint_path(
    value: str | Path, root: Path = SPOT_CHECKPOINTS_DIR
) -> Path:
    """``core.checkpoints.resolve_ref`` with this package's default root."""
    return resolve_ref(value, root)


def build_command(
    *,
    video_path: Path | list[Path],
    checkpoint_path: Path,
    save_dir: Path | list[Path],
    batch_size: int,
    num_workers: int,
    clip_len: int,
    prefetch_factor: int | None = None,
    use_amp: bool = True,
    postprocess: bool = True,
    segments: Sequence[tuple[float, float]] | None = None,
) -> list[str]:
    video_paths = [video_path] if isinstance(video_path, Path) else list(video_path)
    save_dirs = [save_dir] if isinstance(save_dir, Path) else list(save_dir)
    if len(save_dirs) not in (1, len(video_paths)):
        raise ValueError("save_dir must contain one path or one path per video")
    if segments is not None and len(video_paths) != 1:
        raise ValueError("segments requires exactly one video_path")

    cmd = [
        str(SPOT_PYTHON),
        "-m", SPOT_INFERENCE_MODULE,
        "--video_path", *(str(path) for path in video_paths),
        "--checkpoint_path", str(checkpoint_path),
        "--save_dir", *(str(path) for path in save_dirs),
        "--batch_size", str(batch_size),
        "--num_workers", str(num_workers),
        "--clip_len", str(clip_len),
    ]
    if prefetch_factor is not None:
        cmd.extend(["--prefetch_factor", str(prefetch_factor)])
    cmd.append("--amp" if use_amp else "--no-amp")
    if not postprocess:
        # Dense/segment models need every per-frame event; score filtering and
        # NMS would shred contiguous runs.
        cmd.append("--no-postprocess")
    if segments is not None:
        cmd.extend([
            "--segments",
            json.dumps([
                [round(float(start), 3), round(float(end), 3)]
                for start, end in segments
            ]),
        ])
    return cmd


def load_predictions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("SPOT prediction output must contain a list")
    return data


def parse_spot_progress(payload: str) -> dict | None:
    """Parse the JSON body of a ``SPOT_PROGRESS`` line (prefix already stripped).

    Returns the record only for inference-phase ticks; ``None`` for malformed
    lines or other phases. The field schema is the SPOT progress protocol in
    ``yp_video.contracts.action``.
    """
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or data.get("phase") != "inference":
        return None
    return data


def spot_progress_fraction(data: dict) -> float:
    """Inference progress in [0, 1] from a parsed SPOT progress record.

    Prefers ``clips_done/clips_total`` and falls back to frame counts.
    """
    clips_total = int(data.get("clips_total") or 0)
    if clips_total > 0:
        ratio = int(data.get("clips_done") or 0) / clips_total
    else:
        total_frames = max(1, int(data.get("total_frames") or 1))
        ratio = int(data.get("end_frame") or 0) / total_frames
    return max(0.0, min(1.0, ratio))


def spot_progress_message(data: dict) -> str:
    """Human-readable status line from a parsed SPOT progress record."""
    end_frame = int(data.get("end_frame") or 0)
    total_frames = int(data.get("total_frames") or 0)
    batch_done = int(data.get("batch_done") or 0)
    batch_total = int(data.get("batch_total") or 0)
    clips_done = int(data.get("clips_done") or 0)
    clips_total = int(data.get("clips_total") or 0)
    frame_text = f"frame {min(end_frame, total_frames)}/{total_frames}" if total_frames > 0 else ""
    batch_text = f"batch {batch_done}/{batch_total}" if batch_total > 0 else ""
    clip_text = f"clip {clips_done}/{clips_total}" if clips_total > 0 else ""
    parts = [part for part in (batch_text, clip_text, frame_text) if part]
    return "SPOT inference " + " · ".join(parts)


def normalize_event(event: dict, *, num_frames: int, min_score: float) -> dict | None:
    """Validate + normalize one raw SPOT event into the annotation shape.

    Returns ``None`` when the event fails the label whitelist or ``min_score``.
    Shared by ``predictions_to_annotation`` (final path) and the progressive
    partial-event path so both normalize identically.
    """
    label = str(event.get("label", "")).lower()
    if label not in ACTION_LABELS:
        return None
    score = _finite_float(event.get("score"), default=1.0)
    if score < min_score:
        return None
    frame = int(round(_finite_float(event.get("frame"), default=0)))
    if num_frames > 0:
        frame = max(0, min(frame, num_frames - 1))
    xy = event.get("xy") or [event.get("x", 0.5), event.get("y", 0.5)]
    if not isinstance(xy, (list, tuple)) or len(xy) < 2:
        xy = [0.5, 0.5]
    x = _clamp(_finite_float(xy[0], default=0.5), 0.0, 1.0)
    y = _clamp(_finite_float(xy[1], default=0.5), 0.0, 1.0)
    return {
        "frame": frame,
        "label": label,
        "xy": [round(x, 4), round(y, 4)],
        # Visibility-head checkpoints predict the flag; older ones emit
        # events without it, and an unannotated contact defaults visible.
        "visible": bool(event.get("visible", True)),
    }


def predictions_to_annotation(
    predictions: list[dict],
    *,
    video_path: Path,
    metadata: dict,
    checkpoint_path: Path,
    min_score: float,
) -> dict:
    record = predictions[0] if predictions else {}
    raw_events = record.get("events") or []
    num_frames = int(metadata.get("num_frames") or 0)
    fps = float(metadata.get("fps") or 0)

    events = []
    for event in raw_events:
        normalized = normalize_event(event, num_frames=num_frames, min_score=min_score)
        if normalized is not None:
            events.append(normalized)

    events.sort(key=lambda e: (e["frame"], e["label"]))
    return {
        "video": video_path.stem,
        "num_frames": num_frames,
        "fps": fps,
        "num_events": len(events),
        "source": {
            "type": "spot",
            "checkpoint": checkpoint_ref(checkpoint_path),
            "min_score": min_score,
            "prediction_video": record.get("video"),
        },
        "events": events,
    }


def _finite_float(value, *, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))
