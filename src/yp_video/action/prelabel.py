"""Integration helpers for the local yp-spot action spotting model."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from yp_video.config import (
    ACTION_CHECKPOINTS_DIR,
    SPOT_DIR,
    SPOT_INFERENCE_MODULE,
    SPOT_PACKAGE_DIR,
    SPOT_PYTHON,
    VIDEOS_DIR,
)
from yp_video.contracts.action import ACTION_LABELS

_CHECKPOINT_RE = re.compile(r"checkpoint_(\d+)\.pt$")
_BEST_CHECKPOINT = "checkpoint_best.pt"


def spot_available() -> bool:
    return (
        SPOT_DIR.exists()
        and SPOT_PYTHON.exists()
        and (SPOT_PACKAGE_DIR / "inference.py").exists()
    )


def list_checkpoints() -> list[dict]:
    checkpoints = []
    for path in _iter_checkpoint_paths():
        if not path.is_file():
            continue
        match = _CHECKPOINT_RE.match(path.name)
        is_best = path.name == _BEST_CHECKPOINT
        best_metadata = _load_best_metadata(path.parent) if is_best else {}
        epoch = int(match.group(1)) if match else int(best_metadata.get("epoch", -1))
        stat = path.stat()
        rel = checkpoint_ref(path)
        checkpoints.append({
            "path": rel,
            "name": f"{path.parent.name}/{path.name}",
            "experiment": path.parent.name,
            "epoch": epoch,
            "is_best": is_best,
            "best_metric": best_metadata.get("metric"),
            "best_value": best_metadata.get("value"),
            "mtime": stat.st_mtime,
            "size_mb": stat.st_size / (1024 * 1024),
            "source": "action-checkpoints",
        })
    checkpoints.sort(key=lambda c: (c["is_best"], c["mtime"], c["epoch"]), reverse=True)
    return checkpoints


def default_checkpoint() -> Path | None:
    checkpoints = list_checkpoints()
    if not checkpoints:
        return None
    chosen = checkpoints[0]
    return resolve_checkpoint(chosen["path"])


def resolve_checkpoint(value: str | None) -> Path:
    if value:
        path = resolve_checkpoint_path(value)
    else:
        path = default_checkpoint()
        if path is None:
            raise FileNotFoundError(f"No SPOT checkpoint found under {ACTION_CHECKPOINTS_DIR}")

    resolved = path.resolve()
    if not _is_action_checkpoint(resolved):
        raise ValueError("SPOT checkpoint must live under ~/videos/action-checkpoints")
    if not resolved.exists():
        raise FileNotFoundError(f"SPOT checkpoint not found: {resolved}")
    if resolved.suffix != ".pt":
        raise ValueError("SPOT checkpoint must be a .pt file")
    return resolved


def _iter_checkpoint_paths() -> list[Path]:
    if ACTION_CHECKPOINTS_DIR.exists():
        return list(ACTION_CHECKPOINTS_DIR.glob("*/checkpoint_*.pt"))
    return []


def resolve_checkpoint_path(value: str | Path) -> Path:
    """Resolve a possibly-relative action checkpoint path to an absolute one.

    Absolute paths pass through unchanged. A relative path whose first segment is
    the action-checkpoints dir name is taken relative to ``VIDEOS_DIR``; any other
    relative path is taken relative to ``ACTION_CHECKPOINTS_DIR``. Performs no
    existence/containment checks — callers validate.
    """
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == ACTION_CHECKPOINTS_DIR.name:
        return VIDEOS_DIR / path
    return ACTION_CHECKPOINTS_DIR / path


def _is_action_checkpoint(path: Path) -> bool:
    try:
        path.resolve().relative_to(ACTION_CHECKPOINTS_DIR.resolve())
        return True
    except ValueError:
        return False


def checkpoint_ref(path: Path) -> str:
    """Display ref for a checkpoint: path relative to ``VIDEOS_DIR`` if possible.

    All action checkpoints live under ``ACTION_CHECKPOINTS_DIR`` (itself under
    ``VIDEOS_DIR``), so this normally yields ``action-checkpoints/<run>/...``.
    """
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(VIDEOS_DIR.resolve()))
    except ValueError:
        return str(resolved)


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
) -> list[str]:
    video_paths = [video_path] if isinstance(video_path, Path) else list(video_path)
    save_dirs = [save_dir] if isinstance(save_dir, Path) else list(save_dir)
    if len(save_dirs) not in (1, len(video_paths)):
        raise ValueError("save_dir must contain one path or one path per video")

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
    return cmd


def load_predictions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("SPOT prediction output must contain a list")
    return data


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
        label = str(event.get("label", "")).lower()
        if label not in ACTION_LABELS:
            continue
        score = _finite_float(event.get("score"), default=1.0)
        if score < min_score:
            continue
        frame = int(round(_finite_float(event.get("frame"), default=0)))
        if num_frames > 0:
            frame = max(0, min(frame, num_frames - 1))
        xy = event.get("xy") or [event.get("x", 0.5), event.get("y", 0.5)]
        if not isinstance(xy, (list, tuple)) or len(xy) < 2:
            xy = [0.5, 0.5]
        x = _clamp(_finite_float(xy[0], default=0.5), 0.0, 1.0)
        y = _clamp(_finite_float(xy[1], default=0.5), 0.0, 1.0)
        events.append({
            "frame": frame,
            "label": label,
            "xy": [round(x, 4), round(y, 4)],
            "visible": True,
        })

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


def _load_best_metadata(experiment_dir: Path) -> dict:
    path = experiment_dir / "checkpoint_best.json"
    if not path.exists():
        path = experiment_dir / "manifest.json"
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    if "best" in data and isinstance(data["best"], dict):
        return data["best"]
    return data


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))
