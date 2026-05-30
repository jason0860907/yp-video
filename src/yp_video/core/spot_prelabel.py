"""Integration helpers for the local yp-spot action spotting model."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from yp_video.config import SPOT_DIR, SPOT_INFERENCE_SCRIPT, SPOT_PRELABELS_DIR, SPOT_PYTHON

ACTION_LABELS = {"serve", "receive", "set", "spike", "block", "score"}
_CHECKPOINT_RE = re.compile(r"checkpoint_(\d+)\.pt$")
_BEST_CHECKPOINT = "checkpoint_best.pt"


def spot_available() -> bool:
    return SPOT_DIR.exists() and SPOT_PYTHON.exists() and SPOT_INFERENCE_SCRIPT.exists()


def list_checkpoints() -> list[dict]:
    if not SPOT_DIR.exists():
        return []
    checkpoints = []
    for path in SPOT_DIR.glob("exp/*/checkpoint_*.pt"):
        if not path.is_file():
            continue
        match = _CHECKPOINT_RE.match(path.name)
        is_best = path.name == _BEST_CHECKPOINT
        best_metadata = _load_best_metadata(path.parent) if is_best else {}
        epoch = int(match.group(1)) if match else int(best_metadata.get("epoch", -1))
        stat = path.stat()
        rel = path.relative_to(SPOT_DIR)
        checkpoints.append({
            "path": str(rel),
            "name": f"{path.parent.name}/{path.name}",
            "experiment": path.parent.name,
            "epoch": epoch,
            "is_best": is_best,
            "best_metric": best_metadata.get("metric"),
            "best_value": best_metadata.get("value"),
            "mtime": stat.st_mtime,
            "size_mb": stat.st_size / (1024 * 1024),
        })
    checkpoints.sort(key=lambda c: (c["is_best"], c["mtime"], c["epoch"]), reverse=True)
    return checkpoints


def default_checkpoint() -> Path | None:
    checkpoints = list_checkpoints()
    if not checkpoints:
        return None
    official_best = [
        c for c in checkpoints
        if c["experiment"] == "vnl15_official_150" and c["is_best"]
    ]
    any_best = [c for c in checkpoints if c["is_best"]]
    official = [c for c in checkpoints if c["experiment"] == "vnl15_official_150"]
    chosen = (official_best or any_best or official or checkpoints)[0]
    return resolve_checkpoint(chosen["path"])


def resolve_checkpoint(value: str | None) -> Path:
    if value:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = SPOT_DIR / path
    else:
        path = default_checkpoint()
        if path is None:
            raise FileNotFoundError("No SPOT checkpoint found under ~/yp-spot/exp")

    resolved = path.resolve()
    spot_root = SPOT_DIR.resolve()
    try:
        resolved.relative_to(spot_root)
    except ValueError as exc:
        raise ValueError("SPOT checkpoint must live under ~/yp-spot") from exc
    if not resolved.exists():
        raise FileNotFoundError(f"SPOT checkpoint not found: {resolved}")
    if resolved.suffix != ".pt":
        raise ValueError("SPOT checkpoint must be a .pt file")
    return resolved


def build_command(
    *,
    video_path: Path,
    checkpoint_path: Path,
    save_dir: Path,
    batch_size: int,
    num_workers: int,
    clip_len: int,
) -> list[str]:
    return [
        str(SPOT_PYTHON),
        str(SPOT_INFERENCE_SCRIPT),
        "--video_path", str(video_path),
        "--checkpoint_path", str(checkpoint_path),
        "--save_dir", str(save_dir),
        "--batch_size", str(batch_size),
        "--num_workers", str(num_workers),
        "--clip_len", str(clip_len),
    ]


def predictions_path(job_id: str, video_stem: str) -> Path:
    safe_stem = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in video_stem)
    return SPOT_PRELABELS_DIR / f"{safe_stem}-{job_id}" / "predictions.json"


def load_predictions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("SPOT predictions.json must contain a list")
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
        })

    events.sort(key=lambda e: (e["frame"], e["label"]))
    return {
        "video": video_path.stem,
        "num_frames": num_frames,
        "fps": fps,
        "num_events": len(events),
        "source": {
            "type": "spot",
            "checkpoint": str(checkpoint_path.relative_to(SPOT_DIR)),
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
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))
