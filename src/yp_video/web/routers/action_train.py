"""SPOT action-label training router."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from yp_video.config import (
    ACTION_ANNOTATIONS_DIR,
    ACTION_CHECKPOINTS_DIR,
    ACTION_FRAMES_DIR,
    SPOT_DIR,
    SPOT_PYTHON,
    find_cut,
)
from yp_video.action.frames import ensure_action_frame_caches, inspect_action_frame_cache
from yp_video.action.prelabel import resolve_checkpoint_path
from yp_video.core.jsonl import read_jsonl, write_jsonl
from yp_video.web.job_helpers import ProgressParser, stop_vllm_for_job, stream_subprocess
from yp_video.web.jobs import JobStatus, job_manager

log = logging.getLogger(__name__)
router = APIRouter()


@dataclass(slots=True)
class _TrainProgress:
    """Mutable running state for a SPOT training job's progress parsers.

    A dataclass (not a dict) so a mis-typed field raises AttributeError instead
    of silently creating a dead key — the parsers below all mutate this from
    different regex callbacks.
    """

    epochs: int
    completed_epoch: int = -1
    current_epoch: int = 0
    train_total: int = 0
    latest_train_loss: float | None = None
    latest_val_loss: float | None = None
    latest_val_map: float | None = None
    best_epoch: int | None = None
    best_value: float | None = None


class ActionTrainRequest(BaseModel):
    source: str = Field(default="vnl_1_5", pattern="^(vnl_1_5|action_annotations)$")
    training_mode: str = Field(default="split", pattern="^(split|all)$")
    dataset: str | None = None
    frame_dir: str | None = None
    save_dir: str | None = None
    checkpoint_dir: str | None = None
    # None → fall back to the VNL base (_default_init_checkpoint); "" → train
    # from scratch; an explicit path → that checkpoint.
    init_checkpoint: str | None = None
    gpu: int = Field(default=0, ge=0)
    feature_arch: str = "rny008_gsm"
    temporal_arch: str = "gru"
    pred_loc_arch: str = "mlp"
    clip_len: int = Field(default=64, ge=8, le=256)
    batch_size: int = Field(default=8, ge=1, le=64)
    num_epochs: int = Field(default=100, ge=1, le=1000)
    warm_up_epochs: int = Field(default=3, ge=0, le=100)
    learning_rate: float = Field(default=0.0008, gt=0)
    num_workers: int = Field(default=4, ge=0, le=32)
    criterion: str = Field(default="map", pattern="^(map|loss)$")
    start_val_epoch: int = Field(default=0, ge=0)
    epoch_num_frames: int | None = Field(default=None, ge=1)
    val_ratio: float = Field(default=0.2, gt=0, lt=1)
    split_seed: int = 42
    predict_location: bool = True
    stop_vllm: bool = False


def _spot_path(path: str | Path) -> Path:
    p = Path(os.path.expanduser(str(path)))
    if not p.is_absolute():
        p = SPOT_DIR / p
    return p


def _default_frame_dir(source: str) -> str:
    if source == "vnl_1_5":
        return "data/vnl_1.5/frames_224p"
    return str(ACTION_FRAMES_DIR)


def _default_dataset(source: str) -> str:
    return "vnl_1.5" if source == "vnl_1_5" else "yp_actions"


def _default_init_checkpoint() -> str:
    path = SPOT_DIR / "exp" / "vnl15_official_150" / "checkpoint_best.pt"
    return str(path.relative_to(SPOT_DIR)) if path.exists() else ""


def _safe_run_name(dataset: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset).strip("._") or "actions"


def _resolve_save_dir(req: ActionTrainRequest, dataset: str | None = None) -> Path:
    dataset = dataset or req.dataset or _default_dataset(req.source)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return _spot_path(req.save_dir or (Path("exp") / f"{_safe_run_name(dataset)}_{stamp}"))


def _action_checkpoint_path(path: str | Path) -> Path:
    return _validate_action_checkpoint_dir(resolve_checkpoint_path(path))


def _validate_action_checkpoint_dir(path: Path) -> Path:
    root = ACTION_CHECKPOINTS_DIR.resolve()
    resolved = path.expanduser().resolve()
    if resolved.parent != root:
        raise HTTPException(
            400,
            f"Checkpoint dir must be directly under {ACTION_CHECKPOINTS_DIR}",
        )
    return resolved


def _resolve_checkpoint_dir(req: ActionTrainRequest, *, save_dir: Path) -> Path:
    if req.checkpoint_dir:
        return _action_checkpoint_path(req.checkpoint_dir)
    return _validate_action_checkpoint_dir(ACTION_CHECKPOINTS_DIR / save_dir.name)


def _count_jsonl_records(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    meta, records = read_jsonl(path)
    return len(records), int(meta.get("num_events") or sum(len(r.get("events", [])) for r in records))


def _action_annotation_stats() -> dict:
    ACTION_ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    videos = 0
    events = 0
    frames = 0
    for path in sorted(ACTION_ANNOTATIONS_DIR.glob("*_actions.jsonl")):
        try:
            meta, records = read_jsonl(path)
        except (OSError, json.JSONDecodeError):
            continue
        videos += 1
        events += len(records)
        frames += int(meta.get("num_frames") or 0)
    return {
        "label_dir": str(ACTION_ANNOTATIONS_DIR),
        "frame_dir": str(ACTION_FRAMES_DIR),
        "videos": videos,
        "events": events,
        "frames": frames,
        "exists": ACTION_ANNOTATIONS_DIR.exists(),
    }


def _action_frame_items() -> list[tuple[Path, int | None]]:
    items: list[tuple[Path, int | None]] = []
    missing: list[str] = []
    for path in sorted(ACTION_ANNOTATIONS_DIR.glob("*_actions.jsonl")):
        try:
            meta, _records = read_jsonl(path)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot read action labels: {path.name}") from exc

        stem = str(meta.get("video") or path.stem.removesuffix("_actions"))
        video_path = find_cut(f"{stem}.mp4")
        if video_path is None:
            missing.append(f"{stem}.mp4")
            continue

        # Action JSONL metadata can inherit an over-reported MP4 frame count.
        # The training labels are normalized against the extracted cache later.
        items.append((video_path, None))

    if missing:
        sample = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f" and {len(missing) - 5} more"
        raise RuntimeError(f"Missing source video(s) for action labels: {sample}{suffix}")
    return items


# Seconds of slack added on each side of the match window so clips straddling
# the first/last rally boundary are not clipped too tightly.
RALLY_SAMPLE_MARGIN_S = 2.0


def _rally_match_span(meta: dict, num_frames: int) -> tuple[int, int] | None:
    """Frame span ``[first_rally_start, last_rally_end]`` (± margin) for sampling.

    Restricting training clips to this match window keeps the in-rally actions
    *and* the genuine dead time between rallies (real background), while
    excluding the warm-up / post-match regions whose real-but-unlabelled actions
    would otherwise be sampled as background and confuse the model. Returns
    ``None`` when the video has no rallies, so non-rally datasets fall back to
    whole-video sampling.
    """
    rallies = meta.get("rallies") or []
    fps = float(meta.get("fps") or 30.0)
    starts = [float(r["start"]) for r in rallies if r.get("start") is not None]
    ends = [float(r["end"]) for r in rallies if r.get("end") is not None]
    if not starts or not ends:
        return None
    start = max(0, int(round((min(starts) - RALLY_SAMPLE_MARGIN_S) * fps)))
    end = min(num_frames, int(round((max(ends) + RALLY_SAMPLE_MARGIN_S) * fps)))
    if end <= start:
        return None
    return start, end


def _prepare_action_training_labels(*, frame_dir: Path, save_dir: Path) -> dict:
    """Write run-local label copies whose frame counts match the SPOT cache."""

    label_files = sorted(ACTION_ANNOTATIONS_DIR.glob("*_actions.jsonl"))
    if not label_files:
        raise RuntimeError(f"No action JSONL labels found in {ACTION_ANNOTATIONS_DIR}")

    label_dir = save_dir / "labels" / "action-annotations"
    label_dir.mkdir(parents=True, exist_ok=True)
    for stale in label_dir.glob("*_actions.jsonl"):
        stale.unlink()

    videos = 0
    events = 0
    total_frames = 0
    span_frames = 0
    adjusted: list[dict] = []
    for path in label_files:
        try:
            meta, records = read_jsonl(path)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot read action labels: {path.name}") from exc

        stem = str(meta.get("video") or path.stem.removesuffix("_actions"))
        video_path = find_cut(f"{stem}.mp4")
        if video_path is None:
            raise RuntimeError(f"Missing source video for action labels: {stem}.mp4")

        cache = inspect_action_frame_cache(video_path, cache_root=frame_dir)
        cache_frames = int(cache.get("frame_count") or 0)
        if cache_frames <= 0:
            raise RuntimeError(f"Missing action frame cache for {stem}")

        out_of_range = [
            int(round(float(event.get("frame", 0) or 0)))
            for event in records
            if int(round(float(event.get("frame", 0) or 0))) >= cache_frames
        ]
        if out_of_range:
            sample = ", ".join(str(frame) for frame in out_of_range[:5])
            suffix = "" if len(out_of_range) <= 5 else f" and {len(out_of_range) - 5} more"
            raise RuntimeError(
                f"{path.name} has action frame(s) beyond the frame cache "
                f"({cache_frames} frames): {sample}{suffix}"
            )

        original_frames = int(meta.get("num_frames") or 0)
        training_meta = {
            **meta,
            "num_frames": cache_frames,
            "training_num_frames_source": "action_frame_cache",
        }
        if original_frames and original_frames != cache_frames:
            training_meta["source_num_frames"] = original_frames
            adjusted.append({
                "video": stem,
                "source_num_frames": original_frames,
                "training_num_frames": cache_frames,
            })

        match_span = _rally_match_span(meta, cache_frames)
        if match_span is not None:
            training_meta["sample_spans"] = [list(match_span)]
            span_frames += match_span[1] - match_span[0]
        else:
            span_frames += cache_frames

        write_jsonl(label_dir / path.name, training_meta, records)
        videos += 1
        events += len(records)
        total_frames += cache_frames

    return {
        "label_dir": str(label_dir),
        "source_label_dir": str(ACTION_ANNOTATIONS_DIR),
        "videos": videos,
        "events": events,
        "frames": total_frames,
        "sample_frames": span_frames,
        "adjusted": adjusted,
    }


def _load_json_file(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _reset_checkpoint_package_dir(package_dir: Path) -> None:
    package_dir = _validate_action_checkpoint_dir(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    for child in package_dir.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink(missing_ok=True)


def _export_action_checkpoint_package(
    *,
    run_dir: Path,
    package_dir: Path,
    req: ActionTrainRequest,
    cmd: list[str],
    label_summary: dict | None,
) -> dict:
    best_checkpoint = run_dir / "checkpoint_best.pt"
    if not best_checkpoint.exists():
        raise RuntimeError(f"checkpoint_best.pt was not found in {run_dir}")

    _reset_checkpoint_package_dir(package_dir)

    copied: list[str] = []
    for name in (
        "checkpoint_best.pt",
        "checkpoint_best.json",
        "config.json",
        "loss.json",
        "terminal.log",
    ):
        src = run_dir / name
        if src.exists():
            dst = package_dir / name
            shutil.copy2(src, dst)
            copied.append(name)

    src_label_dir = run_dir / "labels" / "action-annotations"
    if src_label_dir.exists():
        dst_label_dir = package_dir / "labels" / "action-annotations"
        dst_label_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_label_dir, dst_label_dir)
        copied.extend(
            str(path.relative_to(package_dir))
            for path in sorted(dst_label_dir.glob("*_actions.jsonl"))
        )

    best = _load_json_file(run_dir / "checkpoint_best.json")
    config = _load_json_file(run_dir / "config.json")
    manifest = {
        "type": "yp-video-action-checkpoint",
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": package_dir.name,
        "source_run_dir": str(run_dir),
        "package_dir": str(package_dir),
        "checkpoint": "checkpoint_best.pt",
        "best": best if isinstance(best, dict) else None,
        "config": config if isinstance(config, dict) else None,
        "training": {
            "source": req.source,
            "training_mode": req.training_mode,
            "dataset": req.dataset or _default_dataset(req.source),
            "frame_dir": str(_spot_path(req.frame_dir or _default_frame_dir(req.source))),
            "init_checkpoint": req.init_checkpoint or "",
            "label_summary": label_summary,
        },
        "command": cmd,
        "files": copied,
        "omitted": [
            "checkpoint_*.pt",
            "optim_*.pt",
            "pred-val.*",
            "*.recall.json.gz",
        ],
    }
    manifest_path = package_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    copied.append("manifest.json")

    return {
        "dir": str(package_dir),
        "checkpoint": str(package_dir / "checkpoint_best.pt"),
        "files": copied,
        "best": manifest["best"],
    }


def _vnl_stats() -> dict:
    base = SPOT_DIR / "data" / "vnl_1.5"
    train_path = base / "train.jsonl"
    val_path = base / "val.jsonl"
    test_path = base / "test.jsonl"
    train_videos, train_events = _count_jsonl_records(train_path)
    val_videos, val_events = _count_jsonl_records(val_path)
    test_videos, test_events = _count_jsonl_records(test_path)
    frame_dir = base / "frames_224p"
    return {
        "dataset": "vnl_1.5",
        "base_dir": str(base),
        "frame_dir": str(frame_dir),
        "frame_dir_exists": frame_dir.exists(),
        "train_jsonl": str(train_path),
        "val_jsonl": str(val_path),
        "train_videos": train_videos,
        "train_events": train_events,
        "val_videos": val_videos,
        "val_events": val_events,
        "test_videos": test_videos,
        "test_events": test_events,
        "ready": train_path.exists() and val_path.exists() and frame_dir.exists(),
    }


def _init_checkpoint_options() -> list[dict]:
    """Selectable init-checkpoint options: the VNL base plus packaged action runs.

    Action checkpoints live under ACTION_CHECKPOINTS_DIR/<run>/checkpoint_best.pt
    and are returned as absolute paths (passed through _spot_path unchanged). The
    VNL base stays SPOT-relative so it resolves under SPOT_DIR.
    """
    options: list[dict] = []
    default = _default_init_checkpoint()
    if default:
        options.append({"label": "VNL 1.5 base (vnl15_official_150)", "value": default})
    if ACTION_CHECKPOINTS_DIR.exists():
        for run_dir in sorted(ACTION_CHECKPOINTS_DIR.iterdir(), reverse=True):
            ckpt = run_dir / "checkpoint_best.pt"
            if not run_dir.is_dir() or not ckpt.is_file():
                continue
            best = _load_json_file(run_dir / "checkpoint_best.json")
            value = best.get("value") if isinstance(best, dict) else None
            label = run_dir.name
            if isinstance(value, (int, float)):
                label = f"{run_dir.name} (mAP {value:.3f})"
            options.append({"label": label, "value": str(ckpt)})
    return options


def _action_checkpoint_stats() -> dict:
    count = 0
    if ACTION_CHECKPOINTS_DIR.exists():
        count = sum(1 for path in ACTION_CHECKPOINTS_DIR.glob("*/checkpoint_best.pt") if path.is_file())
    return {
        "dir": str(ACTION_CHECKPOINTS_DIR),
        "runs": count,
        "exists": ACTION_CHECKPOINTS_DIR.exists(),
    }


def _active_job() -> dict | None:
    for job in job_manager.jobs.values():
        if job.type == "action_train" and job.status == JobStatus.RUNNING:
            return job.to_dict()
    return None


@router.get("/status")
def status() -> dict:
    return {
        "spot_available": SPOT_DIR.exists() and SPOT_PYTHON.exists(),
        "spot_dir": str(SPOT_DIR),
        "spot_python": str(SPOT_PYTHON),
        "default_init_checkpoint": _default_init_checkpoint(),
        "init_checkpoints": _init_checkpoint_options(),
        "vnl_1_5": _vnl_stats(),
        "action_annotations": _action_annotation_stats(),
        "action_checkpoints": _action_checkpoint_stats(),
        "active_job": _active_job(),
    }


def _build_command(
    req: ActionTrainRequest,
    *,
    save_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
    action_label_dir: Path | None = None,
) -> tuple[list[str], Path, dict]:
    if not SPOT_DIR.exists():
        raise HTTPException(503, "SPOT is not available at ~/yp-spot")
    if not SPOT_PYTHON.exists():
        raise HTTPException(503, f"SPOT python not found: {SPOT_PYTHON}")

    dataset = req.dataset or _default_dataset(req.source)
    frame_dir_value = req.frame_dir or _default_frame_dir(req.source)
    frame_dir = _spot_path(frame_dir_value)
    if not frame_dir.exists():
        raise HTTPException(400, f"Frame directory not found: {frame_dir}")

    if req.source == "vnl_1_5":
        for rel in ("data/vnl_1.5/train.jsonl", "data/vnl_1.5/val.jsonl"):
            if not (SPOT_DIR / rel).exists():
                raise HTTPException(400, f"Missing VNL JSONL labels: {SPOT_DIR / rel}")
    init_value = _default_init_checkpoint() if req.init_checkpoint is None else req.init_checkpoint
    if init_value:
        init_checkpoint = _spot_path(init_value)
        if not init_checkpoint.exists():
            raise HTTPException(400, f"Init checkpoint not found: {init_checkpoint}")
    else:
        init_checkpoint = None

    save_dir = save_dir or _resolve_save_dir(req, dataset)
    checkpoint_dir = checkpoint_dir or _resolve_checkpoint_dir(req, save_dir=save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(SPOT_PYTHON),
        "-m",
        "yp_spot.train",
        dataset,
        str(frame_dir),
        "-m",
        req.feature_arch,
        "-t",
        req.temporal_arch,
        "-p",
        req.pred_loc_arch,
        "--clip_len",
        str(req.clip_len),
        "--batch_size",
        str(req.batch_size),
        "--num_epochs",
        str(req.num_epochs),
        "--warm_up_epochs",
        str(req.warm_up_epochs),
        "--learning_rate",
        str(req.learning_rate),
        "--num_workers",
        str(req.num_workers),
        "--criterion",
        req.criterion,
        "--start_val_epoch",
        str(req.start_val_epoch),
        "-s",
        str(save_dir),
    ]
    if req.predict_location:
        cmd.append("--predict_location")
    if init_checkpoint is not None:
        cmd.extend(["--init_checkpoint", str(init_checkpoint)])
    if req.epoch_num_frames is not None:
        cmd.extend(["--epoch_num_frames", str(req.epoch_num_frames)])
    if req.source == "action_annotations":
        label_dir = action_label_dir or ACTION_ANNOTATIONS_DIR
        if not any(label_dir.glob("*_actions.jsonl")):
            raise HTTPException(400, f"No action JSONL labels found in {label_dir}")
        if req.training_mode == "all":
            cmd.extend([
                "--train_labels",
                str(label_dir),
                "--val_labels",
                str(label_dir),
            ])
        else:
            cmd.extend([
                "--label_dir",
                str(label_dir),
                "--val_ratio",
                str(req.val_ratio),
                "--split_seed",
                str(req.split_seed),
            ])

    params = {
        "source": req.source,
        "dataset": dataset,
        "frame_dir": str(frame_dir),
        "save_dir": str(save_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "init_checkpoint": str(init_checkpoint) if init_checkpoint else "",
        "gpu": req.gpu,
        "epochs": req.num_epochs,
        "feature_arch": req.feature_arch,
        "criterion": req.criterion,
    }
    if req.source == "action_annotations":
        params["label_dir"] = str(action_label_dir or ACTION_ANNOTATIONS_DIR)
        params["training_mode"] = req.training_mode
        if req.training_mode == "split":
            params["val_ratio"] = req.val_ratio
            params["split_seed"] = req.split_seed
    return cmd, save_dir, params


@router.post("/start")
async def start(req: ActionTrainRequest) -> dict:
    dataset = req.dataset or _default_dataset(req.source)
    save_dir = _resolve_save_dir(req, dataset)
    checkpoint_dir = _resolve_checkpoint_dir(req, save_dir=save_dir)
    initial_params = {
        "source": req.source,
        "dataset": dataset,
        "frame_dir": str(_spot_path(req.frame_dir or _default_frame_dir(req.source))),
        "save_dir": str(save_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "gpu": req.gpu,
        "epochs": req.num_epochs,
        "feature_arch": req.feature_arch,
        "criterion": req.criterion,
    }
    if req.source == "action_annotations":
        initial_params["training_mode"] = req.training_mode
    job = job_manager.create_job(
        "action_train",
        initial_params,
        name=f"SPOT action training ({dataset})",
    )

    async def run_job() -> None:
        checkpoint_exporter: Callable[..., Awaitable[dict | None]] | None = None
        try:
            await job_manager.update_job(job.id, status="running", message="Preparing action training...")
            frame_dir = _spot_path(req.frame_dir or _default_frame_dir(req.source))
            action_label_dir = None
            label_summary = None
            if req.source == "action_annotations":
                items = await asyncio.to_thread(_action_frame_items)
                if not items:
                    raise RuntimeError(f"No action JSONL labels found in {ACTION_ANNOTATIONS_DIR}")

                loop = asyncio.get_running_loop()

                def frame_progress(done: int, total: int, message: str) -> None:
                    progress = 0.02 + (0.16 * done / total if total else 0.0)
                    loop.call_soon_threadsafe(
                        lambda progress=progress, message=message: asyncio.ensure_future(
                            job_manager.update_job(
                                job.id,
                                progress=progress,
                                message=message,
                            )
                        )
                    )

                summary = await asyncio.to_thread(
                    ensure_action_frame_caches,
                    items,
                    cache_root=frame_dir,
                    progress=frame_progress,
                )
                await job_manager.update_job(
                    job.id,
                    progress=0.18,
                    message="Frame cache ready.",
                    params={**job.params, "frame_cache": summary},
                )
                label_summary = await asyncio.to_thread(
                    _prepare_action_training_labels,
                    frame_dir=frame_dir,
                    save_dir=save_dir,
                )
                action_label_dir = Path(label_summary["label_dir"])
                await job_manager.update_job(
                    job.id,
                    progress=0.2,
                    message="Training labels validated.",
                    params={**job.params, "training_labels": label_summary},
                )

            cmd, resolved_save_dir, params = _build_command(
                req,
                save_dir=save_dir,
                checkpoint_dir=checkpoint_dir,
                action_label_dir=action_label_dir,
            )
            await job_manager.update_job(
                job.id,
                params={**job.params, **params},
                message="Waiting for GPU...",
            )
            async with stop_vllm_for_job(job.id, when=req.stop_vllm):
                async with job_manager.gpu_lock:
                    await job_manager.update_job(job.id, message="Starting SPOT training...")
                    ctx = _TrainProgress(epochs=req.num_epochs)
                    checkpoint_export_lock = asyncio.Lock()
                    checkpoint_export_tasks: set[asyncio.Task] = set()

                    def training_params(**extra) -> dict:
                        return {
                            "action_train_progress": {
                                "epoch": ctx.current_epoch,
                                "epoch_display": ctx.current_epoch + 1,
                                "epochs": max(1, ctx.epochs),
                                "completed_epoch": ctx.completed_epoch,
                                "latest_train_loss": ctx.latest_train_loss,
                                "latest_val_loss": ctx.latest_val_loss,
                                "latest_val_map": ctx.latest_val_map,
                                "best_epoch": ctx.best_epoch,
                                "best_value": ctx.best_value,
                                **extra,
                            }
                        }

                    def phase_progress(epoch: int, phase: str, step: int, total: int) -> float:
                        phase_offsets = {"train": 0.0, "val": 0.78, "map": 0.94}
                        phase_weights = {"train": 0.78, "val": 0.16, "map": 0.06}
                        frac = step / max(1, total)
                        epoch_frac = phase_offsets[phase] + phase_weights[phase] * frac
                        total_epochs = max(1, ctx.epochs)
                        return min(0.99, 0.2 + 0.79 * ((epoch + epoch_frac) / total_epochs))

                    def on_epoch(match: re.Match) -> dict:
                        epoch = int(match.group(1))
                        ctx.completed_epoch = max(ctx.completed_epoch, epoch)
                        ctx.current_epoch = epoch
                        return {
                            "params": training_params(
                                phase="summary",
                                phase_label="Epoch summary",
                            ),
                        }

                    def on_config_epochs(match: re.Match) -> dict | None:
                        ctx.epochs = int(match.group(1))
                        return None

                    def on_tqdm(match: re.Match) -> dict:
                        step = int(match.group("step"))
                        total = int(match.group("total"))
                        tail = match.group("tail") or ""
                        if "sum=" in tail:
                            if total >= int(ctx.train_total or 0):
                                ctx.train_total = total
                                phase = "train"
                                epoch = max(0, int(ctx.completed_epoch) + 1)
                            else:
                                phase = "val"
                                epoch = max(0, int(ctx.current_epoch))
                        else:
                            phase = "map"
                            epoch = max(0, int(ctx.current_epoch))

                        ctx.current_epoch = epoch
                        phase_label = {
                            "train": "Training",
                            "val": "Validation loss",
                            "map": "mAP evaluation",
                        }[phase]
                        loss_match = re.search(r"sum=([0-9.]+)", tail)
                        current_loss = float(loss_match.group(1)) if loss_match else None
                        pct = int(step * 100 / max(1, total))
                        total_epochs = max(1, ctx.epochs)
                        return {
                            "progress": phase_progress(epoch, phase, step, total),
                            "message": (
                                f"Epoch {epoch + 1}/{total_epochs} - "
                                f"{phase_label} {step}/{total} ({pct}%)"
                            ),
                            "params": training_params(
                                phase=phase,
                                phase_label=phase_label,
                                step=step,
                                total=total,
                                phase_progress=step / max(1, total),
                                current_loss=current_loss,
                            ),
                        }

                    def on_train_loss(match: re.Match) -> dict:
                        ctx.latest_train_loss = float(match.group(4))
                        return {"params": training_params()}

                    def on_val_loss(match: re.Match) -> dict:
                        ctx.latest_val_loss = float(match.group(4))
                        return {"params": training_params()}

                    def on_val_map(match: re.Match) -> dict:
                        ctx.latest_val_map = float(match.group(1)) / 100.0
                        return {"params": training_params()}

                    async def export_checkpoint_package_once(
                        *,
                        expected_epoch: int | None,
                        reason: str,
                        update_job: bool = True,
                    ) -> dict | None:
                        for _ in range(120):
                            best = _load_json_file(resolved_save_dir / "checkpoint_best.json")
                            best_epoch = best.get("epoch") if isinstance(best, dict) else None
                            ready = (
                                (resolved_save_dir / "checkpoint_best.pt").exists()
                                and isinstance(best_epoch, int)
                                and (expected_epoch is None or best_epoch == expected_epoch)
                            )
                            if ready:
                                async with checkpoint_export_lock:
                                    summary = await asyncio.to_thread(
                                        _export_action_checkpoint_package,
                                        run_dir=resolved_save_dir,
                                        package_dir=checkpoint_dir,
                                        req=req,
                                        cmd=cmd,
                                        label_summary=label_summary,
                                    )
                                if update_job:
                                    await job_manager.update_job(
                                        job.id,
                                        params={
                                            **job.params,
                                            "checkpoint_package": summary,
                                            "checkpoint_package_reason": reason,
                                        },
                                    )
                                return summary
                            await asyncio.sleep(0.5)

                        log.warning(
                            "Timed out waiting to export action checkpoint package "
                            "for %s (expected_epoch=%s, run_dir=%s)",
                            reason,
                            expected_epoch,
                            resolved_save_dir,
                        )
                        return None

                    checkpoint_exporter = export_checkpoint_package_once

                    def schedule_checkpoint_export(expected_epoch: int | None, reason: str) -> None:
                        task = asyncio.create_task(
                            export_checkpoint_package_once(
                                expected_epoch=expected_epoch,
                                reason=reason,
                            )
                        )
                        checkpoint_export_tasks.add(task)
                        task.add_done_callback(checkpoint_export_tasks.discard)

                    def on_new_best(_match: re.Match) -> dict:
                        ctx.best_epoch = ctx.current_epoch
                        ctx.best_value = (
                            ctx.latest_val_map
                            if req.criterion == "map"
                            else ctx.latest_val_loss
                        )
                        schedule_checkpoint_export(ctx.best_epoch, "new_best")
                        return {"params": training_params()}

                    env = {
                        **os.environ,
                        "PYTHONUNBUFFERED": "1",
                        "PYTHONPATH": (
                            f"{SPOT_DIR}{os.pathsep}{os.environ['PYTHONPATH']}"
                            if os.environ.get("PYTHONPATH")
                            else str(SPOT_DIR)
                        ),
                        "CUDA_VISIBLE_DEVICES": str(req.gpu),
                    }
                    rc, last_line = await stream_subprocess(
                        job.id,
                        cmd,
                        cwd=SPOT_DIR,
                        env=env,
                        parsers=[
                            ProgressParser(r'"num_epochs":\s*(\d+)', on_config_epochs),
                            ProgressParser(
                                r"(?P<pct>\d+)%\|.*?\|\s*(?P<step>\d+)/(?P<total>\d+)\s*\[[^\]]+\](?P<tail>.*)",
                                on_tqdm,
                            ),
                            ProgressParser(r"Epoch:\s*(\d+)", on_epoch),
                            ProgressParser(
                                r"Train loss\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)",
                                on_train_loss,
                            ),
                            ProgressParser(
                                r"Val loss\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)",
                                on_val_loss,
                            ),
                            ProgressParser(
                                r"Harmonic mean \(temporal and spatial mAPs\):\s*([0-9.]+)%",
                                on_val_map,
                            ),
                            ProgressParser(r"New best epoch!", on_new_best),
                        ],
                        is_key_line=lambda line: (
                            "Epoch:" in line
                            or "Best epoch" in line
                            or "New best epoch" in line
                            or "Harmonic mean" in line
                            or "Train loss" in line
                            or "Val loss" in line
                        ),
                        tee_to_terminal=True,
                        terminal_prefix="[action-train] ",
                        log_path=resolved_save_dir / "terminal.log",
                    )
            if rc == 0:
                if checkpoint_exporter is None:
                    raise RuntimeError("Checkpoint package exporter was not initialized")
                checkpoint_summary = await checkpoint_exporter(
                    expected_epoch=None,
                    reason="completed",
                    update_job=False,
                )
                if checkpoint_summary is None:
                    raise RuntimeError(f"Training finished but no checkpoint package was exported to {checkpoint_dir}")
                await job_manager.update_job(
                    job.id,
                    status="completed",
                    progress=1.0,
                    message=f"Training complete: {checkpoint_dir}",
                    params={**job.params, "checkpoint_package": checkpoint_summary},
                )
            else:
                raise RuntimeError(last_line or f"SPOT training exited with code {rc}")
        except asyncio.CancelledError:
            checkpoint_summary = None
            if checkpoint_exporter is not None:
                try:
                    checkpoint_summary = await checkpoint_exporter(
                        expected_epoch=None,
                        reason="cancelled",
                        update_job=False,
                    )
                except Exception:  # noqa: BLE001
                    log.exception("Failed to export action checkpoint package after cancellation")
            await job_manager.update_job(
                job.id,
                status="cancelled",
                message="Training cancelled",
                params={
                    **job.params,
                    **({"checkpoint_package": checkpoint_summary} if checkpoint_summary else {}),
                },
            )
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            print(f"\n[action-train] Failed:\n{tb}", flush=True)
            log.error("Action training failed:\n%s", tb)
            job_obj = job_manager.get_job(job.id)
            checkpoint_summary = None
            if checkpoint_exporter is not None:
                try:
                    checkpoint_summary = await checkpoint_exporter(
                        expected_epoch=None,
                        reason="failed",
                        update_job=False,
                    )
                except Exception:  # noqa: BLE001
                    log.exception("Failed to export action checkpoint package after failure")
            if job_obj:
                job_obj.logs.append(f"{type(exc).__name__}: {exc}")
                job_obj.logs.extend(tb.splitlines())
            await job_manager.update_job(
                job.id,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                message="SPOT action training failed",
                params={
                    **(job_obj.params if job_obj else job.params),
                    **({"checkpoint_package": checkpoint_summary} if checkpoint_summary else {}),
                },
            )

    task = asyncio.create_task(run_job())
    job_manager.attach_task(job, task)
    return job.to_dict()
