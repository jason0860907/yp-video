"""SPOT action-label training router."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import traceback
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from yp_video.config import ACTION_ANNOTATIONS_DIR, ACTION_FRAMES_DIR, SPOT_DIR, SPOT_PYTHON, find_cut
from yp_video.core.action_frames import ensure_action_frame_caches
from yp_video.core.jsonl import read_jsonl
from yp_video.web.job_helpers import ProgressParser, stop_vllm_for_job, stream_subprocess
from yp_video.web.jobs import JobStatus, job_manager

log = logging.getLogger(__name__)
router = APIRouter()


class ActionTrainRequest(BaseModel):
    source: str = Field(default="vnl_1_5", pattern="^(vnl_1_5|action_annotations)$")
    dataset: str | None = None
    frame_dir: str | None = None
    save_dir: str | None = None
    init_checkpoint: str | None = "exp/vnl15_official_150/checkpoint_best.pt"
    gpu: int = Field(default=0, ge=0)
    feature_arch: str = "rny008_gsm"
    temporal_arch: str = "gru"
    pred_loc_arch: str = "mlp"
    clip_len: int = Field(default=64, ge=8, le=256)
    batch_size: int = Field(default=8, ge=1, le=64)
    num_epochs: int = Field(default=150, ge=1, le=1000)
    warm_up_epochs: int = Field(default=3, ge=0, le=100)
    learning_rate: float = Field(default=0.001, gt=0)
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


def _count_jsonl_records(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    meta, records = read_jsonl(path)
    return len(records), int(meta.get("num_events") or sum(len(r.get("events", [])) for r in records))


def _action_annotation_stats() -> dict:
    ACTION_ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    videos = 0
    events = 0
    for path in sorted(ACTION_ANNOTATIONS_DIR.glob("*_actions.jsonl")):
        try:
            _meta, records = read_jsonl(path)
        except (OSError, json.JSONDecodeError):
            continue
        videos += 1
        events += len(records)
    return {
        "label_dir": str(ACTION_ANNOTATIONS_DIR),
        "frame_dir": str(ACTION_FRAMES_DIR),
        "videos": videos,
        "events": events,
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

        num_frames = int(meta.get("num_frames") or 0) or None
        items.append((video_path, num_frames))

    if missing:
        sample = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f" and {len(missing) - 5} more"
        raise RuntimeError(f"Missing source video(s) for action labels: {sample}{suffix}")
    return items


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
        "vnl_1_5": _vnl_stats(),
        "action_annotations": _action_annotation_stats(),
        "active_job": _active_job(),
    }


def _build_command(req: ActionTrainRequest) -> tuple[list[str], Path, dict]:
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
    if req.init_checkpoint:
        init_checkpoint = _spot_path(req.init_checkpoint)
        if not init_checkpoint.exists():
            raise HTTPException(400, f"Init checkpoint not found: {init_checkpoint}")
    else:
        init_checkpoint = None

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = _spot_path(req.save_dir or (Path("exp") / f"{_safe_run_name(dataset)}_{stamp}"))
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
        if not any(ACTION_ANNOTATIONS_DIR.glob("*_actions.jsonl")):
            raise HTTPException(400, f"No action JSONL labels found in {ACTION_ANNOTATIONS_DIR}")
        cmd.extend([
            "--label_dir",
            str(ACTION_ANNOTATIONS_DIR),
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
        "init_checkpoint": str(init_checkpoint) if init_checkpoint else "",
        "gpu": req.gpu,
        "epochs": req.num_epochs,
        "feature_arch": req.feature_arch,
        "criterion": req.criterion,
    }
    return cmd, save_dir, params


@router.post("/start")
async def start(req: ActionTrainRequest) -> dict:
    dataset = req.dataset or _default_dataset(req.source)
    initial_params = {
        "source": req.source,
        "dataset": dataset,
        "frame_dir": str(_spot_path(req.frame_dir or _default_frame_dir(req.source))),
        "gpu": req.gpu,
        "epochs": req.num_epochs,
        "feature_arch": req.feature_arch,
        "criterion": req.criterion,
    }
    job = job_manager.create_job(
        "action_train",
        initial_params,
        name=f"SPOT action training ({dataset})",
    )

    async def run_job() -> None:
        try:
            await job_manager.update_job(job.id, status="running", message="Preparing action training...")
            frame_dir = _spot_path(req.frame_dir or _default_frame_dir(req.source))
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

            cmd, save_dir, params = _build_command(req)
            await job_manager.update_job(
                job.id,
                params={**job.params, **params},
                message="Waiting for GPU...",
            )
            async with stop_vllm_for_job(job.id, when=req.stop_vllm):
                async with job_manager.gpu_lock:
                    await job_manager.update_job(job.id, message="Starting SPOT training...")
                    ctx = {"epochs": req.num_epochs}

                    def on_epoch(match: re.Match) -> dict:
                        epoch = int(match.group(1))
                        total = max(1, ctx["epochs"])
                        return {
                            "progress": min((epoch + 1) / total, 0.99),
                            "message": f"Epoch {epoch + 1}/{total}",
                        }

                    def on_config_epochs(match: re.Match) -> dict | None:
                        ctx["epochs"] = int(match.group(1))
                        return None

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
                            ProgressParser(r"Epoch:\s*(\d+)", on_epoch),
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
                        log_path=save_dir / "terminal.log",
                    )
            if rc == 0:
                await job_manager.update_job(
                    job.id,
                    status="completed",
                    progress=1.0,
                    message=f"Training complete: {save_dir}",
                )
            else:
                raise RuntimeError(last_line or f"SPOT training exited with code {rc}")
        except asyncio.CancelledError:
            await job_manager.update_job(job.id, status="cancelled", message="Training cancelled")
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            print(f"\n[action-train] Failed:\n{tb}", flush=True)
            log.error("Action training failed:\n%s", tb)
            job_obj = job_manager.get_job(job.id)
            if job_obj:
                job_obj.logs.append(f"{type(exc).__name__}: {exc}")
                job_obj.logs.extend(tb.splitlines())
            await job_manager.update_job(
                job.id,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                message="SPOT action training failed",
            )

    task = asyncio.create_task(run_job())
    job_manager.attach_task(job, task)
    return job.to_dict()
