"""SPOT rally prediction router.

Runs a trained rally segment model (``rally-spot-checkpoints``) over cut videos
and writes the merged rally spans to ``rally-spot-pre-annotations`` — same file
format as the VLM detect flow but a separate directory, so the two model
families never overwrite each other. Rally Label loads either source.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from yp_video import rally_spot
from yp_video.config import (
    RALLY_ANNOTATIONS_DIR,
    RALLY_PRE_ANNOTATIONS_DIR,
    RALLY_SPOT_CHECKPOINTS_DIR,
    RALLY_SPOT_PRE_ANNOTATIONS_DIR,
    SPOT_DIR,
    cut_kind_of,
    find_cut,
    iter_all_cuts,
)
from yp_video.action import prelabel
from yp_video.contracts.action import (
    ACTION_CONTRACT_VERSION,
    ACTION_CONTRACT_VERSION_ENV,
)
from yp_video.core.ffmpeg import probe_video_metadata
from yp_video.core.jsonl import write_jsonl
from yp_video.web.job_helpers import (
    TERMINAL_ITEM_STATUSES,
    ProgressParser,
    batch_items_params,
    batch_message,
    fail_job_from_exc,
    finalize_batch_job,
    init_batch_items,
    mark_batch_item,
    stop_vllm_for_job,
    stream_subprocess,
    update_batch_item,
)
from yp_video.web.jobs import job_manager
from yp_video.web.r2_client import sync_to_r2

log = logging.getLogger(__name__)
router = APIRouter()


class RallyPredictRequest(BaseModel):
    videos: list[str]
    checkpoint: str = ""
    # Per-frame score floor before merging; argmax already implies ~0.5 for the
    # binary rally/background model, so this mainly trims low-confidence edges.
    min_score: float = Field(default=0.5, ge=0.0, le=1.0)
    # Frames closer than this join one rally; also bridges the sampling stride.
    max_gap_s: float = Field(default=2.0, ge=0.0, le=30.0)
    min_duration_s: float = Field(default=4.0, ge=0.0, le=60.0)
    batch_size: int = Field(default=8, ge=1, le=64)
    clip_len: int = Field(default=64, ge=8, le=256)
    num_workers: int = Field(default=4, ge=1, le=32)
    prefetch_factor: int | None = Field(default=None, ge=1, le=16)
    use_amp: bool = True
    overwrite: bool = False
    stop_vllm: bool = False


def _pre_annotation_path(stem: str) -> Path:
    return RALLY_SPOT_PRE_ANNOTATIONS_DIR / f"{stem}_annotations.jsonl"


@router.get("/videos")
def list_videos() -> list[dict]:
    results = []
    for f in sorted(iter_all_cuts(), key=lambda p: p.name):
        results.append({
            "name": f.name,
            "kind": cut_kind_of(f),
            "has_annotation": (RALLY_ANNOTATIONS_DIR / f"{f.stem}_annotations.jsonl").exists(),
            "has_pre_annotation": _pre_annotation_path(f.stem).exists(),
            "has_vlm_pre_annotation": (
                RALLY_PRE_ANNOTATIONS_DIR / f"{f.stem}_annotations.jsonl"
            ).exists(),
        })
    return results


@router.get("/spot")
def spot_info() -> dict:
    available = prelabel.spot_available()
    info: dict = {"available": available, "spot_dir": str(SPOT_DIR)}
    if not available:
        info["error"] = f"yp-spot not found at {SPOT_DIR}"
        return info
    checkpoints = prelabel.list_checkpoints(RALLY_SPOT_CHECKPOINTS_DIR)
    default = prelabel.default_checkpoint(RALLY_SPOT_CHECKPOINTS_DIR)
    info["checkpoints"] = checkpoints
    info["default_checkpoint"] = prelabel.checkpoint_ref(default) if default else ""
    if not checkpoints:
        info["error"] = (
            f"No rally checkpoints under {RALLY_SPOT_CHECKPOINTS_DIR}; "
            "train one on the Rally SPOT Train page first."
        )
    return info


def _save_rally_pre_annotation(
    *,
    video_path: Path,
    predictions_file: Path,
    checkpoint: Path,
    req: RallyPredictRequest,
) -> dict:
    predictions = prelabel.load_predictions(predictions_file)
    events = (predictions[0].get("events") or []) if predictions else []
    metadata = probe_video_metadata(video_path)
    segments = rally_spot.events_to_rally_segments(
        events,
        native_fps=float(metadata["fps"]),
        min_score=req.min_score,
        max_gap_s=req.max_gap_s,
        min_duration_s=req.min_duration_s,
    )
    write_jsonl(
        _pre_annotation_path(video_path.stem),
        {
            "video": str(video_path),
            "duration": float(metadata["duration"]),
            "source": {
                "type": "rally-spot",
                "checkpoint": prelabel.checkpoint_ref(checkpoint),
                "min_score": req.min_score,
                "max_gap_s": req.max_gap_s,
                "min_duration_s": req.min_duration_s,
            },
        },
        segments,
    )
    return {"video": video_path.stem, "rallies": len(segments)}


@router.post("/start")
async def start(req: RallyPredictRequest) -> dict:
    if not req.videos:
        raise HTTPException(400, "No videos selected")
    try:
        checkpoint = prelabel.resolve_checkpoint(
            req.checkpoint, root=RALLY_SPOT_CHECKPOINTS_DIR
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc

    video_paths: list[Path] = []
    skipped: list[str] = []
    for name in req.videos:
        path = find_cut(name)
        if path is None:
            raise HTTPException(404, f"Video not found: {name}")
        if not req.overwrite and _pre_annotation_path(path.stem).exists():
            skipped.append(path.stem)
            continue
        video_paths.append(path)

    if not video_paths:
        raise HTTPException(
            400, "All selected videos already have pre-annotations (enable overwrite)"
        )

    total = len(video_paths)
    job = job_manager.create_job(
        "rally_spot_predict",
        {
            "videos": [p.name for p in video_paths],
            "skipped_existing": skipped,
            "checkpoint": prelabel.checkpoint_ref(checkpoint),
            "items": init_batch_items([p.name for p in video_paths]),
        },
        name=f"Rally SPOT Predict ({total} videos)",
    )

    async def run_job() -> None:
        items = job.params["items"]
        try:
            await job_manager.update_job(
                job.id, status="running", message="Waiting for inference slot..."
            )
            with tempfile.TemporaryDirectory(prefix="rally-spot-") as tmp:
                tmp_dir = Path(tmp)
                # One subprocess for the whole batch: the model loads once and
                # yp-spot writes per-video predictions to <tmp>/<stem>/.
                cmd = prelabel.build_command(
                    video_path=video_paths,
                    checkpoint_path=checkpoint,
                    save_dir=[tmp_dir / p.stem for p in video_paths],
                    batch_size=req.batch_size,
                    num_workers=req.num_workers,
                    clip_len=req.clip_len,
                    prefetch_factor=req.prefetch_factor,
                    use_amp=req.use_amp,
                    postprocess=False,
                )

                state = {"index": 0}

                def current_video() -> str:
                    return video_paths[state["index"]].name

                def on_video_start(match: re.Match) -> dict:
                    prev = state["index"]
                    state["index"] = int(match.group(1)) - 1
                    if state["index"] > prev:
                        mark_batch_item(
                            items, prev, progress=1.0,
                            message="inference done — converting after batch",
                        )
                    mark_batch_item(
                        items, state["index"],
                        status="running", message="preparing first batch",
                    )
                    return {
                        "progress": state["index"] / total,
                        "message": batch_message(
                            state["index"], total, current_video(), "preparing first batch"
                        ),
                        "params": batch_items_params(items),
                    }

                def on_spot_progress(match: re.Match) -> dict | None:
                    data = prelabel.parse_spot_progress(match.group(1))
                    if data is None:
                        return None
                    frac = prelabel.spot_progress_fraction(data)
                    detail = prelabel.spot_progress_message(data)
                    mark_batch_item(items, state["index"], progress=frac, message=detail)
                    return {
                        "progress": (state["index"] + frac) / total,
                        "message": batch_message(
                            state["index"], total, current_video(), detail,
                        ),
                        "params": batch_items_params(items),
                    }

                env = {
                    **os.environ,
                    "PYTHONUNBUFFERED": "1",
                    ACTION_CONTRACT_VERSION_ENV: ACTION_CONTRACT_VERSION,
                }
                async with stop_vllm_for_job(job.id, when=req.stop_vllm):
                    async with job_manager.inference_lock:
                        rc, last_line = await stream_subprocess(
                            job.id,
                            cmd,
                            cwd=SPOT_DIR,
                            env=env,
                            parsers=[
                                ProgressParser(
                                    r"Starting inference (\d+)/(\d+): (.+)",
                                    on_video_start,
                                ),
                                ProgressParser(
                                    r"^SPOT_PROGRESS (\{.*\})", on_spot_progress
                                ),
                            ],
                            is_key_line=lambda line: "Starting inference" in line,
                            tee_to_terminal=True,
                        )

                failed = 0
                converted: list[dict] = []
                for i, video_path in enumerate(video_paths):
                    predictions_file = tmp_dir / video_path.stem / "predictions.json"
                    if not predictions_file.exists():
                        failed += 1
                        job_manager.get_job(job.id).logs.append(
                            f"[{video_path.stem}] no predictions written"
                        )
                        await update_batch_item(
                            job.id, items, i, status="failed",
                            message="no predictions written",
                            error="no predictions written",
                        )
                        continue
                    try:
                        await update_batch_item(
                            job.id, items, i, message="converting rallies...",
                        )
                        result = await asyncio.to_thread(
                            _save_rally_pre_annotation,
                            video_path=video_path,
                            predictions_file=predictions_file,
                            checkpoint=checkpoint,
                            req=req,
                        )
                        converted.append(result)
                        # Must run on the event loop: sync_to_r2 is a no-op
                        # inside a worker thread (fire-and-forget needs a loop).
                        sync_to_r2(
                            _pre_annotation_path(video_path.stem),
                            "rally-spot/pre-annotations",
                        )
                        await update_batch_item(
                            job.id, items, i, status="completed", progress=1.0,
                            message=f"{result['rallies']} rallies",
                        )
                    except Exception as exc:  # noqa: BLE001
                        failed += 1
                        log.exception("Rally conversion failed for %s", video_path.stem)
                        job_manager.get_job(job.id).logs.append(
                            f"[{video_path.stem}] {type(exc).__name__}: {exc}"
                        )
                        await update_batch_item(
                            job.id, items, i, status="failed",
                            message=f"{type(exc).__name__}: {exc}",
                            error=str(exc),
                        )
                if rc != 0 and failed == 0:
                    raise RuntimeError(
                        last_line or f"SPOT inference exited with code {rc}"
                    )
                await job_manager.update_job(
                    job.id, params={**job.params, "results": converted}
                )
            await finalize_batch_job(job.id, total, failed)
        except asyncio.CancelledError:
            for i in range(len(items)):
                if items[i].get("status") not in TERMINAL_ITEM_STATUSES:
                    mark_batch_item(items, i, status="cancelled", message="Cancelled")
            current = job_manager.get_job(job.id)
            await job_manager.update_job(
                job.id, status="cancelled", message="Cancelled",
                params={**(current.params if current else {}), **batch_items_params(items)},
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("Rally prediction failed")
            await fail_job_from_exc(job.id, exc)

    task = asyncio.create_task(run_job())
    job_manager.attach_task(job, task)
    return job.to_dict()
