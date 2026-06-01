"""SPOT-style point action annotator router."""

import asyncio
import json
import logging
import os
import subprocess
import traceback
from fractions import Fraction
from pathlib import Path
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field, field_validator

from yp_video.config import (
    ACTION_ANNOTATIONS_DIR,
    ANNOTATIONS_DIR,
    CUT_R2_CATEGORIES,
    PRE_ANNOTATIONS_DIR,
    SPOT_DIR,
    cut_kind_of,
    find_cut,
    iter_all_cuts,
)
from yp_video.core import spot_prelabel
from yp_video.core.action_frames import ensure_action_frame_cache, inspect_action_frame_cache
from yp_video.core.annotation_ids import action_id, rally_id
from yp_video.core.jsonl import read_jsonl
from yp_video.web.job_helpers import ProgressParser, finalize_batch_job, stop_vllm_for_job, stream_subprocess
from yp_video.web.jobs import job_manager
from yp_video.web.r2_client import serve_video_or_r2_redirect, sync_to_r2

log = logging.getLogger(__name__)
router = APIRouter()

ACTION_LABELS = ("serve", "receive", "set", "spike", "block", "score")


class ActionEvent(BaseModel):
    id: str | None = None
    rally_id: str | None = None
    frame: int = Field(ge=0)
    time: float | None = None
    relative_frame: int | None = None
    label: str
    xy: tuple[float, float]

    @field_validator("label")
    @classmethod
    def validate_label(cls, value: str) -> str:
        if value not in ACTION_LABELS:
            raise ValueError(f"label must be one of: {', '.join(ACTION_LABELS)}")
        return value

    @field_validator("xy")
    @classmethod
    def validate_xy(cls, value: tuple[float, float]) -> tuple[float, float]:
        x, y = value
        if not (0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError("xy must be normalized to [0, 1]")
        return value


class SaveActionAnnotationsRequest(BaseModel):
    video: str
    fps: float = Field(gt=0)
    num_frames: int = Field(ge=0)
    events: list[ActionEvent]


class SpotPrelabelOptions(BaseModel):
    checkpoint: str | None = None
    batch_size: int = Field(default=8, ge=1, le=128)
    num_workers: int = Field(default=4, ge=0, le=16)
    clip_len: int = Field(default=64, ge=8, le=256)
    min_score: float = Field(default=0.15, ge=0, le=1)
    overwrite: bool = False
    stop_vllm: bool = False
    use_amp: bool = True


class SpotPrelabelRequest(SpotPrelabelOptions):
    video: str


class SpotPrelabelBatchRequest(SpotPrelabelOptions):
    videos: list[str] = Field(min_length=1)


def _annotation_path(video_name: str) -> Path:
    return ACTION_ANNOTATIONS_DIR / f"{Path(video_name).stem}_actions.jsonl"


def _rally_annotation_path(video_name: str) -> Path | None:
    filename = f"{Path(video_name).stem}_annotations.jsonl"
    for directory in (ANNOTATIONS_DIR, PRE_ANNOTATIONS_DIR):
        path = directory / filename
        if path.exists():
            return path
    return None


def _rally_sources(video_name: str) -> list[str]:
    filename = f"{Path(video_name).stem}_annotations.jsonl"
    sources = []
    if (ANNOTATIONS_DIR / filename).exists():
        sources.append("annotation")
    if (PRE_ANNOTATIONS_DIR / filename).exists():
        sources.append("pre-annotation")
    return sources


def _parse_rate(rate: str | None) -> float:
    if not rate or rate == "0/0":
        return 0.0
    try:
        return float(Fraction(rate))
    except (ValueError, ZeroDivisionError):
        return 0.0


def _video_metadata(path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-of", "json",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise HTTPException(502, f"ffprobe failed: {result.stderr[:200]}")
    try:
        stream = (json.loads(result.stdout).get("streams") or [{}])[0]
    except json.JSONDecodeError as exc:
        raise HTTPException(502, "ffprobe returned invalid JSON") from exc

    fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate")) or 30.0
    duration = float(stream.get("duration") or 0)
    num_frames = int(stream.get("nb_frames") or round(duration * fps))
    return {"fps": fps, "duration": duration, "num_frames": num_frames}


def _load_annotation(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        data, events = read_jsonl(path)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, f"Invalid annotation JSONL: {path.name}") from exc
    data["events"] = sorted(events, key=lambda e: (e.get("frame", 0), e.get("label", "")))
    data["num_events"] = len(data["events"])
    return data


def _load_rallies(video: Path) -> list[dict]:
    path = _rally_annotation_path(video.name)
    if path is None:
        return []
    meta, records = read_jsonl(path)
    source_video = str(meta.get("video") or video.name)
    rallies: list[dict] = []
    for i, record in enumerate(records):
        start = float(record.get("start", record.get("start_time", 0)) or 0)
        end = float(record.get("end", record.get("end_time", 0)) or 0)
        rallies.append({
            "id": rally_id(source_video, record, i),
            "start": start,
            "end": end,
            "label": record.get("label", "rally"),
        })
    rallies.sort(key=lambda r: (r["start"], r["end"], r["id"]))
    return rallies


def _rally_for_event(event: dict, fps: float, rallies: list[dict]) -> dict | None:
    if not rallies:
        return None
    time = float(event.get("time") if event.get("time") is not None else event.get("frame", 0) / fps)
    for rally in rallies:
        if rally["start"] <= time < rally["end"]:
            return rally
    existing_id = event.get("rally_id")
    if existing_id:
        for rally in rallies:
            if rally["id"] == existing_id:
                return rally
    return None


def _normalize_events(video_stem: str, events: list[dict], *, fps: float, num_frames: int, rallies: list[dict]) -> list[dict]:
    normalized = []
    max_frame = max(0, num_frames - 1)
    for i, raw in enumerate(events):
        event = dict(raw)
        frame = max(0, min(int(round(float(event.get("frame", 0) or 0))), max_frame))
        event["frame"] = frame
        event["id"] = action_id(video_stem, event, i)
        time = frame / fps if fps > 0 else float(event.get("time") or 0)
        event["time"] = round(time, 4)
        rally = _rally_for_event(event, fps, rallies)
        if rally:
            event["rally_id"] = rally["id"]
            event["relative_frame"] = max(0, int(round((time - rally["start"]) * fps)))
        else:
            event["rally_id"] = None
            event["relative_frame"] = None
        normalized.append(event)
    normalized.sort(key=lambda e: (e["frame"], e["label"], e["id"]))
    return normalized


def _write_annotation_atomic(output_path: Path, data: dict) -> None:
    tmp_path = output_path.with_suffix(output_path.suffix + f".tmp.{os.getpid()}")
    meta = {k: v for k, v in data.items() if k != "events"}
    meta["_meta"] = True
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for event in data.get("events", []):
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, output_path)


_TERMINAL_ITEM_STATUSES = {"completed", "failed", "cancelled"}


def _batch_counts(items: list[dict]) -> dict:
    return {
        "total": len(items),
        "completed": sum(1 for item in items if item.get("status") == "completed"),
        "failed": sum(1 for item in items if item.get("status") == "failed"),
        "cancelled": sum(1 for item in items if item.get("status") == "cancelled"),
    }


def _batch_progress(index: int, item_progress: float, total: int) -> float:
    return min(0.99, max(0.0, (index + item_progress) / max(1, total)))


async def _update_batch_item(
    job_id: str,
    items: list[dict],
    index: int,
    *,
    status: str | None = None,
    progress: float | None = None,
    message: str | None = None,
    error: str | None = None,
    overall_progress: float | None = None,
    overall_message: str | None = None,
    extra: dict | None = None,
) -> None:
    item = dict(items[index])
    if status is None and item.get("status") in _TERMINAL_ITEM_STATUSES:
        return
    if status is not None:
        item["status"] = status
    if progress is not None:
        item["progress"] = max(0.0, min(float(progress), 1.0))
    if message is not None:
        item["message"] = message
    if error is not None:
        item["error"] = error
    if extra:
        item.update(extra)
    items[index] = item

    job = job_manager.get_job(job_id)
    if job is None:
        return
    params = {
        **job.params,
        "items": [dict(i) for i in items],
        **_batch_counts(items),
    }
    update: dict = {"params": params}
    if overall_progress is not None:
        update["progress"] = overall_progress
    if overall_message is not None:
        update["message"] = overall_message
    await job_manager.update_job(job_id, **update)


def _resolve_prelabel_entries(names: list[str], *, overwrite: bool) -> list[tuple[Path, Path]]:
    entries: list[tuple[Path, Path]] = []
    missing: list[str] = []
    existing: list[str] = []
    seen: set[str] = set()

    for raw_name in names:
        name = Path(str(raw_name)).name
        if not name or name in seen:
            continue
        seen.add(name)
        video = find_cut(name)
        if video is None:
            missing.append(name)
            continue
        ann_path = _annotation_path(video.name)
        if ann_path.exists() and not overwrite:
            existing.append(video.name)
            continue
        entries.append((video, ann_path))

    if missing:
        sample = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f" and {len(missing) - 5} more"
        raise HTTPException(404, f"Video not found: {sample}{suffix}")
    if existing:
        sample = ", ".join(existing[:5])
        suffix = "" if len(existing) <= 5 else f" and {len(existing) - 5} more"
        raise HTTPException(409, f"Action annotation already exists for: {sample}{suffix}; set overwrite=true")
    if not entries:
        raise HTTPException(400, "No valid videos selected")
    return entries


@router.get("/labels")
def labels() -> dict:
    return {"labels": list(ACTION_LABELS)}


@router.get("/spot")
def spot_status() -> dict:
    checkpoints = spot_prelabel.list_checkpoints()
    default = spot_prelabel.default_checkpoint()
    return {
        "available": spot_prelabel.spot_available(),
        "spot_dir": str(SPOT_DIR),
        "default_checkpoint": str(default.relative_to(SPOT_DIR)) if default else "",
        "checkpoints": checkpoints,
    }


@router.get("/videos")
def list_videos() -> list[dict]:
    results = []
    for video in sorted(iter_all_cuts(), key=lambda p: p.name):
        ann_path = _annotation_path(video.name)
        event_count = 0
        if ann_path.exists():
            try:
                event_count = len((_load_annotation(ann_path) or {}).get("events", []))
            except HTTPException:
                event_count = -1
        results.append({
            "name": video.name,
            "kind": cut_kind_of(video),
            "rally_sources": _rally_sources(video.name),
            "has_action_annotation": ann_path.exists(),
            "event_count": event_count,
            "frame_cache": inspect_action_frame_cache(video),
        })
    return results


@router.get("/annotations/{name:path}")
async def get_annotations(name: str) -> dict:
    decoded = unquote(name)
    video = find_cut(Path(decoded).name)
    if video is None:
        raise HTTPException(404, "Video not found")

    meta = await asyncio.to_thread(_video_metadata, video)
    rallies = await asyncio.to_thread(_load_rallies, video)
    ann = _load_annotation(_annotation_path(video.name))
    if ann is not None:
        ann.setdefault("video", video.stem)
        ann["source_video"] = video.name
        ann.setdefault("fps", meta["fps"])
        ann.setdefault("num_frames", meta["num_frames"])
        ann["rallies"] = rallies
        ann["events"] = _normalize_events(
            video.stem,
            ann.get("events", []),
            fps=float(ann["fps"]),
            num_frames=int(ann["num_frames"]),
            rallies=rallies,
        )
        ann["num_events"] = len(ann["events"])
        ann["duration"] = meta["duration"]
        return ann

    return {
        "video": video.stem,
        "source_video": video.name,
        "duration": meta["duration"],
        "fps": meta["fps"],
        "num_frames": meta["num_frames"],
        "num_events": 0,
        "rallies": rallies,
        "events": [],
    }


@router.post("/annotations")
async def save_annotations(req: SaveActionAnnotationsRequest) -> dict:
    video = find_cut(Path(req.video).name)
    if video is None:
        raise HTTPException(404, "Video not found")

    ACTION_ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    rallies = await asyncio.to_thread(_load_rallies, video)
    events = _normalize_events(
        video.stem,
        [event.model_dump(mode="json") for event in req.events],
        fps=req.fps,
        num_frames=req.num_frames,
        rallies=rallies,
    )
    data = {
        "video": video.stem,
        "num_frames": req.num_frames,
        "fps": req.fps,
        "rallies": rallies,
        "num_events": len(events),
        "events": events,
    }
    output_path = _annotation_path(video.name)
    await asyncio.to_thread(_write_annotation_atomic, output_path, data)
    sync_to_r2(output_path, "action-annotations")
    return {"saved": str(output_path), "count": len(events)}


@router.post("/prelabel")
async def start_spot_prelabel(req: SpotPrelabelRequest) -> dict:
    if not spot_prelabel.spot_available():
        raise HTTPException(503, "SPOT is not available at ~/yp-spot")

    video = find_cut(Path(req.video).name)
    if video is None:
        raise HTTPException(404, "Video not found")

    ann_path = _annotation_path(video.name)
    if ann_path.exists() and not req.overwrite:
        raise HTTPException(409, "Action annotation already exists; set overwrite=true")

    try:
        checkpoint = spot_prelabel.resolve_checkpoint(req.checkpoint)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    job = job_manager.create_job(
        "spot-prelabel",
        {
            "video": video.name,
            "checkpoint": str(checkpoint.relative_to(SPOT_DIR)),
            "min_score": req.min_score,
        },
        name=f"SPOT pre-label — {video.name}",
    )

    async def run_job() -> None:
        try:
            await job_manager.update_job(
                job.id,
                status="running",
                progress=0.02,
                message="Reading video metadata...",
            )
            meta = await asyncio.to_thread(_video_metadata, video)
            await job_manager.update_job(
                job.id,
                progress=0.03,
                message="Preparing frame cache...",
            )
            cache_info = await asyncio.to_thread(
                ensure_action_frame_cache,
                video,
                expected_frames=int(meta.get("num_frames") or 0) or None,
            )
            pred_file = spot_prelabel.predictions_path(job.id, video.stem)
            run_log = pred_file.parent / "run.log"
            cmd = spot_prelabel.build_command(
                video_path=video,
                checkpoint_path=checkpoint,
                save_dir=pred_file.parent,
                batch_size=req.batch_size,
                num_workers=req.num_workers,
                clip_len=req.clip_len,
                use_amp=req.use_amp,
            )

            def progress_handler(match):
                end_frame = int(match.group(2))
                total_frames = max(1, int(meta.get("num_frames") or 1))
                frac = min(0.9, 0.08 + 0.82 * (end_frame / total_frames))
                return {
                    "progress": frac,
                    "message": f"SPOT inference frame {min(end_frame, total_frames)}/{total_frames}",
                }

            parsers = [ProgressParser(r"Processed .* from (\d+) to (\d+)", progress_handler)]
            await job_manager.update_job(
                job.id,
                progress=0.05,
                message="Waiting for GPU...",
            )

            async with stop_vllm_for_job(job.id, when=req.stop_vllm):
                async with job_manager.gpu_lock:
                    await job_manager.update_job(
                        job.id,
                        progress=0.08,
                        message="Running SPOT inference...",
                    )
                    rc, last_line = await stream_subprocess(
                        job.id,
                        cmd,
                        SPOT_DIR,
                        parsers=parsers,
                        is_key_line=lambda line: line.startswith("Saved predictions"),
                        push_interval=1.0,
                        tee_to_terminal=True,
                        terminal_prefix=f"[SPOT {job.id} {video.name}] ",
                        log_path=run_log,
                    )
            if rc != 0:
                raise RuntimeError(last_line or f"SPOT exited with code {rc}")
            if not pred_file.exists():
                raise RuntimeError(f"SPOT did not create predictions.json at {pred_file}")

            await job_manager.update_job(job.id, progress=0.93, message="Converting predictions...")
            predictions = await asyncio.to_thread(spot_prelabel.load_predictions, pred_file)
            data = spot_prelabel.predictions_to_annotation(
                predictions,
                video_path=video,
                metadata=meta,
                checkpoint_path=checkpoint,
                min_score=req.min_score,
            )
            rallies = await asyncio.to_thread(_load_rallies, video)
            data["rallies"] = rallies
            data["events"] = _normalize_events(
                video.stem,
                data.get("events", []),
                fps=float(data.get("fps") or meta["fps"]),
                num_frames=int(data.get("num_frames") or meta["num_frames"]),
                rallies=rallies,
            )
            data["num_events"] = len(data["events"])
            ACTION_ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(_write_annotation_atomic, ann_path, data)
            sync_to_r2(ann_path, "action-annotations")
            await job_manager.update_job(
                job.id,
                status="completed",
                progress=1.0,
                message=f"Pre-label complete: {data['num_events']} event(s)",
                params={
                    **job.params,
                    "count": data["num_events"],
                    "frame_cache": cache_info,
                    "saved": str(ann_path),
                    "predictions": str(pred_file),
                    "run_log": str(run_log),
                },
            )
        except asyncio.CancelledError:
            await job_manager.update_job(job.id, status="cancelled", message="Cancelled")
            raise
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            log.error("SPOT pre-label failed for %s:\n%s", video.name, tb)
            job_obj = job_manager.get_job(job.id)
            if job_obj:
                job_obj.logs.append(f"[{video.name}] {type(exc).__name__}: {exc}")
                job_obj.logs.extend(tb.splitlines())
            await job_manager.update_job(
                job.id,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                message="SPOT pre-label failed",
            )

    task = asyncio.create_task(run_job())
    job_manager.attach_task(job, task)
    return job.to_dict()


@router.post("/prelabel-batch")
async def start_spot_prelabel_batch(req: SpotPrelabelBatchRequest) -> dict:
    if not spot_prelabel.spot_available():
        raise HTTPException(503, "SPOT is not available at ~/yp-spot")

    entries = _resolve_prelabel_entries(req.videos, overwrite=req.overwrite)

    try:
        checkpoint = spot_prelabel.resolve_checkpoint(req.checkpoint)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    total = len(entries)
    items = [
        {
            "video": video.name,
            "status": "pending",
            "progress": 0.0,
            "message": "Pending",
        }
        for video, _ann_path in entries
    ]
    job = job_manager.create_job(
        "spot-prelabel-batch",
        {
            "videos": [video.name for video, _ann_path in entries],
            "checkpoint": str(checkpoint.relative_to(SPOT_DIR)),
            "min_score": req.min_score,
            "total": total,
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
            "items": [dict(item) for item in items],
        },
        name=f"SPOT pre-label batch ({total} videos)",
    )

    async def run_job() -> None:
        failed = 0
        try:
            await job_manager.update_job(
                job.id,
                status="running",
                progress=0.0,
                message=f"Queued {total} video(s)",
            )
            async with stop_vllm_for_job(job.id, when=req.stop_vllm):
                for idx, (video, ann_path) in enumerate(entries):
                    try:
                        await _run_prelabel_batch_item(
                            job.id,
                            items,
                            idx,
                            total,
                            video=video,
                            ann_path=ann_path,
                            checkpoint=checkpoint,
                            req=req,
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:  # noqa: BLE001
                        failed += 1
                        tb = traceback.format_exc()
                        log.error("SPOT batch pre-label failed for %s:\n%s", video.name, tb)
                        job_obj = job_manager.get_job(job.id)
                        if job_obj:
                            job_obj.logs.append(f"[{video.name}] {type(exc).__name__}: {exc}")
                            job_obj.logs.extend(tb.splitlines())
                        await _update_batch_item(
                            job.id,
                            items,
                            idx,
                            status="failed",
                            progress=1.0,
                            message="Failed",
                            error=f"{type(exc).__name__}: {exc}",
                            overall_progress=_batch_progress(idx, 1.0, total),
                            overall_message=f"{video.name} failed",
                        )
            await finalize_batch_job(job.id, total, failed)
        except asyncio.CancelledError:
            for idx, item in enumerate(items):
                if item.get("status") not in _TERMINAL_ITEM_STATUSES:
                    await _update_batch_item(
                        job.id,
                        items,
                        idx,
                        status="cancelled",
                        progress=float(item.get("progress") or 0),
                        message="Cancelled",
                    )
            await job_manager.update_job(job.id, status="cancelled", message="Batch cancelled")
            raise
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            log.error("SPOT batch pre-label failed:\n%s", tb)
            job_obj = job_manager.get_job(job.id)
            if job_obj:
                job_obj.logs.append(f"{type(exc).__name__}: {exc}")
                job_obj.logs.extend(tb.splitlines())
            await job_manager.update_job(
                job.id,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                message="SPOT batch pre-label failed",
            )

    task = asyncio.create_task(run_job())
    job_manager.attach_task(job, task)
    return job.to_dict()


async def _run_prelabel_batch_item(
    job_id: str,
    items: list[dict],
    idx: int,
    total: int,
    *,
    video: Path,
    ann_path: Path,
    checkpoint: Path,
    req: SpotPrelabelOptions,
) -> None:
    await _update_batch_item(
        job_id,
        items,
        idx,
        status="running",
        progress=0.02,
        message="Reading metadata",
        overall_progress=_batch_progress(idx, 0.02, total),
        overall_message=f"{video.name}: reading metadata",
    )
    meta = await asyncio.to_thread(_video_metadata, video)

    await _update_batch_item(
        job_id,
        items,
        idx,
        progress=0.08,
        message="Preparing frame cache",
        overall_progress=_batch_progress(idx, 0.08, total),
        overall_message=f"{video.name}: preparing frame cache",
    )
    cache_info = await asyncio.to_thread(
        ensure_action_frame_cache,
        video,
        expected_frames=int(meta.get("num_frames") or 0) or None,
    )

    pred_file = spot_prelabel.predictions_path(job_id, video.stem)
    run_log = pred_file.parent / "run.log"
    cmd = spot_prelabel.build_command(
        video_path=video,
        checkpoint_path=checkpoint,
        save_dir=pred_file.parent,
        batch_size=req.batch_size,
        num_workers=req.num_workers,
        clip_len=req.clip_len,
        use_amp=req.use_amp,
    )

    await _update_batch_item(
        job_id,
        items,
        idx,
        progress=0.18,
        message="Waiting for GPU",
        overall_progress=_batch_progress(idx, 0.18, total),
        overall_message=f"{video.name}: waiting for GPU",
        extra={"frame_cache": cache_info},
    )

    def progress_handler(match):
        end_frame = int(match.group(2))
        total_frames = max(1, int(meta.get("num_frames") or 1))
        item_progress = min(0.9, 0.22 + 0.68 * (end_frame / total_frames))
        message = f"SPOT inference frame {min(end_frame, total_frames)}/{total_frames}"
        asyncio.create_task(
            _update_batch_item(
                job_id,
                items,
                idx,
                progress=item_progress,
                message=message,
                overall_progress=_batch_progress(idx, item_progress, total),
                overall_message=f"{video.name}: {message}",
            )
        )
        return None

    async with job_manager.gpu_lock:
        await _update_batch_item(
            job_id,
            items,
            idx,
            progress=0.22,
            message="Running SPOT inference",
            overall_progress=_batch_progress(idx, 0.22, total),
            overall_message=f"{video.name}: running SPOT inference",
        )
        rc, last_line = await stream_subprocess(
            job_id,
            cmd,
            SPOT_DIR,
            parsers=[ProgressParser(r"Processed .* from (\d+) to (\d+)", progress_handler)],
            is_key_line=lambda line: line.startswith("Saved predictions"),
            push_interval=1.0,
            tee_to_terminal=True,
            terminal_prefix=f"[SPOT {job_id} {video.name}] ",
            log_path=run_log,
            update_job=False,
        )

    if rc != 0:
        raise RuntimeError(last_line or f"SPOT exited with code {rc}")
    if not pred_file.exists():
        raise RuntimeError(f"SPOT did not create predictions.json at {pred_file}")

    await _update_batch_item(
        job_id,
        items,
        idx,
        progress=0.93,
        message="Converting predictions",
        overall_progress=_batch_progress(idx, 0.93, total),
        overall_message=f"{video.name}: converting predictions",
    )
    predictions = await asyncio.to_thread(spot_prelabel.load_predictions, pred_file)
    data = spot_prelabel.predictions_to_annotation(
        predictions,
        video_path=video,
        metadata=meta,
        checkpoint_path=checkpoint,
        min_score=req.min_score,
    )
    rallies = await asyncio.to_thread(_load_rallies, video)
    data["rallies"] = rallies
    data["events"] = _normalize_events(
        video.stem,
        data.get("events", []),
        fps=float(data.get("fps") or meta["fps"]),
        num_frames=int(data.get("num_frames") or meta["num_frames"]),
        rallies=rallies,
    )
    data["num_events"] = len(data["events"])
    ACTION_ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(_write_annotation_atomic, ann_path, data)
    sync_to_r2(ann_path, "action-annotations")
    await _update_batch_item(
        job_id,
        items,
        idx,
        status="completed",
        progress=1.0,
        message=f"Complete: {data['num_events']} event(s)",
        overall_progress=_batch_progress(idx, 1.0, total),
        overall_message=f"{video.name}: complete",
        extra={
            "count": data["num_events"],
            "saved": str(ann_path),
            "predictions": str(pred_file),
            "run_log": str(run_log),
        },
    )


@router.get("/export")
def export_dataset() -> Response:
    ACTION_ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted(ACTION_ANNOTATIONS_DIR.glob("*_actions.jsonl")):
        data = _load_annotation(path)
        if data is not None:
            records.append({
                "video": data.get("video", path.stem.removesuffix("_actions")),
                "num_frames": data.get("num_frames", 0),
                "fps": data.get("fps", 0),
                "rallies": data.get("rallies", []),
                "num_events": len(data.get("events", [])),
                "events": data.get("events", []),
            })
    lines = [
        json.dumps(
            {
                "_meta": True,
                "type": "spot_action_annotations",
                "num_videos": len(records),
                "num_events": sum(record["num_events"] for record in records),
            },
            ensure_ascii=False,
        )
    ]
    lines.extend(json.dumps(record, ensure_ascii=False) for record in records)
    return Response(
        "\n".join(lines) + "\n",
        media_type="application/x-ndjson",
        headers={"Content-Disposition": 'attachment; filename="spot_action_annotations.jsonl"'},
    )


@router.get("/video/{path:path}")
def stream_video(path: str):
    decoded_path = unquote(path)
    video_path = find_cut(Path(decoded_path).name)
    if video_path is None:
        raise HTTPException(404, "Video not found")
    response = serve_video_or_r2_redirect(video_path, CUT_R2_CATEGORIES)
    if response:
        return response
    raise HTTPException(404, "Video not found")
