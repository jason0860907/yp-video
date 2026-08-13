"""SPOT-style point action annotator router."""

import asyncio
import json
import logging
import os
import tempfile
import traceback
from pathlib import Path
from typing import Literal
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import Field, field_validator

from yp_video.action import prelabel
from yp_video.config import (
    ACTION_ANNOTATIONS_DIR,
    CUT_R2_CATEGORIES,
    SPOT_DIR,
    find_cut,
)
from yp_video.contracts.action import (
    ACTION_CONTRACT_VERSION,
    ACTION_CONTRACT_VERSION_ENV,
    ACTION_LABELS_ORDERED,
    SPOT_PROGRESS_PREFIX,
)
from yp_video.core import label_done
from yp_video.core.annotation_ids import action_id
from yp_video.core.ffmpeg import parse_optional_float as _parse_optional_float
from yp_video.core.rallies import load_rallies
from yp_video.web import worklists
from yp_video.web.action_annotations import (
    annotation_path,
    annotation_state,
    load_annotation,
    pre_annotation_path,
)
from yp_video.web.action_waveform import audio_waveform, video_metadata
from yp_video.web.job_helpers import (
    ProgressParser,
    batch_message,
    batch_progress,
    cancel_batch_items,
    fail_job_from_exc,
    finalize_batch_job,
    init_batch_items,
    stop_vllm_for_job,
    stream_subprocess,
    terminal_prefix,
    update_batch_item,
)
from yp_video.web.jobs import JobSummary, JobType, job_manager
from yp_video.web.r2_client import resolve_cut, serve_video_or_r2_redirect, sync_to_r2
from yp_video.web.schemas import StrictModel

log = logging.getLogger(__name__)
router = APIRouter()

ACTION_LABELS = ACTION_LABELS_ORDERED
SPOT_DEFAULT_DECODER: Literal["opencv", "nvdec"] = "nvdec"
SPOT_DEFAULT_DECODE_PRODUCERS = 2
SPOT_DEFAULT_DECODER_THREADS = 1
SPOT_DEFAULT_PREFETCH_FACTOR = 2
SPOT_DEFAULT_DECODE_CHUNK_FRAMES = 256
SPOT_DEFAULT_NVIDIA_VIDEO_LIB_DIR = Path.home() / ".local/lib/nvidia-video"


class ActionEvent(StrictModel):
    id: str | None = None
    rally_id: int | None = None
    frame: int = Field(ge=0)
    time: float | None = None
    relative_frame: int | None = None
    label: str
    xy: tuple[float, float]
    visible: bool = True

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


class SaveActionAnnotationsRequest(StrictModel):
    video: str
    fps: float = Field(gt=0)
    num_frames: int = Field(ge=0)
    events: list[ActionEvent]
    # The human-store revision this edit is based on (from GET /annotations or
    # the previous save's response); None claims the store does not exist yet.
    # A mismatch means another writer saved in between — 409, never overwrite.
    revision: str | None


class SpotPrelabelOptions(StrictModel):
    checkpoint: str | None = None
    batch_size: int = Field(default=16, ge=1, le=128)
    num_workers: int = Field(default=2, ge=0, le=16)
    clip_len: int = Field(default=64, ge=8, le=256)
    decoder: Literal["opencv", "nvdec"] = SPOT_DEFAULT_DECODER
    decode_producers: int = Field(default=SPOT_DEFAULT_DECODE_PRODUCERS, ge=1, le=8)
    decoder_threads: int = Field(default=SPOT_DEFAULT_DECODER_THREADS, ge=1, le=8)
    prefetch_factor: int = Field(default=SPOT_DEFAULT_PREFETCH_FACTOR, ge=1, le=8)
    decode_chunk_frames: int = Field(default=SPOT_DEFAULT_DECODE_CHUNK_FRAMES, ge=1, le=512)
    min_score: float = Field(default=0.15, ge=0, le=1)
    overwrite: bool = False
    stop_vllm: bool = False
    use_amp: bool = True


class SpotPrelabelBatchRequest(SpotPrelabelOptions):
    videos: list[str] = Field(min_length=1)


def _load_rallies(video: Path) -> list[dict]:
    """This video's rally spans (see core/rallies.py for the source priority)."""
    return load_rallies(video.stem)


def _annotation_revision(video_name: str) -> str | None:
    """The human store's identity for optimistic concurrency: its mtime_ns.

    A string, not an int — the token crosses JSON into JS numbers, and ns
    timestamps exceed Number.MAX_SAFE_INTEGER. None = no human file.
    """
    try:
        return str(annotation_path(video_name).stat().st_mtime_ns)
    except FileNotFoundError:
        return None


def _rally_for_event(event: dict, fps: float, rallies: list[dict]) -> dict | None:
    if not rallies:
        return None
    explicit_time = _parse_optional_float(event.get("time"))
    if explicit_time is not None:
        time = explicit_time
    else:
        frame = _parse_optional_float(event.get("frame")) or 0.0
        time = frame / fps if fps > 0 else 0.0
    for rally in rallies:
        if rally["start"] <= time < rally["end"]:
            return rally
    existing_id = _coerce_rally_id(event.get("rally_id"))
    if existing_id:
        for rally in rallies:
            if rally["rally_id"] == existing_id:
                return rally
    return None


def _coerce_rally_id(value: object) -> int | None:
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and value.isdigit() and int(value) > 0:
        return int(value)
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
        event["visible"] = _truthy_event_visible(event.get("visible", True))
        rally = _rally_for_event(event, fps, rallies)
        if rally:
            event["rally_id"] = rally["rally_id"]
            event["relative_frame"] = max(0, int(round((time - rally["start"]) * fps)))
        else:
            event["rally_id"] = None
            event["relative_frame"] = None
        normalized.append(event)
    normalized.sort(key=lambda e: (e["frame"], e["label"], e["id"]))
    return normalized


def _truthy_event_visible(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return value is not False


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


async def _save_spot_action_annotation(
    *,
    video: Path,
    ann_path: Path,
    meta: dict,
    pred_file: Path,
    checkpoint: Path,
    min_score: float,
) -> dict:
    predictions = await asyncio.to_thread(prelabel.load_predictions, pred_file)
    data = prelabel.predictions_to_annotation(
        predictions,
        video_path=video,
        metadata=meta,
        checkpoint_path=checkpoint,
        min_score=min_score,
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
    ann_path.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(_write_annotation_atomic, ann_path, data)
    sync_to_r2(ann_path, "action/pre-annotations")
    return data


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
        # Only the machine store gates a re-run: prelabel writes (and with
        # overwrite, rewrites) pre-annotations exclusively. The human store is
        # never touched — the editor keeps preferring it either way.
        pre_path = pre_annotation_path(video.name)
        if pre_path.exists() and not overwrite:
            existing.append(video.name)
            continue
        entries.append((video, pre_path))

    if missing:
        sample = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f" and {len(missing) - 5} more"
        raise HTTPException(404, f"Video not found: {sample}{suffix}")
    if existing:
        sample = ", ".join(existing[:5])
        suffix = "" if len(existing) <= 5 else f" and {len(existing) - 5} more"
        raise HTTPException(409, f"Action pre-label already exists for: {sample}{suffix}; set overwrite=true")
    if not entries:
        raise HTTPException(400, "No valid videos selected")
    return entries


def _spot_progress_fraction(data: dict, *, start: float, span: float, cap: float) -> float:
    """Map SPOT inference progress into a UI band ``[start, start+span]``."""
    return min(cap, start + span * prelabel.spot_progress_fraction(data))


def _spot_app_log_line(prefix: str, line: str) -> str | None:
    if line.startswith(SPOT_PROGRESS_PREFIX):
        data = prelabel.parse_spot_progress(line.removeprefix(SPOT_PROGRESS_PREFIX))
        if data is None:
            return None
        video = data.get("video_basename") or Path(str(data.get("video") or "")).name
        return f"{prefix}{video}: {prelabel.spot_progress_message(data)}"
    if line.startswith((
        "Starting inference", "Timing ", "Saved predictions",
        "Failed inference", "Failure summary", "Warning:", "Decode pipeline:",
    )):
        return f"{prefix}{line}"
    return None


def _spot_subprocess_env(req: SpotPrelabelOptions) -> dict[str, str]:
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "SPOT_DECODER": req.decoder,
        "SPOT_NUM_PRODUCERS": str(req.decode_producers),
        "SPOT_DECODER_THREADS": str(req.decoder_threads),
        "SPOT_DECODE_CHUNK_FRAMES": str(req.decode_chunk_frames),
        ACTION_CONTRACT_VERSION_ENV: ACTION_CONTRACT_VERSION,
    }
    if req.decoder == "nvdec":
        env["SPOT_ENABLE_EXPERIMENTAL_NVDEC"] = "1"
        env["SPOT_NVDEC_GPU_PREPROCESS"] = "1"
    env.setdefault("MALLOC_ARENA_MAX", "2")
    video_lib_dir = os.environ.get("SPOT_NVIDIA_VIDEO_LIB_DIR")
    if not video_lib_dir and SPOT_DEFAULT_NVIDIA_VIDEO_LIB_DIR.exists():
        video_lib_dir = str(SPOT_DEFAULT_NVIDIA_VIDEO_LIB_DIR)
    if video_lib_dir:
        current = env.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = (
            f"{video_lib_dir}:{current}" if current else video_lib_dir
        )
    return env


def _spot_decode_settings_text(req: SpotPrelabelOptions) -> str:
    return (
        f"decoder={req.decoder} "
        f"prefetch={req.prefetch_factor} "
        f"producers={req.decode_producers} "
        f"threads={req.decoder_threads} "
        f"chunk={req.decode_chunk_frames}"
    )


@router.get("/labels")
def labels() -> dict:
    return {"labels": list(ACTION_LABELS)}


@router.get("/spot")
def spot_status() -> dict:
    checkpoints = prelabel.list_checkpoints()
    default = prelabel.default_checkpoint()
    return {
        "available": prelabel.spot_available(),
        "spot_dir": str(SPOT_DIR),
        "default_checkpoint": prelabel.checkpoint_ref(default) if default else "",
        "checkpoints": checkpoints,
    }


@router.get("/videos")
def list_videos() -> list[dict]:
    return worklists.action_videos()


class DoneRequest(StrictModel):
    done: bool = True


@router.put("/done/{name:path}")
def set_done(name: str, req: DoneRequest) -> dict:
    """Persist the human "action labeling is finished" verdict for one video.

    Deliberately not tied to saving: Save writes the human store while this
    flag is the separate "I'm finished" claim.
    """
    video = resolve_cut(Path(unquote(name)).name)
    if video is None:
        raise HTTPException(404, "Video not found")
    flags = label_done.set_done(video.stem, "action", req.done)
    return {"done": flags["action"]}


@router.get("/annotations/{name:path}")
async def get_annotations(
    name: str,
    source: Literal["annotation", "pre-annotation"] | None = None,
) -> dict:
    """One video's action annotation, from the active store by default.

    ``source`` forces one store — the saved annotation or the machine
    pre-annotation — mirroring the rally editor's Source select; a store
    the video does not have is a 404, not an empty editor.
    """
    decoded = unquote(name)
    video = resolve_cut(Path(decoded).name)
    if video is None:
        raise HTTPException(404, "Video not found")

    meta = await asyncio.to_thread(video_metadata, video)
    rallies = await asyncio.to_thread(_load_rallies, video)
    # Shallow copies throughout — the cached dict is shared; every key below
    # is reassigned wholesale, and _normalize_events copies each event.
    if source is not None:
        path = annotation_path(video.name) if source == "annotation" else pre_annotation_path(video.name)
        forced = load_annotation(path)
        if forced is None:
            raise HTTPException(404, f"No {source} for this video")
        ann = dict(forced)
        loaded = source
    else:
        state = annotation_state(video.name)
        if state.active_error is not None:
            raise state.active_error
        ann = dict(state.active) if state.active is not None else None
        loaded = None
        if ann is not None:
            loaded = "annotation" if state.active_path.parent == ACTION_ANNOTATIONS_DIR else "pre-annotation"
    revision = _annotation_revision(video.name)
    if ann is not None:
        # Which store this payload came from — the editor's "what am I
        # looking at" badge; the file's own provenance stays in `source`.
        ann["loaded_source"] = loaded
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
        ann["revision"] = revision
        return ann

    return {
        "video": video.stem,
        "source_video": video.name,
        "loaded_source": None,
        "duration": meta["duration"],
        "fps": meta["fps"],
        "num_frames": meta["num_frames"],
        "num_events": 0,
        "rallies": rallies,
        "events": [],
        "revision": revision,
    }


@router.get("/waveform/{name:path}")
async def get_waveform(name: str, points: int = Query(default=9600, ge=200, le=96000)) -> dict:
    decoded = unquote(name)
    video = resolve_cut(Path(decoded).name)
    if video is None:
        raise HTTPException(404, "Video not found")
    return await asyncio.to_thread(audio_waveform, video, points)


@router.post("/annotations")
async def save_annotations(req: SaveActionAnnotationsRequest) -> dict:
    video = resolve_cut(Path(req.video).name)
    if video is None:
        raise HTTPException(404, "Video not found")
    if req.revision != _annotation_revision(video.name):
        raise HTTPException(
            409,
            "Annotation changed on disk since it was loaded — reload before saving",
        )

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
        "source": {"type": "manual"},
        "rallies": rallies,
        "num_events": len(events),
        "events": events,
    }
    output_path = annotation_path(video.name)
    await asyncio.to_thread(_write_annotation_atomic, output_path, data)
    sync_to_r2(output_path, "action/annotations")
    return {
        "saved": str(output_path),
        "count": len(events),
        "revision": _annotation_revision(video.name),
    }


@router.post("/prelabel-batch", response_model=JobSummary)
async def start_spot_prelabel_batch(req: SpotPrelabelBatchRequest) -> dict:
    if not prelabel.spot_available():
        raise HTTPException(503, "SPOT is not available at ~/yp-spot")

    entries = _resolve_prelabel_entries(req.videos, overwrite=req.overwrite)

    try:
        checkpoint = prelabel.resolve_checkpoint(req.checkpoint)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    total = len(entries)
    items = init_batch_items([video.name for video, _ann_path in entries])
    job = job_manager.create_job(
        JobType.SPOT_PRELABEL_BATCH,
        {
            "videos": [video.name for video, _ann_path in entries],
            "checkpoint": prelabel.checkpoint_ref(checkpoint),
            "min_score": req.min_score,
            "total": total,
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
            "items": [dict(item) for item in items],
        },
        name=f"Action Predict ({total} videos)",
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
                failed = await _run_prelabel_batch_subprocess(
                    job.id,
                    items,
                    entries,
                    checkpoint=checkpoint,
                    req=req,
                )
            await finalize_batch_job(job.id, total, failed)
        except asyncio.CancelledError:
            await cancel_batch_items(job.id, items)
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("SPOT batch pre-label failed")
            await fail_job_from_exc(job.id, exc)

    task = asyncio.create_task(run_job())
    job_manager.attach_task(job, task)
    return job.to_dict()


async def _run_prelabel_batch_subprocess(
    job_id: str,
    items: list[dict],
    entries: list[tuple[Path, Path]],
    *,
    checkpoint: Path,
    req: SpotPrelabelOptions,
) -> int:
    total = len(entries)
    metas: list[dict] = []
    prefix = terminal_prefix(job_manager.get_job(job_id))

    with tempfile.TemporaryDirectory(prefix=f"yp-spot-batch-{job_id}-") as tmp_root:
        tmp_root_path = Path(tmp_root)
        pred_files: list[Path] = []
        for idx, (video, _ann_path) in enumerate(entries):
            await job_manager.update_job(
                job_id,
                progress=0.02 + 0.04 * ((idx + 1) / total),
                message=batch_message(idx, total, video.name, "reading metadata"),
            )
            metas.append(await asyncio.to_thread(video_metadata, video))
            pred_file = tmp_root_path / f"{idx:05d}" / "predictions.json"
            pred_file.parent.mkdir(parents=True, exist_ok=True)
            pred_files.append(pred_file)

        async def convert_predictions(idx: int) -> bool:
            video, ann_path = entries[idx]
            meta = metas[idx]
            pred_file = pred_files[idx]
            try:
                if not pred_file.exists():
                    raise RuntimeError("SPOT did not create prediction output")
                await update_batch_item(
                    job_id,
                    items,
                    idx,
                    progress=0.92,
                    message="Inference complete; saving pre-label",
                    overall_progress=batch_progress(idx, 0.92, total),
                    overall_message=batch_message(idx, total, video.name, "saving pre-label"),
                )
                data = await _save_spot_action_annotation(
                    video=video,
                    ann_path=ann_path,
                    meta=meta,
                    pred_file=pred_file,
                    checkpoint=checkpoint,
                    min_score=req.min_score,
                )
                log.info("%ssaved %s (%d event(s))", prefix, ann_path.name, data["num_events"])
                await update_batch_item(
                    job_id,
                    items,
                    idx,
                    status="completed",
                    progress=1.0,
                    message=f"Complete: {data['num_events']} event(s)",
                    overall_progress=batch_progress(idx, 1.0, total),
                    overall_message=batch_message(idx, total, video.name, "complete"),
                    extra={
                        "count": data["num_events"],
                        "saved": str(ann_path),
                    },
                )
                return True
            except Exception as exc:  # noqa: BLE001
                tb = traceback.format_exc()
                log.error("SPOT batch conversion failed for %s:\n%s", video.name, tb)
                job_obj = job_manager.get_job(job_id)
                if job_obj:
                    job_obj.logs.append(f"[{video.name}] {type(exc).__name__}: {exc}")
                    job_obj.logs.extend(tb.splitlines())
                await update_batch_item(
                    job_id,
                    items,
                    idx,
                    status="failed",
                    progress=1.0,
                    message="Failed",
                    error=f"{type(exc).__name__}: {exc}",
                    overall_progress=batch_progress(idx, 1.0, total),
                    overall_message=batch_message(idx, total, video.name, "failed"),
                )
                return False

        def missing_output_error(rc: int, last_line: str, failure_lines: list[str]) -> str:
            detail = failure_lines[-1] if failure_lines else last_line
            if detail:
                return f"SPOT failed before creating prediction output: {detail}"
            if rc != 0:
                return f"SPOT exited with code {rc} before creating prediction output"
            return "SPOT did not create prediction output"

        failed = 0
        # inference_lock (not gpu_lock) so the batch can run alongside training;
        # still serialized against other inference jobs.
        async with job_manager.inference_lock:
            await job_manager.update_job(job_id, message="Running SPOT inference", progress=0.08)
            for idx, ((video, _ann_path), pred_file) in enumerate(zip(entries, pred_files)):
                failure_lines: list[str] = []

                await update_batch_item(
                    job_id,
                    items,
                    idx,
                    status="running",
                    progress=0.08,
                    message="Launching SPOT inference",
                    overall_progress=batch_progress(idx, 0.08, total),
                    overall_message=batch_message(idx, total, video.name, "launching SPOT inference"),
                )

                def start_handler(_match, *, item_idx: int = idx, item_video: Path = video):
                    asyncio.create_task(
                        update_batch_item(
                            job_id,
                            items,
                            item_idx,
                            status="running",
                            progress=0.10,
                            message="Preparing first batch (decoding frames)",
                            overall_progress=batch_progress(item_idx, 0.10, total),
                            overall_message=batch_message(
                                item_idx, total, item_video.name, "preparing first batch"
                            ),
                        )
                    )
                    return None

                def progress_handler(match, *, item_idx: int = idx, item_video: Path = video):
                    data = prelabel.parse_spot_progress(match.group(1))
                    if data is None:
                        return None
                    item_progress = _spot_progress_fraction(data, start=0.12, span=0.78, cap=0.9)
                    message = prelabel.spot_progress_message(data)
                    asyncio.create_task(
                        update_batch_item(
                            job_id,
                            items,
                            item_idx,
                            status="running",
                            progress=item_progress,
                            message=message,
                            overall_progress=batch_progress(item_idx, item_progress, total),
                            overall_message=batch_message(item_idx, total, item_video.name, message),
                            extra={
                                "current_frame": int(data.get("end_frame") or 0),
                                "total_frames": int(data.get("total_frames") or 0),
                                "clips_done": int(data.get("clips_done") or 0),
                                "clips_total": int(data.get("clips_total") or 0),
                            },
                        )
                    )
                    return None

                def failure_handler(match):
                    failure_lines.append(match.group(1))
                    return None

                cmd = prelabel.build_command(
                    video_path=video,
                    checkpoint_path=checkpoint,
                    save_dir=pred_file.parent,
                    batch_size=req.batch_size,
                    num_workers=req.num_workers,
                    clip_len=req.clip_len,
                    prefetch_factor=req.prefetch_factor,
                    use_amp=req.use_amp,
                )

                rc, last_line = await stream_subprocess(
                    job_id,
                    cmd,
                    SPOT_DIR,
                    env=_spot_subprocess_env(req),
                    parsers=[
                        ProgressParser(r"Starting inference (\d+)/(\d+): (.+)", start_handler),
                        ProgressParser(SPOT_PROGRESS_PREFIX + r"(.+)", progress_handler),
                        ProgressParser(r"((?:Failed inference \d+/\d+|Failure summary): .+)", failure_handler),
                    ],
                    is_key_line=lambda line: (
                        line.startswith("Starting inference")
                        or line.startswith("SPOT_PROGRESS ")
                        or line.startswith("Saved predictions")
                        or line.startswith("Timing ")
                        or line.startswith("Failed inference")
                        or line.startswith("Failure summary")
                    ),
                    push_interval=1.0,
                    tee_to_terminal=True,
                    log_command=(
                        f"{prefix}start video {idx + 1}/{total}: {video.name} "
                        f"batch={req.batch_size} workers={req.num_workers} "
                        f"{_spot_decode_settings_text(req)}"
                    ),
                    log_line=lambda line: _spot_app_log_line(prefix, line),
                    update_job=False,
                )

                if not pred_file.exists():
                    failed += 1
                    error = missing_output_error(rc, last_line, failure_lines)
                    job_obj = job_manager.get_job(job_id)
                    if job_obj:
                        job_obj.logs.append(f"[{video.name}] RuntimeError: {error}")
                    await update_batch_item(
                        job_id,
                        items,
                        idx,
                        status="failed",
                        progress=1.0,
                        message="Failed",
                        error=error,
                        overall_progress=batch_progress(idx, 1.0, total),
                        overall_message=batch_message(idx, total, video.name, "failed"),
                    )
                    continue

                if not await convert_predictions(idx):
                    failed += 1
                continue

        return failed


@router.get("/export")
def export_dataset() -> Response:
    ACTION_ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted(ACTION_ANNOTATIONS_DIR.glob("*_actions.jsonl")):
        data = load_annotation(path)
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
def stream_video(path: str, request: Request):
    decoded_path = unquote(path)
    video_path = resolve_cut(Path(decoded_path).name)
    if video_path is None:
        raise HTTPException(404, "Video not found")
    response = serve_video_or_r2_redirect(
        video_path, CUT_R2_CATEGORIES, host=request.headers.get("host")
    )
    if response:
        return response
    raise HTTPException(404, "Video not found")
