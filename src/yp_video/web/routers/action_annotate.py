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
from fastapi.responses import JSONResponse
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
from yp_video.core.annotation_ids import action_id, rally_id
from yp_video.core.jsonl import read_jsonl
from yp_video.web.job_helpers import ProgressParser, stop_vllm_for_job, stream_subprocess
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


class SpotPrelabelRequest(BaseModel):
    video: str
    checkpoint: str | None = None
    batch_size: int = Field(default=8, ge=1, le=32)
    num_workers: int = Field(default=4, ge=0, le=16)
    clip_len: int = Field(default=64, ge=8, le=256)
    min_score: float = Field(default=0.15, ge=0, le=1)
    overwrite: bool = False
    stop_vllm: bool = False


def _annotation_path(video_name: str) -> Path:
    return ACTION_ANNOTATIONS_DIR / f"{Path(video_name).stem}_actions.json"


def _rally_annotation_path(video_name: str) -> Path | None:
    filename = f"{Path(video_name).stem}_annotations.jsonl"
    for directory in (ANNOTATIONS_DIR, PRE_ANNOTATIONS_DIR):
        path = directory / filename
        if path.exists():
            return path
    return None


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
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, f"Invalid annotation JSON: {path.name}") from exc
    events = data.get("events", [])
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
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, output_path)


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
            "has_action_annotation": ann_path.exists(),
            "event_count": event_count,
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
            pred_file = spot_prelabel.predictions_path(job.id, video.stem)
            run_log = pred_file.parent / "run.log"
            cmd = spot_prelabel.build_command(
                video_path=video,
                checkpoint_path=checkpoint,
                save_dir=pred_file.parent,
                batch_size=req.batch_size,
                num_workers=req.num_workers,
                clip_len=req.clip_len,
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


@router.get("/export")
def export_dataset() -> JSONResponse:
    ACTION_ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted(ACTION_ANNOTATIONS_DIR.glob("*_actions.json")):
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
    return JSONResponse(
        records,
        headers={"Content-Disposition": 'attachment; filename="spot_action_annotations.json"'},
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
