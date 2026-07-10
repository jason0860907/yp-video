"""SPOT-style point action annotator router."""

import asyncio
import hashlib
import json
import logging
import math
import os
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import Literal
from urllib.parse import unquote

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field, field_validator

from yp_video.config import (
    ACTION_ANNOTATIONS_DIR,
    ACTION_PRE_ANNOTATIONS_DIR,
    ANNOTATIONS_DIR,
    ACTION_WAVEFORMS_DIR,
    CUT_R2_CATEGORIES,
    PRE_ANNOTATIONS_DIR,
    SPOT_DIR,
    cut_kind_of,
    find_cut,
    iter_all_cuts,
)
from yp_video.action import prelabel
from yp_video.action.frames import inspect_action_frame_cache
from yp_video.contracts.action import (
    ACTION_CONTRACT_VERSION,
    ACTION_CONTRACT_VERSION_ENV,
    ACTION_LABELS_ORDERED,
    SPOT_PROGRESS_PREFIX,
)
from yp_video.core.annotation_ids import action_id, rally_id
from yp_video.core.ffmpeg import (
    FFmpegError,
    parse_optional_float as _parse_optional_float,
    probe_video_metadata,
)
from yp_video.core.jsonl import read_jsonl
from yp_video.web.job_helpers import (
    TERMINAL_ITEM_STATUSES,
    ProgressParser,
    batch_message,
    batch_progress,
    fail_job_from_exc,
    finalize_batch_job,
    init_batch_items,
    stop_vllm_for_job,
    stream_subprocess,
    terminal_prefix,
    update_batch_item,
)
from yp_video.web.jobs import job_manager
from yp_video.web.r2_client import serve_video_or_r2_redirect, sync_to_r2

log = logging.getLogger(__name__)
router = APIRouter()

ACTION_LABELS = ACTION_LABELS_ORDERED
AUDIO_WAVEFORM_SAMPLE_RATE = 32000
AUDIO_WAVEFORM_CHANNELS = 2
AUDIO_WAVEFORM_CACHE_VERSION = 7
SPOT_DEFAULT_DECODER: Literal["opencv", "nvdec"] = "nvdec"
SPOT_DEFAULT_DECODE_PRODUCERS = 2
SPOT_DEFAULT_DECODER_THREADS = 1
SPOT_DEFAULT_PREFETCH_FACTOR = 2
SPOT_DEFAULT_DECODE_CHUNK_FRAMES = 256
SPOT_DEFAULT_NVIDIA_VIDEO_LIB_DIR = Path.home() / ".local/lib/nvidia-video"


class ActionEvent(BaseModel):
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


class SaveActionAnnotationsRequest(BaseModel):
    video: str
    fps: float = Field(gt=0)
    num_frames: int = Field(ge=0)
    events: list[ActionEvent]


class SpotPrelabelOptions(BaseModel):
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


class SpotPrelabelRequest(SpotPrelabelOptions):
    video: str


class SpotPrelabelBatchRequest(SpotPrelabelOptions):
    videos: list[str] = Field(min_length=1)


def _annotation_path(video_name: str) -> Path:
    return ACTION_ANNOTATIONS_DIR / f"{Path(video_name).stem}_actions.jsonl"


def _pre_annotation_path(video_name: str) -> Path:
    return ACTION_PRE_ANNOTATIONS_DIR / f"{Path(video_name).stem}_actions.jsonl"


def _active_annotation_path(video_name: str) -> Path:
    final_path = _annotation_path(video_name)
    pre_path = _pre_annotation_path(video_name)
    if final_path.exists():
        try:
            if _annotation_reviewed(_load_annotation(final_path)):
                return final_path
        except HTTPException:
            return final_path
    return pre_path if pre_path.exists() else final_path


def _rally_annotation_path(video_name: str) -> Path | None:
    # Priority: manual rally annotations, then rally pre-annotations.
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


def _video_metadata(path: Path) -> dict:
    """Probe ``{fps, duration, num_frames, start_time}``; HTTP 502 on failure."""
    try:
        return probe_video_metadata(path)
    except FFmpegError as exc:
        raise HTTPException(502, str(exc)) from exc


def _timeline_metadata(path: Path, video_meta: dict) -> dict:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=start_time,duration:stream=codec_type,start_time,duration",
        "-of", "json",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise HTTPException(502, f"ffprobe failed: {result.stderr[:200]}")
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise HTTPException(502, "ffprobe returned invalid JSON") from exc

    streams = data.get("streams") or []
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), {})
    format_info = data.get("format") or {}
    format_start = _parse_optional_float(format_info.get("start_time")) or 0.0
    video_start = (
        _parse_optional_float(video_stream.get("start_time"))
        if video_stream
        else None
    )
    audio_start = (
        _parse_optional_float(audio_stream.get("start_time"))
        if audio_stream
        else None
    )

    return {
        "format_start_time": format_start,
        "format_duration": _parse_optional_float(format_info.get("duration")),
        "video_start_time": video_start if video_start is not None else float(video_meta.get("start_time") or 0.0),
        "audio_start_time": audio_start,
        "audio_duration": _parse_optional_float(audio_stream.get("duration")) if audio_stream else None,
    }


def _safe_cache_stem(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
    return safe[:80] or "video"


def _waveform_cache_path(video: Path, points: int) -> Path:
    stat = video.stat()
    cache_key = hashlib.sha1(
        f"v{AUDIO_WAVEFORM_CACHE_VERSION}:{video.name}:{stat.st_size}:{stat.st_mtime_ns}:{points}".encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()[:16]
    return ACTION_WAVEFORMS_DIR / f"{_safe_cache_stem(video.stem)}-v{AUDIO_WAVEFORM_CACHE_VERSION}-{cache_key}-{points}.json"


def _empty_waveform(video: Path, meta: dict, *, reason: str) -> dict:
    return {
        "video": video.name,
        "has_audio": False,
        "reason": reason,
        "duration": float(meta.get("duration") or 0),
        "sample_rate": AUDIO_WAVEFORM_SAMPLE_RATE,
        "channels_measured": AUDIO_WAVEFORM_CHANNELS,
        "timeline_aligned": True,
        "points": 0,
        "peak": 0,
        "peaks": [],
        "rms": [],
    }


def _write_waveform_cache(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp_path, path)


def _audio_waveform(video: Path, points: int) -> dict:
    meta = _video_metadata(video)
    timeline = _timeline_metadata(video, meta)
    cache_path = _waveform_cache_path(video, points)
    if cache_path.exists():
        try:
            with open(cache_path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass

    duration = float(meta.get("duration") or 0)
    if duration <= 0:
        payload = _empty_waveform(video, meta, reason="unknown_duration")
        _write_waveform_cache(cache_path, payload)
        return payload

    peaks = np.zeros(points, dtype=np.float32)
    sum_squares = np.zeros((points, AUDIO_WAVEFORM_CHANNELS), dtype=np.float64)
    sample_counts = np.zeros(points, dtype=np.int64)
    samples_per_bin = max(1, math.ceil(duration * AUDIO_WAVEFORM_SAMPLE_RATE / points))
    target_samples = max(1, math.ceil(duration * AUDIO_WAVEFORM_SAMPLE_RATE))
    video_start = float(timeline.get("video_start_time") or 0.0)
    audio_start = timeline.get("audio_start_time")
    audio_offset = 0.0 if audio_start is None else float(audio_start) - video_start
    skip_audio_samples = max(0, int(round(-audio_offset * AUDIO_WAVEFORM_SAMPLE_RATE)))
    samples_seen = min(target_samples, max(0, int(round(audio_offset * AUDIO_WAVEFORM_SAMPLE_RATE))))
    reached_target = samples_seen >= target_samples
    stderr = ""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vn",
        "-ac",
        str(AUDIO_WAVEFORM_CHANNELS),
        "-ar",
        str(AUDIO_WAVEFORM_SAMPLE_RATE),
        "-f",
        "s16le",
        "-",
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise HTTPException(500, "ffmpeg not found") from exc

    assert proc.stdout is not None
    try:
        while True:
            chunk = proc.stdout.read(128 * 1024)
            if not chunk:
                break
            frame_bytes = 2 * AUDIO_WAVEFORM_CHANNELS
            remainder = len(chunk) % frame_bytes
            if remainder:
                chunk = chunk[:-remainder]
            if not chunk:
                continue

            samples = np.frombuffer(chunk, dtype="<i2").reshape(-1, AUDIO_WAVEFORM_CHANNELS)
            if skip_audio_samples:
                if samples.shape[0] <= skip_audio_samples:
                    skip_audio_samples -= samples.shape[0]
                    continue
                samples = samples[skip_audio_samples:]
                skip_audio_samples = 0
            if reached_target:
                break
            remaining = target_samples - samples_seen
            if samples.shape[0] > remaining:
                samples = samples[:remaining]
                reached_target = True
            channel_values = samples.astype(np.float32) / 32768.0
            channel_abs = np.abs(channel_values)
            offset = 0
            while offset < channel_values.shape[0] and samples_seen < target_samples:
                bin_idx = min(points - 1, samples_seen // samples_per_bin)
                boundary = (bin_idx + 1) * samples_per_bin
                take = min(channel_values.shape[0] - offset, boundary - samples_seen, target_samples - samples_seen)
                if take <= 0:
                    break
                segment = channel_values[offset:offset + take]
                abs_segment = channel_abs[offset:offset + take]
                if segment.size:
                    peaks[bin_idx] = max(float(peaks[bin_idx]), float(abs_segment.max()))
                    segment64 = segment.astype(np.float64, copy=False)
                    sum_squares[bin_idx] += (segment64 * segment64).sum(axis=0)
                    sample_counts[bin_idx] += segment.shape[0]
                samples_seen += take
                offset += take
            if samples_seen >= target_samples:
                reached_target = True
                proc.terminate()
                break
    finally:
        if proc.stdout:
            proc.stdout.close()

    if proc.stderr:
        stderr = proc.stderr.read().decode("utf-8", errors="replace")
        proc.stderr.close()
    return_code = proc.wait()

    if samples_seen <= 0:
        payload = _empty_waveform(video, meta, reason="no_audio")
        _write_waveform_cache(cache_path, payload)
        return payload
    if return_code != 0 and not reached_target:
        log.warning("ffmpeg waveform finished with code %s for %s: %s", return_code, video, stderr[:300])

    peak = float(peaks.max()) if peaks.size else 0.0
    rms_by_channel = np.zeros_like(sum_squares, dtype=np.float64)
    valid_bins = sample_counts > 0
    rms_by_channel[valid_bins] = np.sqrt(sum_squares[valid_bins] / sample_counts[valid_bins, None])
    rms = rms_by_channel.max(axis=1)
    payload = {
        "video": video.name,
        "has_audio": True,
        "duration": duration,
        "sample_rate": AUDIO_WAVEFORM_SAMPLE_RATE,
        "channels_measured": AUDIO_WAVEFORM_CHANNELS,
        "timeline_aligned": True,
        "video_start_time": round(video_start, 6),
        "audio_start_time": round(float(audio_start), 6) if audio_start is not None else None,
        "audio_offset": round(audio_offset, 6),
        "audio_duration": round(float(timeline["audio_duration"]), 6) if timeline.get("audio_duration") is not None else None,
        "points": points,
        "peak": round(peak, 4),
        "peaks": [round(float(v), 4) for v in peaks.tolist()],
        "rms": [round(float(v), 4) for v in rms.tolist()],
    }
    _write_waveform_cache(cache_path, payload)
    return payload


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


def _annotation_reviewed(data: dict | None) -> bool:
    if not data:
        return False
    if "reviewed" in data:
        return bool(data["reviewed"])
    source = data.get("source")
    if isinstance(source, dict) and source.get("type") == "spot":
        return False
    return True


def _load_rallies(video: Path) -> list[dict]:
    path = _rally_annotation_path(video.name)
    if path is None:
        return []
    meta, records = read_jsonl(path)
    source_video = str(meta.get("video") or video.name)
    parsed_records = []
    for record in records:
        start = float(record.get("start", record.get("start_time", 0)) or 0)
        end = float(record.get("end", record.get("end_time", 0)) or 0)
        parsed_records.append((start, end, record.get("label", "rally"), record))
    parsed_records.sort(key=lambda r: (r[0], r[1], str(r[2])))

    rallies: list[dict] = []
    for i, (start, end, label, record) in enumerate(parsed_records):
        rallies.append({
            "rally_id": rally_id(source_video, record, i),
            "start": start,
            "end": end,
            "label": label,
        })
    return rallies


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
    replace_final: bool = False,
) -> dict:
    predictions = await asyncio.to_thread(prelabel.load_predictions, pred_file)
    data = prelabel.predictions_to_annotation(
        predictions,
        video_path=video,
        metadata=meta,
        checkpoint_path=checkpoint,
        min_score=min_score,
    )
    data["reviewed"] = False
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
    if replace_final:
        final_path = _annotation_path(video.name)
        if final_path != ann_path:
            final_path.unlink(missing_ok=True)
    sync_to_r2(ann_path, "action-pre-annotations")
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
        final_path = _annotation_path(video.name)
        pre_path = _pre_annotation_path(video.name)
        if (final_path.exists() or pre_path.exists()) and not overwrite:
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
    results = []
    for video in sorted(iter_all_cuts(), key=lambda p: p.name):
        final_path = _annotation_path(video.name)
        pre_path = _pre_annotation_path(video.name)
        ann_path = _active_annotation_path(video.name)
        ann = None
        event_count = 0
        training_event_count = 0
        has_training_annotation = final_path.exists()
        if has_training_annotation:
            try:
                final_ann = _load_annotation(final_path)
                training_event_count = len((final_ann or {}).get("events", []))
            except HTTPException:
                training_event_count = -1
        if ann_path.exists():
            try:
                ann = _load_annotation(ann_path)
                event_count = len((ann or {}).get("events", []))
            except HTTPException:
                event_count = -1
        reviewed = _annotation_reviewed(ann)
        has_active = ann_path.exists()
        results.append({
            "name": video.name,
            "kind": cut_kind_of(video),
            "rally_sources": _rally_sources(video.name),
            "has_action_annotation": has_active,
            "has_action_pre_annotation": has_active and not reviewed,
            "has_action_final_annotation": has_active and reviewed,
            "has_action_training_annotation": has_training_annotation,
            "action_annotation_source": "action-annotations" if ann_path == final_path and has_active else ("action-pre-annotations" if has_active else ""),
            "action_reviewed": reviewed,
            "event_count": event_count,
            "training_event_count": training_event_count,
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
    ann = _load_annotation(_active_annotation_path(video.name))
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


@router.get("/waveform/{name:path}")
async def get_waveform(name: str, points: int = Query(default=9600, ge=200, le=96000)) -> dict:
    decoded = unquote(name)
    video = find_cut(Path(decoded).name)
    if video is None:
        raise HTTPException(404, "Video not found")
    return await asyncio.to_thread(_audio_waveform, video, points)


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
        "source": {"type": "manual"},
        "reviewed": True,
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
    if not prelabel.spot_available():
        raise HTTPException(503, "SPOT is not available at ~/yp-spot")

    video = find_cut(Path(req.video).name)
    if video is None:
        raise HTTPException(404, "Video not found")

    final_path = _annotation_path(video.name)
    pre_path = _pre_annotation_path(video.name)
    if (final_path.exists() or pre_path.exists()) and not req.overwrite:
        raise HTTPException(409, "Action pre-label already exists; set overwrite=true")

    try:
        checkpoint = prelabel.resolve_checkpoint(req.checkpoint)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    job = job_manager.create_job(
        "spot_prelabel",
        {
            "video": video.name,
            "checkpoint": prelabel.checkpoint_ref(checkpoint),
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
                progress=0.05,
                message="Waiting for inference slot...",
            )

            def start_handler(_match):
                return {
                    "progress": 0.08,
                    "message": "Preparing first batch (decoding frames)...",
                }

            def progress_handler(match):
                data = prelabel.parse_spot_progress(match.group(1))
                if data is None:
                    return None
                return {
                    "progress": _spot_progress_fraction(data, start=0.08, span=0.82, cap=0.9),
                    "message": prelabel.spot_progress_message(data),
                }

            parsers = [
                ProgressParser(r"Starting inference \d+/\d+: .+", start_handler),
                ProgressParser(SPOT_PROGRESS_PREFIX + r"(.+)", progress_handler),
            ]

            with tempfile.TemporaryDirectory(prefix=f"yp-spot-{job.id}-") as tmp_root:
                pred_file = Path(tmp_root) / "predictions.json"
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

                async with stop_vllm_for_job(job.id, when=req.stop_vllm):
                    # inference_lock (not gpu_lock) so this can run alongside a
                    # training job; still serialized against other inference jobs.
                    async with job_manager.inference_lock:
                        await job_manager.update_job(
                            job.id,
                            progress=0.08,
                            message="Launching SPOT inference...",
                        )
                        rc, last_line = await stream_subprocess(
                            job.id,
                            cmd,
                            SPOT_DIR,
                            env=_spot_subprocess_env(req),
                            parsers=parsers,
                            is_key_line=lambda line: (
                                line.startswith("Starting inference")
                                or line.startswith(SPOT_PROGRESS_PREFIX)
                                or line.startswith("Saved predictions")
                            ),
                            push_interval=1.0,
                            tee_to_terminal=True,
                            log_command=(
                                f"{terminal_prefix(job)}start single video={video.name} "
                                f"batch={req.batch_size} workers={req.num_workers} "
                                f"{_spot_decode_settings_text(req)}"
                            ),
                            log_line=lambda line: _spot_app_log_line(terminal_prefix(job), line),
                        )
                if rc != 0:
                    raise RuntimeError(last_line or f"SPOT exited with code {rc}")
                if not pred_file.exists():
                    raise RuntimeError("SPOT did not create prediction output")

                await job_manager.update_job(job.id, progress=0.93, message="Saving action pre-label...")
                data = await _save_spot_action_annotation(
                    video=video,
                    ann_path=pre_path,
                    meta=meta,
                    pred_file=pred_file,
                    checkpoint=checkpoint,
                    min_score=req.min_score,
                    replace_final=req.overwrite,
                )
                log.info("%ssaved %s (%d event(s))", terminal_prefix(job), pre_path.name, data["num_events"])
            await job_manager.update_job(
                job.id,
                status="completed",
                progress=1.0,
                message=f"Pre-label complete: {data['num_events']} event(s)",
                params={
                    **job.params,
                    "count": data["num_events"],
                    "saved": str(pre_path),
                },
            )
        except asyncio.CancelledError:
            await job_manager.update_job(job.id, status="cancelled", message="Cancelled")
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("SPOT pre-label failed for %s", video.name)
            await fail_job_from_exc(job.id, exc)

    task = asyncio.create_task(run_job())
    job_manager.attach_task(job, task)
    return job.to_dict()


@router.post("/prelabel-batch")
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
        "spot_prelabel_batch",
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
                failed = await _run_prelabel_batch_subprocess(
                    job.id,
                    items,
                    entries,
                    checkpoint=checkpoint,
                    req=req,
                )
            await finalize_batch_job(job.id, total, failed)
        except asyncio.CancelledError:
            for idx, item in enumerate(items):
                if item.get("status") not in TERMINAL_ITEM_STATUSES:
                    await update_batch_item(
                        job.id,
                        items,
                        idx,
                        status="cancelled",
                        progress=float(item.get("progress") or 0),
                        message="Cancelled",
                    )
            await job_manager.update_job(job.id, status="cancelled", message="Cancelled")
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
            metas.append(await asyncio.to_thread(_video_metadata, video))
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
                    replace_final=req.overwrite,
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
