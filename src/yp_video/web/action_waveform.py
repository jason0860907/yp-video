"""Video metadata probing and cached, timeline-aligned audio waveforms."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import subprocess
from pathlib import Path

import numpy as np
from fastapi import HTTPException

from yp_video.config import ACTION_WAVEFORMS_DIR
from yp_video.core.ffmpeg import (
    FFmpegError,
    parse_optional_float,
    probe_video_metadata,
)
from yp_video.web.r2_client import cut_media_source, remote_cut_fingerprint

log = logging.getLogger(__name__)

AUDIO_WAVEFORM_SAMPLE_RATE = 32000
AUDIO_WAVEFORM_CHANNELS = 2
AUDIO_WAVEFORM_CACHE_VERSION = 7


def _media_source(video: Path) -> str:
    """The cut's local file or a presigned R2 URL; HTTP 404 when neither exists."""
    source = cut_media_source(video)
    if source is None:
        raise HTTPException(404, "Video not found")
    return source


def _probe_metadata(source: str) -> dict:
    try:
        return probe_video_metadata(source)
    except FFmpegError as exc:
        raise HTTPException(502, str(exc)) from exc


def video_metadata(path: Path) -> dict:
    """Probe ``{fps, duration, num_frames, start_time}`` from the local file
    or its R2 copy; HTTP 404 when the cut exists nowhere, 502 on probe failure."""
    return _probe_metadata(_media_source(path))


def _timeline_metadata(source: str, video_meta: dict) -> dict:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=start_time,duration:stream=codec_type,start_time,duration",
        "-of", "json",
        source,
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
    format_start = parse_optional_float(format_info.get("start_time")) or 0.0
    video_start = parse_optional_float(video_stream.get("start_time")) if video_stream else None
    audio_start = parse_optional_float(audio_stream.get("start_time")) if audio_stream else None
    return {
        "format_start_time": format_start,
        "format_duration": parse_optional_float(format_info.get("duration")),
        "video_start_time": video_start if video_start is not None else float(video_meta.get("start_time") or 0.0),
        "audio_start_time": audio_start,
        "audio_duration": parse_optional_float(audio_stream.get("duration")) if audio_stream else None,
    }


def _safe_cache_stem(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
    return safe[:80] or "video"


def _waveform_cache_path(video: Path, points: int) -> Path:
    if video.exists():
        stat = video.stat()
        fingerprint = f"{stat.st_size}:{stat.st_mtime_ns}"
    else:
        fingerprint = remote_cut_fingerprint(video.name) or "remote"
    cache_key = hashlib.sha1(
        f"v{AUDIO_WAVEFORM_CACHE_VERSION}:{video.name}:{fingerprint}:{points}".encode("utf-8"),
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


def audio_waveform(video: Path, points: int) -> dict:
    source = _media_source(video)
    meta = _probe_metadata(source)
    timeline = _timeline_metadata(source, meta)
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
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", source,
        "-vn", "-ac", str(AUDIO_WAVEFORM_CHANNELS), "-ar",
        str(AUDIO_WAVEFORM_SAMPLE_RATE), "-f", "s16le", "-",
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
                take = min(
                    channel_values.shape[0] - offset,
                    boundary - samples_seen,
                    target_samples - samples_seen,
                )
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
