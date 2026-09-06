"""The action annotation store: which file is live for a video, parsed once.

Provenance is by location: only the editor's Save writes
ACTION_ANNOTATIONS_DIR, machine output goes to ACTION_PRE_ANNOTATIONS_DIR —
so the final file existing at all means a human wrote it. Shared by the
action-annotate router (editing, prelabel) and web/worklists.py (listing);
routers may not import each other (tests/test_layering.py), so the store
lives here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import NamedTuple

from fastapi import HTTPException

from yp_video.action import prelabel
from yp_video.config import ACTION_ANNOTATIONS_DIR, ACTION_PRE_ANNOTATIONS_DIR
from yp_video.contracts.action import LABEL_FILE_SUFFIX
from yp_video.core.annotation_ids import action_id
from yp_video.core.cache import StatCache
from yp_video.core.ffmpeg import parse_optional_float
from yp_video.core.jsonl import read_jsonl


def annotation_path(video_name: str) -> Path:
    return ACTION_ANNOTATIONS_DIR / f"{Path(video_name).stem}{LABEL_FILE_SUFFIX}"


def pre_annotation_path(video_name: str) -> Path:
    return ACTION_PRE_ANNOTATIONS_DIR / f"{Path(video_name).stem}{LABEL_FILE_SUFFIX}"


_annotation_cache = StatCache()


def load_annotation(path: Path) -> dict | None:
    """Parsed annotation with events sorted, cached per file version.

    The returned dict and its events are shared across callers — a caller
    that mutates must copy first.
    """
    if not path.exists():
        return None

    def compute() -> dict:
        try:
            data, events = read_jsonl(path)
        except json.JSONDecodeError as exc:
            raise HTTPException(400, f"Invalid annotation JSONL: {path.name}") from exc
        data["events"] = sorted(events, key=lambda e: (e.get("frame", 0), e.get("label", "")))
        data["num_events"] = len(data["events"])
        return data

    try:
        return _annotation_cache.get(str(path), [path], compute)
    except FileNotFoundError:
        return None  # deleted between exists() and stat


class AnnotationState(NamedTuple):
    """Which annotation file is live for a video, both payloads parsed once.

    ``active`` is what the editor reads — the final file when it exists (or
    is corrupt, so the parse error surfaces instead of silently opening the
    pre file), otherwise the pre-annotation. A corrupt file parses to
    ``None`` with its HTTPException in the matching ``*_error`` field.
    """

    final: dict | None
    final_error: HTTPException | None
    active_path: Path
    active: dict | None
    active_error: HTTPException | None

    @property
    def human(self) -> bool:
        """A human-saved annotation exists, even one that fails to parse."""
        return self.final is not None or self.final_error is not None


def _try_load(path: Path) -> tuple[dict | None, HTTPException | None]:
    try:
        return load_annotation(path), None
    except HTTPException as exc:
        return None, exc


def annotation_state(video_name: str) -> AnnotationState:
    final_path = annotation_path(video_name)
    final, final_error = _try_load(final_path)
    if final is not None or final_error is not None:
        return AnnotationState(final, final_error, final_path, final, final_error)
    pre_path = pre_annotation_path(video_name)
    if pre_path.exists():
        active, active_error = _try_load(pre_path)
        return AnnotationState(None, None, pre_path, active, active_error)
    return AnnotationState(None, None, final_path, None, None)


def rally_for_event(event: dict, fps: float, rallies: list[dict]) -> dict | None:
    if not rallies:
        return None
    explicit_time = parse_optional_float(event.get("time"))
    if explicit_time is not None:
        time = explicit_time
    else:
        frame = parse_optional_float(event.get("frame")) or 0.0
        time = frame / fps if fps > 0 else 0.0
    for rally in rallies:
        if rally["start"] <= time < rally["end"]:
            return rally
    existing_id = coerce_rally_id(event.get("rally_id"))
    if existing_id:
        for rally in rallies:
            if rally["rally_id"] == existing_id:
                return rally
    return None


def coerce_rally_id(value: object) -> int | None:
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and value.isdigit() and int(value) > 0:
        return int(value)
    return None


def normalize_events(video_stem: str, events: list[dict], *, fps: float, num_frames: int, rallies: list[dict]) -> list[dict]:
    normalized = []
    max_frame = max(0, num_frames - 1)
    for i, raw in enumerate(events):
        event = dict(raw)
        frame = max(0, min(int(round(float(event.get("frame", 0) or 0))), max_frame))
        event["frame"] = frame
        event["id"] = action_id(video_stem, event, i)
        time = frame / fps if fps > 0 else float(event.get("time") or 0)
        event["time"] = round(time, 4)
        event["visible"] = truthy_event_visible(event.get("visible", True))
        rally = rally_for_event(event, fps, rallies)
        if rally:
            event["rally_id"] = rally["rally_id"]
            event["relative_frame"] = max(0, int(round((time - rally["start"]) * fps)))
        else:
            event["rally_id"] = None
            event["relative_frame"] = None
        normalized.append(event)
    normalized.sort(key=lambda e: (e["frame"], e["label"], e["id"]))
    return normalized


#: What the store persists per event: the human's facts, nothing derived.
#: rally_id / relative_frame / time are recomputed from the live rally store
#: on every read (normalize_events) — a stored copy goes stale the moment
#: rallies are re-edited, which is exactly how the Association board once
#: ended up navigating by outdated spans.
PERSISTED_EVENT_FIELDS = ("id", "frame", "label", "xy", "visible")


def persistable_events(events: list[dict]) -> list[dict]:
    return [{key: event[key] for key in PERSISTED_EVENT_FIELDS} for event in events]


def truthy_event_visible(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return value is not False


def write_annotation_atomic(output_path: Path, data: dict) -> None:
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


def save_spot_pre_annotation(
    *,
    video: Path,
    meta: dict,
    predictions: list[dict],
    checkpoint: Path,
    min_score: float,
) -> dict:
    """Write yp-spot action predictions as this video's machine pre-annotation.

    The one path every SPOT action run takes into the store — the Action
    Predict page and the fusion Inference job alike — so the editor always
    finds the same normalized, id-stamped events. Mirroring to R2 is the
    caller's business: it needs the event loop.
    """
    data = prelabel.predictions_to_annotation(
        predictions,
        video_path=video,
        metadata=meta,
        checkpoint_path=checkpoint,
        min_score=min_score,
    )
    data["events"] = persistable_events(normalize_events(
        video.stem,
        data.get("events", []),
        fps=float(data.get("fps") or meta["fps"]),
        num_frames=int(data.get("num_frames") or meta["num_frames"]),
        rallies=[],
    ))
    data["num_events"] = len(data["events"])
    ann_path = pre_annotation_path(video.name)
    ann_path.parent.mkdir(parents=True, exist_ok=True)
    write_annotation_atomic(ann_path, data)
    return data
