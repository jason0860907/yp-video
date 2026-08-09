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
from pathlib import Path
from typing import NamedTuple

from fastapi import HTTPException

from yp_video.config import ACTION_ANNOTATIONS_DIR, ACTION_PRE_ANNOTATIONS_DIR
from yp_video.core.cache import StatCache
from yp_video.core.jsonl import read_jsonl


def annotation_path(video_name: str) -> Path:
    return ACTION_ANNOTATIONS_DIR / f"{Path(video_name).stem}_actions.jsonl"


def pre_annotation_path(video_name: str) -> Path:
    return ACTION_PRE_ANNOTATIONS_DIR / f"{Path(video_name).stem}_actions.jsonl"


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
