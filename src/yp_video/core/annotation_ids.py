"""Stable identifiers for rally segments and action events."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_id(prefix: str, *parts: Any) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def rally_id(video: str, record: dict, index: int = 0) -> str:
    existing = record.get("id")
    if isinstance(existing, str) and existing:
        return existing
    start = round(float(record.get("start", record.get("start_time", 0)) or 0), 3)
    end = round(float(record.get("end", record.get("end_time", 0)) or 0), 3)
    label = record.get("label", "rally")
    return stable_id("rally", video, index, start, end, label)


def action_id(video: str, record: dict, index: int = 0) -> str:
    existing = record.get("id")
    if isinstance(existing, str) and existing:
        return existing
    frame = int(record.get("frame", 0) or 0)
    label = record.get("label", "")
    xy = record.get("xy") or [record.get("x", 0.5), record.get("y", 0.5)]
    try:
        x = round(float(xy[0]), 4)
        y = round(float(xy[1]), 4)
    except (TypeError, ValueError, IndexError):
        x, y = 0.5, 0.5
    return stable_id("act", video, index, frame, label, x, y)
