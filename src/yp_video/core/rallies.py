"""Where a video's rally spans come from — one answer, three possible files.

A rally span is upstream of almost everything: action events are stamped with
the rally they fall in, tracking scans rallies and nothing else, and a
tracklet's identity key is ``"{rally_id}:{track_id}"``. Three producers can
write those spans (a human, the SPOT model, the VLM bootstrap), so "which
file counts" has to have exactly one answer.

It did not. The Rally Label editor knew all three locations; the action
annotator knew only two, and silently missed the SPOT predictor's output —
so a video whose only rally source was SPOT got ``rallies: []``, every action
event got ``rally_id: None``, and the failure surfaced two stages later in
tracking as "no rally spans annotated". This module is that answer, in one
place, for every consumer.

Priority is reviewed truth, then the trained model, then the bootstrap:
``rally-spot/annotations`` → ``rally-spot/pre-annotations`` → ``rally/pre-annotations``.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from yp_video.config import (
    RALLY_ANNOTATIONS_DIR,
    RALLY_PRE_ANNOTATIONS_DIR,
    RALLY_SPOT_PRE_ANNOTATIONS_DIR,
)
from yp_video.core.annotation_ids import rally_id, stable_id
from yp_video.core.jsonl import read_jsonl_cached


class RallySource(NamedTuple):
    tag: str
    directory: Path
    r2_category: str


#: Load priority, highest first. The Rally Label UI can force one by tag.
RALLY_SOURCES = (
    RallySource("annotation", RALLY_ANNOTATIONS_DIR, "rally-spot/annotations"),
    RallySource("spot-pre-annotation", RALLY_SPOT_PRE_ANNOTATIONS_DIR, "rally-spot/pre-annotations"),
    RallySource("pre-annotation", RALLY_PRE_ANNOTATIONS_DIR, "rally/pre-annotations"),
)
SOURCE_BY_TAG = {source.tag: source for source in RALLY_SOURCES}


def annotation_name(stem: str) -> str:
    return f"{stem}_annotations.jsonl"


def rally_annotation_path(stem: str) -> Path | None:
    """The file that owns this video's rallies, or None when nobody does."""
    name = annotation_name(stem)
    for source in RALLY_SOURCES:
        path = source.directory / name
        if path.exists():
            return path
    return None


def rally_sources(stem: str) -> list[str]:
    """Every source tag that has spans for this video, in priority order."""
    name = annotation_name(stem)
    return [s.tag for s in RALLY_SOURCES if (s.directory / name).exists()]


def load_rallies(stem: str) -> list[dict]:
    """The video's rally spans as ``{rally_id, start, end, label}``, in seconds.

    ``rally_id`` is positional (see core/annotation_ids.rally_id), so the
    ordering here — by (start, end, label) — is what makes the ids stable
    between readers. Every consumer must come through this function for that
    to hold.
    """
    path = rally_annotation_path(stem)
    if path is None:
        return []
    _meta, records = read_jsonl_cached(path)  # read-only; new dicts built below
    parsed = [
        (
            float(r.get("start", r.get("start_time", 0)) or 0),
            float(r.get("end", r.get("end_time", 0)) or 0),
            str(r.get("label", "rally")),
            r,
        )
        for r in records
    ]
    # Key explicitly: a plain sort would fall through to comparing the raw
    # record dicts whenever two spans share (start, end, label).
    parsed.sort(key=lambda item: item[:3])
    return [
        {
            "rally_id": rally_id(stem, record, index),
            "start": start,
            "end": end,
            "label": label,
        }
        for index, (start, end, label, record) in enumerate(parsed)
    ]


def rally_fingerprint(stem: str) -> str | None:
    """Identity of the current spans, for detecting that they moved.

    Tracklets are keyed by rally, and a human can label a tracklet. When the
    rallies change, those keys stop meaning what they meant — this is how a
    consumer notices instead of silently mis-resolving.
    """
    rallies = load_rallies(stem)
    if not rallies:
        return None
    return stable_id(
        "rallies",
        [(r["rally_id"], round(r["start"], 3), round(r["end"], 3)) for r in rallies],
    )
