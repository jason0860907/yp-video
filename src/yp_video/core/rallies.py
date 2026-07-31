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

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

from yp_video.config import (
    RALLY_ANNOTATIONS_DIR,
    RALLY_PRE_ANNOTATIONS_DIR,
    RALLY_SPOT_PRE_ANNOTATIONS_DIR,
)
from yp_video.core.annotation_ids import stable_id
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


def resolve_rally_ids(records: Sequence[Mapping]) -> list[int]:
    """The stored rally ids, verified — the file is the ledger.

    Every record must carry a distinct positive integer ``rally_id``. Ids are
    assigned once, at write time (the rally editor's save and the two
    pre-annotation producers), and never recomputed from position: tracklet
    keys and human labels reference these numbers, so a reader that invented
    ids from sort order would silently re-point them all.
    """
    ids: list[int] = []
    for record in records:
        raw = record.get("rally_id")
        if not isinstance(raw, int) or isinstance(raw, bool) or raw < 1:
            raise ValueError(
                f"Rally record without a valid rally_id: {raw!r} — "
                "run scripts/freeze_rally_ids.py on files from before ids were stored"
            )
        ids.append(raw)
    if len(set(ids)) != len(ids):
        seen: set[int] = set()
        duplicates = sorted({i for i in ids if i in seen or seen.add(i)})
        raise ValueError(f"Duplicate rally_id(s): {duplicates}")
    return ids


def number_rallies(segments: Sequence[Mapping]) -> tuple[list[dict], int]:
    """Fresh ids for a model-produced pass: rows stamped 1..N, and N.

    A pre-annotation is regenerated wholesale on every model run, so its ids
    are born here rather than carried over — re-running the model IS
    re-deciding the rallies, and the fingerprint mechanism reports the change
    to anything tracked against the old pass. A human save then freezes
    whatever the editor loaded (see web/routers/annotate.py).
    """
    rows = [
        {**segment, "rally_id": index}
        for index, segment in enumerate(
            sorted(segments, key=lambda s: (s.get("start", 0), s.get("end", 0))),
            start=1,
        )
    ]
    return rows, len(rows)


def load_rallies(stem: str) -> list[dict]:
    """The video's rally spans as ``{rally_id, start, end, label}``, in seconds.

    Sorted by (start, end, label) for the readers' benefit; ``rally_id`` comes
    from the FILE (see resolve_rally_ids), never from that ordering. Every
    consumer must come through this function.
    """
    path = rally_annotation_path(stem)
    if path is None:
        return []
    _meta, records = read_jsonl_cached(path)  # read-only; new dicts built below
    parsed = [
        (
            float(r.get("start", 0) or 0),
            float(r.get("end", 0) or 0),
            str(r.get("label", "rally")),
            rid,
        )
        for r, rid in zip(records, resolve_rally_ids(records))
    ]
    # Key explicitly: a plain sort would fall through to comparing ids
    # whenever two spans share (start, end, label) — harmless, but implicit.
    parsed.sort(key=lambda item: item[:3])
    return [
        {
            "rally_id": rid,
            "start": start,
            "end": end,
            "label": label,
        }
        for start, end, label, rid in parsed
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
