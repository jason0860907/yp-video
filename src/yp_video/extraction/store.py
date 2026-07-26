"""Where an extraction's output lives: the event records and the actor crops.

    records/<stem>.jsonl      one record per action event — the event, every
                              person the detector found on its frame, and the
                              actor finally chosen
    crops/<stem>/<event>.jpg  the chosen actor, cut on the display box
    crops-masked/<stem>/…     background-suppressed variants (same filenames)

Written by extraction/pipeline.py and read by both consumers: ``actor`` wants
the detections and the contact point, ``reid`` wants the crops. Keeping the
layout here rather than inside either of them is what lets the two packages
stay unaware of each other — this module is a leaf, importing nothing but
config and core.

Everything here is derived data: deleting it costs a re-extraction, never a
human label (those live in videos/association/ and videos/reid/).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from yp_video.config import (
    ACTION_ANNOTATIONS_DIR,
    ACTION_PRE_ANNOTATIONS_DIR,
    EXTRACTION_DIR,
)
from yp_video.core.rallies import load_rallies

RECORDS_DIR = EXTRACTION_DIR / "records"
CROPS_DIR = EXTRACTION_DIR / "crops"
# What the masked embedders actually saw (background-suppressed variants of
# crops/, same filenames) — persisted so the UI can show them. Regenerated on
# every masked embed run.
MASKED_CROPS_DIR = EXTRACTION_DIR / "crops-masked"

# Action labels with nobody to re-identify: "score" marks where the ball
# lands, not a person. Applied at extraction AND at read time, so old
# extractions that predate the rule stay filtered too.
SKIP_LABELS = frozenset({"score"})


def labelable(records: Iterable[dict], stem: str, fps: float) -> list[dict]:
    """The events a person can actually be identified in.

    Two exclusions, one reason — there is nobody to name:

    - SKIP_LABELS: a "score" marks where the ball landed, not a player.
    - Outside every rally span: nobody is TRACKED between rallies (see
      tracklets/tracking.py, which scans rally spans and nothing else), so
      there is no tracklet for an actor to be, and in practice these are
      warm-up hits and mis-timed annotations.

    Applied at read time by both labeling pages and by the identity layer, so
    the boards, the clusters and the counts agree about what is on the table.
    Nothing is deleted: an event dropped here keeps whatever was recorded
    about it, and comes back the moment the rally spans cover it.
    """
    spans = load_rallies(stem)
    return [
        record
        for record in records
        if record.get("label") not in SKIP_LABELS
        and _within(record, spans, fps)
    ]


def _within(record: dict, spans: list[dict], fps: float) -> bool:
    """Whether the event falls in a rally. Prefer the stored time, fall back
    to frame/fps — the same rule the sidebar uses, so the two cannot disagree
    about which rally an event belongs to."""
    if not spans:
        # No rally source at all: hiding everything would be a worse answer
        # than showing it, and the pipeline chips already say what is missing.
        return True
    at = record.get("time")
    if at is None:
        at = record["frame"] / fps if fps else None
    if at is None:
        return True  # cannot tell; never hide on a guess
    return any(span["start"] <= at <= span["end"] for span in spans)


def records_path(stem: str) -> Path:
    return RECORDS_DIR / f"{stem}.jsonl"


def crop_dir(stem: str) -> Path:
    return CROPS_DIR / stem


def masked_crop_dir(stem: str) -> Path:
    return MASKED_CROPS_DIR / stem


def action_annotation_path(stem: str) -> Path | None:
    """Manual action annotations win over pre-annotations."""
    for directory in (ACTION_ANNOTATIONS_DIR, ACTION_PRE_ANNOTATIONS_DIR):
        path = directory / f"{stem}_actions.jsonl"
        if path.exists():
            return path
    return None
