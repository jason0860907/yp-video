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
human label (those live in videos/reid/annotations/).
"""

from __future__ import annotations

from pathlib import Path

from yp_video.config import (
    ACTION_ANNOTATIONS_DIR,
    ACTION_PRE_ANNOTATIONS_DIR,
    REID_DIR,
)

RECORDS_DIR = REID_DIR / "records"
CROPS_DIR = REID_DIR / "crops"
# What the masked embedders actually saw (background-suppressed variants of
# crops/, same filenames) — persisted so the UI can show them. Regenerated on
# every masked embed run.
MASKED_CROPS_DIR = REID_DIR / "crops-masked"

# Action labels with nobody to re-identify: "score" marks where the ball
# lands, not a person. Applied at extraction AND at read time, so old
# extractions that predate the rule stay filtered too.
SKIP_LABELS = frozenset({"score"})


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
