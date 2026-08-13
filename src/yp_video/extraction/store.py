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
from yp_video.core.jsonl import read_jsonl_cached
from yp_video.core.rallies import (
    load_rallies,
    rally_annotation_path,
)

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

# These describe the ACTION, not the detector output.  Old extraction files
# contain snapshots of them, but action annotations are the sole authority:
# changing "spike" to "score" must not require another expensive detection
# pass merely to make every downstream reader see the edit.
ACTION_FIELDS = frozenset({"frame", "label", "xy", "visible"})

# Rally-derived copies that annotation files and old records used to carry;
# the live rally store owns them now. Stripped on read, never re-applied.
LEGACY_ACTION_FIELDS = frozenset({"time", "rally_id", "relative_frame"})


def action_source_paths(stem: str) -> list[Path]:
    """Files whose edits can change which extraction records are current.

    Consumers with their own ``StatCache`` include these alongside the
    extraction file, otherwise the merge below would be correct only after a
    server restart.
    """
    return [
        path
        for path in (action_annotation_path(stem), rally_annotation_path(stem))
        if path is not None
    ]


def labelable_actions(stem: str, fps: float = 0) -> list[dict]:
    """Current action-source events that call for identifying a player.

    Unlike ``labelable``, this does not require detector output to exist. It is
    the cheap, authoritative denominator for work-list progress.
    """
    source = action_annotation_path(stem)
    if source is None:
        return []
    meta, rows = read_jsonl_cached(source)
    spans = load_rallies(stem)
    source_fps = fps or float(meta.get("fps") or 0)
    return [
        event
        for event in rows
        if event.get("frame") is not None
        and event.get("label") not in SKIP_LABELS
        and _within(event, spans, source_fps)
    ]


def with_current_actions(records: Iterable[dict], stem: str) -> list[dict]:
    """Join derived extraction rows to their current action event by id.

    The action annotation is the source of truth for event metadata. Extraction
    records are derived detector output joined to it by event id. This matters
    when an action is relabeled after detection: the stale copied ``label`` in
    an old record must not keep a current ``score`` on the Association board.
    Deleted action ids have no current event and are omitted.
    """
    records = list(records)
    source = action_annotation_path(stem)
    if source is None:
        # Compatibility for isolated imports/tests and legacy data whose
        # annotation source is unavailable. This is explicitly the fallback,
        # never the normal authority.
        events = {str(r.get("id")): r for r in records}
    else:
        _meta, rows = read_jsonl_cached(source)
        events = {
            str(event.get("id") or f"f{event['frame']}"): event
            for event in rows
            if event.get("frame") is not None
        }

    out = []
    for stored in records:
        event = events.get(str(stored.get("id")))
        if event is None:
            continue
        # Remove every old action-owned field before applying the current
        # source. Absence in the source is meaningful too (e.g. no xy).
        record = {
            key: value for key, value in stored.items()
            if key not in ACTION_FIELDS and key not in LEGACY_ACTION_FIELDS
        }
        record.update({
            key: event[key] for key in ACTION_FIELDS
            if key in event
        })
        # The join key belongs to both sides and must always survive.
        record["id"] = str(stored.get("id"))
        out.append(record)
    return out


def labelable(records: Iterable[dict], stem: str, fps: float) -> list[dict]:
    """Current action events a person can actually be identified in.

    Three exclusions, one reason — there is nobody/current event to name:

    - SKIP_LABELS: a "score" marks where the ball landed, not a player.
    - Outside every rally span: nobody is TRACKED between rallies (see
      tracklets/tracking.py, which scans rally spans and nothing else), so
      there is no tracklet for an actor to be, and in practice these are
      warm-up hits and mis-timed annotations.
    - Deleted action ids: derived detector output does not keep a removed
      annotation alive.

    Applied at read time by both labeling pages and by the identity layer, so
    the boards, the clusters and the counts agree about what is on the table.
    Nothing is deleted: an event dropped here keeps whatever was recorded
    about it, and comes back the moment the rally spans cover it.
    """
    spans = load_rallies(stem)
    return [
        record
        for record in with_current_actions(records, stem)
        if record.get("label") not in SKIP_LABELS
        and _within(record, spans, fps)
    ]


def _within(record: dict, spans: list[dict], fps: float) -> bool:
    """Whether the event falls in a rally — ``frame / fps`` against the live
    spans, the same rule every reader uses."""
    if not spans:
        # No rally source at all: hiding everything would be a worse answer
        # than showing it, and the pipeline chips already say what is missing.
        return True
    if not fps:
        return True  # cannot tell; never hide on a guess
    at = record["frame"] / fps
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
