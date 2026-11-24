"""The per-event rows both labeling pages show, and the video context the
player draws them in.

ReID Label reads them off the extraction records. Association Label needs no
detector run at all: a pick is a 2XLarge box on the event frame
(actor/box_check.py), not one of the detections, and an event detection never
saw is just an unassociated record (pipeline.unextracted_records). So a video
without a records file is a video whose every event is unassociated — the
records file is an input here, never a precondition.
"""

from __future__ import annotations

from typing import TypedDict

from yp_video.actor import labels as actor_labels
from yp_video.actor.box_style import event_box
from yp_video.contracts.action import event_id
from yp_video.core.cache import StatCache
from yp_video.core.jsonl import read_jsonl_cached
from yp_video.core.rallies import load_rallies
from yp_video.extraction import pipeline
from yp_video.extraction.store import (
    action_annotation_path,
    action_source_paths,
    labelable,
    records_path,
)

# Rebuilt when the detector output (once there is any) or either annotation
# source changes. Values are shared across requests — read-only.
_rows_cache: StatCache = StatCache()


class SidebarEvent(TypedDict):
    id: str
    frame: int
    time: float | None
    label: str | None
    visible: bool


class VideoContext(TypedDict):
    #: The LIVE rally store — the action file carries no rally copy, so a
    #: rally re-edit reaches the boards without re-saving the action file.
    rallies: list[dict]
    #: The full action event list, score / non-visible included (nobody is
    #: identified in those), for the boards' action sidebar.
    action_events: list[SidebarEvent]


def labelable_records(stem: str, fps: float) -> list[dict]:
    """Every event a person can be identified in, as an extraction record in
    frame order: the stored ones, plus an unassociated record for each event
    detection never saw (all of them, before Player Detection ran)."""
    path = records_path(stem)
    sources = [*([path] if path.exists() else []), *action_source_paths(stem)]
    return _rows_cache.get((stem, fps), sources, lambda: _labelable_records(stem, fps))


def _labelable_records(stem: str, fps: float) -> list[dict]:
    path = records_path(stem)
    records = read_jsonl_cached(path)[1] if path.exists() else []
    rows = [*records, *pipeline.unextracted_records(records, stem)]
    return sorted(labelable(rows, stem, fps), key=lambda r: r["frame"])


def with_verdicts(stem: str, rows: list[dict]) -> list[dict]:
    """Copies of ``rows`` carrying each event's actor verdict.

    A row detection never boxed (no record, or one still unassociated) takes
    its label's box on the event frame, so a proposal or a pick on a video
    Player Detection never ran on draws like any other actor.
    """
    labels = actor_labels.load(stem)
    out = []
    for row in rows:
        label = labels.get(row["id"])
        box = row.get("box")
        if box is None and label is not None:
            drawn = event_box(stem, label, int(row["frame"]))
            box = None if drawn is None else list(drawn)
        out.append({
            **row,
            "box": box,
            "actor_review": label.verdict.value if label is not None else "unreviewed",
        })
    return out


def video_context(stem: str, fps: float) -> VideoContext:
    """What the player needs besides the rows: rallies and the action list."""
    ann = action_annotation_path(stem)
    events = read_jsonl_cached(ann)[1] if ann is not None else []
    return {
        "rallies": load_rallies(stem),
        "action_events": [
            {
                "id": event_id(event),
                "frame": event["frame"],
                "time": float(event["frame"]) / fps if fps else None,
                "label": event.get("label"),
                "visible": event.get("visible", True),
            }
            for event in events
            if event.get("frame") is not None
        ],
    }
