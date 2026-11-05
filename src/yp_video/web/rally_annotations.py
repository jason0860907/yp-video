"""The rally editor's save: one row schema, one atomic writer.

Used by the annotate router to mint stable ids for editor saves. Offline
boundary adjustment preserves existing ids and fields through core.jsonl.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

from pydantic import Field

from yp_video.core.jsonl import read_jsonl, read_jsonl_header, write_jsonl
from yp_video.web.annotation_lock import annotation_write_lock
from yp_video.web.schemas import StrictModel

#: Court side the rally's winner played on (camera-frame): left/right for
#: broadcast footage, near/far for sideline.
WinnerSide = Literal["left", "right", "near", "far"]

#: The rally stores' tags (core/rallies.RALLY_SOURCES), as the editor's
#: Source select and loaded-source badge speak them.
RallySourceTag = Literal["annotation", "spot-pre-annotation"]


class RallyRow(TypedDict):
    """One stored rally, as written (editor save) or predicted (SPOT pass)."""

    rally_id: int
    start: float
    end: float
    label: str
    #: Absent (or None) = not annotated yet.
    winner: NotRequired[WinnerSide | None]
    #: SPOT pre-annotations only: the segment's mean rally score, and the
    #: winner head's confidence.
    score: NotRequired[float]
    winner_score: NotRequired[float]


class RallyAnnotationData(TypedDict):
    """One rally result file: its header, its rows, and the store it came from."""

    #: The cut's filename (``<stem>.mp4``), as resolve_cut takes it.
    video: str
    duration: float
    #: The id high-water mark (see _prior_max_rally_id).
    max_rally_id: NotRequired[int]
    source: RallySourceTag
    #: Presentation order (start, end, label); ids come from the file.
    results: list[RallyRow]


class RallySaveResult(TypedDict):
    #: The file written.
    saved: str
    count: int
    #: The rows as written, ids assigned — the editor adopts these so a new
    #: rally does not get a fresh id minted on every autosave.
    annotations: list[RallyRow]


class Annotation(StrictModel):
    id: str | None = None
    #: None = a new rally; the save assigns it a fresh id (see
    #: write_annotations_atomic). A present id is kept verbatim — identity
    #: follows the row, not its position.
    rally_id: int | None = Field(default=None, ge=1)
    start: float
    end: float
    label: str
    #: None = not annotated yet.
    winner: WinnerSide | None = None


def _prior_max_rally_id(output_path: Path) -> int:
    """The id high-water mark of the file being replaced, or 0.

    Persisted in the header so a deleted id is never reused: minting from
    max(present) would hand a re-added rally the id of a deleted one, and
    every stored tracklet key "<id>:<track>" would silently re-attach.
    """
    if not output_path.exists():
        return 0
    try:
        raw = read_jsonl_header(output_path).get("max_rally_id")
        return raw if isinstance(raw, int) and raw > 0 else 0
    except (OSError, ValueError):
        return 0


def _prior_rows(output_path: Path) -> list[dict]:
    """The rows currently on disk, or [] when there is no file yet."""
    if not output_path.exists():
        return []
    try:
        return read_jsonl(output_path)[1]
    except (OSError, ValueError, json.JSONDecodeError):
        # An unreadable prior file is not a reason to refuse the save; it just
        # means every row counts as new for the audit summary.
        return []


def write_annotations_atomic(
    output_path: Path, video: str, duration: float, annotations: list[Annotation]
) -> tuple[list[RallyRow], list[dict]]:
    """Write JSONL via tmp file + atomic rename.

    Returns the rows as written and the rows that were there before, so the
    caller can audit the difference. The comparison itself belongs to the
    handler: this function's job is the file.

    Ids are assigned here and only here: rows that carry one keep it —
    identity follows the row, sorting is presentation order — and new (None)
    rows are minted ids above the high-water mark, in start order.
    """
    with annotation_write_lock:
        before = _prior_rows(output_path)
        high = max(
            _prior_max_rally_id(output_path),
            *(a.rally_id for a in annotations if a.rally_id is not None),
            0,
        )
        ordered = sorted(annotations, key=lambda ann: (ann.start, ann.end, ann.label))
        rows: list[RallyRow] = []
        for a in ordered:
            assigned = a.rally_id
            if assigned is None:
                high += 1
                assigned = high
            row = {
                "start": a.start,
                "end": a.end,
                "label": a.label,
                "rally_id": assigned,
            }
            if a.winner is not None:
                row["winner"] = a.winner
            rows.append(row)
        write_jsonl(
            output_path,
            {"video": video, "duration": duration, "max_rally_id": high},
            rows,
        )
        return rows, before
