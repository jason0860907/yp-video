"""The rally editor's save: one row schema, one atomic writer.

Shared by the annotate router (the editor's save endpoint) and
scripts/anchor_rally_edges.py, which rewrites annotation files outside the
server. Both must mint ids the same way, or the ledger stops being one.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import Field

from yp_video.core.jsonl import read_jsonl, read_jsonl_header
from yp_video.web.schemas import StrictModel


class Annotation(StrictModel):
    id: str | None = None
    #: None = a new rally; the save assigns it a fresh id (see
    #: write_annotations_atomic). A present id is kept verbatim — identity
    #: follows the row, not its position.
    rally_id: int | None = Field(default=None, ge=1)
    start: float
    end: float
    label: str
    #: Court side the rally's winner played on (camera-frame): left/right for
    #: sideline footage, near/far for broadcast. None = not annotated yet.
    winner: Literal["left", "right", "near", "far"] | None = None


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
) -> tuple[list[dict], list[dict]]:
    """Write JSONL via tmp file + atomic rename.

    Returns the rows as written and the rows that were there before, so the
    caller can audit the difference. The comparison itself belongs to the
    handler: this function's job is the file.

    Ids are assigned here and only here: rows that carry one keep it —
    identity follows the row, sorting is presentation order — and new (None)
    rows are minted ids above the high-water mark, in start order.
    """
    before = _prior_rows(output_path)
    high = max(
        _prior_max_rally_id(output_path),
        *(a.rally_id for a in annotations if a.rally_id is not None),
        0,
    )
    ordered = sorted(annotations, key=lambda ann: (ann.start, ann.end, ann.label))
    rows: list[dict] = []
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
    tmp_path = output_path.with_suffix(output_path.suffix + f".tmp.{os.getpid()}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        meta = {
            "_meta": True,
            "video": video,
            "duration": duration,
            "max_rally_id": high,
        }
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, output_path)
    return rows, before
