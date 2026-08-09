"""Server-side Done / In-Progress / Prediction counts for the label sidebar.

Counting on the server means the sidebar polls one small payload instead of
four full listings; the tally itself lives in web/worklists.py next to the
list builders it counts, so the counters and the work lists cannot disagree.
"""

from __future__ import annotations

from fastapi import APIRouter

from yp_video.web import worklists

router = APIRouter()


@router.get("/stats")
def stats() -> dict[str, dict[str, int]]:
    return worklists.label_stats()
