"""The court paint of one tracked video (court/paint.py): the guides, and
the marks when the fit is trusted.

The tracklets know both things the paint finder needs: when the rallies run
— frames the players move through, so their median is the bare court — and
where feet stand, so where the floor ends. One implementation for the
labeling web page and for identify, which ships them in its result so the
app pre-draws them on the first court calibration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from yp_video.core.frames import read_frames
from yp_video.court import paint
from yp_video.tracklets.store import tracklet_data

#: Frames spread over the tracked rallies; their median is the bare court.
FRAMES = 12


def court_prefill(stem: str, video: Path | str) -> paint.Found | None:
    """The court for this video's tracks (which must exist); None when no
    court fits. ``video`` is a path or a URL."""
    data = tracklet_data(stem)
    width, height = data.meta["frame_size"]
    boxes = np.array([box for r in data.records for box in r["boxes"]], np.float64).reshape(-1, 4)
    scores = np.array([score for r in data.records for score in r["scores"]], np.float64)
    if not np.any(scores > paint.FEET_SCORE):
        return None
    tracked = np.unique(np.concatenate([r["frames"] for r in data.records]))
    picks = tracked[np.linspace(0, len(tracked) - 1, FRAMES).astype(int)]
    frames = [image for _, image in read_frames(video, picks) if image is not None]
    if not frames:
        return None
    return paint.find(frames, paint.floor_top(boxes / [width, height, width, height], scores))
