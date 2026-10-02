"""McByte++ association, run by the yp-track package (contracts/track.py).

ByteTrack links boxes by overlap alone, which is where a volleyball rally
breaks it: players cross at the net and ByteTrack swaps or splits them.
McByte++ adds two cues — each track's EdgeTAM mask propagated frame to frame,
consulted when overlap is ambiguous, and OSNet re-ID to rejoin a track that
was lost — at ~14 tracked frames/s on the 4090, against ByteTrack's
thousands.

Measured 2026-10-01 on three human-named videos (same-rally pairwise F1 of
tracklet ≡ player): fusion boxes 0.27 → 0.39 and RF-DETR boxes 0.30 → 0.48
against ByteTrack. Tracking every 2nd frame was faster still and no worse
(0.44 vs 0.39 on fusion boxes), hence ``STRIDE``.

This module only crosses the process boundary; which detections go in, and
what is done with the matched indices that come back, is the caller's.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from yp_video.config import TRACK_MCBYTE_MODULE, TRACK_PKG_DIR, TRACK_PYTHON
from yp_video.contracts.track import (
    TRACK_CONTRACT_VERSION,
    TRACK_CONTRACT_VERSION_ENV,
    TRACK_PROGRESS_PREFIX,
)
from yp_video.core.progress import ProgressFn

#: Tracked-frame step, relative to the detections handed in.
STRIDE = 2
#: Written into the tracks header so a reader can tell the trackers apart.
SOURCE = f"McByte++ (yp-track contract {TRACK_CONTRACT_VERSION})"


class McByteError(RuntimeError):
    """yp-track failed; the message carries the tail of its output."""


def available() -> bool:
    return TRACK_PYTHON.exists()


def track(
    video: Path,
    detections: Mapping[int, np.ndarray],
    spans: Sequence[tuple[int, int, int]],
    *,
    stride: int,
    track_thresh: float,
    cmc: bool,
    on_progress: ProgressFn | None = None,
) -> list[dict]:
    """Tracklets for every span, ``{rally_id, track_id, frames, boxes, scores, det_index}``.

    ``detections`` maps a native frame to its ``(n, 5)`` ``x0 y0 x1 y1 score``
    rows in frame pixels; ``det_index`` in the result indexes those rows.
    ``cmc`` turns on camera motion compensation — a moving camera needs it,
    a fixed sideline camera only pays for it.
    """
    if not available():
        raise McByteError(f"yp-track is not installed at {TRACK_PKG_DIR} (uv sync there)")
    with tempfile.TemporaryDirectory(prefix="yp-track-") as tmp:
        tmp = Path(tmp)
        np.savez(tmp / "dets.npz", **{str(f): np.asarray(d, np.float32).reshape(-1, 5) for f, d in detections.items()})
        (tmp / "spans.json").write_text(json.dumps([list(map(int, s)) for s in spans]))
        out = tmp / "tracks.jsonl"
        cmd = [
            str(TRACK_PYTHON), "-m", TRACK_MCBYTE_MODULE,
            "--video", str(video), "--detections", str(tmp / "dets.npz"),
            "--spans", str(tmp / "spans.json"), "--out", str(out),
            "--stride", str(stride), "--track-thresh", str(track_thresh),
            "--cmc", "orb" if cmc else "none",
        ]
        env = {**os.environ, "PYTHONUNBUFFERED": "1", TRACK_CONTRACT_VERSION_ENV: TRACK_CONTRACT_VERSION}
        proc = subprocess.Popen(
            cmd, cwd=TRACK_PKG_DIR, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        tail: deque[str] = deque(maxlen=20)
        assert proc.stdout is not None
        for raw in proc.stdout:
            for line in raw.rstrip("\n").split("\r"):
                if line.startswith(TRACK_PROGRESS_PREFIX):
                    if on_progress:
                        p = json.loads(line[len(TRACK_PROGRESS_PREFIX):])
                        on_progress(p["done"], p["total"], f"McByte++ frame {p['done']}/{p['total']}")
                elif line.strip():
                    tail.append(line.strip())
        if (rc := proc.wait()) != 0 or not out.exists():
            raise McByteError(f"yp-track failed (rc={rc}): " + " | ".join(list(tail)[-5:]))
        return [json.loads(line) for line in out.read_text().splitlines() if line]
