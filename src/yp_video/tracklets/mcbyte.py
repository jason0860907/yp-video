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
import threading
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
#: Rallies are independent trackers, so they run in this many yp-track
#: processes at once. With mask removal each holds ~3 GB of GPU memory at its
#: peak; three keep the 4090 busy without starving the detector or ReID.
WORKERS = 3
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
    low_thresh: float,
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
    groups = _balanced(spans, WORKERS)
    with tempfile.TemporaryDirectory(prefix="yp-track-") as tmp:
        tmp = Path(tmp)
        np.savez(tmp / "dets.npz", **{str(f): np.asarray(d, np.float32).reshape(-1, 5) for f, d in detections.items()})
        # torch, OpenCV and numpy each size their pools to every core; several
        # workers doing that at once thrash the CPU (measured: 3 × 620 % CPU,
        # load 40 on 28 cores, GPU at 30 %). Split the cores between them.
        threads = str(max(1, (os.cpu_count() or 1) // len(groups)))
        env = {
            **os.environ, "PYTHONUNBUFFERED": "1", TRACK_CONTRACT_VERSION_ENV: TRACK_CONTRACT_VERSION,
            "OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads, "OPENBLAS_NUM_THREADS": threads,
        }
        procs = []
        for i, group in enumerate(groups):
            (tmp / f"spans{i}.json").write_text(json.dumps([list(map(int, s)) for s in group]))
            cmd = [
                str(TRACK_PYTHON), "-m", TRACK_MCBYTE_MODULE,
                "--video", str(video), "--detections", str(tmp / "dets.npz"),
                "--spans", str(tmp / f"spans{i}.json"), "--out", str(tmp / f"tracks{i}.jsonl"),
                "--stride", str(stride), "--track-thresh", str(track_thresh),
                "--low-thresh", str(low_thresh),
                "--cmc", "orb" if cmc else "none",
            ]
            procs.append(subprocess.Popen(
                cmd, cwd=TRACK_PKG_DIR, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            ))
        done = [0] * len(procs)
        totals = [0] * len(procs)
        tails = [deque(maxlen=20) for _ in procs]
        lock = threading.Lock()

        def pump(i: int) -> None:
            assert procs[i].stdout is not None
            for raw in procs[i].stdout:
                for line in raw.rstrip("\n").split("\r"):
                    if line.startswith(TRACK_PROGRESS_PREFIX):
                        p = json.loads(line[len(TRACK_PROGRESS_PREFIX):])
                        with lock:
                            done[i], totals[i] = p["done"], p["total"]
                            if on_progress:
                                on_progress(sum(done), sum(totals), f"McByte++ frame {sum(done)}/{sum(totals)}")
                    elif line.strip():
                        tails[i].append(line.strip())

        pumps = [threading.Thread(target=pump, args=(i,), daemon=True) for i in range(len(procs))]
        for t in pumps:
            t.start()
        failed = []
        for i, proc in enumerate(procs):
            rc = proc.wait()
            pumps[i].join()
            if rc != 0 or not (tmp / f"tracks{i}.jsonl").exists():
                failed.append(f"worker {i} rc={rc}: " + " | ".join(list(tails[i])[-5:]))
        if failed:
            raise McByteError("yp-track failed: " + " || ".join(failed))
        return [
            json.loads(line)
            for i in range(len(procs))
            for line in (tmp / f"tracks{i}.jsonl").read_text().splitlines()
            if line
        ]


def _balanced(spans: Sequence[tuple[int, int, int]], workers: int) -> list[list[tuple[int, int, int]]]:
    """Spans dealt longest-first to the least-loaded worker; empty groups dropped."""
    groups: list[list[tuple[int, int, int]]] = [[] for _ in range(max(1, workers))]
    load = [0] * len(groups)
    for span in sorted(spans, key=lambda s: s[2] - s[1], reverse=True):
        i = load.index(min(load))
        groups[i].append(span)
        load[i] += span[2] - span[1]
    return [g for g in groups if g]
