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
(0.44 vs 0.39 on fusion boxes); callers choose the step.

This module only crosses the process boundary; which detections go in, and
what is done with the matched indices that come back, is the caller's.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
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

#: Rallies are independent trackers, so they run in this many yp-track
#: processes at once. Sized to the 4090: with mask removal each holds ~3 GB
#: of GPU memory at its peak, and three keep the GPU busy without starving
#: the detector or ReID. A different GPU wants a different number.
WORKERS = 3
#: Written into the tracks header so a reader can tell the trackers apart.
SOURCE = f"McByte++ (yp-track contract {TRACK_CONTRACT_VERSION})"
#: How often the parent checks whether a worker has exited.
_POLL_S = 0.2


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
    min_frames: int,
    cmc: bool,
    on_progress: ProgressFn | None = None,
) -> list[dict]:
    """Tracklets for every span, ``{rally_id, track_id, frames, boxes, scores, det_index}``.

    ``detections`` maps a native frame to its ``(n, 5)`` ``x0 y0 x1 y1 score``
    rows in frame pixels; ``det_index`` in the result indexes those rows.
    Tracklets shorter than ``min_frames`` are dropped. ``cmc`` turns on
    camera motion compensation — a moving camera needs it, a fixed sideline
    camera only pays for it.

    The first worker to fail stops the others, and no worker outlives this
    call: each holds GPU memory for as long as it runs.
    """
    if not available():
        raise McByteError(f"yp-track is not installed at {TRACK_PKG_DIR} (uv sync there)")
    groups = _balanced(spans, WORKERS)
    with tempfile.TemporaryDirectory(prefix="yp-track-") as tmp:
        tmp = Path(tmp)
        np.savez(tmp / "dets.npz", **{str(f): np.asarray(d, np.float32).reshape(-1, 5) for f, d in detections.items()})
        # torch, OpenCV and BLAS each size their pools to every core; several
        # workers doing that at once thrash the CPU (measured: 3 × 620 % CPU,
        # load 40 on 28 cores, GPU at 30 %). Split the cores between them.
        threads = max(1, (os.cpu_count() or 1) // len(groups))
        env = {**os.environ, "PYTHONUNBUFFERED": "1", TRACK_CONTRACT_VERSION_ENV: TRACK_CONTRACT_VERSION}
        progress = _Progress(len(groups), on_progress)
        workers: list[_Worker] = []
        try:
            for i, group in enumerate(groups):
                (tmp / f"spans{i}.json").write_text(json.dumps([list(map(int, s)) for s in group]))
                workers.append(_Worker(i, [
                    str(TRACK_PYTHON), "-m", TRACK_MCBYTE_MODULE,
                    "--video", str(video), "--detections", str(tmp / "dets.npz"),
                    "--spans", str(tmp / f"spans{i}.json"), "--out", str(tmp / f"tracks{i}.jsonl"),
                    "--stride", str(stride), "--track-thresh", str(track_thresh),
                    "--low-thresh", str(low_thresh), "--min-frames", str(min_frames),
                    "--threads", str(threads), "--cmc", "orb" if cmc else "none",
                ], env, progress))
            while (running := [w for w in workers if w.proc.poll() is None]) and not any(
                w.proc.returncode for w in workers if w not in running
            ):
                time.sleep(_POLL_S)
        finally:
            for w in workers:
                w.stop()
        failed = [w.failure(tmp / f"tracks{w.index}.jsonl") for w in workers]
        if failed := [f for f in failed if f]:
            raise McByteError("yp-track failed: " + " || ".join(failed))
        if progress.error is not None:
            raise progress.error
        return [
            json.loads(line)
            for w in workers
            for line in (tmp / f"tracks{w.index}.jsonl").read_text().splitlines()
            if line
        ]


class _Progress:
    """Sums the workers' progress into one callback, from their reader threads.

    A callback that raises is remembered and re-raised by ``track`` once the
    workers have stopped — never allowed to kill a reader, whose worker would
    then block on a full pipe.
    """

    def __init__(self, workers: int, on_progress: ProgressFn | None):
        self.done = [0] * workers
        self.totals = [0] * workers
        self.on_progress = on_progress
        self.error: BaseException | None = None
        self._lock = threading.Lock()

    def update(self, index: int, done: int, total: int) -> None:
        with self._lock:
            self.done[index], self.totals[index] = done, total
            if self.on_progress is None or self.error is not None:
                return
            d, t = sum(self.done), sum(self.totals)
            try:
                self.on_progress(d, t, f"McByte++ frame {d}/{t}")
            except Exception as exc:  # noqa: BLE001 — re-raised by track()
                self.error = exc


class _Worker:
    """One yp-track process and the thread draining its output."""

    def __init__(self, index: int, cmd: list[str], env: dict, progress: _Progress):
        self.index = index
        self.killed = False
        self.tail: deque[str] = deque(maxlen=20)
        self.proc = subprocess.Popen(
            cmd, cwd=TRACK_PKG_DIR, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        self._reader = threading.Thread(target=self._drain, args=(progress,), daemon=True)
        self._reader.start()

    def _drain(self, progress: _Progress) -> None:
        assert self.proc.stdout is not None
        for raw in self.proc.stdout:
            for line in raw.rstrip("\n").split("\r"):
                if line.startswith(TRACK_PROGRESS_PREFIX):
                    try:
                        p = json.loads(line[len(TRACK_PROGRESS_PREFIX):])
                        progress.update(self.index, p["done"], p["total"])
                    except (ValueError, KeyError, TypeError):
                        self.tail.append(line.strip())
                elif line.strip():
                    self.tail.append(line.strip())

    def stop(self) -> None:
        """Kill the process if it is still running, then reap it and its reader."""
        if self.proc.poll() is None:
            self.proc.kill()
            self.killed = True
        self.proc.wait()
        self._reader.join()

    def failure(self, out: Path) -> str | None:
        """Why this worker failed, or None — including when it was stopped
        because another one failed (that one carries the reason)."""
        if self.killed or (self.proc.returncode == 0 and out.exists()):
            return None
        return f"worker {self.index} rc={self.proc.returncode}: " + " | ".join(list(self.tail)[-5:])


def _balanced(spans: Sequence[tuple[int, int, int]], workers: int) -> list[list[tuple[int, int, int]]]:
    """Spans dealt longest-first to the least-loaded worker; empty groups dropped."""
    groups: list[list[tuple[int, int, int]]] = [[] for _ in range(max(1, workers))]
    load = [0] * len(groups)
    for span in sorted(spans, key=lambda s: s[2] - s[1], reverse=True):
        i = load.index(min(load))
        groups[i].append(span)
        load[i] += span[2] - span[1]
    return [g for g in groups if g]
