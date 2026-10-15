"""Sparse frame reads: a few hundred single frames of one video, decoded fast.

Identify reads one frame per touch twice over (re-cropping the actor, then
exporting the review photos). A seek decodes from the previous keyframe, about
55 ms a frame at 1080p, which made each of those passes half a minute; touches
sit a second or two apart, so grabbing through the gap is cheaper, and short
runs of them decode in parallel on their own captures (OpenCV releases the GIL
while decoding). The pixels are the ones a seek returns.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

#: A gap up to this many frames is grabbed through rather than sought across.
GRAB_GAP = 150
#: Frames per run; each run opens its own capture and starts with one seek.
RUN_LENGTH = 16
WORKERS = 4


def read_frames(video_path: Path, frames: Iterable[int]) -> Iterator[tuple[int, object | None]]:
    """``(frame, BGR image)`` for each distinct requested frame, ascending.

    The image is None where the frame would not decode. At most a few runs
    are held in memory ahead of the consumer.
    """
    wanted = sorted({int(f) for f in frames})
    runs = [wanted[i:i + RUN_LENGTH] for i in range(0, len(wanted), RUN_LENGTH)]
    with ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="frames") as pool:
        pending: deque = deque()
        for run in runs:
            pending.append(pool.submit(_read_run, video_path, run))
            if len(pending) > 2 * WORKERS:
                yield from pending.popleft().result()
        while pending:
            yield from pending.popleft().result()


def _read_run(video_path: Path, run: list[int]) -> list[tuple[int, object | None]]:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    out: list[tuple[int, object | None]] = []
    position = None  # the frame the last read returned
    try:
        for frame in run:
            gap = frame - position - 1 if position is not None else -1
            if 0 <= gap <= GRAB_GAP:
                for _ in range(gap):
                    capture.grab()
            else:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = capture.read()
            out.append((frame, image if ok else None))
            # After a failed read the decoder's position is unknown: seek next.
            position = frame if ok else None
    finally:
        capture.release()
    return out
