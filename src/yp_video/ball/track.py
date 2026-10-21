"""The ball on one video's flights: detector boxes between its touches.

Between two touches the ball flies free, and that is where a 3D trajectory
can be fitted (court/trajectory.py). So only those frames are detected: the
spans between consecutive touches (score included — it is where the ball
came down) no more than court.positions.MAX_FLIGHT_S apart, and the
MAX_FLIGHT_S after a touch nothing follows that closely — a ball shanked
out, or out of the shot, whose flight is followed from its touch alone. Picking the
ball among a frame's boxes is left to the fit: physics separates the ball
from a head or a lamp far better than a score does.

``<stem>_ball.json`` in BALL_TRACKS_DIR::

    {"fps": 29.97, "frame_size": [1920, 1080], "weights": "<checkpoint>",
     "frames": {"<frame>": [[x, y, score], ...]}}

``x, y`` the box centre normalized to the frame, best score first.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from yp_video.ball import detector
from yp_video.config import BALL_TRACKS_DIR
from yp_video.core.jsonl import atomic_write
from yp_video.core.progress import ProgressFn
from yp_video.court.positions import MAX_FLIGHT_S

#: Boxes kept per frame. The true ball is the detector's top box on 95% of
#: held-out contact frames and 99% of flight frames; three leave the fit room.
MAX_PER_FRAME = 3

#: A video's ball track, as written to disk and carried in results.
BallTrack = dict


def track_path(stem: str) -> Path:
    return BALL_TRACKS_DIR / f"{stem}_ball.json"


def flight_spans(frames: Iterable[int], fps: float) -> list[tuple[int, int]]:
    """[first, last] frame spans covering every touch's flight: up to the
    next touch, or MAX_FLIGHT_S on when none comes that soon; overlapping
    flights merged into one span."""
    ordered = sorted(set(frames))
    longest = int(MAX_FLIGHT_S * fps)
    spans: list[list[int]] = []
    for a, b in zip(ordered, [*ordered[1:], None]):
        end = b if b is not None and b - a <= longest else a + longest
        if spans and a <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], end)
        else:
            spans.append([a, end])
    return [(a, b) for a, b in spans]


def track_video(video_path: Path, touches: Iterable[int], *, on_progress: ProgressFn | None = None) -> BallTrack:
    """Detect the ball on the flights between the ``touches`` (frames), write
    and return the track."""
    spans = flight_spans(touches, detector.video_fps(video_path))
    try:
        found = detector.detect(video_path, spans, on_progress=on_progress)
    finally:
        detector.release()
    frames = {}
    for frame, boxes in zip(found.frames, found.boxes):
        if not len(boxes):
            continue
        top = boxes[(-boxes[:, 4]).argsort()[:MAX_PER_FRAME]]
        frames[str(frame)] = [
            [round(float((b[0] + b[2]) / 2), 4), round(float((b[1] + b[3]) / 2), 4), round(float(b[4]), 3)]
            for b in top
        ]
    track = {
        "fps": found.fps,
        "frame_size": list(found.frame_size),
        # The run and file, e.g. 20261014_medium1024/checkpoint_best_ema.pth.
        "weights": f"{detector.WEIGHTS.parent.name}/{detector.WEIGHTS.name}",
        "frames": frames,
    }
    path = track_path(video_path.stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_write(path) as f:
        json.dump(track, f)
    return track


def load(stem: str) -> BallTrack | None:
    path = track_path(stem)
    return json.loads(path.read_text()) if path.exists() else None
