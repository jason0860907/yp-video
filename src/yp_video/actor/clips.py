"""Per-player action clips: what a person-centric action classifier trains on.

The whole-frame spotting model sees a player as 40–80 px of a 224-px frame,
which is why block (arms up at the net, no approach) barely trains. This
exporter cuts the SAME window the actor head reasons over — one tracked
player's own box at each of ``ACTOR_WINDOW_OFFSETS`` — but from the cut at
native resolution, so the crops keep the arms.

One sample = one (event, tracklet) pair: nine square crops centred on the
player's box, packed row-major into a 3×3 mosaic JPEG, plus a row in the
video's ``_clips.jsonl`` index carrying the label. Positives are the
tracklet a human named as the actor (``candidates.build`` target kind
``track``), labelled with the event's action; negatives are other players
on the same event frame, labelled ``none`` — a fixed number per event,
drawn deterministically so a re-export reproduces the set.

Reads the cut from a local path only: this package may not reach into the
web layer, so fetching the bytes (``web.r2_client.materialized_cut``) is
the driver's job (``web/action_clips.py``).
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from yp_video.actor import candidates
from yp_video.contracts.action import ACTOR_WINDOW_OFFSETS, event_id
from yp_video.core.jsonl import read_jsonl_cached, write_jsonl
from yp_video.core.progress import ProgressFn
from yp_video.extraction.store import SKIP_LABELS, action_annotation_path

#: Side of every crop in pixels. Sideline players stand 100–250 px tall in
#: 1080p, so this keeps them at (or near) native resolution.
CROP_SIZE = 192
#: The square's side relative to the box's longer side — breathing room for
#: raised arms and the ball.
CROP_SCALE = 1.3
#: Other players on the event frame exported as ``none`` per positive.
NEGATIVES_PER_EVENT = 2
NONE_LABEL = "none"
#: The nine offsets as a 3×3 mosaic, row-major.
GRID = 3
INDEX_SUFFIX = "_clips.jsonl"

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Sample:
    event_id: str
    frame: int
    track: str
    #: An action, ``none`` for a negative, None for an unlabelled candidate.
    label: str | None
    #: ``positive`` (the named actor), ``negative`` (another player) or
    #: ``candidate`` (inference: every player, no answer yet).
    role: str
    contact: list[float] | None
    contact_visible: bool
    #: Normalized box per offset; None where the tracklet had no box there.
    boxes: tuple[Box | None, ...]

    @property
    def id(self) -> str:
        return f"{self.event_id}_{self.track.replace(':', '-')}"

    @property
    def file(self) -> str:
        return f"{self.id}.jpg"


def index_path(clips_dir: Path, stem: str) -> Path:
    return clips_dir / f"{stem}{INDEX_SUFFIX}"


def load_events(stem: str) -> list[dict]:
    """The video's action events that name a person — ``score`` marks where
    the ball landed, not who acted."""
    path = action_annotation_path(stem)
    if path is None:
        return []
    _meta, rows = read_jsonl_cached(path)
    return [
        r for r in rows
        if r.get("frame") is not None and r.get("label") not in SKIP_LABELS
    ]


# ------------------------------------------------------------------ planning


def plan_samples(stem: str, events: Sequence[dict]) -> tuple[list[Sample], dict]:
    """Every training sample this video yields, and ``candidates.build``'s
    tally. Only events whose actor resolved to a tracklet produce samples:
    an occluded verdict has no player to crop, and an untracked one has no
    box."""
    rows, tally = candidates.build(stem, events)
    labels = {event_id(e): str(e["label"]) for e in events}
    samples: list[Sample] = []
    for row in rows:
        if row.get("target_kind") != "track":
            continue
        target = int(row["target"])
        cands = row["candidates"]
        samples.append(_sample(row, cands[target], labels[row["id"]], "positive"))
        others = [c for i, c in enumerate(cands) if i != target]
        samples += [
            _sample(row, other, NONE_LABEL, "negative")
            for other in pick_negatives(row["id"], others)
        ]
    return samples, tally


def plan_candidates(stem: str, events: Sequence[dict]) -> list[Sample]:
    """Every candidate of every event, unlabelled — what inference scores.
    Built by the same code training's positives came from."""
    return [
        _sample(row, cand, None, "candidate")
        for row in candidates.candidates_only(stem, events)
        for cand in row["candidates"]
    ]


def _sample(row: dict, cand: dict, label: str | None, role: str) -> Sample:
    return Sample(
        event_id=row["id"], frame=int(row["frame"]), track=cand["track"],
        label=label, role=role, contact=row.get("contact"),
        contact_visible=bool(row.get("contact_visible", True)),
        boxes=_boxes(cand["boxes"]),
    )


def pick_negatives(seed: str, others: Sequence[dict]) -> list[dict]:
    """``NEGATIVES_PER_EVENT`` of the other candidates, the same ones every
    time for the same event id."""
    rng = random.Random(seed)
    ordered = sorted(others, key=lambda c: str(c["track"]))
    return rng.sample(ordered, min(NEGATIVES_PER_EVENT, len(ordered)))


def _boxes(raw: Iterable[Sequence[float] | None]) -> tuple[Box | None, ...]:
    return tuple(None if b is None else (float(b[0]), float(b[1]), float(b[2]), float(b[3])) for b in raw)


# ------------------------------------------------------------------ geometry


def fill_boxes(boxes: Sequence[Box | None]) -> tuple[list[Box], list[bool]]:
    """A box for every offset, and which offsets really had one.

    A tracklet can lose its box for a few frames inside the window; the crop
    then follows the nearest offset that has one, so the clip stays on the
    player rather than going black. The mask tells the trainer which crops
    are borrowed. At least one box must be present — candidates are
    members of the event frame by construction.
    """
    present = [b is not None for b in boxes]
    have = [i for i, p in enumerate(present) if p]
    if not have:
        raise ValueError("a candidate has no box at any offset")
    filled = [boxes[min(have, key=lambda j: abs(j - i))] for i in range(len(boxes))]
    return [b for b in filled if b is not None], present


def crop_square(box: Box, width: int, height: int) -> tuple[int, int, int]:
    """``(x, y, side)`` in pixels: the square centred on the box, sides
    ``CROP_SCALE`` × the box's longer side. May extend past the frame."""
    x0, y0, x1, y1 = box[0] * width, box[1] * height, box[2] * width, box[3] * height
    side = max(x1 - x0, y1 - y0) * CROP_SCALE
    return (
        int(round((x0 + x1) / 2 - side / 2)),
        int(round((y0 + y1) / 2 - side / 2)),
        max(1, int(round(side))),
    )


def cut_square(frame: np.ndarray, x: int, y: int, side: int) -> np.ndarray:
    """The square as a ``CROP_SIZE`` image; pixels outside the frame are black."""
    h, w = frame.shape[:2]
    out = np.zeros((side, side, 3), dtype=np.uint8)
    sx0, sy0, sx1, sy1 = max(x, 0), max(y, 0), min(x + side, w), min(y + side, h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y:sy1 - y, sx0 - x:sx1 - x] = frame[sy0:sy1, sx0:sx1]
    interpolation = cv2.INTER_AREA if side > CROP_SIZE else cv2.INTER_LINEAR
    return cv2.resize(out, (CROP_SIZE, CROP_SIZE), interpolation=interpolation)


def pack(crops: Sequence[np.ndarray]) -> np.ndarray:
    """Nine ``CROP_SIZE`` crops → one ``GRID×GRID`` mosaic, row-major."""
    if len(crops) != GRID * GRID:
        raise ValueError(f"expected {GRID * GRID} crops, got {len(crops)}")
    rows = [np.concatenate(crops[r * GRID:(r + 1) * GRID], axis=1) for r in range(GRID)]
    return np.concatenate(rows, axis=0)


def unpack(mosaic: np.ndarray) -> list[np.ndarray]:
    """The inverse of ``pack``."""
    side = mosaic.shape[0] // GRID
    return [
        mosaic[r * side:(r + 1) * side, c * side:(c + 1) * side]
        for r in range(GRID) for c in range(GRID)
    ]


# ------------------------------------------------------------------- export


def _read_frames(video_path: Path, wanted: Sequence[int]) -> Iterator[tuple[int, np.ndarray | None]]:
    """Yield ``(frame, image)`` for every wanted frame in ascending order.

    Same reader tracking uses (cv2 seek + sequential grab), so a frame index
    here is the frame the tracklet's box was drawn on. Small gaps are read
    through rather than seeked — a seek decodes from the previous keyframe
    anyway. A frame the decoder cannot produce yields None.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    try:
        position = -1
        for frame in wanted:
            if frame < 0:
                yield frame, None
                continue
            if frame - position > 64 or frame <= position:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                position = frame - 1
            image = None
            while position < frame:
                if not cap.grab():
                    position = frame
                    break
                position += 1
                if position == frame:
                    ok, bgr = cap.retrieve()
                    image = bgr if ok else None
            yield frame, image
    finally:
        cap.release()


def export_video(
    stem: str,
    video_path: Path,
    clips_dir: Path,
    samples: Sequence[Sample],
    *,
    counts: dict | None = None,
    on_progress: ProgressFn | None = None,
) -> dict:
    """Write the samples of one video: mosaics under ``clips_dir/<stem>/``
    and the index at ``clips_dir/<stem>_clips.jsonl``. ``counts`` is
    whatever the planner tallied; the export's own counts join it and the
    total is returned as written into the index header."""

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Which (sample, offset) needs which frame, and the square to cut there.
    squares: list[list[tuple[int, int, int]]] = []
    present: list[list[bool]] = []
    needs: dict[int, list[tuple[int, int]]] = {}
    for si, sample in enumerate(samples):
        filled, mask = fill_boxes(sample.boxes)
        squares.append([crop_square(b, width, height) for b in filled])
        present.append(mask)
        for oi, offset in enumerate(ACTOR_WINDOW_OFFSETS):
            needs.setdefault(sample.frame + offset, []).append((si, oi))

    out_dir = clips_dir / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    pending: dict[int, list[np.ndarray | None]] = {}
    written = 0
    missing_frames = 0
    wanted = sorted(needs)
    for done, (frame, image) in enumerate(_read_frames(video_path, wanted), start=1):
        if image is None:
            missing_frames += 1
        for si, oi in needs[frame]:
            crops = pending.setdefault(si, [None] * len(ACTOR_WINDOW_OFFSETS))
            x, y, side = squares[si][oi]
            crops[oi] = (
                np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8) if image is None
                else cut_square(image, x, y, side)
            )
            if all(c is not None for c in crops):
                cv2.imwrite(
                    str(out_dir / samples[si].file), pack(crops),
                    [cv2.IMWRITE_JPEG_QUALITY, 92],
                )
                del pending[si]
                written += 1
        if on_progress and (done % 200 == 0 or done == len(wanted)):
            on_progress(done, len(wanted), f"{written} clips written")

    counts = {
        **(counts or {}),
        "samples": len(samples),
        "positives": sum(s.role == "positive" for s in samples),
        "negatives": sum(s.role == "negative" for s in samples),
        "candidates": sum(s.role == "candidate" for s in samples),
        "missing_frames": missing_frames,
    }
    write_jsonl(
        index_path(clips_dir, stem),
        {
            "video": stem,
            "fps": fps,
            "frame_size": [width, height],
            "offsets": list(ACTOR_WINDOW_OFFSETS),
            "crop_size": CROP_SIZE,
            "crop_scale": CROP_SCALE,
            "layout": f"{GRID}x{GRID}",
            "counts": counts,
        },
        (
            {
                "id": s.id,
                "event_id": s.event_id,
                "frame": s.frame,
                "track": s.track,
                "label": s.label,
                "role": s.role,
                "contact": s.contact,
                "contact_visible": s.contact_visible,
                "boxes": [None if b is None else list(b) for b in s.boxes],
                "present": present[i],
                # Pixel (x, y, side) of each crop in the source frame: what
                # maps a point between crop and frame coordinates.
                "squares": [list(sq) for sq in squares[i]],
                "file": f"{stem}/{s.file}",
            }
            for i, s in enumerate(samples)
        ),
    )
    return counts
