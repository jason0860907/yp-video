"""The tracker's boxes, written as the person head's training labels.

One ``<stem>_person.npz`` per video in the run's ``labels/person-boxes``:
every frame inside a rally span (where tracking ran) with the boxes of
every tracklet on it, normalized against the tracks' frame size. A frame
outside every span is absent from the file — no supervision, not "nobody".

The frame index must be the frame cache's. Tracks index frames the way
cv2 counts them; the cache is ffmpeg's sequential decode. For most videos
those agree, but a 60 fps or 23.976 fps source decodes into a cache at a
different rate (28 of 340 tracked videos) and its tracks would land on the
wrong pictures — those videos are skipped and counted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from yp_video.action.frames import inspect_action_frame_cache
from yp_video.config import ACTION_FRAMES_DIR
from yp_video.contracts.action import TASKS
from yp_video.core.jsonl import read_jsonl
from yp_video.tracklets.store import load_tracklets, tracks_path

PERSON_FILE_SUFFIX = TASKS["person"].label_glob.removeprefix("*")
#: A tracks file's fps may differ from the cache's by this much and still
#: index the same pictures (29.97 vs 30 is rounding, 59.94 vs 30 is not).
FPS_TOLERANCE = 1.0


def write_person_labels(
    items: list[tuple[Path, Path]],
    *,
    label_dir: Path,
    cache_root: Path = ACTION_FRAMES_DIR,
) -> dict:
    """``items`` are the rally source's ``(annotation, video)`` pairs — the
    videos the rally stream samples. Videos without tracks contribute
    nothing (and train nothing); misaligned ones are skipped."""
    label_dir.mkdir(parents=True, exist_ok=True)
    for stale in label_dir.glob(f"*{PERSON_FILE_SUFFIX}"):
        stale.unlink()

    counts = {"videos": 0, "without_tracks": 0, "misaligned": 0, "frames": 0, "boxes": 0}
    for ann_path, video_path in items:
        stem = video_path.stem
        path = tracks_path(stem)
        if not path.exists():
            counts["without_tracks"] += 1
            continue
        cache = inspect_action_frame_cache(video_path, cache_root=cache_root)
        cache_frames = int(cache.get("frame_count") or 0)
        meta, _rows = read_jsonl(ann_path)
        duration = float(meta.get("duration") or 0)
        data = load_tracklets(path)
        if not cache.get("ready") or cache_frames <= 0 or duration <= 0:
            raise RuntimeError(f"Missing frame cache or duration for {stem}")
        cache_fps = cache_frames / duration
        tracks_fps = float(data.meta.get("fps") or 0)
        last_frame = max((max(t["frames"]) for t in data.records if t["frames"]), default=-1)
        if abs(cache_fps - tracks_fps) > FPS_TOLERANCE or last_frame >= cache_frames:
            counts["misaligned"] += 1
            continue

        frames, box_counts, boxes = _frame_boxes(data, ann_path_rows=_rows, fps=cache_fps, num_frames=cache_frames)
        np.savez_compressed(
            label_dir / f"{stem}{PERSON_FILE_SUFFIX}",
            frames=frames, counts=box_counts, boxes=boxes,
        )
        counts["videos"] += 1
        counts["frames"] += int(len(frames))
        counts["boxes"] += int(len(boxes))
    return {"label_dir": str(label_dir), **counts}


def _frame_boxes(data, *, ann_path_rows, fps: float, num_frames: int):
    """Supervised frames (every frame of every rally span) and their boxes."""
    width, height = (float(v) for v in data.meta["frame_size"])
    supervised: set[int] = set()
    for row in ann_path_rows:
        try:
            start, end = float(row["start"]), float(row["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue
        first = max(0, min(int(round(start * fps)), num_frames - 1))
        last = max(first, min(int(round(end * fps)), num_frames - 1))
        supervised.update(range(first, last + 1))

    per_frame: dict[int, list[list[float]]] = {f: [] for f in supervised}
    scale = np.array([width, height, width, height], dtype=np.float32)
    for tracklet in data.records:
        for frame, box in zip(tracklet["frames"], tracklet["boxes"]):
            if frame in per_frame:
                per_frame[frame].append((np.asarray(box, dtype=np.float32) / scale).clip(0, 1))
    frames = np.array(sorted(per_frame), dtype=np.int32)
    box_counts = np.array([len(per_frame[f]) for f in frames], dtype=np.int32)
    boxes = (
        np.stack([b for f in frames for b in per_frame[f]]).astype(np.float16)
        if box_counts.sum() else np.zeros((0, 4), np.float16)
    )
    return frames, box_counts, boxes
