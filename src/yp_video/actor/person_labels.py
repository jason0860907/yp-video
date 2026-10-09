"""RF-DETR Seg 2XLarge pseudo labels plus reviewed human corrections for the person head.

Reviewed frames replace pseudo labels (including confirmed empty frames).
Saved drafts have no supervision. Human labels also work without a dense pass.

One ``<stem>_person.npz`` per video in the run's ``labels/person-boxes``:
every frame the dense pass covered (person/dense.py: every rally-span frame)
with its boxes at or above PERSON_LABEL_MIN_SCORE, normalized to the frame.
A frame the pass did not cover is absent from the file — no supervision, not
"nobody"; a covered frame with no box above the cut is "nobody".

The pass indexes frames the way cv2 counts them, and the frame cache keeps
every native frame in decode order, so the indices agree. A video whose
cache disagrees with the pass on frame rate or length is skipped and counted.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from yp_video.action.frames import inspect_action_frame_cache
from yp_video.config import ACTION_FRAMES_DIR
from yp_video.contracts.action import TASKS
from yp_video.core.jsonl import read_jsonl
from yp_video.person.annotations import apply_annotations, load
from yp_video.person.dense import dense_path

PERSON_FILE_SUFFIX = TASKS["person"].label_glob.removeprefix("*")
#: The dense pass keeps boxes from 0.1 so the cut is chosen here. 0.4 was
#: calibrated 2026-10-08 against the human-reviewed person frames.
PERSON_LABEL_MIN_SCORE = 0.4
#: A cache's frame rate (frames / duration) may differ from the pass's by
#: this much and still index the same pictures (29.97 vs 30 is rounding).
FPS_TOLERANCE = 1.0


def write_person_labels(
    items: list[tuple[Path, Path]],
    *,
    label_dir: Path,
    cache_root: Path = ACTION_FRAMES_DIR,
) -> dict:
    """``items`` are the rally source's ``(annotation, video)`` pairs — the
    videos the rally stream samples. The dense pass supplies pseudo labels;
    human review overrides them and also works on videos without one."""
    label_dir.mkdir(parents=True, exist_ok=True)
    for stale in label_dir.glob(f"*{PERSON_FILE_SUFFIX}"):
        stale.unlink()

    counts = {"videos": 0, "without_dense": 0, "misaligned": 0, "frames": 0, "boxes": 0, "reviewed_frames": 0}
    for ann_path, video_path in items:
        stem = video_path.stem
        path = dense_path(stem)
        if not path.exists() and load(stem) is None:
            counts["without_dense"] += 1
            continue
        cache = inspect_action_frame_cache(video_path, cache_root=cache_root)
        cache_frames = int(cache.get("frame_count") or 0)
        if not cache.get("ready") or cache_frames <= 0:
            raise RuntimeError(f"Missing frame cache for {stem}")
        per_frame: dict[int, list] = {}
        if path.exists():
            meta, _rows = read_jsonl(ann_path)
            duration = float(meta.get("duration") or 0)
            if duration <= 0:
                raise RuntimeError(f"Missing annotation duration for {stem}")
            fps, labels = dense_frame_boxes(path, min_score=PERSON_LABEL_MIN_SCORE)
            if abs(cache_frames / duration - fps) > FPS_TOLERANCE or max(labels, default=-1) >= cache_frames:
                counts["misaligned"] += 1
            else:
                per_frame = labels
        else:
            counts["without_dense"] += 1
        counts["reviewed_frames"] += apply_annotations(stem, cache_frames, per_frame)
        if not per_frame:
            continue
        frames = np.array(sorted(per_frame), dtype=np.int32)
        box_counts = np.array([len(per_frame[int(f)]) for f in frames], dtype=np.int32)
        boxes = np.asarray([b for f in frames for b in per_frame[int(f)]], dtype=np.float16).reshape(-1, 4)
        np.savez_compressed(
            label_dir / f"{stem}{PERSON_FILE_SUFFIX}",
            frames=frames, counts=box_counts, boxes=boxes,
        )
        counts["videos"] += 1
        counts["frames"] += int(len(frames))
        counts["boxes"] += int(len(boxes))
    return {"label_dir": str(label_dir), "min_score": PERSON_LABEL_MIN_SCORE, **counts}


def dense_frame_boxes(path: Path, *, min_score: float) -> tuple[float, dict[int, list[list[float]]]]:
    """The pass's fps and ``{frame: [normalized xyxy, ...]}`` of its boxes at
    or above ``min_score``, one entry per covered frame (empty = nobody).

    A source the pass strode (above 30 fps) gives each skipped frame the
    boxes of the detected frame before it — 17 ms apart — so the labels
    cover every native frame of the span."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        frames, box_counts = data["frames"], data["counts"]
        boxes, scores = data["boxes"].astype(np.float32), data["scores"]
    stride = int(meta.get("stride") or 1)
    ends = np.cumsum(box_counts)
    out: dict[int, list[list[float]]] = {}
    for i, frame in enumerate(frames.tolist()):
        start = ends[i] - box_counts[i]
        kept = boxes[start:ends[i]][scores[start:ends[i]] >= min_score].tolist()
        out[frame] = kept
        # Fill only inside a span: the next detected frame is one stride on.
        if i + 1 < len(frames) and frames[i + 1] == frame + stride:
            for skipped in range(frame + 1, frame + stride):
                out[skipped] = kept
    return float(meta["fps"]), out
