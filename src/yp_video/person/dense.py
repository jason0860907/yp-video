"""Every rally frame's people from the strongest RF-DETR Seg: person-head supervision.

The person head learns boxes per frame. Its labels were the tracker's boxes,
which only hold detections that joined a tracklet — the occluded and
far-side players a tracker drops are exactly the ones the head should learn.
This pass keeps every detection of the largest seg model above a low floor,
with its score, so the label threshold is chosen when the labels are written
rather than baked in here.

One ``<stem>_dense.npz`` per video in PERSON_DENSE_DIR: ``frames`` (int32,
cv2 frame indices — the tracks' convention — of every rally-span frame),
``counts`` (int32, boxes on that frame), ``boxes`` (float16, xyxy normalized
to the frame), ``scores`` (float16) and ``meta`` (JSON: model, floor, fps,
frame size, rally fingerprint). A frame with zero boxes is a real answer.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from yp_video.config import PERSON_DENSE_DIR
from yp_video.core.progress import ProgressFn
from yp_video.core.rallies import load_rallies, rally_fingerprint
from yp_video.person.seg_batch import BatchSegDetector, span_frames

DENSE_VARIANT = "RFDETRSeg2XLarge"
# Batch 4 at res 768: ~52 ms/frame and ~7.6 GB reserved on the 4090 (batch 8
# is only 12% faster for twice the VRAM, which the production worker needs).
DENSE_BATCH_SIZE = 4
# Low enough to keep the occluded players; the label writer picks the cut.
DENSE_SCORE_FLOOR = 0.1
DENSE_SUFFIX = "_dense.npz"

_detector = BatchSegDetector(DENSE_VARIANT, DENSE_BATCH_SIZE)


class DensePaused(Exception):
    """``should_pause`` asked to stop; the detector's VRAM is already freed."""


def dense_path(stem: str) -> Path:
    return PERSON_DENSE_DIR / f"{stem}{DENSE_SUFFIX}"


def release_detector() -> None:
    _detector.release()


def detect_dense(
    video_path: Path,
    *,
    should_pause: Callable[[], bool],
    on_progress: ProgressFn | None = None,
) -> dict:
    """Detect people on every frame of every rally span of one video.

    ``should_pause`` is polled between batches; when it returns True the
    detector is released and DensePaused raised — nothing is written, so the
    video simply runs again later.
    """
    import cv2

    stem = video_path.stem
    rallies = load_rallies(stem)
    if not rallies:
        raise ValueError(f"No rally spans for {stem}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not fps > 0 or width <= 0 or height <= 0:
        cap.release()
        raise ValueError(f"Invalid video geometry: {video_path}")
    spans = [(r["rally_id"], int(round(r["start"] * fps)), int(round(r["end"] * fps))) for r in rallies]
    total = sum(f1 - f0 + 1 for _, f0, f1 in spans)

    _detector.ensure()
    res = _detector.resolution
    scale = np.array([res, res, res, res], dtype=np.float32)
    frames: list[int] = []
    counts: list[int] = []
    boxes: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    try:
        with span_frames(cap, spans, stride=1, resolution=res, name=f"dense-decode-{stem}") as items:
            pending: list = []
            exhausted = False
            while not exhausted or pending:
                if should_pause():
                    _detector.release()
                    raise DensePaused(stem)
                while not exhausted and len(pending) < DENSE_BATCH_SIZE:
                    item = next(items, None)
                    if item is None:
                        exhausted = True
                        break
                    pending.append(item)
                if not pending:
                    break
                # Boxes only: the masks rfdetr's predict() post-processes are ~90% of its time.
                for (_, frame_idx, _), (xyxy, confidence) in zip(
                    pending, _detector.predict_boxes([p[2] for p in pending], DENSE_SCORE_FLOOR)
                ):
                    frames.append(frame_idx)
                    counts.append(len(xyxy))
                    boxes.append((xyxy / scale).clip(0, 1).astype(np.float16))
                    scores.append(confidence.astype(np.float16))
                if on_progress:
                    on_progress(len(frames), total, f"frame {len(frames)}/{total}")
                pending = []
    finally:
        cap.release()

    meta = {
        "video": stem,
        "model": DENSE_VARIANT,
        "score_floor": DENSE_SCORE_FLOOR,
        "fps": fps,
        "frame_size": [width, height],
        "rallies": {"count": len(spans), "fingerprint": rally_fingerprint(stem)},
        "created_at": time.time(),
    }
    out = dense_path(stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part.npz")
    np.savez_compressed(
        part,
        frames=np.asarray(frames, dtype=np.int32),
        counts=np.asarray(counts, dtype=np.int32),
        boxes=np.concatenate(boxes) if boxes else np.zeros((0, 4), np.float16),
        scores=np.concatenate(scores) if scores else np.zeros((0,), np.float16),
        meta=np.array(json.dumps(meta)),
    )
    os.replace(part, out)
    return {"frames": len(frames), "boxes": int(sum(counts)), "rallies": len(spans)}
