"""Per-rally ByteTrack tracklets over dense RF-DETR Seg detections.

The extraction pipeline is tracking-free on purpose — one frame per event.
This module adds the dense complement: within each annotated rally span,
every frame is detected and linked into tracklets (supervision's ByteTrack,
motion-only), so events whose actor boxes land on the same tracklet are the
same player and an identity labeled once propagates along the track.

The seg model gives every tracked detection an instance mask for free; the
masks persist beside the tracklets (box-crop space, packed bits — see
store.save_track_masks), rows aligned with each tracklet's frames.

One tracker per rally: between rallies players reshuffle and broadcasts cut
away, so cross-rally tracks would be fiction. Only rally spans are scanned —
they are where the events live and everything else is replays and crowd.

The dense pass is throughput-tuned (~9 ms/frame vs ~116 ms naive):
a producer thread decodes and preprocesses frames while the GPU runs
fixed-size fp16 batches, so neither side ever waits for the other.

Storage: reid/tracks/<stem>_tracks.jsonl, one record per tracklet
({rally_id, track_id, frames, boxes, scores}), plus <stem>_masks.npz with
the packed per-frame masks keyed "rally:track".
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from yp_video.core.cache import StatCache
from yp_video.core.jsonl import read_jsonl, read_jsonl_cached, write_jsonl
from yp_video.reid.detector import iou
from yp_video.reid.seg import PERSON_CLASS_ID, SEG_WEIGHTS
from yp_video.reid.store import action_annotation_path, reid_path, save_track_masks, tracks_path

# Detection floor for the dense pass — even lower than extraction's 0.1:
# ByteTrack's second association stage recovers these low-score detections
# along confident tracks, and heavily occluded players (the ones remote
# picking exists for) live down here.
TRACK_SCORE_THRESHOLD = 0.05

# Tracklets shorter than this many detections are detector flicker, not a player.
MIN_TRACK_FRAMES = 5
# An event snaps onto a tracklet when at least this fraction of the track box
# lies inside the event's display box (containment, not IoU — the display box
# is a keypoint/contact-point union, a superset of the raw detector box).
LINK_MIN_CONTAINMENT = 0.5

# The traced fp16 graph bakes the batch dimension in, so every call must be
# exactly this size — partial final batches are padded and sliced.
BATCH_SIZE = 16
# Producer→consumer buffer (in frames). Small: it only needs to bridge the
# jitter between decode and inference, not hold a rally.
_QUEUE_FRAMES = 4 * BATCH_SIZE

# Stored mask resolution (box-crop space, tall like people). Sized for the
# Pick Actor silhouettes — 48×96 upscales to a clean outline on 1080p while
# packing to 576 bytes per detection; ~100k boxes/video stays trivial after
# the npz deflate.
MASK_W, MASK_H = 48, 96


def _pack_mask(mask: np.ndarray, box) -> np.ndarray:
    """One res-space instance mask → its box crop as packed MASK_H×MASK_W
    bits; degenerate boxes pack to all-zero."""
    import cv2

    h, w = mask.shape
    x0, y0 = max(int(box[0]), 0), max(int(box[1]), 0)
    x1, y1 = min(int(round(box[2])), w), min(int(round(box[3])), h)
    if x1 <= x0 or y1 <= y0:
        return np.zeros(MASK_H * MASK_W // 8, dtype=np.uint8)
    crop = cv2.resize(mask[y0:y1, x0:x1].astype(np.uint8), (MASK_W, MASK_H), interpolation=cv2.INTER_NEAREST)
    return np.packbits(crop.astype(bool))

ProgressFn = Callable[[int, int, str], None]


class _BatchDetector:
    """fp16 batch-compiled RF-DETR Seg for the dense pass — person boxes,
    scores and instance masks in one forward (3.7 ms/frame at res 432,
    faster than the keypoint model this replaced).

    Separate from PersonDetector on purpose: optimize_for_inference() halves
    latency, and the compiled graph only accepts exactly BATCH_SIZE
    pre-resized tensors.
    """

    def __init__(self):
        self._model = None
        self.resolution: int | None = None

    def ensure(self) -> None:
        if self._model is not None:
            return
        import torch
        from rfdetr import RFDETRSegMedium

        model = RFDETRSegMedium()
        self.resolution = model.model_config.resolution
        model.optimize_for_inference(dtype=torch.float16, batch_size=BATCH_SIZE)
        self._model = model

    def predict_batch(self, tensors: list) -> list:
        """≤BATCH_SIZE preprocessed (C, res, res) tensors → sv.Detections each
        (person class only, masks included), boxes in resolution-pixel space
        (callers scale back to frame pixels)."""
        n = len(tensors)
        padded = tensors + [tensors[-1]] * (BATCH_SIZE - n)
        out = self._model.predict(padded, threshold=TRACK_SCORE_THRESHOLD, include_source_image=False)
        return [det[det.class_id == PERSON_CLASS_ID] for det in out[:n]]


_detector = _BatchDetector()


def track_video(video_path: Path, *, stride: int = 1, on_progress: ProgressFn | None = None) -> dict:
    """Detect + ByteTrack every annotated rally span of one video.

    ``stride`` detects every Nth frame (skipped frames are grabbed but not
    decoded); ByteTrack is told the effective frame rate. Returns the summary
    counts also written to the jsonl header. Synchronous and GPU-bound —
    callers run it in an executor.
    """
    import cv2
    import supervision as sv
    import torch

    stem = video_path.stem
    ann_path = action_annotation_path(stem)
    if ann_path is None:
        raise ValueError(f"No action annotations for {stem}")
    ann_meta, _rows = read_jsonl(ann_path)
    rallies = ann_meta.get("rallies") or []
    if not rallies:
        raise ValueError(f"No rally spans annotated for {stem} — tracking scans rallies only")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or float(ann_meta.get("fps") or 30.0)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    spans = [
        (r["rally_id"], int(round(r["start"] * fps)), int(round(r["end"] * fps)))
        for r in rallies
    ]
    total = sum((f1 - f0) // stride + 1 for _, f0, f1 in spans)

    if on_progress:
        # ensure() below loads + fp16-compiles the model on first use.
        on_progress(0, total, "loading detector weights...")
    _detector.ensure()
    res = _detector.resolution
    box_scale = np.array([frame_w / res, frame_h / res, frame_w / res, frame_h / res])

    # Producer: decode + resize + tensorize on a thread so the GPU never
    # waits on ffmpeg or cv2. INTER_AREA tracks torchvision's antialiased
    # downscale (matched-detection IoU 0.99 vs the in-model resize path).
    frame_q: queue.Queue = queue.Queue(maxsize=_QUEUE_FRAMES)
    stop = threading.Event()
    producer_error: list[BaseException] = []

    def _put(item) -> bool:
        while not stop.is_set():
            try:
                frame_q.put(item, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    def produce() -> None:
        try:
            for rally_id, f0, f1 in spans:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
                for frame_idx in range(f0, f1 + 1):
                    if stop.is_set() or not cap.grab():
                        break
                    if (frame_idx - f0) % stride:
                        continue
                    ok, frame = cap.retrieve()
                    if not ok:
                        break
                    rgb = cv2.resize(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (res, res), interpolation=cv2.INTER_AREA
                    )
                    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255)
                    if not _put((rally_id, frame_idx, tensor)):
                        return
                if stop.is_set():
                    return
        except BaseException as exc:  # noqa: BLE001 — surfaced to the consumer
            producer_error.append(exc)
        finally:
            _put(None)

    producer = threading.Thread(target=produce, name=f"track-decode-{stem}", daemon=True)
    producer.start()

    records: list[dict] = []
    masks_store: dict[str, np.ndarray] = {}
    detected = 0
    current_rally: int | None = None
    tracker = None
    tracks: dict[int, dict] = {}

    def flush_rally() -> None:
        for tid in sorted(tracks):
            t = tracks[tid]
            masks = t.pop("masks")
            if len(t["frames"]) >= MIN_TRACK_FRAMES:
                records.append({"rally_id": current_rally, "track_id": tid, **t})
                masks_store[f"{current_rally}:{tid}"] = np.stack(masks)
        tracks.clear()

    try:
        pending: list[tuple[int, int, object]] = []
        exhausted = False
        while not exhausted or pending:
            while not exhausted and len(pending) < BATCH_SIZE:
                item = frame_q.get()
                if item is None:
                    exhausted = True
                    break
                pending.append(item)
            if not pending:
                break
            detections = _detector.predict_batch([p[2] for p in pending])
            for (rally_id, frame_idx, _), det in zip(pending, detections):
                if rally_id != current_rally:
                    # Rally boundary: batches may span it (detection is
                    # stateless) but the tracker must not.
                    flush_rally()
                    current_rally = rally_id
                    tracker = sv.ByteTrack(
                        frame_rate=max(1, round(fps / stride)),
                        # Two consecutive hits before a track exists — kills the
                        # one-frame ghosts a 0.1 detection floor produces in a crowd.
                        minimum_consecutive_frames=2,
                    )
                det.xyxy = det.xyxy * box_scale
                det = tracker.update_with_detections(det)  # masks ride along, aligned
                for i, (xyxy, score, tid) in enumerate(zip(det.xyxy, det.confidence, det.tracker_id)):
                    t = tracks.setdefault(int(tid), {"frames": [], "boxes": [], "scores": [], "masks": []})
                    t["frames"].append(frame_idx)
                    # Whole pixels: every consumer (overlay, containment) is
                    # pixel-grained, and the file holds ~100k boxes.
                    t["boxes"].append([round(float(v)) for v in xyxy])
                    t["scores"].append(round(float(score), 2))
                    # Crop the res-space mask by the res-space box.
                    t["masks"].append(_pack_mask(det.mask[i], xyxy / box_scale))
                detected += 1
                if on_progress:
                    on_progress(detected, total, f"frame {detected}/{total} · rally {rally_id}")
            pending = []
        flush_rally()
        if producer_error:
            raise producer_error[0]
    finally:
        stop.set()
        producer.join(timeout=5)
        cap.release()

    counts = {
        "rallies": len(spans),
        "frames": detected,
        "tracklets": len(records),
    }
    header = {
        "video": stem,
        "source": {"detector": f"{SEG_WEIGHTS} (fp16 batch)", "tracker": f"supervision.ByteTrack {sv.__version__}"},
        "fps": fps,
        "frame_size": [frame_w, frame_h],
        "stride": stride,
        "mask_res": [MASK_H, MASK_W],
        "created_at": time.time(),
        "counts": counts,
    }
    # Masks land first: the jsonl's mtime is what downstream caches key on,
    # so a reader never sees new tracks with the old masks.
    save_track_masks(stem, (MASK_H, MASK_W), masks_store)
    write_jsonl(tracks_path(stem), header, records)
    return counts


def _containment(track_box: list[float], display_box: list[float]) -> float:
    """Fraction of the track box's area inside the display box."""
    ix0, iy0 = max(track_box[0], display_box[0]), max(track_box[1], display_box[1])
    ix1, iy1 = min(track_box[2], display_box[2]), min(track_box[3], display_box[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area = max(1.0, (track_box[2] - track_box[0]) * (track_box[3] - track_box[1]))
    return inter / area


# link_events results, keyed by stem on both source files. Tiny values
# (one small dict per video), so unbounded is fine.
_links_cache: StatCache = StatCache()


def link_events(stem: str) -> dict[str, dict]:
    """event_id → {rally_id, track_id} for events whose actor box lands on a tracklet.

    Events without a box (miss / "no actor") never link. With a stride > 1
    the event's exact frame may be undetected — the nearest detected frame
    within the stride wins.
    """
    return _links_cache.get(stem, [tracks_path(stem), reid_path(stem)], lambda: _link_events(stem))


def _link_events(stem: str) -> dict[str, dict]:
    tmeta, tracklets = read_jsonl_cached(tracks_path(stem))  # read-only
    _rmeta, records = read_jsonl_cached(reid_path(stem))  # read-only
    stride = int(tmeta.get("stride") or 1)

    by_frame: dict[int, list[tuple[int, int, list[float]]]] = {}
    for t in tracklets:
        for frame, box in zip(t["frames"], t["boxes"]):
            by_frame.setdefault(frame, []).append((t["rally_id"], t["track_id"], box))

    links: dict[str, dict] = {}
    for r in records:
        box = r.get("box")
        if not box:
            continue
        # A cross-frame pick's boxes live on crop_frame, not the event frame
        # (the actor wasn't trackable there) — look the tracklet up THERE.
        # That resolves to exactly the track the user clicked, re-derived
        # geometrically so a re-run of tracking can never leave it stale.
        at = r.get("crop_frame") or r["frame"]
        candidates = next(
            (c for off in sorted(range(-stride + 1, stride), key=abs) if (c := by_frame.get(at + off))),
            None,
        )
        if not candidates:
            continue
        # Rank by IoU against the RAW actor box: the padded display box can
        # fully contain two overlapping players' track boxes, and containment
        # against it picks whoever is bigger — the tight box discriminates.
        # The link GATE keeps the display-box containment semantics.
        anchor = r.get("actor_box") or box
        rally_id, track_id, tbox = max(candidates, key=lambda c: iou(c[2], anchor))
        if _containment(tbox, box) >= LINK_MIN_CONTAINMENT:
            links[r["id"]] = {"rally_id": rally_id, "track_id": track_id}
    return links
