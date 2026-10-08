"""Per-rally tracklets over dense RF-DETR Seg detections.

The extraction pipeline is tracking-free on purpose — one frame per event.
This module adds the dense complement: within each annotated rally span,
every frame is detected and linked into tracklets — by supervision's
ByteTrack (motion-only, inline) or McByte++ (masks + re-ID, run after the
pass; see tracklets/mcbyte.py) — so events whose actor boxes land on the
same tracklet are the same player.

The seg model gives every tracked detection an instance mask for free; the
masks persist beside the tracklets (box-crop space, packed bits — see
store.save_track_masks), rows aligned with each tracklet's frames.

One tracker per rally: between rallies players reshuffle and broadcasts cut
away, so cross-rally tracks would be fiction. Only rally spans are scanned —
they are where the events live and everything else is replays and crowd.

Rally spans come straight from the rally annotation (core/rallies.py), NOT
from the action file's copy of them: tracking needs to know where the rallies
are, not what happened inside them, and reading the action file made this
stage wait for one it does not depend on.

The dense pass overlaps decode and inference: a producer thread decodes and
preprocesses frames (~9 ms/frame) while the GPU runs fixed-size fp16 batches.
Measured 2026-08 on the 4090: the seg model's forward is ~14.5 ms/frame, so
the pass is GPU-bound — a faster decoder would not speed it up; only a faster
or smaller detector would.

Storage lives in tracklets/store.py. Resolving a box back to the tracklet it
belongs to lives in tracklets/geometry.py; joining tracklets to extracted
events needs both files and therefore happens a layer up, in
extraction/links.py.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from yp_video.core.jsonl import write_jsonl
from yp_video.core.progress import ProgressFn
from yp_video.core.rallies import load_rallies, rally_fingerprint
from yp_video.person.detector import DETECTOR_NAME
from yp_video.person.seg import SEG_WEIGHTS
from yp_video.person.seg_batch import BatchSegDetector, span_frames
from yp_video.tracklets import mcbyte
from yp_video.tracklets.store import (
    Tracker,
    save_span_detections,
    save_track_masks,
    tracks_path,
)

# Detection floor for the dense pass — even lower than extraction's 0.1:
# ByteTrack's second association stage recovers these low-score detections
# along confident tracks, and heavily occluded players (the ones remote
# picking exists for) live down here. McByte++'s second association takes
# the same floor (upstream: a fixed 0.1): it only lets these extend tracks
# that already exist.
TRACK_SCORE_THRESHOLD = 0.05

# Tracklets shorter than this many detections are detector flicker, not a player.
MIN_TRACK_FRAMES = 5

# McByte++ over RF-DETR boxes extends tracks from detections above this in its
# first association and starts new ones from this + 0.1. Its published 0.6
# left players RF-DETR scores below 0.7 untracked; 0.4 (new tracks from 0.5)
# picked up the half-occluded and far-side players on review (10-03). 0.3
# (new tracks from 0.4) then resolved 7 of the 9 human actor labels 0.4 still
# left without a tracklet on 0427小窩季打 11 (10-04), at +17% tracklets with
# the same median length — players picked up, not players split.
RFDETR_MCBYTE_TRACK_THRESH = 0.3

# The traced fp16 graph bakes the batch dimension in, so every call must be
# exactly this size — partial final batches are padded and sliced.
BATCH_SIZE = 16

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

_detector = BatchSegDetector("RFDETRSegMedium", BATCH_SIZE)


def track_video(
    video_path: Path,
    *,
    moving_camera: bool,
    stride: int = 1,
    tracker: Tracker = "bytetrack",
    event_frames: set[int] | None = None,
    on_progress: ProgressFn | None = None,
) -> dict:
    """Detect + track every annotated rally span of one video.

    ``moving_camera`` says whether the shot pans or zooms (broadcast) or is
    fixed (a sideline phone); McByte++ pays for camera motion compensation
    only when it does. The caller knows — a file path does not: an uploaded
    clip sits in a temp dir, not under a cut-kind folder.

    ``stride`` detects every Nth frame (skipped frames are grabbed but not
    decoded); the tracker sees the effective frame rate. ``tracker`` picks
    the association step: ByteTrack runs inline, frame by frame, while
    McByte++ (tracklets/mcbyte.py) runs after the detection pass over the
    detections it collected. Returns the summary counts also written to the
    jsonl header. Synchronous and GPU-bound — callers run it in an executor.

    ``event_frames`` (native indices, supplied by the caller so this stage
    keeps not reading the action file) marks frames whose raw detections are
    also persisted as a sidecar: the sparse detect stage
    (extraction/pipeline.detect_video) reads them instead of re-decoding and
    re-detecting frames this dense pass already paid for.
    """
    import cv2
    import supervision as sv

    stem = video_path.stem
    rallies = load_rallies(stem)
    if not rallies:
        raise ValueError(
            f"No rally spans for {stem} — label rallies or run Rally SPOT Predict; "
            "tracking scans rallies only"
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not fps > 0 or frame_w <= 0 or frame_h <= 0:
        cap.release()
        raise ValueError(f"Invalid video geometry: {video_path}")

    spans = [
        (r["rally_id"], int(round(r["start"] * fps)), int(round(r["end"] * fps)))
        for r in rallies
    ]
    total = sum((f1 - f0) // stride + 1 for _, f0, f1 in spans)
    # McByte++ tracks after detection; give each pass half the bar.
    scale = 2 if tracker == "mcbyte" else 1

    if on_progress:
        # ensure() below loads + fp16-compiles the model on first use.
        on_progress(0, total * scale, "loading detector weights...")
    _detector.ensure()
    res = _detector.resolution
    box_scale = np.array([frame_w / res, frame_h / res, frame_w / res, frame_h / res])

    records: list[dict] = []
    masks_store: dict[str, np.ndarray] = {}
    span_detections: dict[int, np.ndarray] = {}
    # McByte++ input, collected over the pass: frame -> rows, frame -> packed masks.
    frame_detections: dict[int, np.ndarray] = {}
    frame_masks: dict[int, list[np.ndarray]] = {}
    detected = 0
    current_rally: int | None = None
    bytetrack = None
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
        with span_frames(cap, spans, stride=stride, resolution=res, name=f"track-decode-{stem}") as frames:
            pending: list[tuple[int, int, object]] = []
            exhausted = False
            while not exhausted or pending:
                while not exhausted and len(pending) < BATCH_SIZE:
                    item = next(frames, None)
                    if item is None:
                        exhausted = True
                        break
                    pending.append(item)
                if not pending:
                    break
                detections = _detector.predict_batch([p[2] for p in pending], TRACK_SCORE_THRESHOLD)
                for (rally_id, frame_idx, _), det in zip(pending, detections):
                    if tracker == "bytetrack" and rally_id != current_rally:
                        # Rally boundary: batches may span it (detection is
                        # stateless) but the tracker must not.
                        flush_rally()
                        current_rally = rally_id
                        bytetrack = sv.ByteTrack(
                            frame_rate=max(1, round(fps / stride)),
                            # Two consecutive hits before a track exists — kills the
                            # one-frame ghosts a 0.1 detection floor produces in a crowd.
                            minimum_consecutive_frames=2,
                        )
                    det.xyxy = det.xyxy * box_scale
                    if event_frames and frame_idx in event_frames:
                        # Full frame-pixel candidate set, pre-tracker: the sparse
                        # detect stage wants flicker too.
                        span_detections[frame_idx] = np.concatenate(
                            (det.xyxy, det.confidence[:, None]), axis=1
                        ).astype(np.float32)
                    if tracker == "mcbyte":
                        frame_detections[frame_idx] = np.concatenate(
                            (det.xyxy, det.confidence[:, None]), axis=1
                        ).astype(np.float32)
                        frame_masks[frame_idx] = [
                            _pack_mask(m, xyxy / box_scale) for m, xyxy in zip(det.mask, det.xyxy)
                        ]
                        detected += 1
                        if on_progress:
                            on_progress(detected, total * scale, f"frame {detected}/{total} · rally {rally_id}")
                        continue
                    det = bytetrack.update_with_detections(det)  # masks ride along, aligned
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
    finally:
        cap.release()

    if tracker == "mcbyte":
        # McByte++ runs in its own process with EdgeTAM + Re-ID; holding the
        # detector meanwhile would put both models on the GPU at once.
        _detector.release()
        records, masks_store = _mcbyte_tracks(
            video_path, spans, stride, frame_detections, frame_masks, moving_camera=moving_camera,
            on_progress=(lambda done, n, msg: on_progress(total + done, total * scale, msg)) if on_progress else None,
        )

    counts = {
        "rallies": len(spans),
        "frames": detected,
        "tracklets": len(records),
    }
    header = {
        "video": stem,
        "source": {
            "detector": f"{SEG_WEIGHTS} (fp16 batch)",
            "tracker": mcbyte.SOURCE if tracker == "mcbyte" else f"supervision.ByteTrack {sv.__version__}",
        },
        "fps": fps,
        "frame_size": [frame_w, frame_h],
        "stride": stride,
        "mask_res": [MASK_H, MASK_W],
        # Which rallies these tracklets were cut from. A track key is
        # "{rally_id}:{track_id}" and rally_id is positional, so if the spans
        # move every key silently means something else — this is how a reader
        # finds out instead of mis-resolving a human's tracklet label.
        "rallies": {"count": len(spans), "fingerprint": rally_fingerprint(stem)},
        "created_at": time.time(),
        "counts": counts,
    }
    if event_frames:
        save_span_detections(stem, DETECTOR_NAME, span_detections)
    # Masks land first: the jsonl's mtime is what downstream caches key on,
    # so a reader never sees new tracks with the old masks.
    save_track_masks(stem, (MASK_H, MASK_W), masks_store)
    write_jsonl(tracks_path(stem), header, records)
    return counts


def _mcbyte_tracks(
    video_path: Path,
    spans: list[tuple[int, int, int]],
    stride: int,
    detections: dict[int, np.ndarray],
    masks: dict[int, list[np.ndarray]],
    *,
    moving_camera: bool,
    on_progress: ProgressFn | None,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """McByte++ over the collected detections; masks re-attached by det index."""
    tracklets = mcbyte.track(
        video_path, detections, spans, stride=stride, track_thresh=RFDETR_MCBYTE_TRACK_THRESH,
        low_thresh=TRACK_SCORE_THRESHOLD, min_frames=MIN_TRACK_FRAMES, cmc=moving_camera,
        on_progress=on_progress,
    )
    records: list[dict] = []
    masks_store: dict[str, np.ndarray] = {}
    for t in tracklets:
        det_index = t.pop("det_index")
        records.append(t)
        masks_store[f"{t['rally_id']}:{t['track_id']}"] = np.stack(
            [masks[f][i] for f, i in zip(t["frames"], det_index)]
        )
    return records, masks_store
