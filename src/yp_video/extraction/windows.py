"""Tracklet-window embeddings: one event seen across a second of its tracklet.

A single crop is one pose, one blur, one occlusion. The actor's tracklet
already says where the same person stands on the surrounding frames, so the
event's identity vector here is the mean over up to ``WINDOW_CROPS`` of those
boxes within ±``WINDOW_FRAMES`` — each cut from the video, background-masked
by the production segmenter, and embedded by the base clip-reident model.

Measured on six labeled sideline videos with McByte++ tracks (cross-tracklet
mAP): single masked crop 0.608 → this window 0.711. It needs tracklets that
hold one person, which is why only advanced identify (McByte++) uses it.

An event with no linked tracklet has a window of one: its own actor box on
its own frame. An event nobody acted in keeps a NaN row, as in embed_video.

Unlike the crop-file embedders (reid/embedder.py), this one needs the video
and the tracks, so it is not a registered embedder: the Embed page cannot
backfill it, and a single-row actor fix blanks its row (an unregistered
matrix is never patched with a stale vector — see pipeline._patch_embedding_row).
"""

from __future__ import annotations

import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from yp_video.core.jsonl import read_jsonl
from yp_video.core.progress import ProgressFn
from yp_video.extraction.cropping import (
    DISPLAY_MARGIN_FRAC,
    DISPLAY_MARGIN_MIN_PX,
    clamp_box,
)
from yp_video.extraction.links import track_keys
from yp_video.extraction.store import records_path
from yp_video.reid.embedder import build_embedders
from yp_video.reid.store import (
    clear_embedding_refreshes,
    embedding_write_transaction,
    save_embedding_matrix,
)
from yp_video.tracklets.store import tracklet_data

WINDOWED_EMBEDDER = "clip-reident-masked-win30"
BASE_EMBEDDER = "clip-reident"
WINDOW_FRAMES = 30
WINDOW_CROPS = 8
# A sequential grab beats a seek up to about this many frames.
_SEEK_GAP = 120

Box = tuple[float, float, float, float]


def window_picks(frames: np.ndarray, frame: int) -> list[int]:
    """Indices of the tracklet frames that represent ``frame``: up to
    WINDOW_CROPS spread evenly over those within ±WINDOW_FRAMES, else the
    single nearest one."""
    inside = np.nonzero(np.abs(frames - frame) <= WINDOW_FRAMES)[0]
    if not len(inside):
        return [int(np.argmin(np.abs(frames - frame)))]
    spread = np.linspace(0, len(inside) - 1, min(WINDOW_CROPS, len(inside))).round().astype(int)
    return sorted({int(inside[i]) for i in spread})


def window_boxes(records: list[dict], links: dict[str, str], tracks: dict[str, dict]) -> list[list[tuple[int, Box]]]:
    """Per record, the (frame, box) views its identity is averaged over."""
    out: list[list[tuple[int, Box]]] = []
    for record in records:
        if not record.get("crop"):
            out.append([])
            continue
        tracklet = tracks.get(links.get(record["id"], ""))
        if tracklet is None:
            box = record.get("actor_box") or record["box"]
            out.append([(int(record["frame"]), tuple(box))])
            continue
        frames = np.asarray(tracklet["frames"])
        out.append([(int(frames[i]), tuple(tracklet["boxes"][i])) for i in window_picks(frames, int(record["frame"]))])
    return out


def _crop_bounds(box: Box, w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    mx = DISPLAY_MARGIN_FRAC * (x1 - x0) + DISPLAY_MARGIN_MIN_PX
    my = DISPLAY_MARGIN_FRAC * (y1 - y0) + DISPLAY_MARGIN_MIN_PX
    return clamp_box((x0 - mx, y0 - my, x1 + mx, y1 + my), w, h)


def _cut_masked(video_path: Path, views: list[list[tuple[int, Box]]], out_dir: Path,
                on_progress: ProgressFn | None) -> dict[tuple[int, Box], Path]:
    """Decode every needed frame once, in order; write each view masked."""
    import cv2

    from yp_video.person.seg import crop_masker

    need: dict[int, set[Box]] = defaultdict(set)
    for record_views in views:
        for frame, box in record_views:
            need[frame].add(box)
    total = sum(len(b) for b in need.values())
    paths: dict[tuple[int, Box], Path] = {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    try:
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        pos = 0
        for frame in sorted(need):
            if frame < pos or frame - pos > _SEEK_GAP:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                pos = frame
            while pos < frame:
                cap.grab()
                pos += 1
            ok, image = cap.read()
            pos += 1
            if not ok:
                continue
            for box in need[frame]:
                x0, y0, x1, y1 = _crop_bounds(box, w, h)
                if x1 <= x0 or y1 <= y0:
                    continue
                target = [box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0]
                path = out_dir / f"{len(paths):06d}.jpg"
                cv2.imwrite(str(path), crop_masker().mask_crop(image[y0:y1, x0:x1], target))
                paths[(frame, box)] = path
                if on_progress and len(paths) % 200 == 0:
                    on_progress(len(paths), total * 2, "masking window crops...")
    finally:
        cap.release()
    return paths


def embed_tracklet_windows(stem: str, video_path: Path, *, on_progress: ProgressFn | None = None) -> dict:
    """Records + tracks + video → the WINDOWED_EMBEDDER matrix, rows aligned
    with the records. Returns ``{"models": [...], "crops": N}`` like embed_video."""
    _meta, records = read_jsonl(records_path(stem))
    tracks = {f"{t['rally_id']}:{t['track_id']}": t for t in tracklet_data(stem).records}
    views = window_boxes(records, track_keys(stem), tracks)
    embedder = build_embedders()[BASE_EMBEDDER]
    with tempfile.TemporaryDirectory(prefix="window-crops-") as tmp:
        crops = _cut_masked(video_path, views, Path(tmp), on_progress)
        order = list(crops)
        offset = len(order)

        def progress(done: int, total: int, msg: str) -> None:
            if on_progress:
                on_progress(offset + done, offset + total, f"{BASE_EMBEDDER} · {msg}")

        vectors = embedder.embed_paths([crops[k] for k in order], on_progress=progress)
    row_of = {k: i for i, k in enumerate(order)}
    dim = vectors.shape[1]
    full = np.full((len(records), dim), np.nan, dtype=np.float32)
    for i, record_views in enumerate(views):
        rows = vectors[[row_of[v] for v in record_views if v in row_of]]
        rows = rows[~np.isnan(rows).any(axis=1)]
        if len(rows):
            mean = rows.mean(axis=0)
            full[i] = mean / (np.linalg.norm(mean) + 1e-12)
    with embedding_write_transaction():
        save_embedding_matrix(stem, WINDOWED_EMBEDDER, full)
        clear_embedding_refreshes(stem, WINDOWED_EMBEDDER)
    return {"models": [WINDOWED_EMBEDDER], "crops": len(order)}
