"""Per-video ReID extraction: action events → person crops → embeddings.

Writes, per video:
  player-reid/embeddings/<stem>_reid.jsonl   header + one record per event
  player-reid/crops/<stem>/<event_id>.jpg    the associated person crop

Records keep the association outcome (ok / multi / miss) so downstream
matching and the UI can treat ambiguous events differently.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

from yp_video.config import (
    ACTION_ANNOTATIONS_DIR,
    ACTION_PRE_ANNOTATIONS_DIR,
    PLAYER_REID_DIR,
)
from yp_video.core.jsonl import read_jsonl, write_jsonl
from yp_video.reid.detector import DETECTOR_NAME, PersonDetector, associate
from yp_video.reid.embedder import EMBEDDER_WEIGHTS, build_embedders

EMBEDDINGS_DIR = PLAYER_REID_DIR / "embeddings"
CROPS_DIR = PLAYER_REID_DIR / "crops"

# One instance per process: the models stay loaded across jobs.
_detector = PersonDetector()
_embedders = build_embedders()

ProgressFn = Callable[[int, int, str], None]


def reid_path(stem: str) -> Path:
    return EMBEDDINGS_DIR / f"{stem}_reid.jsonl"


def crop_dir(stem: str) -> Path:
    return CROPS_DIR / stem


def action_annotation_path(stem: str) -> Path | None:
    """Manual action annotations win over pre-annotations."""
    for directory in (ACTION_ANNOTATIONS_DIR, ACTION_PRE_ANNOTATIONS_DIR):
        path = directory / f"{stem}_actions.jsonl"
        if path.exists():
            return path
    return None


def load_events(stem: str) -> list[dict]:
    """Visible action events with a location, sorted by frame."""
    path = action_annotation_path(stem)
    if path is None:
        return []
    _meta, rows = read_jsonl(path)
    events = [
        r for r in rows
        if r.get("visible", True) and r.get("xy") and r.get("frame") is not None
    ]
    events.sort(key=lambda e: e["frame"])
    return events


def _clamp_box(box: tuple[float, float, float, float], w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(w, int(x1)), min(h, int(y1))
    return x0, y0, x1, y1


def _display_box(
    person: tuple[float, float, float, float], x: float, y: float, w: int, h: int
) -> tuple[int, int, int, int]:
    """Person box grown to include the contact point (the ball), plus margin.

    The saved crop uses this so a human reviewer sees the ball AND the player;
    the embedding still uses the tight person box, which the ball would only
    pollute.
    """
    x0, y0, x1, y1 = person
    ux0, uy0, ux1, uy1 = min(x0, x), min(y0, y), max(x1, x), max(y1, y)
    mx, my = 0.04 * (ux1 - ux0) + 4, 0.04 * (uy1 - uy0) + 4
    return _clamp_box((ux0 - mx, uy0 - my, ux1 + mx, uy1 + my), w, h)


def extract_video(video_path: Path, *, on_progress: ProgressFn | None = None) -> dict:
    """Run the full detect → associate → crop → embed pass for one video.

    Returns the summary counts also written to the jsonl header.
    Synchronous and GPU-bound — callers run it in an executor.
    """
    import cv2

    stem = video_path.stem
    events = load_events(stem)
    if not events:
        raise ValueError(f"No action events for {video_path.name}")

    out_crops = crop_dir(stem)
    out_crops.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    records: list[dict] = []
    crops: list = []
    crop_owners: list[int] = []  # records index each crop belongs to
    total = len(events)
    try:
        for i, event in enumerate(events):
            cap.set(cv2.CAP_PROP_POS_FRAMES, event["frame"])
            ok, frame = cap.read()
            record = {
                "id": event.get("id") or f"f{event['frame']}",
                "frame": event["frame"],
                "time": event.get("time"),
                "label": event.get("label"),
                "xy": event["xy"],
                "status": "miss",
                "box": None,
                "score": None,
                "candidates": 0,
                "crop": None,
            }
            if ok:
                x, y = event["xy"][0] * frame_w, event["xy"][1] * frame_h
                candidates = associate(_detector.detect(frame), x, y)
                record["candidates"] = len(candidates)
                if candidates:
                    best = candidates[0]
                    x0, y0, x1, y1 = _clamp_box(best.xyxy, frame_w, frame_h)
                    if x1 > x0 and y1 > y0:
                        dx0, dy0, dx1, dy1 = _display_box(best.xyxy, x, y, frame_w, frame_h)
                        crop = frame[dy0:dy1, dx0:dx1]
                        crop_file = out_crops / f"{record['id']}.jpg"
                        cv2.imwrite(str(crop_file), crop)
                        # Keypoints ship as crop-relative data; the UI draws
                        # the skeleton as a toggleable overlay, so the jpg
                        # stays raw — identical to what the embedder sees.
                        keypoints = None
                        if best.keypoints is not None and best.keypoint_conf is not None:
                            cw, ch = max(dx1 - dx0, 1), max(dy1 - dy0, 1)
                            keypoints = [
                                [round(float(px - dx0) / cw, 4), round(float(py - dy0) / ch, 4), round(float(c), 2)]
                                for (px, py), c in zip(best.keypoints, best.keypoint_conf)
                            ]
                        record.update(
                            status="ok" if len(candidates) == 1 else "multi",
                            box=[dx0, dy0, dx1, dy1],
                            score=best.score,
                            crop=crop_file.name,
                            keypoints=keypoints,
                        )
                        crops.append(crop)
                        crop_owners.append(len(records))
            records.append(record)
            if on_progress:
                on_progress(i + 1, total, record["status"])
    finally:
        cap.release()

    # Every registered embedder runs on the same crops so models can be
    # A/B-compared on identical inputs without re-extracting.
    for name, embedder in _embedders.items():
        matrix = embedder.embed(crops)
        for owner, emb in zip(crop_owners, matrix):
            records[owner].setdefault("embeddings", {})[name] = [round(float(v), 5) for v in emb]

    counts = {
        "events": total,
        "ok": sum(r["status"] == "ok" for r in records),
        "multi": sum(r["status"] == "multi" for r in records),
        "miss": sum(r["status"] == "miss" for r in records),
    }
    header = {
        "video": stem,
        "source": {"detector": DETECTOR_NAME, "embedders": EMBEDDER_WEIGHTS},
        "frame_size": [frame_w, frame_h],
        "created_at": time.time(),
        **counts,
    }
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(reid_path(stem), header, records)
    return counts
