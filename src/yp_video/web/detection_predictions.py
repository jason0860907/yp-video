"""Named prediction sources for human review; never substitute one silently."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import Literal

import numpy as np

from yp_video.core.cache import StatCache
from yp_video.core.jsonl import read_jsonl_cached, read_jsonl_header
from yp_video.core.person_boxes import PersonBoxes, person_boxes_path
from yp_video.extraction.store import records_path
from yp_video.tracklets.store import load_tracklets, tracks_path

Source = Literal["fusion", "tracking", "detection"]
_cache = StatCache(max_source_bytes=64 * 1024 * 1024)
LABELS = {"fusion": "Fusion", "tracking": "Tracking", "detection": "Action detections"}


def _path(stem: str, source: Source) -> Path:
    return {"fusion": person_boxes_path, "tracking": tracks_path, "detection": records_path}[source](stem)


def _catalog(stem: str, source: Source) -> dict:
    path = _path(stem, source)

    def read():
        if source == "fusion":
            data = PersonBoxes.load(path)
            return {"frames": data.frames[data.counts > 0].tolist(), "people": data,
                    "num_frames": data.num_frames, "model": read_checkpoint(path)}
        meta = read_jsonl_header(path)
        if source == "tracking":
            records = load_tracklets(path).records
            frames = sorted({f for r in records for f in r["frames"]})
            return {"frames": frames, "meta": meta}
        _, records = read_jsonl_cached(path)
        by_frame = {int(r["frame"]): r["detections"] for r in records if r.get("detections")}
        return {"frames": sorted(by_frame), "meta": meta, "by_frame": by_frame}

    return _cache.get((stem, source), [path], read)


def read_checkpoint(path: Path) -> str:
    with np.load(path, allow_pickle=False) as data:
        return str(data["checkpoint"]) if "checkpoint" in data else "Fusion person head"


def sources(stem: str, num_frames: int, fps: float) -> list[dict]:
    result = []
    for source in LABELS:
        if not _path(stem, source).exists():
            continue
        data = _catalog(stem, source)
        frames = data["frames"]
        if source == "fusion":
            aligned = data["num_frames"] == num_frames
            model = data["model"]
        else:
            meta = data["meta"]
            aligned = abs(float(meta.get("fps") or 0) - fps) <= 1 and (not frames or frames[-1] < num_frames)
            model = (meta.get("source") or {}).get("detector", LABELS[source])
        result.append({"id": source, "label": LABELS[source], "model": model,
                       "aligned": aligned, "first_frame": frames[0] if frames and aligned else None,
                       "frame_count": len(frames) if aligned else 0})
    return result


def prediction(stem: str, frame: int, num_frames: int, fps: float, source: Source | None) -> dict:
    available = sources(stem, num_frames, fps)
    if source is None:
        source = next((s["id"] for s in available if s["aligned"] and s["frame_count"]), None)
    chosen = next((s for s in available if s["id"] == source), None)
    empty = {"boxes": [], "scores": [], "frame": None, "source": source,
             "next_frame": None, "previous_frame": None, "message": "No predictions on this frame."}
    if chosen is None:
        return {**empty, "message": "No saved predictions for this source."}
    if not chosen["aligned"]:
        return {**empty, "message": "Prediction frame rate/count does not match this video. Run inference again."}
    data = _catalog(stem, source)
    frames = data["frames"]
    next_index = bisect_right(frames, frame)
    prev_index = bisect_left(frames, frame) - 1
    empty.update(next_frame=frames[next_index] if next_index < len(frames) else None,
                 previous_frame=frames[prev_index] if prev_index >= 0 else None)
    boxes, scores, at = [], [], frame
    if source == "fusion":
        people = data["people"]
        index = min((frame + (people.stride - 1) // 2) // people.stride, len(people.frames) - 1)
        a, b = people.offsets[index:index + 2]
        boxes, scores, at = people.boxes[a:b].tolist(), people.scores[a:b].tolist(), int(people.frames[index])
    else:
        width, height = data["meta"]["frame_size"]
        if source == "tracking":
            for row in load_tracklets(_path(stem, source)).records:
                i = bisect_left(row["frames"], frame)
                if i < len(row["frames"]) and row["frames"][i] == frame:
                    boxes.append(row["boxes"][i])
                    scores.append(row["scores"][i])
        else:
            rows = data["by_frame"].get(frame, [])
            boxes, scores = [r["box"] for r in rows], [r["score"] for r in rows]
        boxes = (np.asarray(boxes).reshape(-1, 4) / [width, height, width, height]).clip(0, 1).tolist()
    valid = [(b, float(s)) for b, s in zip(boxes, scores) if b[2] > b[0] and b[3] > b[1]]
    return {**empty, "boxes": [b for b, _ in valid], "scores": [s for _, s in valid],
            "frame": at if valid else None,
            "message": f'{chosen["label"]} · {chosen["model"]}' if valid else empty["message"]}
