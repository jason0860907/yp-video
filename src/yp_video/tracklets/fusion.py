"""Fusion person boxes → per-rally ByteTrack, without decoding or a detector.

RF-DETR tracking remains the offline label producer; the full Inference
page uses this consumer of the same SPOT pass that spotted its actions.
"""

import time
from pathlib import Path

import numpy as np

from yp_video.core.jsonl import read_jsonl_header, write_jsonl
from yp_video.core.person_boxes import DETECTOR_NAME, PersonBoxes, person_boxes_path
from yp_video.core.progress import ProgressFn
from yp_video.core.rallies import load_rallies, rally_fingerprint
from yp_video.tracklets.store import tracks_current, tracks_masks_path, tracks_path
from yp_video.tracklets.tracking import MIN_TRACK_FRAMES


def fusion_tracks_current(stem: str) -> bool:
    if not tracks_current(stem):
        return False
    boxes_path = person_boxes_path(stem)
    if not boxes_path.exists():
        return False
    source = read_jsonl_header(tracks_path(stem)).get("source") or {}
    return (
        source.get("detector") == DETECTOR_NAME
        and source.get("person_boxes_mtime_ns") == boxes_path.stat().st_mtime_ns
    )


def track_person_boxes(video_path: Path, *, on_progress: ProgressFn | None = None) -> dict:
    """Consume every sampled rally frame, including empty ones, in time order."""
    import cv2
    import supervision as sv

    stem = video_path.stem
    rallies = load_rallies(stem)
    if not rallies:
        raise ValueError(f"No rally spans for {stem}")
    path = person_boxes_path(stem)
    people = PersonBoxes.load(path)
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()
    if width <= 0 or height <= 0 or fps <= 0:
        raise ValueError(f"Invalid video geometry for {video_path}")

    spans = [
        (r["rally_id"], np.searchsorted(people.frames, round(r["start"] * fps)),
         np.searchsorted(people.frames, round(r["end"] * fps), side="right"))
        for r in rallies
    ]
    total = sum(int(end - start) for _, start, end in spans)
    records = []
    done = 0
    for rally_id, start, end in spans:
        tracker = sv.ByteTrack(frame_rate=fps / people.stride, minimum_consecutive_frames=2)
        tracks: dict[int, dict] = {}
        for index in range(start, end):
            rows = people.pixels(index, width, height)
            detections = sv.Detections(xyxy=rows[:, :4], confidence=rows[:, 4])
            tracked = tracker.update_with_detections(detections)
            for box, score, tid in zip(tracked.xyxy, tracked.confidence, tracked.tracker_id):
                track = tracks.setdefault(int(tid), {"frames": [], "boxes": [], "scores": []})
                track["frames"].append(int(people.frames[index]))
                track["boxes"].append([round(float(v)) for v in box])
                track["scores"].append(round(float(score), 2))
            done += 1
            if on_progress:
                on_progress(done, total, f"frame {done}/{total} · rally {rally_id}")
        records.extend(
            {"rally_id": rally_id, "track_id": tid, **track}
            for tid, track in sorted(tracks.items())
            if len(track["frames"]) >= MIN_TRACK_FRAMES
        )

    counts = {"rallies": len(rallies), "frames": done, "tracklets": len(records)}
    # A box-only model has no masks. Never pair new track IDs with old silhouettes.
    tracks_masks_path(stem).unlink(missing_ok=True)
    write_jsonl(tracks_path(stem), {
        "video": stem,
        "source": {
            "detector": DETECTOR_NAME,
            "person_boxes_mtime_ns": path.stat().st_mtime_ns,
            "tracker": f"supervision.ByteTrack {sv.__version__}",
        },
        "fps": fps, "frame_size": [width, height], "stride": people.stride,
        "rallies": {"count": len(rallies), "fingerprint": rally_fingerprint(stem)},
        "created_at": time.time(), "counts": counts,
    }, records)
    return counts
