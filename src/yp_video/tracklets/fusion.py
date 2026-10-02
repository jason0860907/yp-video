"""Fusion person boxes → per-rally tracklets, without a detector.

RF-DETR tracking remains the offline label producer; the full Inference
page uses this consumer of the same SPOT pass that spotted its actions.
ByteTrack needs only the boxes; McByte++ also decodes the frames it tracks
(EdgeTAM masks, re-ID crops).
"""

import time
from pathlib import Path

import numpy as np

from yp_video.config import cut_kind_of
from yp_video.core.jsonl import read_jsonl_header, write_jsonl
from yp_video.core.person_boxes import DETECTOR_NAME, PersonBoxes, person_boxes_path
from yp_video.core.progress import ProgressFn
from yp_video.core.rallies import load_rallies, rally_fingerprint
from yp_video.tracklets import mcbyte
from yp_video.tracklets.store import (
    Tracker,
    tracks_current,
    tracks_masks_path,
    tracks_path,
    tracks_tracker,
)
from yp_video.tracklets.tracking import MIN_TRACK_FRAMES

# Fusion boxes score lower than RF-DETR's: at 0.4 they leave as many boxes per
# frame (~9) as RF-DETR does at McByte++'s default 0.6 (measured 2026-10-01).
MCBYTE_TRACK_THRESH = 0.4


def fusion_tracks_current(stem: str, tracker: Tracker) -> bool:
    if not tracks_current(stem) or tracks_tracker(stem) != tracker:
        return False
    boxes_path = person_boxes_path(stem)
    if not boxes_path.exists():
        return False
    source = read_jsonl_header(tracks_path(stem)).get("source") or {}
    return (
        source.get("detector") == DETECTOR_NAME
        and source.get("person_boxes_mtime_ns") == boxes_path.stat().st_mtime_ns
    )


def track_person_boxes(
    video_path: Path, *, tracker: Tracker = "bytetrack", on_progress: ProgressFn | None = None
) -> dict:
    """Track every rally over the sampled person boxes, empty frames included.

    ByteTrack consumes every sample; McByte++ every ``mcbyte.STRIDE``-th.
    """
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
    if tracker == "mcbyte":
        records, done = _mcbyte_tracks(video_path, people, spans, width, height, on_progress)
        stride = people.stride * mcbyte.STRIDE
        source = mcbyte.SOURCE
    else:
        records, done = _bytetrack_tracks(people, spans, fps, width, height, on_progress)
        stride = people.stride
        source = f"supervision.ByteTrack {sv.__version__}"

    counts = {"rallies": len(rallies), "frames": done, "tracklets": len(records)}
    # A box-only model has no masks. Never pair new track IDs with old silhouettes.
    tracks_masks_path(stem).unlink(missing_ok=True)
    write_jsonl(tracks_path(stem), {
        "video": stem,
        "source": {
            "detector": DETECTOR_NAME,
            "person_boxes_mtime_ns": path.stat().st_mtime_ns,
            "tracker": source,
        },
        "fps": fps, "frame_size": [width, height], "stride": stride,
        "rallies": {"count": len(rallies), "fingerprint": rally_fingerprint(stem)},
        "created_at": time.time(), "counts": counts,
    }, records)
    return counts


def _bytetrack_tracks(people, spans, fps, width, height, on_progress) -> tuple[list[dict], int]:
    import supervision as sv

    total = sum(int(end - start) for _, start, end in spans)
    records = []
    done = 0
    for rally_id, start, end in spans:
        bytetrack = sv.ByteTrack(frame_rate=fps / people.stride, minimum_consecutive_frames=2)
        tracks: dict[int, dict] = {}
        for index in range(start, end):
            rows = people.pixels(index, width, height)
            detections = sv.Detections(xyxy=rows[:, :4], confidence=rows[:, 4])
            tracked = bytetrack.update_with_detections(detections)
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

    return records, done


def _mcbyte_tracks(video_path, people, spans, width, height, on_progress) -> tuple[list[dict], int]:
    """McByte++ over every ``mcbyte.STRIDE``-th sample of each rally."""
    detections, rally_spans = {}, []
    for rally_id, start, end in spans:
        indices = range(int(start), int(end), mcbyte.STRIDE)
        if not indices:
            continue
        for index in indices:
            detections[int(people.frames[index])] = people.pixels(index, width, height)
        rally_spans.append((rally_id, int(people.frames[indices[0]]), int(people.frames[indices[-1]])))
    tracklets = mcbyte.track(
        video_path, detections, rally_spans, stride=people.stride * mcbyte.STRIDE,
        track_thresh=MCBYTE_TRACK_THRESH,
        # A fixed sideline camera gains nothing from motion compensation.
        cmc=cut_kind_of(video_path) == "broadcast", on_progress=on_progress,
    )
    for t in tracklets:
        del t["det_index"]  # box-only: nothing per detection to carry over
    return tracklets, len(detections)
