"""Rally tracking: who is on court over time.

Dense per-rally detection + ByteTrack (see tracklets/tracking.py), and the
two read endpoints that serve the resulting tracklets to any page drawing
them — the ReID board and the actor picker both do.

Deliberately its own router. Tracking depends on rally spans and NOTHING
else: not on the action labels, not on extraction, and not on ReID, which is
three stages further down. It lived under /reid only because that is the page
the button happened to sit on, and that made a stage look like it belonged to
the one thing it is independent of.
"""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException
from pydantic import Field

from yp_video.config import find_cut
from yp_video.core.rallies import rally_sources
from yp_video.extraction import links
from yp_video.extraction import store as extraction_store
from yp_video.extraction.pipeline import load_events
from yp_video.tracklets import store as tracks_store
from yp_video.tracklets import tracking
from yp_video.web.job_helpers import init_batch_items, spawn_batch_video_job
from yp_video.web.jobs import JobSummary, JobType, job_manager
from yp_video.web.schemas import StrictModel

log = logging.getLogger(__name__)
router = APIRouter()


class TrackRequest(StrictModel):
    videos: list[str] = Field(min_length=1)
    overwrite: bool = False
    stop_vllm: bool = False
    # Detect every Nth rally frame; ByteTrack is told the effective rate.
    stride: int = Field(1, ge=1, le=10)


@router.post("/run", response_model=JobSummary)
async def run(req: TrackRequest) -> dict:
    """Dense per-rally detection + ByteTrack (see tracklets/tracking.py)."""
    video_paths: list[Path] = []
    skipped: list[str] = []
    for name in req.videos:
        path = find_cut(name)
        if path is None:
            raise HTTPException(404, f"Video not found: {name}")
        # Tracking needs rally spans and nothing else — deliberately NOT the
        # action annotation, so it can run alongside action labeling.
        if not rally_sources(path.stem):
            raise HTTPException(
                400,
                f"No rally spans for: {name} — label rallies or run Rally SPOT Predict",
            )
        if not req.overwrite and tracks_store.tracks_path(path.stem).exists():
            skipped.append(path.stem)
            continue
        video_paths.append(path)

    if not video_paths:
        raise HTTPException(400, "All selected videos already have tracking (enable overwrite)")

    job = job_manager.create_job(
        JobType.PLAYER_TRACKING,
        {
            "videos": [p.name for p in video_paths],
            "skipped_existing": skipped,
            "items": init_batch_items([p.name for p in video_paths]),
        },
        name=f"Rally Tracking ({len(video_paths)} videos)",
    )
    spawn_batch_video_job(
        job,
        video_paths,
        stop_vllm=req.stop_vllm,
        # Event frames ride along (this layer may join action + tracking;
        # the tracking stage itself stays action-free): their raw detections
        # persist as a sidecar so the sparse detect stage skips re-decoding.
        work=lambda p, cb: tracking.track_video(
            p,
            stride=req.stride,
            event_frames={e["frame"] for e in load_events(p.stem)},
            on_progress=cb,
        ),
        done_message=lambda c: f"{c['tracklets']} tracklets over {c['frames']} frames",
        start_message="tracking rallies...",
    )
    return job.to_dict()


@router.get("/masks/{name}")
def masks(name: str, rally: int) -> dict:
    """One rally's instance masks, whole tracklets at once — the overlay
    silhouettes. Each entry is the tracklet's packed mask rows (base64,
    box-crop space, see tracklets/store.save_track_masks), row i ↔ the tracklet's
    i-th frame in the tracks jsonl the client already holds."""
    import base64

    import numpy as np

    stem = Path(unquote(name)).stem
    masks_path = tracks_store.tracks_masks_path(stem)
    if not masks_path.exists():
        raise HTTPException(404, f"No track masks for {stem} — re-run tracking")
    records = tracks_store.tracklet_data(stem).records
    tracks: dict[str, str] = {}
    with np.load(masks_path) as z:
        h, w = (int(v) for v in z["_shape"])
        for t in records:
            key = f"{t['rally_id']}:{t['track_id']}"
            if t["rally_id"] == rally and key in z:
                tracks[key] = base64.b64encode(z[key].tobytes()).decode()
    return {"mask_hw": [h, w], "tracks": tracks}


@router.get("/{name}")
def tracklets(name: str) -> dict:
    """Tracklets (for the video overlay) + event→tracklet links (for crop
    badges and propagation). Scores stay server-side — the overlay only
    draws boxes, and the payload holds ~286k of them (8.5 MB of JSON, but
    ~1.5 MB over the wire once GZipMiddleware has had it)."""
    stem = Path(unquote(name)).stem
    if not tracks_store.tracks_path(stem).exists():
        raise HTTPException(404, f"No tracking for {stem} — run Rally Tracking first")
    if not extraction_store.records_path(stem).exists():
        raise HTTPException(404, f"No extraction records for {stem}")

    records = tracks_store.tracklet_data(stem).records
    return {
        # Cheap wrapper dicts per response; their large arrays stay owned by
        # the one bounded tracks cache rather than a second unbounded cache.
        "tracklets": [
            {k: tracklet[k] for k in ("rally_id", "track_id", "frames", "boxes")}
            for tracklet in records
        ],
        "links": links.link_payload(stem),
    }
