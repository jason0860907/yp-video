"""Which tracklet each extracted event's actor sits on.

Tracklets know nothing about events and extraction records know nothing about
tracklets — joining them needs both, so it happens here, in the one layer
allowed to see both. The join is geometric and derived: it is recomputed from
the two files rather than stored, so re-running tracking can never leave a
stale pointer behind.

Deliberately NOT written into the record jsonl. Embedding freshness is an
mtime comparison against that file (reid/store.stale_embedding_models), so a
pass that rewrote records to add a link would mark every matrix of every
video stale. The link is ~250 entries per video and free to recompute.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from yp_video.core.cache import StatCache
from yp_video.core.jsonl import read_jsonl_cached
from yp_video.extraction.store import records_path
from yp_video.person.detector import iou
from yp_video.tracklets.geometry import BoxQuery, TrackRef, link_boxes
from yp_video.tracklets.store import (
    load_track_masks,
    tracks_masks_path,
    tracks_path,
)

# Keyed by stem on both source files. Tiny values (one small dict per video).
_links_cache: StatCache = StatCache()


def event_tracks(stem: str) -> dict[str, TrackRef]:
    """event_id → the tracklet its actor box lands on.

    Events with no actor box (a miss, or an occluded verdict) never link —
    there is nothing to resolve, which is an absent entry rather than an
    error.
    """
    tracks = tracks_path(stem)
    records = records_path(stem)
    if not tracks.exists() or not records.exists():
        return {}
    return _links_cache.get(stem, [tracks, records], lambda: _event_tracks(stem))


def _event_tracks(stem: str) -> dict[str, TrackRef]:
    tmeta, tracklets = read_jsonl_cached(tracks_path(stem))  # read-only
    _rmeta, records = read_jsonl_cached(records_path(stem))  # read-only
    queries = [
        BoxQuery(
            key=record["id"],
            # A cross-frame pick's box lives on crop_frame, not the event
            # frame (the actor was not trackable there) — look it up THERE.
            frame=record.get("crop_frame") or record["frame"],
            anchor=record.get("actor_box") or record["box"],
            gate=record["box"],
        )
        for record in records
        if record.get("box")
    ]
    return link_boxes(tracklets, queries, stride=int(tmeta.get("stride") or 1))


def track_keys(stem: str) -> dict[str, str]:
    """event_id → "rally:track", the shape reid takes as an injected link map.

    ``reid`` may not import this module (deriving a link needs tracklets AND
    extraction records, and reid must not depend on both), so the routers
    hand it in. A plain string map keeps the boundary free of shared types.
    """
    return {event_id: ref.key for event_id, ref in event_tracks(stem).items()}


def link_payload(stem: str) -> dict[str, dict]:
    """``event_tracks`` in the shape the UI has always received."""
    return {event_id: ref.payload() for event_id, ref in event_tracks(stem).items()}


# ── Resolving a picked tracklet back to a croppable box ───────────
# The arbitration below used to live in the browser (masks.ts::resolveActorFix)
# because that is where the masks were already decoded. It belongs here: the
# crop it chooses feeds the embedder, so it has to be reproducible from the
# label alone, long after the click.

#: Coverage of the tracklet's mask a stored detection needs to be accepted as
#: that player. Coverage, not IoU: a partially occluded player's mask is a
#: fragment, and a fragment always loses an IoU contest to the occluder in
#: front of it.
MASK_COVERAGE_MIN = 0.6
#: Without masks, box IoU is the best available signal.
PICK_IOU_MIN = 0.3
#: How far from the event the tracklet may be sampled before it counts as
#: "never reaches the action" and the crop comes from elsewhere.
EVENT_TRACK_MAX_DELTA = 3
#: A mask row this far from the sampled frame still describes the same pose.
MASK_NEAR_OFFSETS = (0, -1, 1)


Box = tuple[float, float, float, float]


def _as_box(value: Sequence[float]) -> Box:
    x0, y0, x1, y1 = (float(v) for v in value)
    return x0, y0, x1, y1


@dataclass(frozen=True)
class TrackPick:
    """Where to crop for a tracklet label."""

    box: Box
    #: The frame to cut from — the event's, unless the track never reaches it.
    frame: int
    #: Whether an IoU snap onto a fresh detection may still apply. False when
    #: no stored detection covered the mask: snapping could only attach the
    #: occluder that the mask just ruled out.
    snap: bool


def _box_near(tracklet: dict, frame: int, *, window: int) -> tuple[list[float], int] | None:
    """The tracklet's box at (or nearest to) ``frame``, and where it was found."""
    at = {f: box for f, box in zip(tracklet["frames"], tracklet["boxes"])}
    for delta in range(window + 1):
        for candidate in (frame,) if delta == 0 else (frame - delta, frame + delta):
            if candidate in at:
                return at[candidate], candidate
    return None


def _mask_coverage(mask, track_box: Sequence[float], det_box: Sequence[float]) -> float:
    """Fraction of the mask's on-pixels whose cells fall inside ``det_box``.

    The mask grid is stretched over the track box, so a cell's centre is its
    position in frame pixels.
    """
    import numpy as np

    rows, cols = np.nonzero(mask)
    if not len(rows):
        return 0.0
    x0, y0, x1, y1 = track_box
    cx = x0 + (cols + 0.5) * (x1 - x0) / mask.shape[1]
    cy = y0 + (rows + 0.5) * (y1 - y0) / mask.shape[0]
    inside = (cx >= det_box[0]) & (cx < det_box[2]) & (cy >= det_box[1]) & (cy < det_box[3])
    return float(inside.sum()) / len(rows)


def _mask_at(stem: str, tracklet: dict, ref: TrackRef, frame: int):
    """The tracklet's mask row nearest ``frame``, or None when it has none."""
    if not tracks_masks_path(stem).exists():
        return None
    row_of = {f: i for i, f in enumerate(tracklet["frames"])}
    try:
        masks = load_track_masks(stem, ref.rally_id, ref.track_id)
    except (FileNotFoundError, KeyError):
        return None
    for offset in MASK_NEAR_OFFSETS:
        row = row_of.get(frame + offset)
        if row is not None and row < len(masks):
            return masks[row]
    return None


def resolve_track(stem: str, record: dict, ref: TrackRef) -> TrackPick | None:
    """Where to crop the person this tracklet follows, for one event.

    Prefers a stored detection — the extraction detector's box is what every
    automatic crop was cut from, and a tracklet's segmentation box runs wider,
    so cropping it directly would give manual crops different statistics than
    automatic ones and quietly poison the embedder.

    Returns None only when the tracklet has no box anywhere near the event.
    """
    tracks = tracks_path(stem)
    if not tracks.exists():
        return None
    _meta, tracklets = read_jsonl_cached(tracks)  # read-only
    tracklet = next(
        (
            t
            for t in tracklets
            if t["rally_id"] == ref.rally_id and t["track_id"] == ref.track_id
        ),
        None,
    )
    if tracklet is None or not tracklet["frames"]:
        return None

    event_frame = record["frame"]
    found = _box_near(tracklet, event_frame, window=EVENT_TRACK_MAX_DELTA)
    if found is None:
        # The track never reaches the action — the actor was undetected
        # around it. Crop where the player demonstrably IS. The client used to
        # need a hand-clicked frame for this; the tracklet already knows one.
        nearest = min(tracklet["frames"], key=lambda f: abs(f - event_frame))
        elsewhere = _box_near(tracklet, nearest, window=0)
        if elsewhere is None:
            return None
        return TrackPick(box=_as_box(elsewhere[0]), frame=elsewhere[1], snap=False)

    track_box, at = found
    detections = record.get("detections") or []
    mask = _mask_at(stem, tracklet, ref, at)

    if mask is not None:
        covered = [
            d for d in detections
            if _mask_coverage(mask, track_box, d["box"]) >= MASK_COVERAGE_MIN
        ]
        if covered:
            # The mask has already decided WHO; among the boxes that cover
            # them, take the one the detector is most confident in.
            #
            # The browser used to take the SMALLEST instead. That resolves
            # overlapping people — which is why the pointer hit-test still
            # does it — but once coverage has picked the person, the boxes
            # left are near-duplicates of one player and "smallest" means
            # "flimsiest": measured over 239 events it chose a detection
            # scoring 0.14 (median) where the automatic pick scored 1.54, and
            # agreed with the automatic pick on only 11.7% of events against
            # 81.2% for this rule. A manual crop cut from a worse box than an
            # automatic one is exactly the input skew that quietly degrades
            # the embedder.
            best = max(covered, key=lambda d: (d.get("score") or 0.0, -_area(d["box"])))
            return TrackPick(box=_as_box(best["box"]), frame=event_frame, snap=True)
        # No stored detection is this player. The track box goes through with
        # snapping vetoed, so it cannot re-attach the occluder.
        return TrackPick(box=_as_box(track_box), frame=event_frame, snap=False)

    # Tracked before instance masks existed — box IoU is all there is.
    best_box, best_iou = track_box, PICK_IOU_MIN
    for detection in detections:
        overlap = iou(detection["box"], track_box)
        if overlap >= best_iou:
            best_box, best_iou = detection["box"], overlap
    return TrackPick(box=_as_box(best_box), frame=event_frame, snap=True)


def _area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
