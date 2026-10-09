"""Which tracklet each extracted event's actor IS.

Tracklets know nothing about events and extraction records know nothing about
tracklets — joining them needs both, so it happens here, in the one layer
allowed to see both.

Two answers, in this order:

1. a HUMAN label's box (actor/labels.py), on the event frame
2. the POLICY's pick — the tracklet it named (``record["track"]``) when that
   still exists, else the box it cropped

A box becomes a tracklet by geometry (tracklets/geometry.link_boxes). A label
carries no tracklet on purpose: ``track_id`` restarts per rally, so every
re-track renumbers, while a box on a frame means the same person forever.

Nothing is stored. The answer is recomputed from the label file, the records
and the tracklets, so re-running tracking can never leave a stale pointer
behind.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from yp_video.actor import labels as actor_labels
from yp_video.actor.box_style import event_box
from yp_video.actor.labels import ActorVerdict
from yp_video.core.cache import StatCache
from yp_video.core.jsonl import read_jsonl_cached
from yp_video.extraction.store import (
    action_source_paths,
    labelable,
    records_path,
)
from yp_video.person.dense import dense_path
from yp_video.person.detector import iou
from yp_video.tracklets.geometry import (
    BOX_MATCH_IOU,
    EVENT_TRACK_MAX_DELTA,
    BoxQuery,
    TrackRef,
    box_near,
    link_boxes,
)
from yp_video.tracklets.store import (
    TrackMasks,
    load_track_masks,
    load_tracklets,
    tracklet_index,
    tracks_masks_path,
    tracks_path,
)

# Keyed by stem on its source files. Tiny values (one small dict per video).
_links_cache: StatCache = StatCache()


def event_tracks(stem: str) -> dict[str, TrackRef]:
    """event_id → the tracklet its actor is (see the module docstring).

    Events with no actor at all (a miss, or an occluded verdict) never link —
    there is nothing to resolve, which is an absent entry rather than an
    error. Neither does a label an action edit moved off its frame that the
    dense boxes cannot follow back.
    """
    tracks = tracks_path(stem)
    records = records_path(stem)
    if not tracks.exists() or not records.exists():
        return {}
    sources = [tracks, records, *action_source_paths(stem)]
    # The label file and the dense pass join the cache key only once they
    # exist: before that there are no human answers to honour, or no frames
    # to follow a moved one through.
    for optional in (actor_labels.actors_path(stem), dense_path(stem)):
        if optional.exists():
            sources.append(optional)
    return _links_cache.get(stem, sources, lambda: _event_tracks(stem))


def _event_tracks(stem: str) -> dict[str, TrackRef]:
    tmeta = load_tracklets(tracks_path(stem)).meta
    rmeta, records = read_jsonl_cached(records_path(stem))  # read-only
    records = labelable(records, stem, float(rmeta.get("fps") or 0))
    index = tracklet_index(stem)
    verdicts = actor_labels.load(stem)

    out: dict[str, TrackRef] = {}
    queries: list[BoxQuery] = []
    for record in records:
        event_id = record["id"]
        label = verdicts.get(str(event_id))
        if label is not None:
            if label.verdict is ActorVerdict.OCCLUDED:
                continue
            # The human box is tight and is its own gate: no display box
            # stands between it and the tracklet it sits on.
            box = event_box(stem, label, record["frame"])
            if box is not None:
                queries.append(BoxQuery(event_id, record["frame"], list(box), list(box)))
            continue
        if not record.get("box"):
            continue
        stored = record.get("track")
        named = TrackRef.parse(stored) if stored else None
        if named is not None and index.tracklet(named) is not None:
            out[event_id] = named
            continue
        # A policy pick cut from another frame has its box THERE.
        queries.append(
            BoxQuery(
                event_id,
                int(record.get("crop_frame") or record["frame"]),
                list(record.get("actor_box") or record["box"]),
                record["box"],
            )
        )
    out.update(link_boxes(index, queries, stride=int(tmeta.get("stride") or 1)))
    return out


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


# ── Resolving a policy's tracklet pick back to a croppable box ────
# The crop it chooses feeds the embedder, so it has to be reproducible from
# the record alone, long after the pick.

#: Coverage of the tracklet's mask a stored detection needs to be accepted as
#: that player. Coverage, not IoU: a partially occluded player's mask is a
#: fragment, and a fragment always loses an IoU contest to the occluder in
#: front of it.
MASK_COVERAGE_MIN = 0.6
#: Box IoU a covering detection also needs with the tracklet's own box.
#: Coverage alone cannot tell the player from an occluder standing in front:
#: the occluder's box encloses the visible fragment too, covers it fully, and
#: then wins on detector score. Measured 2026-10-02 over 1,346 masked events:
#: the occluder won 74 times (IoU 0.05–0.42 with the tracklet) while the
#: player's own detection sat at 0.76–0.99; no event lost every candidate.
TRACK_SHAPE_IOU = 0.5
#: A mask row this far from the sampled frame still describes the same pose.
MASK_NEAR_OFFSETS = (0, -1, 1)


Box = tuple[float, float, float, float]


def _as_box(value: Sequence[float]) -> Box:
    x0, y0, x1, y1 = (float(v) for v in value)
    return x0, y0, x1, y1


@dataclass(frozen=True)
class TrackPick:
    """Where to crop for a policy's tracklet pick."""

    box: Box
    #: The frame to cut from — the event's, unless the track never reaches it.
    frame: int
    #: Whether an IoU snap onto a fresh detection may still apply. False when
    #: no stored detection covered the mask: snapping could only attach the
    #: occluder that the mask just ruled out.
    snap: bool


def _box_on(boxes: dict[int, Sequence[float]], frame: int) -> tuple[list[float], int] | None:
    """The tracklet's box ON ``frame``, and the frame its mask row sits on.

    At stride 2 an odd event frame has no box of its own. The detections it is
    compared against were taken on the event frame, so the two boxes either
    side are interpolated — a hitter covers a lot of ground in two frames, and
    comparing their event-frame detection with a box from a neighbouring frame
    would read them as someone else. One side only (the track starts or ends
    there) is taken as is.
    """
    found = box_near(boxes, frame)
    if found is None:
        return None
    box, at = found
    if at != frame:
        reach = range(1, EVENT_TRACK_MAX_DELTA + 1)
        f0 = next((frame - d for d in reach if frame - d in boxes), None)
        f1 = next((frame + d for d in reach if frame + d in boxes), None)
        if f0 is not None and f1 is not None:
            t = (frame - f0) / (f1 - f0)
            return [a + t * (b - a) for a, b in zip(boxes[f0], boxes[f1])], at
    return list(box), at


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


def _silhouettes(stem: str, ref: TrackRef, masks: TrackMasks | None):
    """One tracklet's mask rows, from the caller's open archive if it has one.

    A whole-video archive is ~12 MB compressed, and opening it per event was
    most of what re-deciding a video cost. Callers that already hold one
    (tracklets/store.open_track_masks) pass it in; a one-off resolve still
    reads the file for itself.
    """
    if masks is not None:
        return masks.get(ref.key)
    if not tracks_masks_path(stem).exists():
        return None
    try:
        return load_track_masks(stem, ref.rally_id, ref.track_id)
    except (FileNotFoundError, KeyError):
        return None


def _mask_at(
    stem: str, tracklet: dict, ref: TrackRef, frame: int, masks: TrackMasks | None
):
    """The tracklet's mask row nearest ``frame``, or None when it has none."""
    silhouettes = _silhouettes(stem, ref, masks)
    if silhouettes is None:
        return None
    row_of = {f: i for i, f in enumerate(tracklet["frames"])}
    for offset in MASK_NEAR_OFFSETS:
        row = row_of.get(frame + offset)
        if row is not None and row < len(silhouettes):
            return silhouettes[row]
    return None


def resolve_track(
    stem: str,
    record: dict,
    ref: TrackRef,
    *,
    masks: TrackMasks | None = None,
) -> TrackPick | None:
    """Where to crop the person this tracklet follows, for one event.

    Prefers a stored detection — the extraction detector's box is what every
    automatic crop was cut from, and a tracklet's segmentation box runs wider,
    so cropping it directly would give tracklet picks different statistics
    than box picks and quietly poison the embedder.

    ``masks`` is the caller's already-open silhouette archive, when it has
    one; without it this opens the file itself, which is only affordable for
    a single event.

    Returns None only when the tracklet has no box anywhere near the event.
    """
    if not tracks_path(stem).exists():
        return None
    tracklet = tracklet_index(stem).tracklet(ref)
    if tracklet is None or not tracklet["frames"]:
        return None

    event_frame = record["frame"]
    boxes = dict(zip(tracklet["frames"], tracklet["boxes"]))
    found = _box_on(boxes, event_frame)
    if found is None:
        # The track never reaches the action — the actor was undetected
        # around it. Crop where the player demonstrably IS. The client used to
        # need a hand-clicked frame for this; the tracklet already knows one.
        nearest = min(boxes, key=lambda f: abs(f - event_frame))
        return TrackPick(box=_as_box(boxes[nearest]), frame=nearest, snap=False)

    track_box, at = found
    detections = record.get("detections") or []
    mask = _mask_at(stem, tracklet, ref, at, masks)

    if mask is not None:
        covered = [
            d for d in detections
            if _mask_coverage(mask, track_box, d["box"]) >= MASK_COVERAGE_MIN
            and iou(d["box"], track_box) >= TRACK_SHAPE_IOU
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
    best_box, best_iou = track_box, BOX_MATCH_IOU
    for detection in detections:
        overlap = iou(detection["box"], track_box)
        if overlap >= best_iou:
            best_box, best_iou = detection["box"], overlap
    return TrackPick(box=_as_box(best_box), frame=event_frame, snap=True)


def _area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
