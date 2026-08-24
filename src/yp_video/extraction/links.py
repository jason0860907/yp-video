"""Which tracklet each extracted event's actor IS.

Tracklets know nothing about events and extraction records know nothing about
tracklets — joining them needs both, so it happens here, in the one layer
allowed to see both.

Three answers, in this order, and the order is the whole point:

1. the tracklet a HUMAN named (actor/labels.py), when it still exists
2. the tracklet a POLICY named (``record["track"]``)
3. failing both, the tracklet the stored box geometrically sits on

Geometry is the FALLBACK — it exists because the rule policy answers with a
box and somebody still has to say which player that box is. Running it over
an answer that already named a tracklet is how a deliberate pick got
overwritten: two overlapping players resolve to boxes that each match the
other's tracklet, so clicking the right one changed nothing on screen.
Measured at 6.7% of picks, concentrated on exactly the overlapping players
that get picked by hand in the first place.

Nothing is stored. The answer is recomputed from the label file, the records
and the tracklets, so re-running tracking can never leave a stale pointer
behind — a name the anchor contradicts falls through to geometry rather than
pointing at whoever inherited its id.

"Contradicts" and not "no longer exists": ``track_id`` restarts per rally and
gets reused, so after a re-track the stored pair almost always still
resolves, just to somebody else. Existence cannot tell a surviving id from an
inherited one. The ANCHOR can — the box whoever picked was pointing at, which
no re-run moves — so a person's named tracklet stops being honoured once the
anchor names somebody else (``anchor_names_another``), and is re-derived from
that same anchor instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from yp_video.actor import labels as actor_labels
from yp_video.actor.labels import ActorLabel, ActorVerdict
from yp_video.core.cache import StatCache
from yp_video.core.jsonl import read_jsonl_cached
from yp_video.extraction.store import (
    action_source_paths,
    labelable,
    records_path,
)
from yp_video.person.detector import iou
from yp_video.tracklets.geometry import (
    BOX_MATCH_IOU,
    LINK_MIN_MARGIN,
    BoxQuery,
    TrackletIndex,
    TrackRef,
    anchor_names_another,
    link_boxes,
)
from yp_video.tracklets.store import (
    TrackMasks,
    load_track_masks,
    tracklet_index,
    tracks_masks_path,
    tracks_path,
)

# Keyed by stem on its source files. Tiny values (one small dict per video).
_links_cache: StatCache = StatCache()
_unresolved_cache: StatCache = StatCache()


def event_tracks(stem: str) -> dict[str, TrackRef]:
    """event_id → the tracklet its actor is (see the module docstring).

    Events with no actor at all (a miss, or an occluded verdict) never link —
    there is nothing to resolve, which is an absent entry rather than an
    error.
    """
    tracks = tracks_path(stem)
    records = records_path(stem)
    if not tracks.exists() or not records.exists():
        return {}
    sources = [tracks, records, *action_source_paths(stem)]
    # The label file joins the cache key only once it exists. Before that
    # there are no human answers to honour, and the write that creates it
    # rewrites the records too (extraction/actor_fix.py), so the entry is
    # invalidated either way.
    labels = actor_labels.actors_path(stem)
    if labels.exists():
        sources.append(labels)
    return _links_cache.get(stem, sources, lambda: _event_tracks(stem))


def _anchor(label: ActorLabel | None, record: dict) -> tuple[int, list[float]]:
    """(frame, box) the pick was made at — what re-derives it when ids move.

    A person's own box outranks the record's. For a tracklet label the box IS
    the anchor and not the answer (actor/labels.py): it is where they pointed,
    while the record's box is the POLICY's answer to the same event and can
    name the other player of an overlapping pair. Re-deriving from the record
    would quietly hand the pick back to whoever the policy preferred — the
    very thing naming a tracklet was for.
    """
    frame = record.get("crop_frame") or record["frame"]
    if label is not None and label.track is not None and label.box is not None:
        # A cross-frame pick was drawn on its own frame; look there.
        return int(frame if label.frame is None else label.frame), list(label.box)
    return int(frame), list(record.get("actor_box") or record["box"])


def _borne_out(
    index: TrackletIndex,
    named: TrackRef,
    *,
    by_human: bool,
    frame: int,
    anchor: list[float],
    stride: int,
) -> bool:
    """Whether the tracklets still bear out the name somebody gave.

    A PERSON's pick is checked against its anchor as well: after a re-track
    the pair usually still exists but wears a different player, and only the
    box they pointed at tells those apart. It takes a positive contradiction
    to overturn one — silence is not evidence, see ``anchor_names_another``.

    A POLICY's pick is checked for existence, as before. It lives in the
    records file, which extraction regenerates from the tracklets it is being
    compared against, so it is not a durable answer that has to survive a
    re-run — and its box is the display box for the same event, which for an
    overlapping pair can name the other player. Anchoring it would reject
    answers that are simply the policy disagreeing with geometry, which is the
    disagreement naming a tracklet exists to settle.
    """
    if index.tracklet(named) is None:
        return False
    return not by_human or not anchor_names_another(
        index, named, frame, anchor, stride=stride
    )


def _named_track(label: ActorLabel | None, record: dict) -> TrackRef | None:
    """The tracklet somebody NAMED for this event, human before policy.

    A human's pick outranks a policy's for the same reason it does everywhere
    else: they looked. Neither is checked for existence here — the caller does
    that, because "named a tracklet that is gone" and "named nothing" lead to
    the same place but are not the same fact.
    """
    if label is not None and label.track is not None:
        return label.track
    stored = record.get("track")
    return TrackRef.parse(stored) if stored else None


def _event_tracks(stem: str) -> dict[str, TrackRef]:
    tmeta, _tracklets = read_jsonl_cached(tracks_path(stem))  # read-only
    rmeta, records = read_jsonl_cached(records_path(stem))  # read-only
    records = labelable(records, stem, float(rmeta.get("fps") or 0))
    index = tracklet_index(stem)
    verdicts = actor_labels.load(stem)

    stride = int(tmeta.get("stride") or 1)

    out: dict[str, TrackRef] = {}
    queries: list[BoxQuery] = []
    for record in records:
        if not record.get("box"):
            continue
        event_id = record["id"]
        label = verdicts.get(str(event_id))
        by_human = label is not None and label.track is not None
        named = _named_track(label, record)
        # A cross-frame pick's box lives on crop_frame, not the event frame
        # (the actor was not trackable there) — look it up THERE.
        frame, anchor = _anchor(label, record)
        if named is not None and _borne_out(
            index, named, by_human=by_human, frame=frame, anchor=anchor, stride=stride
        ):
            out[event_id] = named
            continue
        queries.append(
            BoxQuery(
                key=event_id,
                frame=frame,
                anchor=anchor,
                gate=record["box"],
                # Re-deriving what a PERSON chose: a near-tie between two
                # overlapping players is not an answer here. Refusing sends the
                # event to unresolved_labels, which is already the re-pick
                # worklist, instead of silently reassigning their pick.
                margin=LINK_MIN_MARGIN if by_human else 0.0,
            )
        )
    out.update(link_boxes(index, queries, stride=stride))
    return out


def unresolved_labels(stem: str) -> set[str]:
    """The re-pick worklist: labeled events no tracklet can be derived for.

    A verdict that names a person (so, not occluded) but resolves to no
    tracklet TODAY — neither by the key it stored nor by the geometry
    fallback. These are the labels tracklet training drops, whatever their
    verdict: a legacy hand-drawn box, a detection-fallback pick, or a confirm
    whose box no longer sits on anything tracked. Recomputed like the links
    themselves, so a better tracking run shrinks the list by itself.

    Empty while the video has no tracking run at all: nothing is resolvable
    then, but the remedy is running tracking, not re-picking players — that
    gap is the pipeline's to report.

    Cached per file version — deriving it parses the full records file, and
    the association work list asks for every extracted video on each page
    load. The returned set is shared; callers only test membership.
    """
    tracks = tracks_path(stem)
    records = records_path(stem)
    if not tracks.exists() or not records.exists():
        return set()
    # Same sources as event_tracks, and for the same reason: the labels file
    # joins the key only once it exists.
    sources = [tracks, records, *action_source_paths(stem)]
    labels = actor_labels.actors_path(stem)
    if labels.exists():
        sources.append(labels)
    return _unresolved_cache.get(stem, sources, lambda: _unresolved_labels(stem))


def _unresolved_labels(stem: str) -> set[str]:
    meta, rows = read_jsonl_cached(records_path(stem))
    current = {
        str(row["id"])
        for row in labelable(rows, stem, float(meta.get("fps") or 0))
    }
    linked = event_tracks(stem)
    return {
        event_id
        for event_id, label in actor_labels.load(stem).items()
        if event_id in current
        and label.verdict is not ActorVerdict.OCCLUDED
        and event_id not in linked
    }


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
    so cropping it directly would give manual crops different statistics than
    automatic ones and quietly poison the embedder.

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
    mask = _mask_at(stem, tracklet, ref, at, masks)

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
    best_box, best_iou = track_box, BOX_MATCH_IOU
    for detection in detections:
        overlap = iou(detection["box"], track_box)
        if overlap >= best_iou:
            best_box, best_iou = detection["box"], overlap
    return TrackPick(box=_as_box(best_box), frame=event_frame, snap=True)


def _area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
