"""Turning "this person, on this frame" into the pixels a record points at.

Three decisions end here and every one of them is the same three questions —
which detection does the answer really name, what should the crop be centred
on, and was it cut from the event's own frame:

- extraction's automatic pick (extraction/pipeline.py)
- a human label, applied by the fix endpoint (same file) or materialized by
  reassociation (extraction/reassociate.py)
- a re-decided automatic pick (same file)

Each used to answer them with its own copy of the rules, and the copies had
already drifted — one deleted the crop it superseded and the others leaked it,
one anchored a cross-frame crop on the box and another on a contact point that
belongs to a frame the player is not in. Two functions now: ``person_for``
answers WHO, ``cut`` answers WHERE. What to do when the answer is nothing
stays with the callers, because there they genuinely differ — extraction skips
the event, the fix endpoint raises, reassociation clears the record.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from yp_video.actor.box_style import event_box
from yp_video.actor.labels import ActorLabel
from yp_video.extraction.links import resolve_track
from yp_video.person.detector import PersonBox, iou, person_from_detection
from yp_video.tracklets.geometry import TrackRef
from yp_video.tracklets.store import TrackMasks

Box = tuple[float, float, float, float]

# Version of the crop geometry contract persisted on each materialized
# record. Version 2 is segmentation person box ∪ ball; records without it
# were cut by the retired pose-hull contract and are rebuilt on association.
CROP_SCHEMA_VERSION = 2

# A fix box must overlap a stored segmentation detection this much to snap
# onto it; below that the box is embedded as drawn.
FIX_SNAP_IOU = 0.5

# Breathing room around the display box, so the crop isn't flush against the
# player: a fraction of each side, plus a floor that keeps far (small) boxes
# from getting a margin of a pixel or two.
DISPLAY_MARGIN_FRAC = 0.04
DISPLAY_MARGIN_MIN_PX = 4


def clamp_box(box: Box, w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(w, int(x1)), min(h, int(y1))
    return x0, y0, x1, y1


def display_box(person: PersonBox, x: float, y: float, w: int, h: int) -> tuple[int, int, int, int]:
    """The union of the segmentation person box and ball, plus a margin."""
    x0, y0, x1, y1 = person.xyxy
    ux0, uy0, ux1, uy1 = min(x0, x), min(y0, y), max(x1, x), max(y1, y)
    mx = DISPLAY_MARGIN_FRAC * (ux1 - ux0) + DISPLAY_MARGIN_MIN_PX
    my = DISPLAY_MARGIN_FRAC * (uy1 - uy0) + DISPLAY_MARGIN_MIN_PX
    return clamp_box((ux0 - mx, uy0 - my, ux1 + mx, uy1 + my), w, h)


def snap_to_detection(detections: list[dict], box: list[float]) -> PersonBox | None:
    """The stored detection a box refers to, matched by IoU."""
    best, best_iou = None, FIX_SNAP_IOU
    for d in detections:
        overlap = iou(d["box"], box)
        if overlap >= best_iou:
            best, best_iou = d, overlap
    return person_from_detection(best) if best else None


@dataclass(frozen=True)
class CropTarget:
    """Where a decision says to cut, whoever made it."""

    box: Box
    #: The frame to cut from — the event's, unless a policy's tracklet never
    #: reaches it and the answer points at a nearby one.
    frame: int
    #: Whether an IoU snap onto a stored detection may still apply. False for
    #: a human's box (it is the answer as clicked), and when no stored
    #: detection IS this player: snapping could then only attach the occluder
    #: that the silhouettes just ruled out.
    snap: bool


def crop_target(
    stem: str,
    record: dict,
    track: TrackRef | None,
    fallback: CropTarget | None,
    *,
    masks: TrackMasks | None = None,
) -> CropTarget | None:
    """Where a policy's answer says to crop, resolving its tracklet if it
    named one.

    A tracklet is re-resolved from the tracklet every time (see
    extraction/links.resolve_track), so a re-extraction with fresh detections
    self-heals instead of IoU-guessing which box the old answer meant.

    ``fallback`` is what the answer means without a resolvable tracklet: the
    policy's own box, or nothing at all — a policy can simply abstain.
    """
    if track is not None:
        pick = resolve_track(stem, record, track, masks=masks)
        if pick is not None:
            return CropTarget(pick.box, pick.frame, pick.snap)
    return fallback


def label_target(stem: str, record: dict, label: ActorLabel) -> CropTarget | None:
    """Where a human's verdict says to crop: exactly its box on the event frame.

    The box IS the answer (actor/labels.py) — a 2XLarge box the person
    clicked — so nothing snaps it onto a stored detection. None when there is
    no box there: an occluded verdict, or a label an action edit moved off its
    frame that the dense boxes cannot follow back (box_style.event_box). The
    callers treat that as an unresolved event; guessing would crop a stranger.
    """
    box = event_box(stem, label, record["frame"])
    return None if box is None else CropTarget(box, record["frame"], snap=False)


def person_for(record: dict, target: CropTarget) -> PersonBox:
    """Who the target names: a stored detection where one is it, else the box.

    No snap across frames — the stored detections belong to the event frame,
    and on another frame the nearest one is somebody else standing there.
    """
    cross_frame = target.frame != record["frame"]
    snapped = (
        snap_to_detection(record.get("detections") or [], list(target.box))
        if target.snap and not cross_frame
        else None
    )
    return snapped or PersonBox(xyxy=target.box, score=0.0)


def cut(
    record: dict,
    frame_img,
    person: PersonBox,
    *,
    source_frame: int,
    contact: tuple[float, float] | None,
    frame_size: tuple[int, int],
    out_dir: Path,
    suffix: str = "",
):
    """Point ``record`` at ``person`` and write its crop.

    Returns the crop image, or None when the box is degenerate — what that
    means is the caller's to decide.

    The display box unions the contact point, which is meaningless on another
    frame (the player has moved) or when the event has none; those crops are
    anchored on the box itself.
    """
    import cv2

    w, h = frame_size
    x0, y0, x1, y1 = clamp_box(person.xyxy, w, h)
    if x1 <= x0 or y1 <= y0:
        return None
    cross_frame = source_frame != record["frame"]
    ax, ay = (
        contact
        if contact is not None and not cross_frame
        else ((person.xyxy[0] + person.xyxy[2]) / 2, (person.xyxy[1] + person.xyxy[3]) / 2)
    )
    dx0, dy0, dx1, dy1 = display_box(person, ax, ay, w, h)
    crop = frame_img[dy0:dy1, dx0:dx1]
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_file = out_dir / f"{record['id']}{suffix}.jpg"
    cv2.imwrite(str(crop_file), crop)
    record.update(
        box=[dx0, dy0, dx1, dy1],
        # The raw detector box (the display box is a padded superset): the
        # seg masker and the event->tracklet link both need the tight box.
        actor_box=[x0, y0, x1, y1],
        score=person.score,
        crop=crop_file.name,
        crop_schema=CROP_SCHEMA_VERSION,
    )
    # Which frame the pixels came from is part of pointing at them: absent
    # means "the event's own", and a stale value would send every later
    # reader — the tracklet link, the next re-crop — to the wrong frame.
    if cross_frame:
        record["crop_frame"] = source_frame
    else:
        record.pop("crop_frame", None)
    return crop
