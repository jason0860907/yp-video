"""Turning "this person, on this frame" into the pixels a record points at.

Four decisions end here and every one of them is the same three questions —
which detection does the answer really name, what should the crop be centred
on, and was it cut from the event's own frame:

- extraction's automatic pick (extraction/pipeline.py)
- a saved label replayed on re-extraction (same file)
- a fresh label applied by the fix endpoint (same file)
- a re-decided automatic pick (extraction/reassociate.py)

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

from yp_video.actor.labels import ActorLabel
from yp_video.extraction.links import resolve_track
from yp_video.person.detector import PersonBox, iou, person_from_detection
from yp_video.tracklets.geometry import TrackRef
from yp_video.tracklets.store import TrackMasks

Box = tuple[float, float, float, float]

# A fix box must overlap a stored detection this much to snap onto it (and
# inherit its keypoints); below that the box is embedded as drawn.
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
    """The union of the person box, ALL predicted keypoints (regardless of
    confidence), and the contact point (the ball), plus a margin.

    Keypoints join the union so a fully extended limb (spike, jump serve) is
    never cropped off; the detector box stays in it because keypoints mark
    joints, not extremities — eyes/ankles alone would cut the scalp and feet.
    The saved crop and the video overlay both use this box.
    """
    x0, y0, x1, y1 = person.xyxy
    ux0, uy0, ux1, uy1 = min(x0, x), min(y0, y), max(x1, x), max(y1, y)
    if person.keypoints is not None:
        for px, py in person.keypoints:
            ux0, uy0 = min(ux0, float(px)), min(uy0, float(py))
            ux1, uy1 = max(ux1, float(px)), max(uy1, float(py))
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
    #: The frame to cut from — the event's, unless the actor was undetected
    #: there and the answer points at a nearby one.
    frame: int
    #: Whether an IoU snap onto a stored detection may still apply. False when
    #: no stored detection IS this player: snapping could then only attach the
    #: occluder that the silhouettes just ruled out.
    snap: bool


def crop_target(
    stem: str,
    record: dict,
    track: TrackRef | None,
    fallback: CropTarget | None,
    *,
    masks: TrackMasks | None = None,
) -> CropTarget | None:
    """Where an answer says to crop, resolving a tracklet if that is the answer.

    A tracklet is re-resolved from the tracklet every time (see
    extraction/links.resolve_track), so a re-extraction with fresh detections
    self-heals instead of IoU-guessing which box the old answer meant. The
    same function serves an automatic tracklet pick and a hand-placed one, so
    the two cannot drift into cropping different pixels for one tracklet.

    ``fallback`` is what the answer means without a resolvable tracklet: the
    box a human clicked (which stays meaningful — re-tracking renumbers every
    ``track_id``), or nothing at all for a policy that can simply abstain.
    """
    if track is not None:
        pick = resolve_track(stem, record, track, masks=masks)
        if pick is not None:
            return CropTarget(pick.box, pick.frame, pick.snap)
    return fallback


def label_target(
    stem: str,
    record: dict,
    label: ActorLabel,
    masks: TrackMasks | None = None,
) -> CropTarget | None:
    """Where a human's verdict says to crop.

    Its box is only the anchor when it names a tracklet, and takes over when
    the tracklet cannot be resolved — ``track_id`` restarts per rally, so
    re-tracking renumbers everything. Falling back to what the human actually
    clicked keeps the label meaningful; dropping it would not.
    """
    return crop_target(
        stem,
        record,
        label.track,
        (
            CropTarget(
                label.box,
                label.frame if label.frame is not None else record["frame"],
                label.snap,
            )
            if label.box is not None
            else None
        ),
        masks=masks,
    )


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
    # Keypoints ship as crop-relative data; the UI draws the skeleton as a
    # toggleable overlay, so the jpg stays raw — identical to what the
    # embedder sees.
    keypoints = None
    if person.keypoints is not None and person.keypoint_conf is not None:
        cw, ch = max(dx1 - dx0, 1), max(dy1 - dy0, 1)
        keypoints = [
            [round(float(px - dx0) / cw, 4), round(float(py - dy0) / ch, 4), round(float(c), 2)]
            for (px, py), c in zip(person.keypoints, person.keypoint_conf)
        ]
    record.update(
        box=[dx0, dy0, dx1, dy1],
        # The raw detector box (the display box is a padded superset): the
        # seg masker and the event->tracklet link both need the tight box.
        actor_box=[x0, y0, x1, y1],
        score=person.score,
        crop=crop_file.name,
        keypoints=keypoints,
    )
    # Which frame the pixels came from is part of pointing at them: absent
    # means "the event's own", and a stale value would send every later
    # reader — the tracklet link, the next re-crop — to the wrong frame.
    if cross_frame:
        record["crop_frame"] = source_frame
    else:
        record.pop("crop_frame", None)
    return crop
