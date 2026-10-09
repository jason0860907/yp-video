"""Which RF-DETR Seg 2XLarge box a human actor box is — the person head's box style.

Human actor boxes were picked from Medium detections (or drawn), but the
person head learns the dense pass's 2XLarge boxes (person/dense.py). An
event's human box is replaced by the dense box it clearly matches; when none
does, or another person's box competes, nothing is guessed. The actor
snapshot (yp-spot scripts/prepare_person_action.py) and the Association
Label box check (actor/box_check.py) both decide by these functions, so the
queue a reviewer works through holds every box training leaves unsnapped.

Boxes here are normalized xyxy lists; ``people`` maps a frame to every dense
box on it at or above DENSE_SCORE_FLOOR. A frame absent from ``people`` was
not covered by the pass.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from yp_video.actor.labels import ActorLabel, ActorVerdict
from yp_video.person.detector import iou

# A human box takes the dense box it overlaps by this much ...
SNAP_MIN_IOU = .5
# ... unless another person's box (not a re-detection of the same one) comes
# within this margin of the same overlap.
SNAP_MARGIN = .15
DUPLICATE_IOU = .7

SNAPPED = "snapped"
#: Another person's box overlaps about as well as the best one.
CONTESTED = "contested"
#: No dense box overlaps the human box by SNAP_MIN_IOU.
UNMATCHED = "unmatched"
#: The dense pass did not cover the event frame.
NOT_COVERED = "not_covered"
#: A box drawn on another frame lost the person on the way to the event frame.
CROSS_FRAME_UNRESOLVED = "cross_frame_unresolved"


def normalize(box, width: float, height: float) -> list[float]:
    b = np.asarray(box, dtype=float) / [width, height, width, height]
    if b.shape != (4,) or not np.isfinite(b).all():
        raise ValueError('Invalid actor box')
    b = b.clip(0, 1)
    if not (b[2:] > b[:2]).all():
        raise ValueError('Empty actor box')
    return b.tolist()


def match(box: list[float], dense: list[list[float]]) -> tuple[list[float] | None, str]:
    """The dense box ``box`` clearly is with SNAPPED, else None and why not."""
    overlaps = [iou(box, d) for d in dense]
    if not overlaps or max(overlaps) < SNAP_MIN_IOU:
        return None, UNMATCHED
    best = max(range(len(dense)), key=overlaps.__getitem__)
    rivals = [o for i, (o, d) in enumerate(zip(overlaps, dense))
              if i != best and o >= SNAP_MIN_IOU and iou(dense[best], d) < DUPLICATE_IOU]
    if rivals and overlaps[best] - max(rivals) < SNAP_MARGIN:
        return None, CONTESTED
    return dense[best], SNAPPED


def snap(box: list[float], dense: list[list[float]]) -> list[float] | None:
    """The dense box ``box`` clearly is, or None."""
    return match(box, dense)[0]


def settle(box: list[float], dense: list[list[float]] | None) -> tuple[list[float] | None, str]:
    """``match`` on the event frame, whose ``dense`` is None when not covered."""
    return (None, NOT_COVERED) if dense is None else match(box, dense)


def resolve_target(
    label: ActorLabel,
    frame: int,
    people: Mapping[int, list[list[float]]],
    width: float,
    height: float,
) -> tuple[list[float] | None, str]:
    """Normalized event-frame box, never a nearby box silently treated as current.

    The saved human-approved box is authoritative at its own frame; drawn on
    another frame, it is followed to the event frame one frame at a time, and
    an unclear step leaves the event unresolved rather than guessed."""
    if label.verdict == ActorVerdict.OCCLUDED:
        return None, 'occluded'
    if label.box is None:
        return None, 'missing_box'
    box = normalize(label.box, width, height)
    if label.frame is None or label.frame == frame:
        return box, 'human_box_exact_frame'
    at, step = label.frame, 1 if frame > label.frame else -1
    box = snap(box, people.get(at, []))
    while box is not None and at != frame:
        at += step
        box = snap(box, people.get(at, []))
    return box, 'cross_frame_resolved' if box is not None else CROSS_FRAME_UNRESOLVED
