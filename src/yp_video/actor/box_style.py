"""Which RF-DETR Seg 2XLarge box an actor box is, and where a label's box is
on its event's frame.

Actor labels are boxes from the dense pass (person/dense.py) — the box style
the person head learns. A box from anywhere else (an automatic pick's
segmentation box) is replaced by the dense box it clearly matches; when none
does, or another person's box competes, nothing is guessed. The person-head
snapshot (yp-spot scripts/prepare_person_action.py), the confirmation of an
automatic pick (extraction/done.py) and the Association Label box check
(actor/box_check.py) all decide by these functions, so the queue a reviewer
works through holds every box training leaves unsnapped.

A label's box is on its own ``frame``. When an action edit has since moved
the event, the box is followed to the event's frame one dense frame at a
time by the same clear-match rule (``resolve_target``), and every consumer
that needs "the actor on the event frame" asks ``event_box``.

Boxes in the rule are normalized xyxy lists; ``people`` maps a frame to every
dense box on it at or above DENSE_SCORE_FLOOR. A frame absent from ``people``
was not covered by the pass.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from yp_video.actor.labels import ActorLabel, Box
from yp_video.actor.person_labels import DensePass
from yp_video.core.cache import StatCache
from yp_video.person.dense import DENSE_SCORE_FLOOR, dense_path
from yp_video.person.detector import iou

# A box takes the dense box it overlaps by this much ...
SNAP_MIN_IOU = .5
# ... unless another person's box (not a re-detection of the same one) comes
# within this margin of the same overlap.
SNAP_MARGIN = .15
DUPLICATE_IOU = .7

SNAPPED = "snapped"
#: Another person's box overlaps about as well as the best one.
CONTESTED = "contested"
#: No dense box overlaps the box by SNAP_MIN_IOU.
UNMATCHED = "unmatched"
#: The dense pass did not cover the event frame.
NOT_COVERED = "not_covered"

#: How ``resolve_target`` reached the event frame: the label is on it ...
EVENT_FRAME = "event_frame"
#: ... or was followed there from its own frame ...
FOLLOWED = "followed"
#: ... or lost the person on the way.
UNRESOLVED = "unresolved"
OCCLUDED = "occluded"

# Whole-video dense passes are a few MB of npz each and decompress several
# times larger; a labeling session touches a handful of videos at a time.
_dense_cache = StatCache(max_source_bytes=32 * 1024 * 1024)


def normalize(box, width: float, height: float) -> list[float]:
    b = np.asarray(box, dtype=float) / [width, height, width, height]
    if b.shape != (4,) or not np.isfinite(b).all():
        raise ValueError('Invalid actor box')
    b = b.clip(0, 1)
    if not (b[2:] > b[:2]).all():
        raise ValueError('Empty actor box')
    return b.tolist()


def pixels(box: list[float], width: float, height: float) -> Box:
    return (round(box[0] * width, 1), round(box[1] * height, 1),
            round(box[2] * width, 1), round(box[3] * height, 1))


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
    """The label's normalized box on event ``frame``, and how it got there.

    On its own frame the label's box is authoritative. Drawn on another, it
    is followed to the event frame one frame at a time, and an unclear step
    leaves the event UNRESOLVED rather than guessed. ``people`` must cover
    every frame between the two."""
    if label.box is None:
        return None, OCCLUDED
    box = normalize(label.box, width, height)
    if label.frame == frame:
        return box, EVENT_FRAME
    at, step = label.frame, 1 if frame > label.frame else -1
    box = snap(box, people.get(at, []))
    while box is not None and at != frame:
        at += step
        box = snap(box, people.get(at, []))
    return (box, FOLLOWED) if box is not None else (None, UNRESOLVED)


def dense_pass(stem: str) -> DensePass | None:
    """One video's dense pass, cached on its file; None when it has none."""
    path = dense_path(stem)
    try:
        return _dense_cache.get(stem, [path], lambda: DensePass(path))
    except FileNotFoundError:
        return None


def event_box(stem: str, label: ActorLabel, frame: int) -> Box | None:
    """``label``'s pixel box on event ``frame``, or None when there is none.

    None for an occluded verdict, and for a label from another frame whose
    person the dense boxes lose on the way (or a video without a dense pass
    to follow it through) — the event is then unresolved, never guessed.
    """
    if label.box is None or label.frame == frame:
        return label.box
    dense = dense_pass(stem)
    if dense is None:
        return None
    width, height = dense.meta["frame_size"]
    lo, hi = sorted((label.frame, frame))
    box, _ = resolve_target(label, frame, dense.people(range(lo, hi + 1)), width, height)
    return None if box is None else pixels(box, width, height)


def settle_box(stem: str, box: Box, frame: int) -> Box:
    """``box`` (pixels on ``frame``) as the dense box it clearly is, else as is."""
    dense = dense_pass(stem)
    if dense is None:
        return box
    width, height = dense.meta["frame_size"]
    on_frame = dense.at(frame, DENSE_SCORE_FLOOR)
    snapped, _ = settle(normalize(box, width, height), None if on_frame is None else on_frame[0])
    return box if snapped is None else pixels(snapped, width, height)
