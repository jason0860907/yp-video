"""The numeric contract for ranking TRACKLETS against a contact point.

A separate contract from the box ranker's on purpose. A tracklet is a
different domain object: it exists over time, so it has features a single box
cannot have (how long it lived, whether it was even detected at the event
frame, whether it was moving toward the ball), and several box features have
no tracklet meaning at all. Overloading one name list would force every
feature to invent a reading for the other side — which is exactly how
``has_wrist`` ended up in the box contract as a constant that never moved.

Two things the box contract got wrong are fixed here rather than inherited:

- ``detection_score`` is CLAMPED. RF-DETR's confidence is not a probability
  (measured max 3.79, 13% above 1.0), and it was the only feature with no
  bound while every other one was capped.
- No dead features. Anything constant across the corpus is a weight the
  optimizer can never move.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from yp_video.actor.ranking import X_PAD_FRAC, Y_ABOVE_FRAC
from yp_video.person.detector import WRIST_IDXS, iou
from yp_video.tracklets.geometry import TrackletIndex, TrackRef

#: Frames either side of the event a tracklet may be sampled over. Wide
#: enough to see a spiker's approach, short enough to stay the same rally.
WINDOW = 5
#: A stored detection must overlap the track box this much to be treated as
#: the same person (and lend its keypoints).
DET_MATCH_IOU = 0.3

#: A contact point this far outside every silhouette is "nowhere near anyone";
#: past it the exact value stops meaning anything, so it is clamped.
MASK_DISTANCE_CAP = 6.0

TRACK_CANDIDATE_FEATURE_NAMES = (
    # Presence — the tracklet's own answer to "were you even there".
    "present_at_event",
    "frame_gap",
    "window_coverage",
    # Geometry against the contact point, at the sampled frame.
    "contact_dx_width",
    "contact_dy_height",
    "center_distance_height",
    "contact_in_box",
    "min_distance_in_window",
    # The same question asked of the SILHOUETTE rather than the box. A box
    # contains the contact point for two players at once in a third of all
    # events; the segmenter's outline resolves those, because it knows which
    # of the two bodies the pixel under the ball actually belongs to.
    "mask_distance_height",
    # Time — what a box cannot say.
    "track_length_log",
    "approach_speed",
    # Confidence, clamped.
    "score_at_event",
    "score_median",
    # What the extraction detector saw at the same place.
    "det_iou",
    "wrist_distance_height",
)

TRACK_CONTEXT_FEATURE_NAMES = (
    "bias",
    "log_track_count",
    "top_center_distance",
    "top_score_median",
    "top_two_margin",
    "present_fraction",
    "event_visible",
    "no_track_alive",
    # Abstention is a question about the BEST candidate, so it has to see the
    # measurements that actually separate an actor from a bystander. Centre
    # distance alone cannot: it moves with height and stance, and on occluded
    # events the nearest player's centre looks unremarkable while their hands
    # sit a body-height away from the ball.
    "top_wrist_distance",
    "top_mask_distance",
)


@dataclass(frozen=True)
class TrackCandidate:
    """One tracklet, reduced to what it did around one event."""

    ref: TrackRef
    #: (frame, box, score) inside the window, nearest-first is NOT assumed.
    frames: Sequence[int]
    boxes: Sequence[Sequence[float]]
    scores: Sequence[float]
    #: Box-cropped silhouettes aligned with ``frames``, or () when the video
    #: was tracked without them. Carried on the candidate rather than looked
    #: up later so a candidate stays the whole answer to "what did this
    #: tracklet do around this event".
    masks: Sequence[np.ndarray] = ()


@dataclass(frozen=True)
class TrackFeatures:
    """Variable-length candidates + one fixed-size NONE context.

    Structurally the same as the box contract's so the model and the training
    loop take either — only the NAMES differ, which is what the checkpoint
    records so a loader can tell them apart.
    """

    refs: tuple[TrackRef, ...]
    candidates: np.ndarray
    context: np.ndarray


def candidates_near(
    index: TrackletIndex,
    frame: int,
    *,
    window: int = WINDOW,
    masks: Mapping[str, np.ndarray | None] | None = None,
) -> list[TrackCandidate]:
    """Every tracklet detected within ``window`` frames of ``frame``.

    This is the whole candidate set: ~9 tracklets against the ~62 boxes the
    box ranker had to choose from, and without the spectators — a tracklet
    only exists inside a rally span.

    Takes the video's ``TrackletIndex`` rather than its tracklet list: this
    runs once per event, and finding ~9 live tracklets by scanning ~180k
    stored detections is the difference between seconds and minutes per video.

    ``masks`` maps a tracklet key to its silhouettes for the WHOLE video, rows
    aligned with that tracklet's frames; the window's rows are sliced out here
    so nothing downstream has to know that alignment.
    """
    out: list[TrackCandidate] = []
    for ref, tracklet, rows in index.near(frame, window=window):
        silhouettes = masks.get(ref.key) if masks is not None else None
        out.append(
            TrackCandidate(
                ref=ref,
                frames=[tracklet["frames"][i] for i in rows],
                boxes=[tracklet["boxes"][i] for i in rows],
                scores=[tracklet["scores"][i] for i in rows],
                masks=(
                    tuple(silhouettes[i] for i in rows if i < len(silhouettes))
                    if silhouettes is not None
                    else ()
                ),
            )
        )
    return out


def _centre_distance(box: Sequence[float], x: float, y: float) -> float:
    height = max(float(box[3] - box[1]), 1.0)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return float(np.hypot(x - cx, y - cy)) / height


def _mask_distance(
    mask: np.ndarray | None,
    box: Sequence[float],
    x: float,
    y: float,
    fallback: float,
) -> float:
    """Distance from the contact point to this player's OUTLINE, in heights.

    The stored silhouette is the tracklet's box content resampled to a fixed
    grid, so a cell maps back to image space through the box it was cut from.
    Zero means the point lands on the player.

    ``fallback`` is used when the video was tracked without masks or the
    segmenter produced nothing for this frame — the box-centre distance, which
    measures the same thing more crudely, rather than the cap. A "missing"
    sentinel would need a companion has_mask column that is constant 1.0 on
    any corpus tracked since masks existed: a weight the optimizer can never
    move, which is the one thing this contract refuses to carry.
    """
    if mask is None or not mask.any():
        return min(fallback, MASK_DISTANCE_CAP)
    x0, y0, x1, y1 = (float(v) for v in box)
    width, height = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    rows, columns = mask.shape

    # On the player is zero, exactly. Measuring to the nearest cell CENTRE
    # instead would charge a point standing on the silhouette up to half a
    # cell of distance, which is grid resolution leaking into the feature.
    if x0 <= x <= x1 and y0 <= y <= y1:
        column = min(int((x - x0) / width * columns), columns - 1)
        row = min(int((y - y0) / height * rows), rows - 1)
        if mask[row, column]:
            return 0.0

    ys, xs = np.nonzero(mask)
    cell_x = x0 + (xs + 0.5) * width / columns
    cell_y = y0 + (ys + 0.5) * height / rows
    nearest = float(np.hypot(cell_x - x, cell_y - y).min())
    return min(nearest / max(height, 1.0), MASK_DISTANCE_CAP)


def _wrist_distance(detections: Sequence[dict], box: Sequence[float], x: float, y: float):
    """(IoU, wrist distance) of the stored detection that IS this tracklet.

    The tracklet knows where the player is; only the extraction detector
    knows where their hands are, so the two are joined here.
    """
    best, best_iou = None, DET_MATCH_IOU
    for d in detections:
        overlap = iou(d["box"], list(box))
        if overlap >= best_iou:
            best, best_iou = d, overlap
    if best is None:
        return 0.0, 4.0
    keypoints = best.get("keypoints")
    if not keypoints:
        return best_iou, 4.0
    height = max(float(box[3] - box[1]), 1.0)
    distance = min(
        float(np.hypot(keypoints[i][0] - x, keypoints[i][1] - y)) / height
        for i in WRIST_IDXS
    )
    return best_iou, min(distance, 4.0)


def _candidate_row(
    candidate: TrackCandidate,
    x: float,
    y: float,
    event_frame: int,
    detections: Sequence[dict],
) -> list[float]:
    frames = list(candidate.frames)
    nearest = min(range(len(frames)), key=lambda i: abs(frames[i] - event_frame))
    box = candidate.boxes[nearest]
    x0, y0, x1, y1 = (float(v) for v in box)
    width, height = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    gap = abs(frames[nearest] - event_frame)

    # The rule's box test, kept as a feature so the model starts from what the
    # rule already knows instead of rediscovering it.
    in_box = float(
        x0 - X_PAD_FRAC * width <= x <= x1 + X_PAD_FRAC * width
        and y0 - Y_ABOVE_FRAC * height <= y <= y1
    )
    distances = [_centre_distance(b, x, y) for b in candidate.boxes]

    # Did it close on the ball? A spiker approaches; a bystander does not.
    span = max(frames[-1] - frames[0], 1)
    approach = (distances[0] - distances[-1]) / span * WINDOW if len(frames) > 1 else 0.0

    scores = [min(max(float(s), 0.0), 1.0) for s in candidate.scores]
    det_iou, wrist = _wrist_distance(detections, box, x, y)
    centre = min(_centre_distance(box, x, y), 6.0)
    mask = (
        candidate.masks[nearest] if nearest < len(candidate.masks) else None
    )

    return [
        float(gap == 0),
        min(float(gap), float(WINDOW)),
        len(frames) / (2.0 * WINDOW + 1.0),
        min(abs(x - (x0 + x1) / 2) / width, 4.0),
        min((y - y0) / height, 4.0),
        centre,
        in_box,
        min(min(distances), 6.0),
        _mask_distance(mask, box, x, y, centre),
        float(np.log1p(len(frames))),
        float(np.clip(approach, -4.0, 4.0)),
        scores[nearest],
        float(np.median(scores)),
        det_iou,
        wrist,
    ]


def extract_track_features(
    candidates: Sequence[TrackCandidate],
    x: float,
    y: float,
    event_frame: int,
    *,
    detections: Sequence[dict] = (),
    visible: bool = True,
) -> TrackFeatures:
    """The versioned numeric contract shared by training and inference."""
    rows = [
        _candidate_row(candidate, x, y, event_frame, detections)
        for candidate in candidates
    ]
    matrix = np.asarray(rows, dtype=np.float64).reshape(
        len(rows), len(TRACK_CANDIDATE_FEATURE_NAMES)
    )

    column = TRACK_CANDIDATE_FEATURE_NAMES.index
    if rows:
        centres = matrix[:, column("center_distance_height")]
        order = np.argsort(centres)
        top = int(order[0])
        margin = (
            float(centres[int(order[1])] - centres[top]) if len(order) > 1 else 6.0
        )
        context = [
            1.0,
            float(np.log1p(len(rows))),
            float(centres[top]),
            float(matrix[top, column("score_median")]),
            min(margin, 6.0),
            float(matrix[:, column("present_at_event")].mean()),
            float(visible),
            0.0,
            float(matrix[top, column("wrist_distance_height")]),
            float(matrix[top, column("mask_distance_height")]),
        ]
    else:
        # No tracklet alive at all — the ~7% of events where the answer is
        # very likely "nobody". The model is given the fact explicitly so it
        # can learn to abstain rather than infer it from an empty list.
        context = [1.0, 0.0, 6.0, 0.0, 0.0, 0.0, float(visible), 1.0, 4.0, 6.0]

    return TrackFeatures(
        refs=tuple(c.ref for c in candidates),
        candidates=matrix,
        context=np.asarray(context, dtype=np.float64),
    )
