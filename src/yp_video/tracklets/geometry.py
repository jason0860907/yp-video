"""Resolving a box back to the tracklet it belongs to.

Pure geometry over tracklet records — no files, no records, no labels. The
callers that DO know about those live one layer up (extraction/links.py),
which is what lets ``actor`` and ``reid`` both use this without either
importing the other's storage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Sequence

from yp_video.person.detector import iou

#: A candidate must lie this far inside the display box to count as the same
#: person. Deliberately containment and not IoU: the display box is a union of
#: the detector box, every keypoint and the contact point, so it is a superset
#: of the track box and IoU would punish it for being big.
LINK_MIN_CONTAINMENT = 0.5


class TrackRef(NamedTuple):
    """A tracklet's identity. The pair — track_id alone restarts per rally."""

    rally_id: int
    track_id: int

    @property
    def key(self) -> str:
        return f"{self.rally_id}:{self.track_id}"

    def payload(self) -> dict:
        return {"rally_id": self.rally_id, "track_id": self.track_id}

    @classmethod
    def parse(cls, key: str) -> "TrackRef":
        rally, _, track = key.partition(":")
        return cls(int(rally), int(track))


@dataclass(frozen=True)
class BoxQuery:
    """One "which tracklet is this box?" question."""

    key: str
    frame: int
    #: The TIGHT detector box — ranks candidates. A padded display box can
    #: fully contain two overlapping players' track boxes, and then whoever is
    #: bigger wins; the tight box discriminates between them.
    anchor: list[float]
    #: The DISPLAY box — gates the winner by containment (see above).
    gate: list[float]


def containment(track_box: Sequence[float], display_box: Sequence[float]) -> float:
    """Fraction of the track box's area inside the display box."""
    ix0, iy0 = max(track_box[0], display_box[0]), max(track_box[1], display_box[1])
    ix1, iy1 = min(track_box[2], display_box[2]), min(track_box[3], display_box[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area = max(1.0, (track_box[2] - track_box[0]) * (track_box[3] - track_box[1]))
    return inter / area


def frame_index(tracklets: Sequence[dict]) -> dict[int, list[tuple[TrackRef, list[float]]]]:
    """frame → every tracklet detected on it, with its box."""
    index: dict[int, list[tuple[TrackRef, list[float]]]] = {}
    for t in tracklets:
        ref = TrackRef(t["rally_id"], t["track_id"])
        for frame, box in zip(t["frames"], t["boxes"]):
            index.setdefault(frame, []).append((ref, box))
    return index


def tracks_near(
    index: dict[int, list[tuple[TrackRef, list[float]]]],
    frame: int,
    *,
    window: int,
) -> list[tuple[TrackRef, list[float]]]:
    """The nearest detected frame's tracklets, searching outward from ``frame``.

    Nearest rather than merged: at stride > 1 the exact frame may simply not
    have been detected, and one real frame's boxes are the truth — pooling a
    window would list the same player two or three times.
    """
    for offset in sorted(range(-window + 1, window), key=abs):
        found = index.get(frame + offset)
        if found:
            return found
    return []


def link_boxes(
    tracklets: Sequence[dict],
    queries: Sequence[BoxQuery],
    *,
    stride: int = 1,
) -> dict[str, TrackRef]:
    """Resolve each query to the tracklet its box sits on, where one does.

    A query resolves to nothing when no tracklet was detected near its frame,
    or when the best candidate is not contained in the gate box — an absent
    answer, never a bad one.
    """
    index = frame_index(tracklets)
    out: dict[str, TrackRef] = {}
    for query in queries:
        candidates = tracks_near(index, query.frame, window=stride)
        if not candidates:
            continue
        ref, box = max(candidates, key=lambda c: iou(c[1], query.anchor))
        if containment(box, query.gate) >= LINK_MIN_CONTAINMENT:
            out[query.key] = ref
    return out
