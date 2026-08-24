"""Resolving a box back to the tracklet it belongs to.

Pure geometry over tracklet records — no files, no records, no labels. The
callers that DO know about those live one layer up (extraction/links.py),
which is what lets ``actor`` and ``reid`` both use this without either
importing the other's storage.

``TrackletIndex`` is the shape everything reads tracklets through. The raw
jsonl is a list of ~1200 tracklets holding ~180k stored detections between
them, and every consumer asks the same two questions of it — "who was
detected near frame N" and "where is tracklet R" — once per event, ~300
times per video. Asked of the list those are scans; asked of the index they
are dict lookups.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Sequence

from yp_video.person.detector import iou

#: A candidate must lie this far inside the display box to count as the same
#: person. Deliberately containment and not IoU: the display box is a union of
#: the segmentation detector box and contact point, so it is a superset
#: of the track box and IoU would punish it for being big.
LINK_MIN_CONTAINMENT = 0.5

#: How much a loose box must overlap a tracklet's box to be calling the same
#: person. One number, here, because three layers ask it of the same data and
#: an answer that differs between them is a silent disagreement about WHO was
#: named: extraction resolves a picked tracklet to a croppable box, training
#: resolves a labelled box to its tracklet, and the evaluator resolves a
#: policy's box answer before scoring it.
BOX_MATCH_IOU = 0.3

#: How far ahead of the runner-up a re-derived answer must be before it may
#: stand in for one a person gave. Overlapping players each match the other's
#: box (6.7% of picks — see extraction/links.py), and there a wrong answer is
#: worse than none: it silently reassigns a deliberate pick, where no answer
#: becomes visible re-pick work instead.
LINK_MIN_MARGIN = 0.1


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
    #: How far the winner must beat the runner-up by. Zero — the default —
    #: takes the best candidate outright, which is what a policy's box answer
    #: wants: some answer beats none. Re-deriving a pick a PERSON made sets
    #: LINK_MIN_MARGIN instead, because there a near-tie is not an answer.
    margin: float = 0.0


def containment(track_box: Sequence[float], display_box: Sequence[float]) -> float:
    """Fraction of the track box's area inside the display box."""
    ix0, iy0 = max(track_box[0], display_box[0]), max(track_box[1], display_box[1])
    ix1, iy1 = min(track_box[2], display_box[2]), min(track_box[3], display_box[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area = max(1.0, (track_box[2] - track_box[0]) * (track_box[3] - track_box[1]))
    return inter / area


class TrackWindow(NamedTuple):
    """One tracklet's presence inside a frame window.

    ``rows`` index the tracklet's parallel ``frames``/``boxes``/``scores``
    arrays (and its mask rows), ascending — the caller slices them directly.
    """

    ref: TrackRef
    tracklet: dict
    rows: list[int]


class TrackletIndex:
    """One video's tracklets, indexed by frame and by identity.

    Built once per video and shared: the alternative — and what every
    consumer used to do — is a full scan of the tracklet list per event,
    which is ~180k Python iterations to find the ~9 tracklets alive around
    one action.

    A tracklet is detected at most once per frame, so a frame maps to at most
    one row of each tracklet and the returned rows never repeat a player.
    """

    def __init__(self, tracklets: Sequence[dict]):
        self._tracklets = tuple(tracklets)
        self._refs = tuple(
            TrackRef(t["rally_id"], t["track_id"]) for t in self._tracklets
        )
        by_frame: dict[int, list[tuple[int, int]]] = {}
        for position, tracklet in enumerate(self._tracklets):
            for row, frame in enumerate(tracklet["frames"]):
                by_frame.setdefault(frame, []).append((position, row))
        self._by_frame = by_frame
        self._by_ref = {ref: i for i, ref in enumerate(self._refs)}

    def __len__(self) -> int:
        """How many tracklets the video has — the candidate universe."""
        return len(self._tracklets)

    def tracklet(self, ref: TrackRef) -> dict | None:
        """The record for one tracklet, or None when the video has no such
        pair. Absent rather than raising: ``track_id`` restarts per rally, so
        a label written before re-tracking may name a pair that no longer
        exists."""
        position = self._by_ref.get(ref)
        return self._tracklets[position] if position is not None else None

    def at(self, frame: int) -> list[tuple[TrackRef, list[float]]]:
        """Every tracklet detected on exactly ``frame``, with its box there."""
        return [
            (self._refs[position], self._tracklets[position]["boxes"][row])
            for position, row in self._by_frame.get(frame, ())
        ]

    def at_box(
        self, frame: int, box: Sequence[float] | None
    ) -> TrackRef | None:
        """The tracklet ``box`` names on ``frame``, by overlap, or None.

        None is an answer, not a failure: a box drawn on somebody tracking
        never found has no tracklet, and inventing the nearest one would put
        a stranger's identity on the label.
        """
        if box is None:
            return None
        best, best_overlap = None, BOX_MATCH_IOU
        for ref, track_box in self.at(frame):
            overlap = iou(list(box), list(track_box))
            if overlap >= best_overlap:
                best, best_overlap = ref, overlap
        return best

    def nearest(self, frame: int, *, window: int) -> list[tuple[TrackRef, list[float]]]:
        """The nearest detected frame's tracklets, searching outward.

        Nearest rather than merged: at stride > 1 the exact frame may simply
        not have been detected, and one real frame's boxes are the truth —
        pooling a window would list the same player two or three times.
        """
        for offset in sorted(range(-window + 1, window), key=abs):
            if frame + offset in self._by_frame:
                return self.at(frame + offset)
        return []

    def near(self, frame: int, *, window: int) -> list[TrackWindow]:
        """Every tracklet detected within ``window`` frames, and where.

        Merged, unlike ``nearest``: a candidate set wants everything the
        tracklet did around the event, not one frame's snapshot of it.
        """
        found: dict[int, list[int]] = {}
        for at in range(frame - window, frame + window + 1):
            for position, row in self._by_frame.get(at, ()):
                found.setdefault(position, []).append(row)
        # Tracklet order, so a candidate set is the same list however the
        # window was walked — a learned ranker's tie-breaks read the order.
        return [
            TrackWindow(self._refs[position], self._tracklets[position], found[position])
            for position in sorted(found)
        ]


def anchor_names_another(
    index: TrackletIndex,
    ref: TrackRef,
    frame: int,
    box: Sequence[float],
    *,
    stride: int = 1,
) -> bool:
    """Whether the anchor positively names a tracklet OTHER than ``ref``.

    The question a stored pick has to answer after a re-track: ``track_id``
    restarts per rally and gets reused, so the pair almost always still
    exists, wearing whoever inherited the number. Existence cannot tell those
    apart. The anchor can — somebody pointed at a person, and those pixels did
    not move.

    Only a positive contradiction counts. A tracklet simply not detected at
    the anchor frame, or an anchor nothing clears ``BOX_MATCH_IOU`` against,
    is no evidence either way, and no evidence is not grounds to overturn a
    pick a person made. Measured over the real labels that silence covers
    1.6% of them, against 0.2% the anchor actually contradicts.

    Best match rather than merely a passing one: two players standing on top
    of each other both clear ``BOX_MATCH_IOU`` against either's box, so a
    threshold alone would go on honouring a swapped id.
    """
    best, best_overlap = None, BOX_MATCH_IOU
    for candidate, track_box in index.nearest(frame, window=stride):
        overlap = iou(list(track_box), list(box))
        if overlap >= best_overlap:
            best, best_overlap = candidate, overlap
    return best is not None and best != ref


def link_boxes(
    index: TrackletIndex,
    queries: Sequence[BoxQuery],
    *,
    stride: int = 1,
) -> dict[str, TrackRef]:
    """Resolve each query to the tracklet its box sits on, where one does.

    A query resolves to nothing when no tracklet was detected near its frame,
    when the best candidate is not contained in the gate box, or when it does
    not beat the runner-up by the query's margin — an absent answer, never a
    bad one.
    """
    out: dict[str, TrackRef] = {}
    for query in queries:
        candidates = index.nearest(query.frame, window=stride)
        if not candidates:
            continue
        ranked = sorted(candidates, key=lambda c: iou(c[1], query.anchor), reverse=True)
        ref, box = ranked[0]
        if query.margin:
            runner_up = iou(ranked[1][1], query.anchor) if len(ranked) > 1 else 0.0
            if iou(box, query.anchor) - runner_up < query.margin:
                continue
        if containment(box, query.gate) >= LINK_MIN_CONTAINMENT:
            out[query.key] = ref
    return out
