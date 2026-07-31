"""Every human-reviewed event, joined to what a policy would need to decide.

`build_track_dataset` already performs this join, but folds it straight into
feature vectors; a policy evaluator needs the same join and none of the
features. Sharing it here keeps one definition of "a reviewed event" — two
would drift, and the drift would show up as a policy that scores well against
a slightly different question than the one the ranker was scored on.

Note that `reassociate` deliberately SKIPS labelled events, so an evaluator
cannot reuse it: scoring a policy means asking it exactly the questions a
human already answered.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from yp_video.actor import labels as actor_labels
from yp_video.actor.labels import ActorLabel, ActorVerdict
from yp_video.actor.policy import EventContext
from yp_video.core.jsonl import read_jsonl_cached, read_jsonl_header
from yp_video.extraction.store import (
    labelable,
    labelable_actions,
    records_path,
)
from yp_video.tracklets.geometry import TrackletIndex, TrackRef
from yp_video.tracklets.store import open_track_masks, tracklet_index, tracks_path


@dataclass(frozen=True)
class ReviewedEvent:
    """One event a human ruled on, and everything a policy may look at."""

    stem: str
    event_id: str
    record: dict
    label: ActorLabel
    context: EventContext

    @property
    def truth(self) -> TrackRef | None:
        """The tracklet the human named, or None when they saw no actor."""
        return None if self.label.verdict is ActorVerdict.OCCLUDED else self.label.track

    @property
    def is_occluded(self) -> bool:
        return self.label.verdict is ActorVerdict.OCCLUDED

    @property
    def candidate_count(self) -> int:
        return len(self.context.tracks) if self.context.tracks is not None else 0


@dataclass(frozen=True)
class ReviewProgress:
    """One video's current Association review progress."""

    event_count: int
    reviewed: int
    unreviewed: int
    verdicts: dict[str, int]
    #: Reviewed, but the verdict names no tracklet (ActorLabel.box_only) —
    #: counted apart because these need re-picking before training sees them.
    box_only: int = 0

    @property
    def started(self) -> bool:
        return self.reviewed > 0

    @property
    def done(self) -> bool:
        return self.event_count > 0 and self.unreviewed == 0


@dataclass(frozen=True)
class ReviewSummary:
    """Video counts shown as ``done / started`` in corpus summaries."""

    done: int
    started: int


@dataclass(frozen=True)
class VideoEvents:
    """One video's labelable events with everything consumers join them to."""

    stem: str
    meta: dict
    records: list[dict]
    tracks: TrackletIndex
    width: int
    height: int
    verdicts: dict[str, ActorLabel]
    masks: Mapping[str, np.ndarray | None] | None


@contextmanager
def open_video_events(stem: str) -> Iterator[VideoEvents]:
    """The per-video join every consumer shares.

    Labelable records, the tracklet index, frame geometry, the human verdicts
    and the open mask archive — opened once, and the archive is closed on
    exit, so consume the events inside the ``with`` block.
    """
    meta, records = read_jsonl_cached(records_path(stem))
    width, height = meta.get("frame_size") or [0, 0]
    masks = open_track_masks(stem)
    try:
        yield VideoEvents(
            stem=stem,
            meta=meta,
            records=labelable(records, stem, float(meta.get("fps") or 0)),
            tracks=tracklet_index(stem),
            width=int(width or 0),
            height=int(height or 0),
            verdicts=actor_labels.load(stem),
            masks=masks,
        )
    finally:
        if masks is not None:
            masks.close()


def review_progress(stem: str, fps: float = 0) -> ReviewProgress:
    """Compare durable labels with the video's current labelable events."""
    current_ids = {
        str(record["id"])
        for record in labelable_actions(stem, fps)
    }
    labels = actor_labels.load(stem)
    verdicts: dict[str, int] = {}
    box_only = 0
    for event_id, label in labels.items():
        if event_id not in current_ids:
            continue
        verdicts[label.verdict.value] = verdicts.get(label.verdict.value, 0) + 1
        box_only += label.box_only
    reviewed_ids = current_ids & set(labels)
    return ReviewProgress(
        event_count=len(current_ids),
        reviewed=len(reviewed_ids),
        unreviewed=len(current_ids - reviewed_ids),
        verdicts=verdicts,
        box_only=box_only,
    )


def review_summary(stems: Sequence[str] | None = None) -> ReviewSummary:
    """Count completed and started Association-labelled videos.

    ``started`` includes both Done and In Progress. Stale label files whose
    events are no longer part of the current Action/Rally sources count as
    neither, matching the Association work list.
    """
    selected = list(stems) if stems is not None else actor_labels.labeled_stems()
    progress = []
    for stem in selected:
        record_file = records_path(stem)
        if not record_file.exists():
            continue
        header = read_jsonl_header(record_file)
        row = review_progress(stem, float(header.get("fps") or 0))
        if row.started:
            progress.append(row)
    return ReviewSummary(
        done=sum(row.done for row in progress),
        started=len(progress),
    )


def iter_reviewed(stems: Sequence[str] | None = None) -> Iterator[ReviewedEvent]:
    """Human-reviewed events, video by video, with their tracklets and masks.

    Consume this as a stream. Each event borrows its video's open mask
    archive, which is closed as soon as the iterator leaves that video —
    collecting the events into a list first leaves the masks behind.
    """
    selected = list(stems) if stems is not None else actor_labels.labeled_stems()
    for stem in selected:
        if not (records_path(stem).exists() and tracks_path(stem).exists()):
            continue
        with open_video_events(stem) as video:
            if not video.verdicts:
                continue
            for record in video.records:
                label = video.verdicts.get(str(record.get("id")))
                if label is None:
                    continue
                yield ReviewedEvent(
                    stem=stem,
                    event_id=str(record.get("id")),
                    record=record,
                    label=label,
                    context=EventContext.for_event(
                        record,
                        width=video.width,
                        height=video.height,
                        tracks=video.tracks,
                        masks=video.masks,
                    ),
                )
