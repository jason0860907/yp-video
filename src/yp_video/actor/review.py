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

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from yp_video.actor import labels as actor_labels
from yp_video.actor.labels import ActorLabel, ActorVerdict
from yp_video.actor.policy import EventContext
from yp_video.core.jsonl import read_jsonl_cached
from yp_video.extraction.store import records_path
from yp_video.tracklets.geometry import TrackRef
from yp_video.tracklets.store import open_track_masks, tracks_path


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
        return len(self.context.tracklets)


def iter_reviewed(stems: Sequence[str] | None = None) -> Iterator[ReviewedEvent]:
    """Human-reviewed events, video by video, with their tracklets and masks.

    Consume this as a stream. Each event borrows its video's open mask
    archive, which is closed as soon as the iterator leaves that video —
    collecting the events into a list first leaves the masks behind.
    """
    selected = list(stems) if stems is not None else actor_labels.labeled_stems()
    for stem in selected:
        record_file, track_file = records_path(stem), tracks_path(stem)
        if not (record_file.exists() and track_file.exists()):
            continue
        meta, records = read_jsonl_cached(record_file)
        _tmeta, tracklets = read_jsonl_cached(track_file)
        width, height = meta.get("frame_size") or [0, 0]
        verdicts = actor_labels.load(stem)
        if not verdicts:
            continue

        masks = open_track_masks(stem)
        try:
            for record in records:
                label = verdicts.get(str(record.get("id")))
                if label is None:
                    continue
                xy = record.get("xy")
                yield ReviewedEvent(
                    stem=stem,
                    event_id=str(record.get("id")),
                    record=record,
                    label=label,
                    context=EventContext(
                        frame=int(record["frame"]),
                        event_id=str(record.get("id")),
                        contact=(
                            (float(xy[0]) * width, float(xy[1]) * height)
                            if xy and width and height
                            else None
                        ),
                        visible=bool(record.get("visible", True)),
                        detections=record.get("detections") or [],
                        tracklets=tracklets,
                        masks=masks,
                    ),
                )
        finally:
            if masks is not None:
                masks.close()
