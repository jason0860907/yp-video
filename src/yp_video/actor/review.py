"""How far the human Association review of one video has come."""

from __future__ import annotations

from dataclasses import dataclass

from yp_video.actor import labels as actor_labels
from yp_video.extraction.store import labelable_actions


@dataclass(frozen=True)
class ReviewProgress:
    """One video's current Association review progress."""

    event_count: int
    reviewed: int
    unreviewed: int
    verdicts: dict[str, int]

    @property
    def started(self) -> bool:
        return self.reviewed > 0

    @property
    def done(self) -> bool:
        return self.event_count > 0 and self.unreviewed == 0


def review_progress(stem: str, fps: float = 0) -> ReviewProgress:
    """Compare durable labels with the video's current labelable events."""
    current_ids = {
        str(record["id"])
        for record in labelable_actions(stem, fps)
    }
    labels = actor_labels.load(stem)
    verdicts: dict[str, int] = {}
    for event_id, label in labels.items():
        if event_id not in current_ids:
            continue
        verdicts[label.verdict.value] = verdicts.get(label.verdict.value, 0) + 1
    reviewed_ids = current_ids & set(labels)
    return ReviewProgress(
        event_count=len(current_ids),
        reviewed=len(reviewed_ids),
        unreviewed=len(current_ids - reviewed_ids),
        verdicts=verdicts,
    )
