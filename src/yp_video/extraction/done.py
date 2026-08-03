"""Marking a video's labeling finished, and what that implies about actors.

"Done" is a ReID verdict: the user says the player names on this video are
settled. But a user who named every crop has also, implicitly, agreed with
every automatic actor pick behind those crops — they looked at the person and
called them by name. Turning that implication into an explicit
``confirmed_auto`` label is what gives the association model positive
training truth without ever inventing it (see actor/labels.py).

Implicit, so it stays opt-in: ``confirm_auto`` is a parameter, and only
events that actually carry an assignment are confirmed. An unassigned auto
pick is output nobody has looked at.
"""

from __future__ import annotations

from collections.abc import Sequence

from yp_video.actor import labels as actor_labels
from yp_video.actor.labels import ActorLabel
from yp_video.core import label_done
from yp_video.core.jsonl import read_jsonl_cached
from yp_video.extraction.links import track_keys
from yp_video.extraction.store import labelable, records_path
from yp_video.reid import identity
from yp_video.reid import store as reid_store


def confirmable_actors(
    stem: str, records: Sequence[dict]
) -> dict[str, ActorLabel]:
    """Automatic picks this page is entitled to confirm.

    Here the endorsement is the player name: naming an identity means the
    user looked at that crop and called the person by name, which is also a
    statement that the right person was cropped. A name given to the whole
    tracklet counts — it was given while looking at these crops. An unnamed
    auto pick is output nobody has looked at, so Done leaves it alone; the
    Association Label page confirms those, on its own evidence.
    """
    assignments = identity.load_assignments(stem, track_keys(stem))
    return {
        event_id: label
        for event_id, label in actor_labels.confirmations_for(records).items()
        if event_id in assignments
    }


def mark_done(stem: str, done: bool, *, confirm_auto: bool) -> int:
    """Persist the Done verdict; return how many actors it confirmed."""
    confirmed = 0
    if done and confirm_auto:
        meta, records = read_jsonl_cached(records_path(stem))
        records = labelable(records, stem, float(meta.get("fps") or 0))
        confirmed = len(
            actor_labels.confirm_auto(stem, confirmable_actors(stem, records))
        )
    reid_store.save_done(stem, done)
    return confirmed


def confirm_reviewed(stem: str) -> int:
    """Association-Done's standing endorsement, applied to current answers.

    The Association Done flag says a human reviewed this video's actors.
    A predict re-run then invents answers for events that had none — and
    those would arrive unendorsed, un-reviewing a video its reviewer already
    declared finished. So a Done video keeps its endorsement: every current
    automatic answer gets the same ``confirmed_auto``/``occluded`` label the
    per-rally sweep writes, video-wide. Existing verdicts always win
    (actor/labels.confirm_auto). Not-Done videos are left alone — new
    machine output nobody vouched for stays visibly unreviewed.
    """
    if not label_done.is_done(stem, "association"):
        return 0
    path = records_path(stem)
    if not path.exists():
        return 0
    meta, records = read_jsonl_cached(path)
    records = labelable(records, stem, float(meta.get("fps") or 0))
    return len(
        actor_labels.confirm_auto(stem, actor_labels.confirmations_for(records))
    )


def backfill_confirmed_done(stems: Sequence[str]) -> dict[str, int]:
    """Explicit migration for videos marked Done before confirmation existed."""
    counts: dict[str, int] = {}
    for stem in stems:
        if not reid_store.load_done(stem):
            raise ValueError(f"{stem} is not marked Done")
        if not records_path(stem).exists():
            raise FileNotFoundError(f"No extraction records for {stem}")
        counts[stem] = mark_done(stem, True, confirm_auto=True)
    return counts
