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
from yp_video.actor.box_style import event_box, settle_box
from yp_video.actor.labels import ActorLabel, ActorVerdict
from yp_video.actor.resolution import ActorResolution, actor_resolution
from yp_video.core import label_done
from yp_video.core.jsonl import read_jsonl_cached
from yp_video.extraction.store import labelable, records_path
from yp_video.reid import identity


def confirmations_for(stem: str, records: Sequence[dict]) -> dict[str, ActorLabel]:
    """Every automatic pick a human could endorse, as the ``confirmed_auto``
    label it would be.

    The label snapshots the pick's box on the event frame, in the 2XLarge
    style every actor label has (actor/box_style.py): the dense box it
    clearly is, or the pick's own box when none is. A later re-extraction
    cannot then quietly reinterpret what was endorsed. A pick cut from
    another frame (the policy's tracklet never reached the event) is followed
    to the event frame first, and is not endorsable when it cannot be.

    WHO may endorse them is the caller's question, and the two labeling pages
    answer it differently: naming the crop (ReID Label) and reviewing the
    video (Association Label) are both evidence a human looked.
    """
    out: dict[str, ActorLabel] = {}
    for record in records:
        try:
            resolution = actor_resolution(record)
        except ValueError:
            continue  # unmigrated record; never guess what it was
        if resolution is not ActorResolution.AUTO or record.get("actor_box") is None:
            continue
        frame = int(record["frame"])
        picked = ActorLabel(
            ActorVerdict.CONFIRMED_AUTO,
            frame=int(record.get("crop_frame") or frame),
            box=tuple(float(v) for v in record["actor_box"]),
        )
        box = event_box(stem, picked, frame)
        if box is not None:
            out[str(record["id"])] = ActorLabel(
                ActorVerdict.CONFIRMED_AUTO, frame=frame, box=settle_box(stem, box, frame)
            )
    return out


def confirmable_actors(
    stem: str, records: Sequence[dict]
) -> dict[str, ActorLabel]:
    """Automatic picks this page is entitled to confirm.

    Here the endorsement is the player name: naming an identity means the
    user looked at that crop and called the person by name, which is also a
    statement that the right person was cropped. Naming a whole tracklet
    writes the name onto each of its events, so it counts the same. An
    unnamed auto pick is output nobody has looked at, so Done leaves it
    alone; the Association Label page confirms those, on its own evidence.
    """
    assignments = identity.load_assignments(stem)
    return {
        event_id: label
        for event_id, label in confirmations_for(stem, records).items()
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
    label_done.set_done(stem, "reid", done)
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
        actor_labels.confirm_auto(stem, confirmations_for(stem, records))
    )


