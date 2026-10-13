"""The actor head's training rows: one per action event a human gave an actor.

One ``<stem>_actor_events.jsonl`` per video in the run's ``labels/actor-events``,
built from the run's action label snapshot (event ids, frames, labels) and
the association annotations (actor/labels.py). One row per line:

    {"video", "id", "frame", "label", "boxes", "box_source", "snapped", "verdict"}

``boxes`` are normalized source-frame xyxy on the event frame, in the
RF-DETR Seg 2XLarge style the person head learns (person/dense.py): the
actor's box first, then every other dense box at or above
PERSON_LABEL_MIN_SCORE that is not a duplicate of it — candidates, not
"no action" negatives. The actor's box is the label's box replaced by the
dense box it clearly matches (actor/box_style.py); when none does, or
another person's box competes, the label box is kept as drawn (``snapped``
false), and the Association Label box check queues that event for review.
A label drawn on another frame than its event's (the action moved after the
pick) is followed there through the dense boxes; ``box_source`` says which
(event_frame | followed).

Events left out are counted, not written: occluded or unresolved actors,
events without a human actor (no label, or an agent proposal nobody has
confirmed — actor/labels.py), two touches on one frame (one clip, two
answers), and videos without actor labels or a dense pass. Which rows train
and which validate is the trainer's call: it splits them by the action
stream's videos, so the actor head validates on the run's held-out videos.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from yp_video.actor import labels as actor_labels
from yp_video.actor.box_style import resolve_target, settle
from yp_video.actor.person_labels import PERSON_LABEL_MIN_SCORE, DensePass
from yp_video.contracts.action import ACTOR_LABELS, TASKS
from yp_video.core.jsonl import read_jsonl
from yp_video.person.dense import dense_path
from yp_video.person.detector import iou

ACTOR_FILE_SUFFIX = TASKS["actor"].label_glob.removeprefix("*")
#: A dense box overlapping the actor's this much is the actor again.
DUPLICATE_IOU = .5


def write_actor_labels(action_label_dir: Path, *, label_dir: Path) -> dict:
    """Rows for every video in ``action_label_dir`` (the run's action
    snapshot) into ``label_dir``; returns what was written and skipped."""
    label_dir.mkdir(parents=True, exist_ok=True)
    for stale in label_dir.glob(f"*{ACTOR_FILE_SUFFIX}"):
        stale.unlink()

    counts: Counter = Counter()
    by_class: Counter = Counter()
    for path in sorted(action_label_dir.glob(TASKS["action"].label_glob)):
        meta, events = read_jsonl(path)
        stem = str(meta["video"])
        actors = actor_labels.load(stem)
        if not actors:
            counts["videos_without_actor_labels"] += 1
            continue
        dense_file = dense_path(stem)
        if not dense_file.exists():
            counts["videos_without_dense"] += 1
            continue
        rows = _video_rows(stem, events, actors, DensePass(dense_file), counts)
        if not rows:
            continue
        with (label_dir / f"{stem}{ACTOR_FILE_SUFFIX}").open("w", encoding="utf-8") as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts["videos"] += 1
        counts["events"] += len(rows)
        by_class.update(row["label"] for row in rows)
    return {"label_dir": str(label_dir), "classes": dict(by_class), **counts}


def _video_rows(stem: str, events: list[dict], actors: dict, dense: DensePass, counts: Counter) -> list[dict]:
    ids = [event["id"] for event in events]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate action ids: {stem}")
    counts["orphan_actor_labels"] += len(set(actors) - set(ids))
    width, height = dense.meta["frame_size"]
    touches = Counter(event["frame"] for event in events if event["label"] in ACTOR_LABELS)
    rows = []
    for event in events:
        frame = event["frame"]
        if event["label"] not in ACTOR_LABELS:
            counts["score_events"] += 1
            continue
        if touches[frame] > 1:
            counts["shared_frame"] += 1
            continue
        label = actors.get(event["id"])
        if label is None:
            counts["no_human_actor"] += 1
            continue
        if not label.human:
            counts["agent_proposals"] += 1
            continue
        lo, hi = sorted((frame, frame if label.frame is None else label.frame))
        people = dense.people(range(lo, hi + 1))
        target, source = resolve_target(label, frame, people, width, height)
        counts[source] += 1
        if target is None:
            continue
        snapped, _ = settle(target, people.get(frame))
        if snapped is None:
            counts["unsnapped"] += 1
        else:
            target = snapped
        proposals = dense.at(frame, PERSON_LABEL_MIN_SCORE)
        boxes = [target, *(box for box in (proposals[0] if proposals else [])
                           if iou(target, box) < DUPLICATE_IOU)]
        rows.append({
            "video": stem, "id": event["id"], "frame": frame, "label": event["label"],
            "boxes": boxes, "box_source": source, "snapped": snapped is not None,
            "verdict": label.verdict.value,
        })
    return rows
