"""One-off: rewrite every actor label file to the {verdict, frame, box} shape.

Delete this script (and tests/test_convert_actor_labels_20261009.py) once the
real run has landed — code and data switch together, and nothing reads the
old shape afterwards.

The old labels named a tracklet and kept the clicked box only as an anchor;
the new label is the actor's 2XLarge box on the event frame
(actor/labels.py). Per event of ``videos/association/annotations/*_actors.json``:

- An event id the action annotation no longer has, or one whose action is a
  SKIP_LABELS label (nobody to identify), is dropped and counted.
- ``occluded`` stays ``{"verdict": "occluded"}``.
- Otherwise the box is the record's MATERIALIZED pick — the extraction record
  of the same id whose resolution matches the verdict (manual / auto) and
  carries an ``actor_box``: the box every crop, embedding and tracklet link
  uses today. It sits on the record's ``crop_frame`` when the crop came from
  another frame, else on the record's own frame (which an action edit may
  since have moved away from the event's). When that frame is not the
  event's, the box is followed there through the dense boxes
  (box_style.event_box).
- No usable record: the old label's own box, on its own frame (the event's
  when it has none), followed the same way. The old track is ignored — it is
  exactly the pointer this shape retires.
- The event-frame box is then snapped to the 2XLarge box it clearly is
  (the rule of box_style.settle_box), else kept as is; ``frame`` = the event
  frame.
- A box the dense boxes lose on the way to the event frame keeps its own
  frame (snapped there the same way): the new shape's "followed" case, which
  every reader resolves — or reports unresolved — by itself.

Records are not converted: they are derived data whose picks stay valid (the
same person), and nothing in them is read as a label.

Reports (always under ``--out``): ``summary.json`` (counts per outcome) and
``person_changes.jsonl`` — every label whose new box's centre is not inside
the record pick, or the pick's centre not inside the new box. Those would be a
different person, and must be near zero.

    .venv/bin/python scripts/convert_actor_labels_20261009.py --dry-run --out /tmp/x
    .venv/bin/python scripts/convert_actor_labels_20261009.py --out /tmp/x   # in place
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from yp_video.actor import labels as actor_labels
from yp_video.actor.box_style import dense_pass, event_box, normalize, pixels, settle
from yp_video.actor.labels import SCHEMA_VERSION, ActorLabel, ActorVerdict
from yp_video.config import ASSOCIATION_ANNOTATIONS_DIR
from yp_video.core.jsonl import atomic_write, read_jsonl
from yp_video.extraction.store import SKIP_LABELS, action_annotation_path, records_path
from yp_video.person.dense import DENSE_SCORE_FLOOR

_RESOLUTION = {"manual": "manual", "confirmed_auto": "auto"}


def _inside(box, other) -> bool:
    """Whether ``box``'s centre lies inside ``other``."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return other[0] <= cx <= other[2] and other[1] <= cy <= other[3]


def _settle(stem: str, box: tuple, frame: int) -> tuple[tuple, str]:
    """box_style.settle_box, and why it snapped or not (``no_dense`` when the
    video has no dense pass)."""
    dense = dense_pass(stem)
    if dense is None:
        return box, "no_dense"
    width, height = dense.meta["frame_size"]
    hit = dense.at(frame, DENSE_SCORE_FLOOR)
    snapped, status = settle(normalize(box, width, height), None if hit is None else hit[0])
    return (box if snapped is None else pixels(snapped, width, height)), status


def convert_label(
    stem: str, payload: dict, event: dict | None, record: dict | None
) -> tuple[ActorLabel | None, str, dict | None]:
    """One old label as the new one, the outcome to count, and a
    person-change report row (None when the person did not change)."""
    if event is None:
        return None, "dropped_orphan", None
    if event.get("label") in SKIP_LABELS:
        return None, "dropped_skip_label", None
    verdict = ActorVerdict(payload["verdict"])
    if verdict is ActorVerdict.OCCLUDED:
        return ActorLabel(verdict), "occluded", None

    frame = int(event["frame"])
    usable = (
        record is not None
        and record.get("resolution") == _RESOLUTION[verdict.value]
        and record.get("actor_box") is not None
    )
    if usable:
        source = "record"
        at = int(record.get("crop_frame") or record["frame"])
        box = tuple(float(v) for v in record["actor_box"])
    else:
        source = "label"
        at = int(payload.get("frame") or frame)
        box = tuple(float(v) for v in payload["box"])
    followed = event_box(stem, ActorLabel(verdict, at, box), frame)
    if followed is not None and at != frame and not (_inside(followed, box) and _inside(box, followed)):
        # The walk ended on a box that no longer overlaps where the person
        # was: a person change is possible, so a human decides (the label
        # stays on its own frame, which the box check lists as unresolved).
        followed = None
    if followed is None:
        (on, snapped), outcome_frame = _settle(stem, box, at), at
        how = "kept_own_frame"
    else:
        (on, snapped), outcome_frame = _settle(stem, followed, frame), frame
        how = "event_frame" if at == frame else "followed"
    label = ActorLabel(verdict, outcome_frame, tuple(round(v, 1) for v in on))

    change = None
    if usable:
        old = record["actor_box"]
        if not (_inside(label.box, old) and _inside(old, label.box)):
            change = {
                "video": stem, "verdict": verdict.value, "event_frame": frame,
                "record_frame": at, "record_box": old, "new_frame": label.frame,
                "new_box": list(label.box), "how": how, "snap": snapped,
                "old_label": payload,
            }
    return label, f"{verdict.value}:{source}:{how}:{snapped}", change


def convert_video(stem: str, raw: dict) -> tuple[dict, Counter, list[dict]]:
    """The video's new label file payload, outcome counts and person changes."""
    if raw.get("version") != 2:
        raise ValueError(f"{stem}: expected a version 2 label file, got {raw.get('version')!r}")
    action = action_annotation_path(stem)
    events = read_jsonl(action)[1] if action is not None else []
    by_id = {str(e["id"]): e for e in events if e.get("frame") is not None}
    records = {}
    if records_path(stem).exists():
        _meta, rows = read_jsonl(records_path(stem))
        records = {str(r["id"]): r for r in rows}

    counts: Counter = Counter()
    changes: list[dict] = []
    out = {}
    for event_id, payload in raw["actors"].items():
        label, outcome, change = convert_label(stem, payload, by_id.get(event_id), records.get(event_id))
        counts[outcome] += 1
        if change is not None:
            changes.append({"id": event_id, **change})
        if label is not None:
            out[event_id] = label.payload()
    return {"version": SCHEMA_VERSION, "actors": dict(sorted(out.items()))}, counts, changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="report directory (and converted files on --dry-run)")
    parser.add_argument("--dry-run", action="store_true", help="write converted files under --out, not in place")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    files = args.out / "actors" if args.dry_run else None
    if files is not None:
        files.mkdir(exist_ok=True)

    total: Counter = Counter()
    changes: list[dict] = []
    videos = 0
    for path in sorted(ASSOCIATION_ANNOTATIONS_DIR.glob(f"*{actor_labels.LABEL_SUFFIX}")):
        stem = path.name[: -len(actor_labels.LABEL_SUFFIX)]
        dense = dense_pass(stem)
        if dense is None:
            total["videos_without_dense"] += 1
        elif records_path(stem).exists():
            meta = read_jsonl(records_path(stem))[0]
            if list(meta.get("frame_size") or []) != list(dense.meta["frame_size"]):
                raise ValueError(f"{stem}: records and dense pass disagree on frame size")
        payload, counts, video_changes = convert_video(stem, json.loads(path.read_text(encoding="utf-8")))
        total.update(counts)
        changes.extend(video_changes)
        videos += 1
        target = (files / path.name) if files is not None else path
        with atomic_write(target) as file:
            json.dump(payload, file, ensure_ascii=False, indent=1)

    summary = {"videos": videos, "dry_run": args.dry_run, "person_changes": len(changes),
               "outcomes": dict(sorted(total.items()))}
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
    with (args.out / "person_changes.jsonl").open("w", encoding="utf-8") as file:
        for row in changes:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
