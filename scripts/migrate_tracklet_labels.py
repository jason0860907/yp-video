"""One-off migration of the human labels onto tracklets.

Two files change, both under ``videos/reid/annotations/``:

1. ``<stem>_actors.json`` — every verdict that names a box gets the tracklet
   that box sits on. The box stays as the ANCHOR: ``track_id`` restarts in
   every rally, so re-running tracking renumbers everything, and the anchor
   is what re-derives the label when that happens (see ``--reanchor``).

2. ``<stem>_players.json`` — names move from events onto the tracklet they
   agree on. Where a tracklet's events DISAGREE, nothing is written: that is
   an identity switch mid-track, and picking a winner by majority would bury
   the one thing worth looking at. Those events stay as event overrides and
   are reported as conflicts.

Nothing is invented. A verdict that resolves to no tracklet keeps its box and
keeps working — the box path is the supported fallback, not a leftover.

    uv run python scripts/migrate_tracklet_labels.py            # show the plan
    uv run python scripts/migrate_tracklet_labels.py --apply    # do it
    uv run python scripts/migrate_tracklet_labels.py --reanchor # after re-tracking
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

from yp_video.actor import labels as actor_labels
from yp_video.actor.labels import ActorLabel
from yp_video.config import REID_ANNOTATIONS_DIR
from yp_video.core.jsonl import read_jsonl_cached
from yp_video.extraction.links import track_keys
from yp_video.extraction.store import records_path
from yp_video.reid import identity
from yp_video.tracklets.geometry import BoxQuery, TrackRef, link_boxes
from yp_video.tracklets.store import tracks_path

BACKUP = REID_ANNOTATIONS_DIR.with_name(REID_ANNOTATIONS_DIR.name + ".pre-tracklet")


@dataclass
class StemPlan:
    stem: str
    resolved: dict[str, TrackRef]
    unresolved: list[str]
    occluded: int
    tracks: dict[str, str]
    overrides: dict[str, str]
    conflicts: list[tuple[str, list[str], int]]


def _labels_to_tracks(
    stem: str, labels: dict[str, ActorLabel], *, only_stale: bool = False
) -> tuple[dict[str, TrackRef], list[str]]:
    """Resolve each boxed label to the tracklet its ANCHOR sits on.

    Anchored on what the human clicked, not on the record's current box: the
    record is derived and may be re-cut, the click is the evidence.
    """
    if not tracks_path(stem).exists() or not records_path(stem).exists():
        return {}, sorted(labels)
    tmeta, tracklets = read_jsonl_cached(tracks_path(stem))
    _rmeta, records = read_jsonl_cached(records_path(stem))
    by_id = {r["id"]: r for r in records}
    live = {TrackRef(t["rally_id"], t["track_id"]) for t in tracklets}

    queries, skipped = [], []
    for event_id, label in labels.items():
        if label.box is None:  # occluded — nothing to resolve, by design
            continue
        if only_stale and label.track is not None and label.track in live:
            continue
        record = by_id.get(event_id)
        if record is None:
            skipped.append(event_id)
            continue
        queries.append(
            BoxQuery(
                key=event_id,
                frame=label.frame if label.frame is not None else record["frame"],
                anchor=list(label.box),
                # The display box is the superset the containment gate is
                # written against; without a record we have only the anchor.
                gate=record.get("box") or list(label.box),
            )
        )
    resolved = link_boxes(tracklets, queries, stride=int(tmeta.get("stride") or 1))
    unresolved = sorted(
        {q.key for q in queries} - set(resolved) | set(skipped)
    )
    return resolved, unresolved


def plan_stem(stem: str, *, only_stale: bool = False) -> StemPlan:
    labels = actor_labels.load(stem)
    resolved, unresolved = _labels_to_tracks(stem, labels, only_stale=only_stale)
    occluded = sum(1 for label in labels.values() if label.box is None)

    # Phase 2: names follow the unit their events agree on.
    #
    # The unit of an event is the one the RUNTIME resolves (extraction/links),
    # not just the one a label names: most named events are unconfirmed
    # automatic picks and carry no actor label at all. Using a different
    # source here would make the migration something other than the
    # normalization it claims to be.
    players = identity.load_players(stem)
    units = track_keys(stem)
    by_unit: dict[str, set[str]] = {}
    events_of: dict[str, list[str]] = {}
    for event_id, name in players.assignments.items():
        track = units.get(event_id)
        if track is None:
            continue
        by_unit.setdefault(track, set()).add(name)
        events_of.setdefault(track, []).append(event_id)

    tracks, conflicts = dict(players.tracks), []
    settled: set[str] = set()
    for track_key, names in sorted(by_unit.items()):
        if len(names) == 1:
            tracks[track_key] = next(iter(names))
            settled.update(events_of[track_key])
        else:
            conflicts.append((track_key, sorted(names), len(events_of[track_key])))
    overrides = {
        event_id: name
        for event_id, name in players.assignments.items()
        if event_id not in settled
    }
    return StemPlan(stem, resolved, unresolved, occluded, tracks, overrides, conflicts)


def _write(stem: str, plan: StemPlan) -> None:
    labels = actor_labels.load(stem)
    updated = {
        event_id: (
            replace(label, track=plan.resolved[event_id])
            if event_id in plan.resolved
            else label
        )
        for event_id, label in labels.items()
    }
    path = actor_labels.actors_path(stem)
    before = path.stat() if path.exists() else None
    path.write_text(
        json.dumps(
            {
                "version": actor_labels.SCHEMA_VERSION,
                "actors": {k: updated[k].payload() for k in sorted(updated)},
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    if before is not None:
        # A migration is not a labeling edit; downstream caches key on mtime.
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    identity.save_players(stem, tracks=plan.tracks, assignments=plan.overrides)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write (default: plan only)")
    parser.add_argument(
        "--reanchor",
        action="store_true",
        help="only labels whose tracklet no longer exists — run after re-tracking",
    )
    args = parser.parse_args()

    stems = actor_labels.labeled_stems()
    if args.apply and not BACKUP.exists():
        shutil.copytree(REID_ANNOTATIONS_DIR, BACKUP)
        print(f"backed up annotations → {BACKUP}\n")

    totals: Counter[str] = Counter()
    print(f"{'video':26} {'labels':>7} {'→track':>7} {'box only':>9} {'occluded':>9} {'tracks':>7} {'conflict':>9}")
    print("-" * 82)
    for stem in stems:
        plan = plan_stem(stem, only_stale=args.reanchor)
        n = len(actor_labels.load(stem))
        print(
            f"{stem[:24]:26} {n:7} {len(plan.resolved):7} "
            f"{len(plan.unresolved):9} {plan.occluded:9} "
            f"{len(plan.tracks):7} {len(plan.conflicts):9}"
        )
        totals.update(
            labels=n,
            resolved=len(plan.resolved),
            unresolved=len(plan.unresolved),
            occluded=plan.occluded,
            tracks=len(plan.tracks),
            conflicts=len(plan.conflicts),
        )
        for key, names, count in plan.conflicts:
            print(f"      ⚠ {key} — {count} events labeled {names}")
        if args.apply:
            _write(stem, plan)

    print(
        f"\n{totals['labels']} labels · {totals['resolved']} resolved to a tracklet "
        f"({totals['resolved'] / max(totals['labels'] - totals['occluded'], 1):.1%} of the boxed ones) · "
        f"{totals['unresolved']} stay box-only · {totals['occluded']} occluded"
    )
    print(
        f"{totals['tracks']} tracklets named · {totals['conflicts']} with conflicting names"
    )
    print("\ndone" if args.apply else "\nplan only — re-run with --apply")


if __name__ == "__main__":
    main()
