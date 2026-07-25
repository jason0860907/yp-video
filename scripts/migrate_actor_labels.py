"""One-off migration to the split actor / player annotation layout.

Two things move:

1. ``annotations/<stem>_players.json`` carried four things — player
   assignments, ``done``, ``actor_fixes`` and ``actor_reviews``. The last two
   were the same fact written twice (a pick stored a box in one and a
   ``manual`` verdict in the other, and the reader synthesized reviews from
   fixes anyway, silently overriding what was stored). They become one entry
   per event in ``<stem>_actors.json``; the players file keeps assignments
   and ``done``.

2. ``embeddings/<stem>_reid.jsonl`` moves to ``records/<stem>.jsonl`` — the
   records are extraction output shared by actor and reid, not embeddings —
   and every record gains its explicit ``resolution``, dropping the
   ``box_source`` flag that encoded the same state a second time.

Fixes win over stored reviews wherever both exist, reproducing exactly what
the old reader did. Idempotent: already-migrated files are left alone.

    uv run python scripts/migrate_actor_labels.py           # show the plan
    uv run python scripts/migrate_actor_labels.py --apply   # do it
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from yp_video.actor.labels import ActorLabel, ActorVerdict, SCHEMA_VERSION
from yp_video.actor.resolution import ActorResolution
from yp_video.config import REID_ANNOTATIONS_DIR, REID_DIR
from yp_video.extraction.store import RECORDS_DIR

LEGACY_RECORDS_DIR = REID_DIR / "embeddings"


def _box(value: object) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    return tuple(float(v) for v in value)  # type: ignore[return-value]


def _frame(value: object) -> int | None:
    return int(value) if isinstance(value, int) else None


def legacy_labels(data: dict) -> dict[str, ActorLabel]:
    """Collapse actor_reviews + actor_fixes into one label per event."""
    labels: dict[str, ActorLabel] = {}
    for event_id, value in (data.get("actor_reviews") or {}).items():
        payload = value if isinstance(value, dict) else {}
        try:
            verdict = ActorVerdict(str(payload.get("verdict", value)))
        except ValueError:
            continue
        labels[str(event_id)] = ActorLabel(
            verdict=verdict,
            box=_box(payload.get("box")),
            frame=_frame(payload.get("frame")),
        )
    # A fix is the stronger record and the one the old reader trusted.
    for event_id, fix in (data.get("actor_fixes") or {}).items():
        if not isinstance(fix, dict):
            continue
        labels[str(event_id)] = ActorLabel(
            verdict=(
                ActorVerdict.OCCLUDED
                if fix.get("none")
                else ActorVerdict.MANUAL
            ),
            box=_box(fix.get("box")),
            frame=_frame(fix.get("frame")),
            snap=fix.get("snap") is not False,
        )
    return labels


def legacy_resolution(record: dict) -> ActorResolution:
    """The state the old reader inferred from the record's shape."""
    if record.get("box_source") == "manual":
        return (
            ActorResolution.MANUAL
            if record.get("crop")
            else ActorResolution.OCCLUDED
        )
    return (
        ActorResolution.AUTO
        if record.get("crop")
        else ActorResolution.UNRESOLVED
    )


def migrate_annotations(*, apply: bool) -> list[str]:
    lines: list[str] = []
    for path in sorted(REID_ANNOTATIONS_DIR.glob("*_players.json")):
        stem = path.name[: -len("_players.json")]
        data = json.loads(path.read_text(encoding="utf-8"))
        if not (data.get("actor_fixes") or data.get("actor_reviews")):
            lines.append(f"  {stem}: already split")
            continue

        labels = legacy_labels(data)
        counts: dict[str, int] = {}
        for label in labels.values():
            counts[label.verdict.value] = counts.get(label.verdict.value, 0) + 1
        summary = ", ".join(f"{n} {v}" for v, n in sorted(counts.items()))
        lines.append(
            f"  {stem}: {len(labels)} labels ({summary}) → {stem}_actors.json"
        )
        if not apply:
            continue

        actors_path = REID_ANNOTATIONS_DIR / f"{stem}_actors.json"
        actors_path.write_text(
            json.dumps(
                {
                    "version": SCHEMA_VERSION,
                    "actors": {
                        event_id: labels[event_id].payload()
                        for event_id in sorted(labels)
                    },
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        path.write_text(
            json.dumps(
                {k: v for k, v in data.items() if k in ("assignments", "done")},
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
    return lines


def migrate_records(*, apply: bool) -> list[str]:
    lines: list[str] = []
    for path in sorted(LEGACY_RECORDS_DIR.glob("*_reid.jsonl")):
        stem = path.name[: -len("_reid.jsonl")]
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        header, records = rows[0], rows[1:]
        normalized = sum(1 for r in records if "resolution" not in r)
        lines.append(
            f"  {stem}: {len(records)} records "
            f"({normalized} gaining an explicit resolution) → records/{stem}.jsonl"
        )
        if not apply:
            continue

        for record in records:
            if "resolution" not in record:
                record["resolution"] = legacy_resolution(record).value
            record.pop("box_source", None)
        RECORDS_DIR.mkdir(parents=True, exist_ok=True)
        out = RECORDS_DIR / f"{stem}.jsonl"
        with open(out, "w", encoding="utf-8") as file:
            for row in (header, *records):
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        # Embedding freshness is an mtime comparison against this file
        # (reid/store.stale_embedding_models). A migration is not an
        # extraction, so the records must not look newer than the matrices
        # built from them — every model would go stale for no reason.
        source = path.stat()
        os.utime(out, ns=(source.st_atime_ns, source.st_mtime_ns))
        path.unlink()
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true", help="write the changes (default: plan only)"
    )
    args = parser.parse_args()

    if args.apply:
        backup = REID_ANNOTATIONS_DIR.with_name(
            REID_ANNOTATIONS_DIR.name + ".pre-actor-split"
        )
        if not backup.exists():
            shutil.copytree(REID_ANNOTATIONS_DIR, backup)
            print(f"backed up annotations → {backup}")

    print("annotations:")
    for line in migrate_annotations(apply=args.apply) or ["  (none)"]:
        print(line)
    print("records:")
    for line in migrate_records(apply=args.apply) or ["  (none)"]:
        print(line)
    print("\ndone" if args.apply else "\nplan only — re-run with --apply")


if __name__ == "__main__":
    main()
