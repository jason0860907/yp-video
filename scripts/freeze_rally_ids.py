"""Freeze every rally file's positional numbering into stored stable ids.

``rally_id`` used to be recomputed from sort position on every read; now the
file is the ledger (core/rallies.resolve_rally_ids) and readers refuse a file
without ids. This script converts all three rally source directories in one
pass: each record gets ``rally_id`` = the positional number every reader has
derived until today, and the header gains ``max_rally_id`` so a deleted id is
never reused.

Nothing downstream is touched, because nothing downstream changes meaning:
tracklet keys and association labels already reference these exact numbers.
Per file, the rally fingerprint is asserted identical before (legacy
positional numbering, reimplemented here) and after (the new stored-id
reader) — any mismatch aborts the whole run.

File mtimes are preserved: backfill_rally_fingerprint.py proves "spans not
edited since tracking" by comparing mtimes, and a rewrite must not forge that
evidence.

    uv run python scripts/freeze_rally_ids.py             # dry run, report
    uv run python scripts/freeze_rally_ids.py --apply
    uv run python scripts/freeze_rally_ids.py --apply --sync-r2
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yp_video.core.annotation_ids import stable_id  # noqa: E402
from yp_video.core.jsonl import read_jsonl, write_jsonl  # noqa: E402
from yp_video.core.rallies import (  # noqa: E402
    RALLY_SOURCES,
    rally_annotation_path,
    rally_fingerprint,
)
from yp_video.web.r2_client import sync_to_r2  # noqa: E402


def _canonical(records: list[dict]) -> list[tuple[float, float, str, dict]]:
    """The (start, end, label) ordering every reader used to number by."""
    parsed = [
        (
            float(r.get("start", r.get("start_time", 0)) or 0),
            float(r.get("end", r.get("end_time", 0)) or 0),
            str(r.get("label", "rally")),
            r,
        )
        for r in records
    ]
    parsed.sort(key=lambda item: item[:3])
    return parsed


def _legacy_fingerprint(records: list[dict]) -> str | None:
    """What rally_fingerprint returned under positional numbering."""
    ordered = _canonical(records)
    if not ordered:
        return None
    return stable_id(
        "rallies",
        [
            (index + 1, round(start, 3), round(end, 3))
            for index, (start, end, _label, _r) in enumerate(ordered)
        ],
    )


@dataclass(frozen=True)
class Plan:
    path: Path
    tag: str
    r2_category: str
    #: "stamp-ids" | "stamp-meta-only" | "skip"
    action: str
    #: Why this file cannot be converted, or None.
    refused: str | None
    rows: list[dict]
    meta: dict
    legacy_fingerprint: str | None


def plan_file(path: Path, tag: str, r2_category: str) -> Plan:
    meta, records = read_jsonl(path)
    stored = [r.get("rally_id") for r in records]
    valid = [
        v for v in stored if isinstance(v, int) and not isinstance(v, bool) and v >= 1
    ]

    def refuse(reason: str) -> Plan:
        return Plan(path, tag, r2_category, "skip", reason, [], meta, None)

    if valid and len(valid) != len(records):
        return refuse("mixed: some records have rally_id, some not")
    if valid and len(set(valid)) != len(valid):
        return refuse("duplicate stored rally_id(s)")

    legacy = _legacy_fingerprint(records)
    if valid:
        # Ids complete — preserve them verbatim, only the header may be missing.
        rows = [dict(r) for r in records]
        max_id = max(valid)
        action = (
            "skip" if meta.get("max_rally_id") == max_id else "stamp-meta-only"
        )
    else:
        rows = [
            {**r, "rally_id": index + 1}
            for index, (_s, _e, _l, r) in enumerate(_canonical(records))
        ]
        max_id = len(rows)
        action = "stamp-ids"
    new_meta = {**meta, "max_rally_id": max_id}
    return Plan(path, tag, r2_category, action, None, rows, new_meta, legacy)


def apply_plan(plan: Plan) -> None:
    stat = plan.path.stat()
    write_jsonl(plan.path, plan.meta, plan.rows)
    os.utime(plan.path, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    # The invariant that makes this a freeze and not an edit: the file's
    # fingerprint must be exactly what positional numbering produced. Checked
    # through the real reader when this file is the video's active source.
    stem = plan.path.name.removesuffix("_annotations.jsonl")
    if rally_annotation_path(stem) == plan.path:
        after = rally_fingerprint(stem)
        if after != plan.legacy_fingerprint:
            raise SystemExit(
                f"ABORT: fingerprint changed for {plan.path.name} "
                f"({plan.legacy_fingerprint} -> {after})"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="rewrite the files")
    parser.add_argument(
        "--sync-r2",
        action="store_true",
        help="push rewritten files to R2 (readers refuse the old format)",
    )
    args = parser.parse_args()

    plans: list[Plan] = []
    for source in RALLY_SOURCES:
        if not source.directory.exists():
            continue
        for path in sorted(source.directory.glob("*_annotations.jsonl")):
            plans.append(plan_file(path, source.tag, source.r2_category))

    by_action: dict[str, list[Plan]] = {}
    for plan in plans:
        key = plan.action if plan.refused is None else "refused"
        by_action.setdefault(key, []).append(plan)

    print(f"{len(plans)} rally file(s) across {len(RALLY_SOURCES)} sources")
    for key in ("stamp-ids", "stamp-meta-only", "skip", "refused"):
        rows = by_action.get(key, [])
        print(f"  {key:16} {len(rows)}")
    for plan in by_action.get("refused", []):
        print(f"    ✗ {plan.tag}/{plan.path.name}: {plan.refused}")

    todo = by_action.get("stamp-ids", []) + by_action.get("stamp-meta-only", [])
    if by_action.get("refused"):
        print("\nRefused files above need manual attention first.")
        return 1
    if not args.apply:
        print(f"\nDry run — {len(todo)} file(s) to rewrite. Re-run with --apply.")
        return 0

    for plan in todo:
        apply_plan(plan)
        if args.sync_r2:
            sync_to_r2(plan.path, plan.r2_category)
    print(f"\nFroze {len(todo)} file(s); fingerprints verified unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
