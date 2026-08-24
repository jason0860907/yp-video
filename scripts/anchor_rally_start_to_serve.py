"""Anchor every rally's start to a fixed lead-in before its serve.

A rally span's start had no common basis. The serve that opens it sits
anywhere from 0.01 s to 12.5 s after it (median ~2 s), and that spread rides
straight into the SPOT sampling window while leaving rally boundaries with no
comparable baseline to review against. This rewrites ``start`` to the serve's
time minus ``LEAD_S``, clamped at 0. ``end`` is never touched, and neither is
``rally_id``, ``label`` or ``side``.

Only rallies whose FIRST in-span action is a serve are moved. A span whose
first action is something else — or that holds no action at all — is an
annotation problem in its own right, and gets left for a human.

Shortening is safe by construction: the serve is already the first event in
the span, so the region trimmed away holds no action events. Lengthening can
pull in events that used to sit outside every rally, which is new labelling
work rather than lost work.

Association labels are deliberately untouched. They live in their own files
keyed by event id and reference a tracklet as ``"{rally_id}:{track_id}"``;
every rally_id survives here, so those keys go on meaning what they meant.
What WOULD cost them is re-running rally tracking, which renumbers track_id.

Which is the catch worth spelling out. ``rally_fingerprint`` hashes
``(rally_id, start, end)``, so moving a start flips it and
``extraction.prerequisites._tracks_stale`` then tells every tracked video to
re-run tracking — the one action that costs those labels. So the tracks
header fingerprint is re-stamped, but only where the claim can be PROVEN:
tracking scanned the old spans, so the stored tracklets still serve the new
ones unless a newly admitted event now sits in a lead-in that was never
scanned. Videos where that happens keep the stale flag, because there it is
honest. (``_tracks_stale`` still describes rally_id as positional; that has
not been true since freeze_rally_ids.py made it a stable id.)

Only the human stores take part: a video needs both
``rally-spot/annotations/`` and ``action/annotations/``. Pre-annotations are
model output and are regenerated, never edited here.

    uv run python scripts/anchor_rally_start_to_serve.py            # dry run
    uv run python scripts/anchor_rally_start_to_serve.py --apply    # do it
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from yp_video.config import ACTION_ANNOTATIONS_DIR, RALLY_ANNOTATIONS_DIR
from yp_video.contracts.action import LABEL_FILE_SUFFIX
from yp_video.core.jsonl import atomic_write, read_jsonl, read_jsonl_header
from yp_video.core.rallies import annotation_name, rally_fingerprint
from yp_video.tracklets.store import tracks_path
from yp_video.web.r2_client import r2_client
from yp_video.web.routers.annotate import Annotation, _write_annotations_atomic

#: How long before the serve a rally should begin.
LEAD_S = 1.5
SERVE = "serve"
R2_CATEGORY = "rally-spot/annotations"
#: Matches rally_fingerprint's own rounding, so a re-stamp cannot disagree
#: with the value it is about to compute.
PRECISION = 3


@dataclass
class Tally:
    trimmed: int = 0
    extended: int = 0
    unchanged: int = 0
    no_actions: int = 0
    not_serve: int = 0
    seconds_trimmed: float = 0.0
    admitted: int = 0

    def add(self, other: "Tally") -> None:
        self.trimmed += other.trimmed
        self.extended += other.extended
        self.unchanged += other.unchanged
        self.no_actions += other.no_actions
        self.not_serve += other.not_serve
        self.seconds_trimmed += other.seconds_trimmed
        self.admitted += other.admitted


@dataclass
class Plan:
    stem: str
    path: Path
    meta: dict
    rows: list[dict]
    tally: Tally = field(default_factory=Tally)
    #: False when a newly admitted event landed in a never-scanned lead-in.
    restampable: bool = True

    @property
    def changed(self) -> bool:
        return bool(self.tally.trimmed or self.tally.extended)


def _load_actions(stem: str) -> tuple[float, list[tuple[float, str]]] | None:
    """(fps, [(time, label)]) sorted the way every reader sorts events."""
    path = ACTION_ANNOTATIONS_DIR / f"{stem}{LABEL_FILE_SUFFIX}"
    if not path.exists():
        return None
    try:
        meta, records = read_jsonl(path)
    except (json.JSONDecodeError, OSError):
        print(f"  {stem}: action file unparseable, left for a human")
        return None
    fps = float(meta.get("fps") or 30.0) or 30.0
    events = sorted(
        (int(r["frame"]) / fps, str(r.get("label") or ""))
        for r in records
        if r.get("frame") is not None
    )
    return fps, events


def _plan(stem: str, rally_path: Path) -> Plan | None:
    """Work out this video's new spans without writing anything."""
    loaded = _load_actions(stem)
    if loaded is None:
        return None
    _fps, events = loaded
    try:
        meta, records = read_jsonl(rally_path)
    except (json.JSONDecodeError, OSError):
        print(f"  {stem}: rally file unparseable, left for a human")
        return None

    plan = Plan(stem=stem, path=rally_path, meta=meta, rows=[])
    for record in records:
        row = dict(record)
        start, end = float(row["start"]), float(row["end"])
        # The membership rule every reader uses (extraction.store._within):
        # inclusive on both ends.
        inside = [e for e in events if start <= e[0] <= end]
        if not inside:
            plan.tally.no_actions += 1
        elif inside[0][1] != SERVE:
            plan.tally.not_serve += 1
        else:
            new_start = round(max(0.0, inside[0][0] - LEAD_S), PRECISION)
            if new_start > start:
                plan.tally.trimmed += 1
                plan.tally.seconds_trimmed += new_start - start
            elif new_start < start:
                plan.tally.extended += 1
                # Events the longer span now admits. Tracking never scanned
                # this stretch, so a tracklet cannot exist for them — which is
                # exactly what makes the fingerprint unprovable here.
                admitted = sum(1 for e in events if new_start <= e[0] < start)
                plan.tally.admitted += admitted
                if admitted:
                    plan.restampable = False
            else:
                plan.tally.unchanged += 1
            row["start"] = new_start
        plan.rows.append(row)
    return plan


def _write(plan: Plan) -> None:
    """Through the router's own writer, so ids and max_rally_id stay sane."""
    annotations = [
        Annotation(
            rally_id=row.get("rally_id"),
            start=float(row["start"]),
            end=float(row["end"]),
            label=str(row["label"]),
            side=row.get("side"),
        )
        for row in plan.rows
    ]
    _write_annotations_atomic(
        plan.path,
        str(plan.meta.get("video") or plan.stem),
        float(plan.meta.get("duration") or 0.0),
        annotations,
    )


def _restamp(stem: str) -> bool:
    """Re-stamp the tracks header fingerprint; True when a file was rewritten.

    Streams the records through untouched — a tracks jsonl runs to several
    megabytes and only line 0 is changing.
    """
    path = tracks_path(stem)
    if not path.exists():
        return False
    header = read_jsonl_header(path)
    if not (header.get("rallies") or {}).get("fingerprint"):
        return False  # never claimed freshness; do not start now
    header.pop("_meta", None)
    header["rallies"] = {**header["rallies"], "fingerprint": rally_fingerprint(stem)}
    with open(path, "r", encoding="utf-8") as src:
        src.readline()  # drop the old header
        with atomic_write(path) as out:
            out.write(json.dumps({"_meta": True, **header}, ensure_ascii=False) + "\n")
            shutil.copyfileobj(src, out)
    return True


def run(apply: bool) -> None:
    files = sorted(RALLY_ANNOTATIONS_DIR.glob(annotation_name("*")))
    print(f"{RALLY_ANNOTATIONS_DIR.name}/: {len(files)} rally file(s)")

    plans: list[Plan] = []
    for path in files:
        stem = path.name[: -len(annotation_name(""))]
        plan = _plan(stem, path)
        if plan is not None:
            plans.append(plan)
    paired = len(plans)
    changed = [p for p in plans if p.changed]
    print(f"  {paired} paired with an action annotation")

    total = Tally()
    for plan in plans:
        total.add(plan.tally)
    print(f"  {'moved' if apply else 'would move'} {len(changed)} video(s):")
    print(f"    shortened      {total.trimmed}  (−{total.seconds_trimmed:.0f}s total)")
    print(f"    lengthened     {total.extended}  (+{total.admitted} event(s) admitted)")
    print(f"    already at {LEAD_S}s  {total.unchanged}")
    print(f"    left alone     {total.not_serve} first action not a serve"
          f", {total.no_actions} with no actions")

    if apply and changed:
        stamp = date.today().strftime("%Y%m%d")
        backup = RALLY_ANNOTATIONS_DIR.with_name(f"{RALLY_ANNOTATIONS_DIR.name}.bak-{stamp}")
        if not backup.exists():
            shutil.copytree(RALLY_ANNOTATIONS_DIR, backup)
            print(f"  backed up to {backup}")
        for plan in changed:
            _write(plan)
        print(f"  wrote {len(changed)} file(s)")

        if r2_client.configured:
            failed = 0
            for plan in changed:
                try:
                    r2_client.upload_file(plan.path, f"{R2_CATEGORY}/{plan.path.name}")
                except Exception as exc:  # noqa: BLE001 — R2 must not undo local truth
                    failed += 1
                    print(f"  R2 re-upload failed for {plan.path.name}: {exc}")
            print(f"  R2 re-uploaded {len(changed) - failed}/{len(changed)}")

    # ── Tracklet freshness ──
    tracked = [p for p in changed if tracks_path(p.stem).exists()]
    provable = [p for p in tracked if p.restampable]
    refused = [p for p in tracked if not p.restampable]
    print(f"\ntracks/: {len(tracked)} touched video(s) have tracklets")
    if apply:
        stamped = sum(1 for p in provable if _restamp(p.stem))
        print(f"  re-stamped {stamped} fingerprint(s)")
    else:
        print(f"  would re-stamp {len(provable)} fingerprint(s)")
    print(f"  left stale {len(refused)} — a newly admitted event sits in a"
          f" stretch tracking never scanned:")
    for plan in refused:
        print(f"    {plan.stem}  (+{plan.tally.admitted} event(s))")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    args = parser.parse_args()
    run(args.apply)
    if not args.apply:
        print("\nDry run — re-run with --apply to write.")


if __name__ == "__main__":
    main()
