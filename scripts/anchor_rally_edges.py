"""Anchor a rally edge to the action that defines it.

A rally opens on a `serve` and closes on a `score`, but its boundaries sat
wherever they were drawn — the serve anywhere from 0.01 s to 12.5 s after
`start`, the score anywhere before `end`. That spread rides into the SPOT
sampling window and leaves boundaries with no comparable baseline to review
against. This pins one edge to its action:

    start = serve - 1.5s        end = score + 1.0s

Only rallies whose edge action is already AT that edge are moved. A span whose
first action is not a serve, whose last is not a score, or that holds no
action at all is an annotation problem of its own and gets left for a human —
scripts/scan_rally_edges.py lists them.

Shortening is safe by construction: the edge action is already the outermost
one, so the stretch trimmed away holds no events. Lengthening can pull in
events that used to fall outside every rally, which is new labelling work
rather than lost work — and, where those turn out to be the same action
annotated twice, work worth doing before this runs.

Association labels are untouched. They key on event id and name a tracklet as
"{rally_id}:{track_id}"; every rally_id survives here, so those keys go on
meaning what they meant. What would cost them is re-running rally tracking,
which renumbers track_id — and rally_fingerprint hashes (rally_id, start,
end), so a moved edge flips it and _tracks_stale then tells every tracked
video to re-track. The tracks header fingerprint is therefore re-stamped where
that can be PROVEN: tracking scanned the old spans, so the stored tracklets
still serve the new ones unless a newly admitted event now sits in a stretch
that was never scanned. Videos where that happens keep the stale flag.

Only the human stores take part: a video needs both
`rally-spot/annotations/` and `action/annotations/`. Pre-annotations are model
output, regenerated rather than edited.

    uv run python scripts/anchor_rally_edges.py --edge end
    uv run python scripts/anchor_rally_edges.py --edge end --apply
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

R2_CATEGORY = "rally-spot/annotations"
#: Matches rally_fingerprint's own rounding, so a re-stamp cannot disagree
#: with the value it is about to compute.
PRECISION = 3


@dataclass(frozen=True)
class Edge:
    """One end of a rally, the action that defines it, and how far off it."""

    name: str
    label: str
    #: True for the opening edge; False for the close.
    opening: bool
    #: Seconds before the serve, or after the score.
    lead: float

    @property
    def field(self) -> str:
        return "start" if self.opening else "end"


EDGES = {
    "start": Edge("start", "serve", opening=True, lead=1.5),
    "end": Edge("end", "score", opening=False, lead=1.0),
}


@dataclass
class Tally:
    trimmed: int = 0
    extended: int = 0
    unchanged: int = 0
    no_actions: int = 0
    wrong_action: int = 0
    seconds_trimmed: float = 0.0
    admitted: int = 0

    def add(self, other: "Tally") -> None:
        for name in vars(self):
            setattr(self, name, getattr(self, name) + getattr(other, name))


@dataclass
class Plan:
    stem: str
    path: Path
    meta: dict
    rows: list[dict]
    tally: Tally = field(default_factory=Tally)
    #: False when a newly admitted event landed in a never-scanned stretch.
    restampable: bool = True

    @property
    def changed(self) -> bool:
        return bool(self.tally.trimmed or self.tally.extended)


def _actions(stem: str) -> list[tuple[float, str]] | None:
    """[(time, label)] sorted the way every reader sorts events."""
    path = ACTION_ANNOTATIONS_DIR / f"{stem}{LABEL_FILE_SUFFIX}"
    if not path.exists():
        return None
    try:
        meta, records = read_jsonl(path)
    except (json.JSONDecodeError, OSError):
        print(f"  {stem}: action file unparseable, left for a human")
        return None
    fps = float(meta.get("fps") or 30.0) or 30.0
    return sorted(
        (int(r["frame"]) / fps, str(r.get("label") or ""))
        for r in records
        if r.get("frame") is not None
    )


def _plan(stem: str, rally_path: Path, edge: Edge) -> Plan | None:
    events = _actions(stem)
    if events is None:
        return None
    try:
        meta, records = read_jsonl(rally_path)
    except (json.JSONDecodeError, OSError):
        print(f"  {stem}: rally file unparseable, left for a human")
        return None
    duration = float(meta.get("duration") or 0.0)

    plan = Plan(stem=stem, path=rally_path, meta=meta, rows=[])
    for record in records:
        row = dict(record)
        start, end = float(row["start"]), float(row["end"])
        # The membership rule every reader uses (extraction.store._within):
        # inclusive on both ends.
        inside = [e for e in events if start <= e[0] <= end]
        if not inside:
            plan.tally.no_actions += 1
        else:
            at_edge = inside[0] if edge.opening else inside[-1]
            if at_edge[1] != edge.label:
                plan.tally.wrong_action += 1
            else:
                old = start if edge.opening else end
                new = at_edge[0] - edge.lead if edge.opening else at_edge[0] + edge.lead
                new = max(0.0, new)
                if not edge.opening and duration > 0:
                    new = min(new, duration)
                new = round(new, PRECISION)
                # A trim must never cross the other edge; it cannot, since the
                # edge action lies inside, but a clamp could make it degenerate.
                if (new >= end) if edge.opening else (new <= start):
                    plan.tally.unchanged += 1
                    plan.rows.append(row)
                    continue
                grew = new < old if edge.opening else new > old
                if new == old:
                    plan.tally.unchanged += 1
                elif grew:
                    plan.tally.extended += 1
                    # Tracking never scanned this stretch, so no tracklet can
                    # exist for what it admits — which is what makes the
                    # fingerprint unprovable here.
                    # Spans are inclusive at both ends, so the event
                    # sitting exactly on the old edge was already inside.
                    admitted = sum(
                        1
                        for t, _ in events
                        if ((new <= t < old) if edge.opening else (old < t <= new))
                    )
                    plan.tally.admitted += admitted
                    if admitted:
                        plan.restampable = False
                else:
                    plan.tally.trimmed += 1
                    plan.tally.seconds_trimmed += abs(new - old)
                row[edge.field] = new
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


def run(edge: Edge, apply: bool) -> None:
    files = sorted(RALLY_ANNOTATIONS_DIR.glob(annotation_name("*")))
    print(f"{RALLY_ANNOTATIONS_DIR.name}/: {len(files)} rally file(s)")
    print(f"anchoring `{edge.field}` to {edge.label} "
          f"{'−' if edge.opening else '+'} {edge.lead}s")

    plans: list[Plan] = []
    for path in files:
        stem = path.name[: -len(annotation_name(""))]
        plan = _plan(stem, path, edge)
        if plan is not None:
            plans.append(plan)
    changed = [p for p in plans if p.changed]
    print(f"  {len(plans)} paired with an action annotation")

    total = Tally()
    for plan in plans:
        total.add(plan.tally)
    print(f"  {'moved' if apply else 'would move'} {len(changed)} video(s):")
    print(f"    shortened      {total.trimmed}  (−{total.seconds_trimmed:.0f}s total)")
    print(f"    lengthened     {total.extended}  (+{total.admitted} event(s) admitted)")
    print(f"    already there  {total.unchanged}")
    print(f"    left alone     {total.wrong_action} edge action is not a"
          f" {edge.label}, {total.no_actions} with no actions")

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
    if refused:
        print(f"  left stale {len(refused)} — a newly admitted event sits in a"
              f" stretch tracking never scanned:")
        for plan in refused:
            print(f"    {plan.stem}  (+{plan.tally.admitted} event(s))")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge", choices=sorted(EDGES), required=True,
                        help="which boundary to anchor")
    parser.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    args = parser.parse_args()
    run(EDGES[args.edge], args.apply)
    if not args.apply:
        print("\nDry run — re-run with --apply to write.")


if __name__ == "__main__":
    main()
