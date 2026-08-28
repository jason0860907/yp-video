"""Snap every rally's edges to a fixed lead-in before its serve and a fixed
tail after its score.

A rally opens on a serve and closes on a score; the span should begin a
constant ``LEAD_S`` before the one and end a constant ``TAIL_S`` after the
other, so the model sees the same run-up and the same landing everywhere.
Only edges the actions vouch for are touched: a start whose first action
inside the span is a ``serve``, an end whose last action is a ``score``. The
rest are a labelling question (see scan_rally_edges.py), not a boundary one.

A tail can reach an action the old end excluded (a score labelled twice, a
touch after the ball landed); that rally then shows up in the edge scan as
"score not last", which is where it belongs.

Rally ids do not change, so tracklet keys stay valid; but the rally
fingerprint does, so videos with tracks will report ``tracks_stale`` until
re-tracked. That is the check working as designed, not a fault.

    uv run python scripts/snap_rally_lead_in.py          # report only
    uv run python scripts/snap_rally_lead_in.py --apply
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yp_video.config import ACTION_ANNOTATIONS_DIR, RALLY_ANNOTATIONS_DIR  # noqa: E402
from yp_video.contracts.action import LABEL_FILE_SUFFIX  # noqa: E402
from yp_video.core.jsonl import read_jsonl, write_jsonl  # noqa: E402
from yp_video.core.rallies import annotation_name  # noqa: E402

LEAD_S = 1.5
TAIL_S = 1.5
#: Edges within this of the target are already there; float noise, not a move.
TOLERANCE_S = 0.05


def ts(seconds: float) -> str:
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--apply", action="store_true", help="write the moved edges")
    args = parser.parse_args()

    moved = videos = 0
    for rally_path in sorted(RALLY_ANNOTATIONS_DIR.glob(annotation_name("*"))):
        stem = rally_path.name[: -len(annotation_name(""))]
        action_path = ACTION_ANNOTATIONS_DIR / f"{stem}{LABEL_FILE_SUFFIX}"
        if not action_path.exists():
            continue
        meta, events = read_jsonl(action_path)
        fps = float(meta.get("fps") or 30.0) or 30.0
        timeline = sorted(
            (int(e["frame"]) / fps, str(e.get("label") or ""))
            for e in events
            if e.get("frame") is not None
        )

        rally_meta, rows = read_jsonl(rally_path)
        changed = False
        for row in rows:
            start, end = float(row["start"]), float(row["end"])
            inside = [x for x in timeline if start <= x[0] <= end]
            if not inside:
                continue
            targets = {}
            if inside[0][1] == "serve":
                targets["start"] = round(inside[0][0] - LEAD_S, 3)
            if inside[-1][1] == "score":
                targets["end"] = round(inside[-1][0] + TAIL_S, 3)
            for edge, target in targets.items():
                current = float(row[edge])
                if abs(target - current) < TOLERANCE_S:
                    continue
                moved += 1
                changed = True
                print(
                    f"{stem} r{row.get('rally_id')} {edge}: {ts(current)} → {ts(target)} "
                    f"({target - current:+.2f}s)"
                )
                row[edge] = target
        if changed:
            videos += 1
            if args.apply:
                write_jsonl(rally_path, rally_meta, rows)

    verb = "moved" if args.apply else "would move"
    print(f"\n{verb} {moved} rally edge(s) across {videos} video(s)")


if __name__ == "__main__":
    main()
