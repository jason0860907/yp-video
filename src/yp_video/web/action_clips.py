"""``yp-action-clips``: export per-player action clips for every sideline
video with action labels (see ``actor/clips.py``).

Lives in the web layer for one reason: the cut's bytes are in R2, and
``materialized_cut`` is the web layer's. The camera view comes from the
cut's canonical path (its parent dir), never from an annotation's recorded
source path — older rally files still name the pre-split ``cuts/`` layout.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from yp_video.actor import clips
from yp_video.config import ACTION_ANNOTATIONS_DIR, ACTION_CLIPS_DIR, cut_kind_of
from yp_video.contracts.action import LABEL_FILE_GLOB, LABEL_FILE_SUFFIX
from yp_video.web.r2_client import materialized_cut, resolve_cut

VIEW = "sideline"


def labelled_cuts(view: str) -> list[tuple[str, Path]]:
    """``(stem, cut path)`` for every action-labelled video of this view."""
    out = []
    for path in sorted(ACTION_ANNOTATIONS_DIR.glob(LABEL_FILE_GLOB)):
        stem = path.name[: -len(LABEL_FILE_SUFFIX)]
        cut = resolve_cut(f"{stem}.mp4")
        if cut is not None and cut_kind_of(cut) == view:
            out.append((stem, cut))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stems", nargs="*", help="only these videos (default: every sideline video)")
    parser.add_argument("--overwrite", action="store_true", help="re-export videos that already have an index")
    parser.add_argument("--out", type=Path, default=ACTION_CLIPS_DIR)
    args = parser.parse_args()

    videos = labelled_cuts(VIEW)
    if args.stems:
        wanted = set(args.stems)
        videos = [(s, p) for s, p in videos if s in wanted]
        missing = wanted - {s for s, _ in videos}
        if missing:
            sys.exit(f"not a {VIEW} video with action labels: {sorted(missing)}")

    totals: dict[str, int] = {}
    for n, (stem, cut) in enumerate(videos, start=1):
        if not args.overwrite and clips.index_path(args.out, stem).exists():
            print(f"[{n}/{len(videos)}] {stem}: kept existing clips", flush=True)
            continue
        started = time.monotonic()
        with materialized_cut(cut) as local:
            counts = clips.export_video(stem, local, args.out)
        for key, value in counts.items():
            totals[key] = totals.get(key, 0) + value
        print(
            f"[{n}/{len(videos)}] {stem}: {counts['positives']} positives, "
            f"{counts['negatives']} negatives, {counts['missing_frames']} missing frames, "
            f"{time.monotonic() - started:.0f}s",
            flush=True,
        )
    if totals:
        print("total:", ", ".join(f"{k}={v}" for k, v in totals.items()))


if __name__ == "__main__":
    main()
