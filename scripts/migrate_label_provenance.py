"""One-off migration: provenance by store, one home for every Done flag.

Two historical conventions die here:

1. ``videos/action/annotations/`` becomes human-only. Files that the old
   in-band flag called machine output (``reviewed: false``, or a ``spot``
   source from before the flag existed) move to
   ``videos/action/pre-annotations/`` — where machine output has been written
   for a while now. The ``reviewed`` key itself is stripped everywhere: with
   provenance encoded by location, workflow state inside a training file is
   noise. When both stores hold the same machine file, the stale copy in
   annotations/ is deleted — the UI was already showing the pre file.

2. ``videos/reid/annotations/<stem>_players.json`` loses its ``done`` key.
   The verdict moves to the shared per-video sidecar (core/label_done.py),
   where rally / action / association already keep theirs.

R2 mirrors of moved action files are re-homed too (delete under
``action/annotations/``, upload under ``action/pre-annotations/``) when R2
is configured; failures there are reported, never fatal.

    uv run python scripts/migrate_label_provenance.py            # show the plan
    uv run python scripts/migrate_label_provenance.py --apply    # do it
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from yp_video.config import (
    ACTION_ANNOTATIONS_DIR,
    ACTION_PRE_ANNOTATIONS_DIR,
    REID_ANNOTATIONS_DIR,
)
from yp_video.core import label_done
from yp_video.core.jsonl import atomic_write
from yp_video.reid.store import PLAYERS_SUFFIX
from yp_video.web.r2_client import r2_client


def _was_reviewed(meta: dict) -> bool:
    """The retired in-band rule, applied one last time to sort old files."""
    if "reviewed" in meta:
        return bool(meta["reviewed"])
    source = meta.get("source")
    if isinstance(source, dict) and source.get("type") == "spot":
        return False
    return True


def _rewrite_without_reviewed(path: Path, lines: list[str], meta: dict) -> None:
    meta.pop("reviewed", None)
    with atomic_write(path) as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        f.writelines(lines[1:])


def _move_r2(name: str) -> str:
    if not r2_client.configured:
        return ""
    try:
        r2_client.upload_file(ACTION_PRE_ANNOTATIONS_DIR / name, f"action/pre-annotations/{name}")
        r2_client.delete_object(f"action/annotations/{name}")
        return "r2 re-homed"
    except Exception as exc:  # noqa: BLE001 — R2 trouble must not undo local truth
        return f"R2 re-home failed ({exc}); fix manually"


def migrate_action(apply: bool) -> None:
    for path in sorted(ACTION_ANNOTATIONS_DIR.glob("*_actions.jsonl")):
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        if not lines:
            print(f"[action] {path.name}: empty file, skipped")
            continue
        try:
            meta = json.loads(lines[0])
        except json.JSONDecodeError:
            print(f"[action] {path.name}: unparseable meta, left for a human")
            continue

        if _was_reviewed(meta):
            if "reviewed" in meta:
                print(f"[action] {path.name}: human file, strip 'reviewed'")
                if apply:
                    _rewrite_without_reviewed(path, lines, meta)
            continue

        pre_path = ACTION_PRE_ANNOTATIONS_DIR / path.name
        if pre_path.exists():
            print(f"[action] {path.name}: stale machine copy, pre file already exists — delete")
            if apply:
                path.unlink()
                note = _move_r2(path.name)
                if note:
                    print(f"         {note}")
            continue

        print(f"[action] {path.name}: machine output — move to pre-annotations")
        if apply:
            ACTION_PRE_ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
            _rewrite_without_reviewed(pre_path, lines, meta)
            path.unlink()
            note = _move_r2(path.name)
            if note:
                print(f"         {note}")


def migrate_reid_done(apply: bool) -> None:
    for path in sorted(REID_ANNOTATIONS_DIR.glob(f"*{PLAYERS_SUFFIX}")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "done" not in data:
            continue
        stem = path.name[: -len(PLAYERS_SUFFIX)]
        done = bool(data.pop("done"))
        print(f"[reid] {stem}: done={done} → label-done sidecar")
        if apply:
            label_done.set_done(stem, "reid", done)
            with atomic_write(path) as f:
                json.dump(data, f, ensure_ascii=False, indent=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    args = parser.parse_args()

    if ACTION_ANNOTATIONS_DIR.exists():
        migrate_action(args.apply)
    if REID_ANNOTATIONS_DIR.exists():
        migrate_reid_done(args.apply)
    if not args.apply:
        print("\nDry run — re-run with --apply to write.")


if __name__ == "__main__":
    main()
