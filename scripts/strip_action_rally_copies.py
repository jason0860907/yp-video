"""One-off migration: action label files stop carrying rally copies.

The annotation store now persists only the human's facts (event
``id/frame/label/xy/visible`` and meta ``video/num_frames/fps/source``);
rally spans and the fields derived from them are joined from the live rally
store on every read. This strips the historical copies — meta ``rallies``
and per-event ``rally_id`` / ``relative_frame`` / ``time`` — from every
existing file in ``action/annotations/`` and ``action/pre-annotations/``.

Both directories are backed up wholesale to ``<dir>.bak-<today>/`` before
the first write. Changed human annotations are re-uploaded to R2 so the
mirror matches; failures there are reported, never fatal.

    uv run python scripts/strip_action_rally_copies.py            # dry run
    uv run python scripts/strip_action_rally_copies.py --apply    # do it
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import date
from pathlib import Path

from yp_video.config import ACTION_ANNOTATIONS_DIR, ACTION_PRE_ANNOTATIONS_DIR
from yp_video.core.jsonl import atomic_write
from yp_video.web.r2_client import r2_client

META_STRIP = ("rallies",)
EVENT_STRIP = ("rally_id", "relative_frame", "time")

R2_CATEGORY = {
    ACTION_ANNOTATIONS_DIR: "action/annotations",
    ACTION_PRE_ANNOTATIONS_DIR: "action/pre-annotations",
}


def _strip_file(path: Path, apply: bool) -> bool:
    """Rewrite one label file without its rally copies; True when it changed."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        print(f"  {path.name}: empty file, skipped")
        return False
    try:
        rows = [json.loads(line) for line in lines if line.strip()]
    except json.JSONDecodeError:
        print(f"  {path.name}: unparseable, left for a human")
        return False

    meta, events = rows[0], rows[1:]
    dirty = any(key in meta for key in META_STRIP) or any(
        key in event for event in events for key in EVENT_STRIP
    )
    if not dirty:
        return False

    for key in META_STRIP:
        meta.pop(key, None)
    for event in events:
        for key in EVENT_STRIP:
            event.pop(key, None)

    if apply:
        with atomic_write(path) as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            for event in events:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
    return True


def migrate(apply: bool) -> None:
    stamp = date.today().strftime("%Y%m%d")
    for directory in (ACTION_ANNOTATIONS_DIR, ACTION_PRE_ANNOTATIONS_DIR):
        if not directory.exists():
            continue
        files = sorted(directory.glob("*_actions.jsonl"))
        print(f"{directory.name}/: {len(files)} file(s)")

        backup = directory.with_name(f"{directory.name}.bak-{stamp}")
        if apply and not backup.exists():
            shutil.copytree(directory, backup)
            print(f"  backed up to {backup}")

        changed = [path for path in files if _strip_file(path, apply)]
        print(f"  {'stripped' if apply else 'would strip'} {len(changed)} file(s)")

        if apply and changed and r2_client.configured:
            category = R2_CATEGORY[directory]
            failed = 0
            for path in changed:
                try:
                    r2_client.upload_file(path, f"{category}/{path.name}")
                except Exception as exc:  # noqa: BLE001 — R2 must not undo local truth
                    failed += 1
                    print(f"  R2 re-upload failed for {path.name}: {exc}")
            print(f"  R2 re-uploaded {len(changed) - failed}/{len(changed)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    args = parser.parse_args()
    migrate(args.apply)
    if not args.apply:
        print("\nDry run — re-run with --apply to write.")


if __name__ == "__main__":
    main()
