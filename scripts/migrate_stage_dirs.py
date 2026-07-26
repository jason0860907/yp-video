"""One-off migration: give each pipeline stage its own directory.

Everything used to live under ``videos/reid/`` — extraction records, actor
crops, tracklets, association checkpoints and both kinds of human label —
because ReID was the stage that consumed all of it. Consuming is not owning,
and the layout said otherwise to every reader that went looking.

    videos/reid/records/       →  videos/extraction/records/
    videos/reid/crops/         →  videos/extraction/crops/
    videos/reid/crops-masked/  →  videos/extraction/crops-masked/
    videos/reid/tracks/        →  videos/tracks/
    videos/reid/association/   →  videos/association/
    videos/reid/annotations/*_actors.json
                               →  videos/association/annotations/

``*_players.json`` stays in ``videos/reid/annotations/`` — naming a player IS
ReID, and it is the only human label that ever was.

The mapping is spelled out here rather than read from config, because a
migration must describe the move, not ask the code it is migrating where
things are now.

Dry-run by default; ``--apply`` performs it. Renames only — no copying, so
either the whole move happens or the source is untouched.

    uv run python scripts/migrate_stage_dirs.py
    uv run python scripts/migrate_stage_dirs.py --apply
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yp_video.config import VIDEOS_DIR  # noqa: E402

#: (source, destination) relative to VIDEOS_DIR. Whole directories.
DIRECTORIES: tuple[tuple[str, str], ...] = (
    ("reid/records", "extraction/records"),
    ("reid/crops", "extraction/crops"),
    ("reid/crops-masked", "extraction/crops-masked"),
    ("reid/tracks", "tracks"),
    ("reid/association", "association"),
)

#: Files matching a glob inside one directory, moved to another.
FILES: tuple[tuple[str, str, str], ...] = (
    ("reid/annotations", "*_actors.json", "association/annotations"),
)

#: Exported ReID datasets record an absolute ``crops_root`` and reference
#: crops relative to it (yp-reid Contract A). crops/ and crops-masked/ move
#: together, so every relative path still holds — only the root is stale, and
#: rewriting it is the difference between a valid dataset and one where not a
#: single sample resolves.
DATASETS_DIR = "reid/datasets"
CROPS_ROOT_WAS = "reid"
CROPS_ROOT_NOW = "extraction"


@dataclass(frozen=True)
class Move:
    src: Path
    dst: Path
    #: Files under src, for the report. A directory move is one rename
    #: whatever this says.
    files: int

    def describe(self, root: Path) -> str:
        return (
            f"  {self.src.relative_to(root)}  →  {self.dst.relative_to(root)}"
            f"   ({self.files} files)"
        )


def _count(path: Path) -> int:
    return sum(1 for p in path.rglob("*") if p.is_file())


def stale_manifests(root: Path) -> list[Path]:
    """Dataset manifests whose crops_root still points at the old layout."""
    out = []
    for manifest in sorted((root / DATASETS_DIR).glob("*/manifest.json")):
        try:
            crops_root = json.loads(manifest.read_text(encoding="utf-8"))["crops_root"]
        except (OSError, ValueError, KeyError):
            continue
        if Path(crops_root) == root / CROPS_ROOT_WAS:
            out.append(manifest)
    return out


def retarget(manifest: Path, root: Path) -> None:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["crops_root"] = str(root / CROPS_ROOT_NOW)
    tmp = manifest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, manifest)


def plan(root: Path) -> tuple[list[Move], list[str]]:
    """Every move to make, and every reason one was skipped."""
    moves: list[Move] = []
    skipped: list[str] = []

    for src_rel, dst_rel in DIRECTORIES:
        src, dst = root / src_rel, root / dst_rel
        if not src.exists():
            skipped.append(f"{src_rel}: not present")
            continue
        if dst.exists():
            skipped.append(f"{dst_rel}: destination already exists — move by hand")
            continue
        moves.append(Move(src, dst, _count(src)))

    for src_rel, pattern, dst_rel in FILES:
        src, dst = root / src_rel, root / dst_rel
        if not src.exists():
            skipped.append(f"{src_rel}: not present")
            continue
        matched = sorted(src.glob(pattern))
        clashes = [p for p in matched if (dst / p.name).exists()]
        if clashes:
            skipped.append(
                f"{dst_rel}: {len(clashes)} file(s) already there — move by hand"
            )
            continue
        for path in matched:
            moves.append(Move(path, dst / path.name, 1))

    return moves, skipped


def apply(moves: list[Move]) -> None:
    for move in moves:
        move.dst.parent.mkdir(parents=True, exist_ok=True)
        # os.rename moves a file or a whole directory in one atomic step —
        # which is the point: a half-migrated crops/ would look to every
        # reader like a video that lost most of its crops. Only a different
        # filesystem needs the copying fallback.
        try:
            os.rename(move.src, move.dst)
        except OSError:
            shutil.move(str(move.src), str(move.dst))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="perform the moves")
    args = parser.parse_args()

    root = VIDEOS_DIR
    print(f"videos root: {root}\n")
    moves, skipped = plan(root)
    manifests = stale_manifests(root)

    if moves:
        print(f"{len(moves)} move(s):")
        for move in moves:
            print(move.describe(root))
    else:
        print("nothing to move")
    if manifests:
        print(f"\n{len(manifests)} dataset manifest(s) to retarget:")
        for manifest in manifests:
            print(f"  {manifest.parent.relative_to(root)}  crops_root → {CROPS_ROOT_NOW}/")
    if skipped:
        print(f"\n{len(skipped)} skipped:")
        for reason in skipped:
            print(f"  {reason}")

    if not args.apply:
        print("\nDry run. Re-run with --apply to perform it.")
        return 0
    if not moves and not manifests:
        return 0

    apply(moves)
    for manifest in manifests:
        retarget(manifest, root)
    print(f"\nMoved {len(moves)} item(s), retargeted {len(manifests)} manifest(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
