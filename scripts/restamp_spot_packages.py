"""One-off: stamp pre-registry checkpoint packages with the task list.

Packages exported before the task registry declare their heads as
``config.json`` ``predict_*`` booleans and a per-family manifest ``type``.
Every reader now asks ``manifest["tasks"]`` / ``config["tasks"]`` instead, so
run this once over each package directory (after moving them into
``videos/spot/checkpoints``)::

    python scripts/restamp_spot_packages.py videos/spot/checkpoints [--dry-run]

Association packages (``yp-video-association-checkpoint``) are left alone.
Already-stamped packages are skipped, so the script is safe to rerun.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yp_video.contracts.action import (  # noqa: E402
    ACTION_CONTRACT_VERSION,
    ASSOCIATION_PACKAGE_TYPE,
    RECIPES,
    SPOT_PACKAGE_TYPE,
    validate_tasks,
)

LEGACY_TYPES = {
    "yp-video-action-checkpoint",
    "actor-association-spot",
    "yp-video-rally-spot-checkpoint",
}


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def legacy_tasks(package: Path, manifest: dict, config: dict) -> tuple[str, ...]:
    """The heads a pre-registry package trained, from what it left behind."""
    if manifest.get("type") == "yp-video-rally-spot-checkpoint" or any(
        (package / "labels").glob("*/*_rally.jsonl")
    ):
        tasks = ["rally"]
        if config.get("predict_winner"):
            tasks.append("winner")
    else:
        tasks = ["action"]
        if config.get("predict_location", True):
            tasks.append("location")
        if config.get("predict_actor"):
            tasks.append("actor")
    return validate_tasks(tasks)


def recipe_for(tasks: tuple[str, ...]) -> str | None:
    return next((r.id for r in RECIPES.values() if r.tasks == tasks), None)


def restamp(root: Path, *, dry_run: bool) -> None:
    for package in sorted(p for p in root.iterdir() if p.is_dir()):
        manifest_path, config_path = package / "manifest.json", package / "config.json"
        if not manifest_path.is_file() or not config_path.is_file():
            print(f"skip  {package.name}: no manifest/config")
            continue
        manifest, config = _read(manifest_path), _read(config_path)
        if manifest.get("type") == ASSOCIATION_PACKAGE_TYPE:
            print(f"skip  {package.name}: association package")
            continue
        if manifest.get("type") == SPOT_PACKAGE_TYPE and manifest.get("tasks") and config.get("tasks"):
            print(f"ok    {package.name}: already {manifest['tasks']}")
            continue
        if manifest.get("type") not in LEGACY_TYPES:
            print(f"skip  {package.name}: unknown type {manifest.get('type')!r}")
            continue
        tasks = legacy_tasks(package, manifest, config)
        manifest.update(
            {
                "type": SPOT_PACKAGE_TYPE,
                "tasks": list(tasks),
                "recipe": recipe_for(tasks),
                "contract_version": ACTION_CONTRACT_VERSION,
            }
        )
        for key in ("predict_location", "predict_vis", "predict_actor", "predict_winner"):
            config.pop(key, None)
        config["tasks"] = list(tasks)
        config["contract_version"] = ACTION_CONTRACT_VERSION
        print(f"stamp {package.name}: {list(tasks)}{' (dry run)' if dry_run else ''}")
        if not dry_run:
            _write(manifest_path, manifest)
            _write(config_path, config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", type=Path, help="checkpoint package directory")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    restamp(args.root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
