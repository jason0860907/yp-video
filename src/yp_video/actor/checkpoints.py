"""Association checkpoint repository.

A trained candidate is a file and nothing more. Nothing here activates: a
model is used by naming it in an Association Predict run, which writes its
answers into the records under its own name. There is deliberately no
"current model" setting to drift out of sync with what produced a record.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path

from yp_video.config import ASSOCIATION_DIR
from yp_video.core.cache import StatCache
from yp_video.core.jsonl import atomic_write
from yp_video.actor.model import AssociationModel

CHECKPOINTS_DIR = ASSOCIATION_DIR / "checkpoints"
MODEL_FILE = "model.json"
MANIFEST_FILE = "manifest.json"

_model_cache: StatCache = StatCache()


def _validate_name(name: str) -> str:
    if not name or "/" in name or name.startswith("."):
        raise ValueError(f"Invalid association checkpoint name: {name!r}")
    return name


def checkpoint_dir(name: str) -> Path:
    return CHECKPOINTS_DIR / _validate_name(name)


def save_candidate(
    model: AssociationModel,
    manifest: dict,
) -> Path:
    root = checkpoint_dir(model.name)
    if root.exists():
        raise FileExistsError(f"Association checkpoint {model.name} exists")
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    staging = CHECKPOINTS_DIR / f".{model.name}.{uuid.uuid4().hex}.tmp"
    staging.mkdir()
    try:
        with atomic_write(staging / MODEL_FILE) as file:
            json.dump(model.payload(), file, ensure_ascii=False, indent=2)
        with atomic_write(staging / MANIFEST_FILE) as file:
            json.dump(manifest, file, ensure_ascii=False, indent=2)
        os.replace(staging, root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return root


def load(name: str) -> AssociationModel:
    path = checkpoint_dir(name) / MODEL_FILE
    if not path.exists():
        raise FileNotFoundError(f"Unknown association checkpoint: {name}")

    def parse() -> AssociationModel:
        with open(path, encoding="utf-8") as file:
            return AssociationModel.from_payload(json.load(file))

    return _model_cache.get(name, [path], parse)


def usable_rejection(name: str) -> str | None:
    """Why this checkpoint cannot be run, or None when it can.

    A checkpoint that no longer LOADS is one of the answers. Feature contracts
    get retired and the checkpoints trained against them stay on disk; a
    listing that let that raise would take down the page for every other
    checkpoint too, over one file nobody can select anyway.
    """
    try:
        load(name)
    except (OSError, ValueError, KeyError) as exc:
        return str(exc)
    return None


def list_candidates() -> list[dict]:
    if not CHECKPOINTS_DIR.exists():
        return []
    rows: list[dict] = []
    for path in CHECKPOINTS_DIR.glob(f"*/{MANIFEST_FILE}"):
        # Only an unreadable MANIFEST hides a checkpoint: without it there is
        # nothing to show. A model that no longer loads still gets listed,
        # because a checkpoint silently missing from the page is a worse
        # answer than one shown with the reason it cannot be used.
        try:
            with open(path, encoding="utf-8") as file:
                manifest = json.load(file)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        rows.append(
            {
                **manifest,
                "name": path.parent.name,
                "mtime": path.stat().st_mtime,
            }
        )
    return sorted(rows, key=lambda row: row["mtime"], reverse=True)
