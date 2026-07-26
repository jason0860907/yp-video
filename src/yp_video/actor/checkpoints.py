"""Association checkpoint repository with explicit shadow activation."""

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
SHADOW_CONFIG = ASSOCIATION_DIR / "shadow.json"
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


def active_shadow_name() -> str | None:
    if not SHADOW_CONFIG.exists():
        return None
    with open(SHADOW_CONFIG, encoding="utf-8") as file:
        value = json.load(file).get("checkpoint")
    return str(value) if value else None


def load_active_shadow() -> AssociationModel | None:
    name = active_shadow_name()
    return load(name) if name is not None else None


def set_active_shadow(name: str | None) -> None:
    if name is not None:
        load(name)
    with atomic_write(SHADOW_CONFIG) as file:
        json.dump({"checkpoint": name}, file, ensure_ascii=False, indent=2)


def list_candidates() -> list[dict]:
    if not CHECKPOINTS_DIR.exists():
        return []
    active = active_shadow_name()
    rows: list[dict] = []
    for path in CHECKPOINTS_DIR.glob(f"*/{MANIFEST_FILE}"):
        # Only an unreadable MANIFEST hides a checkpoint: without it there is
        # nothing to show. A model that no longer loads — a retired feature
        # contract, a partial write — still gets listed, because a checkpoint
        # silently missing from the page is a worse answer than one shown with
        # the reason it cannot be used (see the router's shadow_blocked_on).
        try:
            with open(path, encoding="utf-8") as file:
                manifest = json.load(file)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        rows.append(
            {
                **manifest,
                "name": path.parent.name,
                "active_shadow": path.parent.name == active,
                "mtime": path.stat().st_mtime,
            }
        )
    return sorted(
        rows,
        key=lambda row: (row["active_shadow"], row["mtime"]),
        reverse=True,
    )
