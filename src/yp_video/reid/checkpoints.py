"""yp-reid checkpoint package discovery and resolution (Contract B, consumer side).

Packages live under ``REID_CHECKPOINTS_DIR/<run>/`` and are self-describing
via their manifest (see contracts/reid.py). This module only reads: packages
are written exclusively by yp-reid (training and import_weights), which is
what keeps one writer per format.

Mirrors action/prelabel.py's checkpoint handling: ``list_checkpoints`` for
the UI, ``default_checkpoint`` for unconfigured callers, ``resolve_checkpoint``
with containment checks for anything user-supplied.
"""

from __future__ import annotations

import json
from pathlib import Path

from yp_video.config import REID_CHECKPOINTS_DIR, REID_PYTHON, VIDEOS_DIR
from yp_video.contracts.reid import CHECKPOINT_MANIFEST_NAME, CHECKPOINT_TYPE, REID_CONTRACT_VERSION


def read_manifest(package: Path) -> dict:
    """Load and validate a package manifest; raise on anything off."""
    path = package / CHECKPOINT_MANIFEST_NAME
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("type") != CHECKPOINT_TYPE:
        raise ValueError(f"{path}: type {manifest.get('type')!r} != {CHECKPOINT_TYPE!r}")
    if manifest.get("contract_version") != REID_CONTRACT_VERSION:
        raise ValueError(
            f"{path}: contract_version {manifest.get('contract_version')!r} != {REID_CONTRACT_VERSION!r}"
        )
    if not (package / manifest["checkpoint"]).exists():
        raise FileNotFoundError(f"{package}: manifest names missing checkpoint {manifest['checkpoint']!r}")
    return manifest


def list_checkpoints() -> list[dict]:
    """Every readable package, best-metric first (imported packages without
    metrics sort last — a run fine-tuned on our own crops outranks generic
    released weights until its numbers say otherwise)."""
    rows = []
    if not REID_CHECKPOINTS_DIR.exists():
        return rows
    for manifest_path in REID_CHECKPOINTS_DIR.glob(f"*/{CHECKPOINT_MANIFEST_NAME}"):
        package = manifest_path.parent
        try:
            manifest = read_manifest(package)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue  # half-written or foreign directory; not a package
        best = manifest.get("best") or {}
        rows.append({
            "path": checkpoint_ref(package),
            "run_name": manifest.get("run_name", package.name),
            "source": manifest.get("source"),
            "architecture": manifest.get("model", {}).get("architecture"),
            "embedding_dim": manifest.get("model", {}).get("embedding_dim"),
            "best_metric": best.get("metric"),
            "best_value": best.get("value"),
            "metrics": manifest.get("metrics"),
            "created_at": manifest.get("created_at"),
            "note": manifest.get("note"),
            "mtime": manifest_path.stat().st_mtime,
        })
    rows.sort(
        key=lambda r: (r["best_value"] if r["best_value"] is not None else float("-inf"), r["mtime"]),
        reverse=True,
    )
    return rows


def default_checkpoint() -> Path | None:
    """The package the clip-reident embedder binds to; None when none exist."""
    rows = list_checkpoints()
    return resolve_checkpoint(rows[0]["path"]) if rows else None


def resolve_checkpoint(value: str | Path | None) -> Path:
    """Resolve a package reference to its validated directory.

    Accepts an absolute path, a ``reid/checkpoints/<run>`` ref (relative to
    VIDEOS_DIR), or a bare run name. Anything resolving outside
    REID_CHECKPOINTS_DIR or without a valid manifest raises.
    """
    if value is None:
        package = default_checkpoint()
        if package is None:
            raise FileNotFoundError(f"No ReID checkpoint package under {REID_CHECKPOINTS_DIR}")
        return package
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        root_rel = REID_CHECKPOINTS_DIR.relative_to(VIDEOS_DIR).parts
        path = VIDEOS_DIR / path if path.parts[: len(root_rel)] == root_rel else REID_CHECKPOINTS_DIR / path
    resolved = path.resolve()
    try:
        resolved.relative_to(REID_CHECKPOINTS_DIR.resolve())
    except ValueError:
        raise ValueError(f"ReID checkpoint must live under {REID_CHECKPOINTS_DIR}") from None
    read_manifest(resolved)
    return resolved


def checkpoint_ref(package: Path) -> str:
    """Display/API ref: path relative to VIDEOS_DIR (``reid/checkpoints/<run>``)."""
    try:
        return str(package.resolve().relative_to(VIDEOS_DIR.resolve()))
    except ValueError:
        return str(package.resolve())


def reid_engine_available() -> bool:
    """Whether the clip-reident embedder can run: yp-reid venv + a package."""
    return REID_PYTHON.exists() and default_checkpoint() is not None
