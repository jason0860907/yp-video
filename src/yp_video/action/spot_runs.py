"""What a SPOT training run leaves on disk, readable without the web.

Both trainers (action spotting and rally segments) shell out to
``yp_spot.train`` and write the same run layout under ``yp-spot/exp/`` —
optimizer snapshots, ``metrics.jsonl`` / ``loss.json``, ``checkpoint_best.*``
— and finished runs are exported as checkpoint packages. This module owns
that on-disk knowledge once: run discovery, package export, per-epoch metric
reading. The live stdout protocol and job plumbing stay in
``yp_video.web.spot_runs``, which is the only part that needs a web server.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path

from yp_video.config import SPOT_DIR


def load_json_file(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


# ── Run discovery ──────────────────────────────────────────────────

# ── Run discovery (resume + init-checkpoint options) ──────────────


def last_resumable_epoch(run_dir: Path) -> int | None:
    """Latest epoch with optimizer state in ``run_dir``, or None if not resumable.

    Mirrors SPOT's ``get_last_epoch`` (globs ``optim_*.pt``): ``--resume`` needs
    the optimizer/scheduler snapshot, and SPOT prunes all but the latest one.
    """
    epochs = [
        int(m.group(1))
        for p in run_dir.glob("optim_*.pt")
        if (m := re.fullmatch(r"optim_(\d+)", p.stem))
    ]
    return max(epochs) if epochs else None


def resumable_run_options(prefix: str | tuple[str, ...] | None = None) -> list[dict]:
    """Runs under ``exp/`` that ``--resume`` can continue (have optimizer state).

    ``prefix`` restricts to one trainer's runs (the trainers share the
    ``exp/`` dir but use distinct run-name prefixes); a tuple matches any of
    its prefixes, exactly as ``str.startswith`` does.
    """
    exp_dir = SPOT_DIR / "exp"
    if not exp_dir.exists():
        return []
    options: list[dict] = []
    for run_dir in sorted(exp_dir.iterdir(), reverse=True):
        if not run_dir.is_dir() or (prefix and not run_dir.name.startswith(prefix)):
            continue
        last_epoch = last_resumable_epoch(run_dir)
        if last_epoch is None:
            continue
        best = load_json_file(run_dir / "checkpoint_best.json")
        best_value = best.get("value") if isinstance(best, dict) else None
        label = f"{run_dir.name} (E{last_epoch + 1}"
        if isinstance(best_value, (int, float)):
            label += f", best {best_value:.3f}"
        label += ")"
        options.append({"label": label, "value": str(run_dir)})
    return options


def checkpoint_package_options(checkpoints_dir: Path) -> list[dict]:
    """Selectable init-checkpoint options: packaged runs under ``checkpoints_dir``."""
    options: list[dict] = []
    if checkpoints_dir.exists():
        for run_dir in sorted(checkpoints_dir.iterdir(), reverse=True):
            ckpt = run_dir / "checkpoint_best.pt"
            if not run_dir.is_dir() or not ckpt.is_file():
                continue
            best = load_json_file(run_dir / "checkpoint_best.json")
            value = best.get("value") if isinstance(best, dict) else None
            label = run_dir.name
            if isinstance(value, (int, float)):
                metric = best.get("metric") if isinstance(best, dict) else None
                label = f"{run_dir.name} ({'mAP' if metric == 'map' else metric or 'best'} {value:.3f})"
            options.append({"label": label, "value": str(ckpt)})
    return options

# ── Checkpoint packages ────────────────────────────────────────────


def validate_checkpoint_dir(path: Path, *, root: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.parent != root.resolve():
        raise ValueError(f"Checkpoint dir must be directly under {root}")
    return resolved


def _reset_package_dir(package_dir: Path) -> None:
    package_dir.mkdir(parents=True, exist_ok=True)
    for child in package_dir.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink(missing_ok=True)


def export_checkpoint_package(
    *,
    run_dir: Path,
    package_dir: Path,
    checkpoints_root: Path,
    package_type: str,
    label_subdir: str,
    label_glob: str,
    training: dict,
    cmd: list[str],
) -> dict:
    """Copy a finished run's durable artifacts into a checkpoint package.

    Heavy per-epoch files (``checkpoint_*.pt``, ``optim_*.pt``, prediction
    dumps) stay in the run dir; the package holds the best checkpoint, config,
    metrics, terminal log, the ``labels/<label_subdir>`` snapshot, and a
    ``manifest.json`` describing how it was trained.
    """
    best_checkpoint = run_dir / "checkpoint_best.pt"
    if not best_checkpoint.exists():
        raise RuntimeError(f"checkpoint_best.pt was not found in {run_dir}")

    package_dir = validate_checkpoint_dir(package_dir, root=checkpoints_root)
    _reset_package_dir(package_dir)

    copied: list[str] = []
    for name in (
        "checkpoint_best.pt",
        "checkpoint_best.json",
        "config.json",
        "metrics.jsonl",
        "loss.json",
        "terminal.log",
    ):
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, package_dir / name)
            copied.append(name)

    src_label_dir = run_dir / "labels" / label_subdir
    if src_label_dir.exists():
        dst_label_dir = package_dir / "labels" / label_subdir
        dst_label_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_label_dir, dst_label_dir)
        copied.extend(
            str(path.relative_to(package_dir))
            for path in sorted(dst_label_dir.glob(label_glob))
        )

    best = load_json_file(run_dir / "checkpoint_best.json")
    config = load_json_file(run_dir / "config.json")
    manifest = {
        "type": package_type,
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": package_dir.name,
        "source_run_dir": str(run_dir),
        "package_dir": str(package_dir),
        "checkpoint": "checkpoint_best.pt",
        "best": best if isinstance(best, dict) else None,
        "config": config if isinstance(config, dict) else None,
        "training": training,
        "command": cmd,
        "files": copied,
        "omitted": [
            "checkpoint_*.pt",
            "optim_*.pt",
            "pred-val.*",
            "*.recall.json.gz",
        ],
    }
    manifest_path = package_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    copied.append("manifest.json")

    return {
        "dir": str(package_dir),
        "checkpoint": str(package_dir / "checkpoint_best.pt"),
        "files": copied,
        "best": manifest["best"],
    }

# ── Per-epoch metrics for the performance charts ──────────────────


def _normalize_metrics_entry(rec: dict) -> dict:
    """Flatten one epoch record into the flat shape the UI reads.

    Handles both the new ``metrics.jsonl`` schema (nested ``mAP``/``loss`` +
    ``lr``/``per_class``) and the legacy ``loss.json`` schema (flat ``val_mAP*``).
    """
    if "mAP" in rec:  # new metrics.jsonl schema
        m = rec.get("mAP") or {}
        loss = rec.get("loss") or {}
        return {
            "epoch": rec.get("epoch"),
            "lr": rec.get("lr"),
            "val_mAP": m.get("harmonic", 0),
            "val_mAP_temporal": m.get("temporal", 0),
            "val_mAP_spatial": m.get("spatial", 0),
            "train_loss": loss.get("train"),
            "val_loss": loss.get("val"),
            "per_class": rec.get("per_class") or {},
            "val_per_video": rec.get("per_video") or [],
            "tasks": rec.get("tasks") or {},
            "selection": rec.get("selection") or {},
        }
    return {  # legacy loss.json schema
        "epoch": rec.get("epoch"),
        "lr": rec.get("lr"),
        "val_mAP": rec.get("val_mAP", 0),
        "val_mAP_temporal": rec.get("val_mAP_temporal", 0),
        "val_mAP_spatial": rec.get("val_mAP_spatial", 0),
        "train_loss": rec.get("train"),
        "val_loss": rec.get("val"),
        "per_class": rec.get("per_class") or {},
        "val_per_video": rec.get("val_per_video") or [],
        "tasks": rec.get("tasks") or {},
        "selection": rec.get("selection") or {},
    }


def _read_run_metrics(run_dir: Path) -> tuple[dict | None, list[dict]]:
    """Read a run's per-epoch metrics, preferring metrics.jsonl over loss.json.

    Returns ``(meta, entries)`` where entries are normalized to the flat UI shape.
    """
    jsonl = run_dir / "metrics.jsonl"
    if jsonl.exists():
        meta: dict | None = None
        entries: list[dict] = []
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("_meta"):
                meta = rec
            else:
                entries.append(_normalize_metrics_entry(rec))
        return meta, entries

    loss = load_json_file(run_dir / "loss.json")
    if isinstance(loss, list):
        return None, [_normalize_metrics_entry(r) for r in loss]
    return None, []


def _freshest_metrics_dir(package_dir: Path) -> Path:
    """The dir whose metrics.jsonl is most current for this run.

    The checkpoint package only re-exports on a new best epoch, so mid-run the
    live training dir (manifest's source_run_dir) is ahead of the package —
    read it directly whenever it is fresher, so the per-epoch chart advances
    every epoch instead of every personal best.
    """
    manifest = load_json_file(package_dir / "manifest.json")
    src_value = manifest.get("source_run_dir") if isinstance(manifest, dict) else None
    if not src_value:
        return package_dir
    live = Path(src_value) / "metrics.jsonl"
    packaged = package_dir / "metrics.jsonl"
    try:
        if live.exists() and (
            not packaged.exists() or live.stat().st_mtime > packaged.stat().st_mtime
        ):
            return live.parent
    except OSError:
        pass
    return package_dir


def performance_payload(
    checkpoints_dir: Path,
    run: str | None = None,
    *,
    run_prefixes: tuple[str, ...] | None = None,
) -> dict:
    """Per-epoch validation metrics (lr, mAP, per-class, per-video) for a run.

    Reads ``metrics.jsonl`` (falling back to the legacy ``loss.json``) from a
    checkpoint package. Defaults to the most recently modified run; pass
    ``run`` to select one by name. ``runs`` lists the runs (newest first).
    ``run_prefixes`` lets a shared checkpoint root expose only one model
    family without duplicating the metrics reader.
    """
    if not checkpoints_dir.exists():
        return {"entries": [], "runs": []}

    def has_metrics(d: Path) -> bool:
        return (d / "metrics.jsonl").exists() or (d / "loss.json").exists()

    runs = sorted(
        (
            d
            for d in checkpoints_dir.iterdir()
            if d.is_dir()
            and has_metrics(d)
            and (
                run_prefixes is None
                or d.name.startswith(run_prefixes)
            )
        ),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        return {"entries": [], "runs": []}

    run_dir = (
        next((candidate for candidate in runs if candidate.name == run), None)
        if run
        else runs[0]
    )
    if run_dir is None:
        raise LookupError(f"No metrics for run {run!r}")

    meta, entries = _read_run_metrics(_freshest_metrics_dir(run_dir))
    best = load_json_file(run_dir / "checkpoint_best.json")
    return {
        "run": run_dir.name,
        "meta": meta,
        "best": best if isinstance(best, dict) else None,
        "entries": entries,
        "runs": [d.name for d in runs],
    }
