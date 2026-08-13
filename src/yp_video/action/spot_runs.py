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
import logging
import shutil
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from yp_video.contracts.action import ACTION_PACKAGE_TYPE, FUSION_PACKAGE_TYPE

log = logging.getLogger(__name__)


# ── Run naming ────────────────────────────────────────────────────

def spot_run_name(*, view: str, task: str, feature_arch: str) -> str:
    """Canonical run name {date}_{view}_{task}_{model}.

    Mirrors yp-spot's ``RunName`` (its env is separate, so importing it is
    not an option): the model token is the backbone base with any temporal
    suffix stripped, and the "all" view is spelled out as "all_view".
    Tasks: ``act`` action spotting, ``ass_act`` +actor, ``ral`` rally.
    """
    model = feature_arch
    for suffix in ("_tsm", "_gsm"):
        model = model.removesuffix(suffix)
    view = "all_view" if view == "all" else view
    return f"{datetime.now():%Y%m%d}_{view}_{task}_{model}"


def dedupe_run_name(name: str, exp_dir: Path) -> str:
    """First of name, name-2, name-3... without a run dir in ``exp_dir``.

    The canonical name carries only the date, so a same-day rerun with the
    same config would land in the finished run's directory and clobber it.
    """
    candidate, i = name, 1
    while (exp_dir / candidate).exists():
        i += 1
        candidate = f"{name}-{i}"
    return candidate


def load_json_file(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


# ── Init-checkpoint options ───────────────────────────────────────

#: Package types whose weights are a SPOT model — what the yp-spot trainers
#: (Action, Fusion, Rally) can warm-start from. The independent association
#: package is deliberately absent: its weights are an AssociationModel, and
#: the SPOT loader's shape-matching init would load zero tensors from it
#: without a word of complaint.
SPOT_INIT_PACKAGE_TYPES = (ACTION_PACKAGE_TYPE, FUSION_PACKAGE_TYPE)


def checkpoint_package_options(
    checkpoints_dir: Path, *, package_types: Sequence[str]
) -> list[dict]:
    """Selectable init-checkpoint options: packaged runs under ``checkpoints_dir``.

    ``package_types`` names what the trainer behind the picker can actually
    load — several families share one checkpoints directory, so every caller
    must say which it eats. A package with no readable manifest type is
    excluded too: what it contains cannot be verified.
    """
    options: list[dict] = []
    if checkpoints_dir.exists():
        for run_dir in sorted(checkpoints_dir.iterdir(), reverse=True):
            ckpt = run_dir / "checkpoint_best.pt"
            if not run_dir.is_dir() or not ckpt.is_file():
                continue
            manifest = load_json_file(run_dir / "manifest.json")
            declared = (
                manifest.get("type") if isinstance(manifest, dict) else None
            )
            if declared not in package_types:
                continue
            options.append(
                {"label": _package_label(run_dir, manifest), "value": str(ckpt)}
            )
    return options


#: How a task's primary metric reads in a one-line picker label.
_METRIC_LABELS = {
    "harmonic_mAP": "mAP",
    "spatial_mAP": "loc mAP",
    "player_top1": "Top-1",
}


def _package_label(run_dir: Path, manifest: object) -> str:
    """One line describing a package by what it can actually serve.

    A multi-task package is labelled with EVERY serveable task's own best —
    ``(action mAP 0.225 · actor Top-1 0.628)`` — because a picker showing
    only the headline metric silently misdescribes every other task (the
    actor head's quality is not the action mAP). Packages without a
    ``best_per_task`` record fall back to the headline metric.
    """
    per_task = (
        manifest.get("best_per_task") if isinstance(manifest, dict) else None
    )
    parts = [
        f"{task} {_METRIC_LABELS.get(entry['metric'], entry['metric'])} "
        f"{entry['value']:.3f}"
        for task, entry in (per_task or {}).items()
        if entry.get("file") and isinstance(entry.get("value"), (int, float))
    ]
    if parts:
        return f"{run_dir.name} ({' · '.join(parts)})"
    best = load_json_file(run_dir / "checkpoint_best.json")
    value = best.get("value") if isinstance(best, dict) else None
    if not isinstance(value, (int, float)):
        return run_dir.name
    metric = best.get("metric")
    return f"{run_dir.name} ({'mAP' if metric == 'map' else metric or 'best'} {value:.3f})"

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


def best_epochs_per_task(run_dir: Path) -> dict[str, dict]:
    """Each declared task's best validation epoch, from the run's own records.

    Pure selection, no policy of its own: ``metrics.jsonl`` opens with a
    ``{"_meta": true}`` header where the trainer declares its tasks and each
    task's primary metric (``task_definitions``), and every epoch record
    carries each task's validation metrics (``tasks``). Both sides of the
    selection come from that one file. A task the trainer never declared or
    never validated is simply absent — which is how a future task (rally)
    joins without a code change here. A metric named ``loss`` ranks
    ascending; everything else descending.

    Each entry carries the FULL validation metrics of the winning epoch, so
    consumers that poll (checkpoint pickers) never have to re-read the
    multi-megabyte metrics file to describe a package.
    """
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        return {}
    definitions: dict = {}
    epochs: list[dict] = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("_meta") is True:
            declared = record.get("task_definitions")
            if isinstance(declared, dict):
                definitions = declared
        elif isinstance(record.get("epoch"), int):
            epochs.append(record)

    out: dict[str, dict] = {}
    for task, definition in definitions.items():
        metric = (definition or {}).get("primary_metric")
        if not metric:
            continue
        scored: list[tuple[int, float, dict]] = []
        for record in epochs:
            validation = (
                (record.get("tasks") or {}).get(task) or {}
            ).get("validation") or {}
            metrics = validation.get("metrics") or {}
            value = metrics.get(metric)
            if value is None and metric == "loss":
                value = validation.get("loss")
            if isinstance(value, (int, float)):
                scored.append((record["epoch"], float(value), metrics))
        if not scored:
            continue
        pick = min if metric.endswith("loss") else max
        epoch, value, metrics = pick(scored, key=lambda item: item[1])
        out[task] = {"epoch": epoch, "metric": metric, "value": value,
                     "metrics": metrics}
    return out


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
    serveable_tasks: Sequence[str] = (),
) -> dict:
    """Copy a finished run's durable artifacts into a checkpoint package.

    Heavy per-epoch files (``checkpoint_*.pt``, ``optim_*.pt``, prediction
    dumps) stay in the run dir; the package holds the best checkpoint, config,
    metrics, terminal log, the ``labels/<label_subdir>`` snapshot, and a
    ``manifest.json`` describing how it was trained.

    ``serveable_tasks`` names the tasks a predict surface can load on their
    own — the caller knows what its package serves, the same way it declares
    ``package_type``. Every declared task's best epoch is recorded in the
    manifest, but only a serveable task earns its own weights file: a task
    that can only ever be loaded alongside another (location rides with
    action) would duplicate ~80 MB nobody can use.
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

    # One best PER TASK, not one per run: a fusion run's action-best and
    # actor-best epochs rarely coincide, and shipping only the selection
    # criterion's epoch quietly serves every other task a compromised model.
    # ``file`` decouples the logical best from the physical weights, so a
    # task whose best IS the headline epoch points at checkpoint_best.pt
    # instead of duplicating it.
    headline_epoch = best.get("epoch") if isinstance(best, dict) else None
    best_per_task: dict[str, dict] = {}
    for task, pick in best_epochs_per_task(run_dir).items():
        entry = dict(pick)
        if task in serveable_tasks:
            if pick["epoch"] == headline_epoch:
                entry["file"] = "checkpoint_best.pt"
            else:
                source = run_dir / f"checkpoint_{pick['epoch']:03d}.pt"
                if source.is_file():
                    entry["file"] = f"checkpoint_best_{task}.pt"
                    shutil.copy2(source, package_dir / entry["file"])
                    copied.append(entry["file"])
                else:
                    log.warning(
                        "%s: %s-best epoch %d has no checkpoint file; the "
                        "task keeps only the headline best",
                        run_dir.name, task, pick["epoch"],
                    )
        best_per_task[task] = entry

    manifest = {
        "type": package_type,
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": package_dir.name,
        "source_run_dir": str(run_dir),
        "package_dir": str(package_dir),
        "checkpoint": "checkpoint_best.pt",
        "best": best if isinstance(best, dict) else None,
        "best_per_task": best_per_task,
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
        "best_per_task": best_per_task,
    }

# ── Per-epoch metrics for the performance charts ──────────────────

# The independent association trainer reports each phase as one flat dict;
# these keys are tallies, everything else is a rate. The task-metrics
# contract keeps them apart (``counts`` vs ``metrics``), so split here —
# the one place that reshapes trainer records for the UI.
_ACTOR_COUNT_KEYS = ("events", "player_events", "player_correct")


def actor_task_metrics(record: dict) -> dict:
    """Task-metrics contract for one independent-association epoch record."""
    loss = record.get("loss") or {}

    def phase(loss_value: float | None, flat: dict | None) -> dict:
        flat = flat or {}
        return {
            "loss": loss_value,
            "metrics": {k: v for k, v in flat.items() if k not in _ACTOR_COUNT_KEYS},
            "counts": {k: flat[k] for k in _ACTOR_COUNT_KEYS if k in flat},
        }

    return {
        "actor": {
            "primary_metric": "player_top1",
            "train": phase(loss.get("train"), record.get("train")),
            "validation": phase(loss.get("val"), record.get("val")),
        }
    }


def _normalize_metrics_entry(rec: dict) -> dict:
    """Flatten one epoch record into the flat shape the UI reads.

    Handles both the new ``metrics.jsonl`` schema (nested ``mAP``/``loss`` +
    ``lr``/``per_class``) and the legacy ``loss.json`` schema (flat ``val_mAP*``).
    """
    if isinstance(rec.get("val"), dict) or isinstance(rec.get("train"), dict):
        # Independent association trainer schema: per-epoch train/val metric
        # dicts, no mAP. Reshaped into the task-metrics contract so the same
        # performance card renders actor curves without a bespoke chart.
        loss = rec.get("loss") or {}
        return {
            "epoch": rec.get("epoch"),
            "lr": rec.get("lr"),
            "val_mAP": 0,
            "val_mAP_temporal": 0,
            "val_mAP_spatial": 0,
            "train_loss": loss.get("train"),
            "val_loss": loss.get("val"),
            "per_class": {},
            "val_per_video": [],
            "tasks": actor_task_metrics(rec),
            "selection": {"task": "actor", "metric": "player_top1", "mode": "max"},
        }
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
    package_types: Sequence[str] | None = None,
) -> dict:
    """Per-epoch validation metrics (lr, mAP, per-class, per-video) for a run.

    Reads ``metrics.jsonl`` (falling back to the legacy ``loss.json``) from a
    checkpoint package. Defaults to the most recently modified run; pass
    ``run`` to select one by name. ``runs`` lists the runs (newest first).
    ``package_types`` filters a shared checkpoint root to one model family by
    manifest type — the same declaration every other package reader keys on.
    """
    if not checkpoints_dir.exists():
        return {"entries": [], "runs": []}

    def has_metrics(d: Path) -> bool:
        return (d / "metrics.jsonl").exists() or (d / "loss.json").exists()

    def family_matches(d: Path) -> bool:
        if package_types is None:
            return True
        manifest = load_json_file(d / "manifest.json")
        declared = manifest.get("type") if isinstance(manifest, dict) else None
        return declared in package_types

    runs = sorted(
        (
            d
            for d in checkpoints_dir.iterdir()
            if d.is_dir() and has_metrics(d) and family_matches(d)
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
