"""Shared plumbing for SPOT training runs (action spotting and rally segments).

Both trainers shell out to ``yp_spot.train`` and speak the same stdout
protocol, write the same ``metrics.jsonl`` / ``checkpoint_best.*`` run layout
under ``yp-spot/exp/``, and export finished runs as checkpoint packages. This
module owns that machinery once; the routers keep only what genuinely differs
(labels, request models, run naming).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from fastapi import HTTPException

from yp_video.config import SPOT_DIR
from yp_video.web.job_helpers import ProgressParser
from yp_video.web.jobs import job_manager

log = logging.getLogger(__name__)


def load_json_file(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


# ── Live progress parsing (yp-spot stdout → job params) ───────────


@dataclass(slots=True)
class TrainProgress:
    """Mutable running state for a SPOT training job's progress parsers.

    A dataclass (not a dict) so a mis-typed field raises AttributeError instead
    of silently creating a dead key — the parsers below all mutate this from
    different regex callbacks.
    """

    epochs: int
    completed_epoch: int = -1
    current_epoch: int = 0
    train_total: int = 0
    latest_train_loss: float | None = None
    latest_val_loss: float | None = None
    latest_val_map: float | None = None
    latest_val_breakdown: dict | None = None
    best_epoch: int | None = None
    best_value: float | None = None
    best_breakdown: dict | None = None


def make_train_parsers(
    ctx: TrainProgress,
    *,
    params_key: str,
    criterion: str,
    headline_pattern: str,
    on_new_best: Callable[[], None] | None = None,
    base_progress: float = 0.2,
) -> tuple[list[ProgressParser], Callable[[str], bool]]:
    """Build the stdout parsers for one ``yp_spot.train`` subprocess.

    ``params_key`` is where the live snapshot lands in ``job.params``;
    ``headline_pattern`` matches the per-epoch validation metric line (one
    percent-valued group) — "Harmonic mean …" for actions, "Segment mAP …" for
    rally. ``on_new_best`` runs after the best-epoch state updates (checkpoint
    package export). Job progress maps preparation to ``[0, base_progress)``
    and training to the rest.
    """

    def training_params(**extra) -> dict:
        return {
            params_key: {
                "epoch": ctx.current_epoch,
                "epoch_display": ctx.current_epoch + 1,
                "epochs": max(1, ctx.epochs),
                "completed_epoch": ctx.completed_epoch,
                "latest_train_loss": ctx.latest_train_loss,
                "latest_val_loss": ctx.latest_val_loss,
                "latest_val_map": ctx.latest_val_map,
                "latest_val_breakdown": ctx.latest_val_breakdown,
                "best_epoch": ctx.best_epoch,
                "best_value": ctx.best_value,
                "best_breakdown": ctx.best_breakdown,
                **extra,
            }
        }

    def phase_progress(epoch: int, phase: str, step: int, total: int) -> float:
        phase_offsets = {"train": 0.0, "val": 0.78, "map": 0.94}
        phase_weights = {"train": 0.78, "val": 0.16, "map": 0.06}
        frac = step / max(1, total)
        epoch_frac = phase_offsets[phase] + phase_weights[phase] * frac
        total_epochs = max(1, ctx.epochs)
        return min(
            0.99,
            base_progress
            + (0.99 - base_progress) * ((epoch + epoch_frac) / total_epochs),
        )

    def on_epoch(match: re.Match) -> dict:
        epoch = int(match.group(1))
        ctx.completed_epoch = max(ctx.completed_epoch, epoch)
        ctx.current_epoch = epoch
        return {
            "params": training_params(phase="summary", phase_label="Epoch summary"),
        }

    def on_config_epochs(match: re.Match) -> None:
        ctx.epochs = int(match.group(1))
        return None

    def on_tqdm(match: re.Match) -> dict:
        step = int(match.group("step"))
        total = int(match.group("total"))
        tail = match.group("tail") or ""
        if "sum=" in tail:
            if total >= int(ctx.train_total or 0):
                ctx.train_total = total
                phase = "train"
                epoch = max(0, int(ctx.completed_epoch) + 1)
            else:
                phase = "val"
                epoch = max(0, int(ctx.current_epoch))
        else:
            phase = "map"
            epoch = max(0, int(ctx.current_epoch))

        ctx.current_epoch = epoch
        phase_label = {
            "train": "Training",
            "val": "Validation loss",
            "map": "mAP evaluation",
        }[phase]
        loss_match = re.search(r"sum=([0-9.]+)", tail)
        current_loss = float(loss_match.group(1)) if loss_match else None
        pct = int(step * 100 / max(1, total))
        total_epochs = max(1, ctx.epochs)
        return {
            "progress": phase_progress(epoch, phase, step, total),
            "message": (
                f"Epoch {epoch + 1}/{total_epochs} - "
                f"{phase_label} {step}/{total} ({pct}%)"
            ),
            "params": training_params(
                phase=phase,
                phase_label=phase_label,
                step=step,
                total=total,
                phase_progress=step / max(1, total),
                current_loss=current_loss,
            ),
        }

    def on_train_loss(match: re.Match) -> dict:
        ctx.latest_train_loss = float(match.group(4))
        return {"params": training_params()}

    def on_val_loss(match: re.Match) -> dict:
        ctx.latest_val_loss = float(match.group(4))
        return {"params": training_params()}

    def on_val_map(match: re.Match) -> dict:
        ctx.latest_val_map = float(match.group(1)) / 100.0
        return {"params": training_params()}

    def on_val_metrics(match: re.Match) -> dict | None:
        try:
            ctx.latest_val_breakdown = json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
        return {"params": training_params()}

    def on_new_best_line(_match: re.Match) -> dict:
        ctx.best_epoch = ctx.current_epoch
        ctx.best_value = (
            ctx.latest_val_map if criterion == "map" else ctx.latest_val_loss
        )
        ctx.best_breakdown = ctx.latest_val_breakdown
        if on_new_best is not None:
            on_new_best()
        return {"params": training_params()}

    parsers = [
        ProgressParser(r'"num_epochs":\s*(\d+)', on_config_epochs),
        ProgressParser(
            r"(?P<pct>\d+)%\|.*?\|\s*(?P<step>\d+)/(?P<total>\d+)\s*\[[^\]]+\](?P<tail>.*)",
            on_tqdm,
        ),
        ProgressParser(r"Epoch:\s*(\d+)", on_epoch),
        ProgressParser(
            r"Train loss\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)",
            on_train_loss,
        ),
        ProgressParser(
            r"Val loss\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)",
            on_val_loss,
        ),
        ProgressParser(headline_pattern, on_val_map),
        ProgressParser(r"SPOT_METRICS (\{.*\})", on_val_metrics),
        ProgressParser(r"New best epoch!", on_new_best_line),
    ]

    def is_key_line(line: str) -> bool:
        return (
            "Epoch:" in line
            or "Best epoch" in line
            or "New best epoch" in line
            or "Harmonic mean" in line
            or "Segment mAP" in line
            or "SPOT_METRICS" in line
            or "Train loss" in line
            or "Val loss" in line
        )

    return parsers, is_key_line


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


def resumable_run_options(prefix: str | None = None) -> list[dict]:
    """Runs under ``exp/`` that ``--resume`` can continue (have optimizer state).

    ``prefix`` restricts to one trainer's runs (action and rally share the
    ``exp/`` dir but use distinct run-name prefixes).
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
        raise HTTPException(400, f"Checkpoint dir must be directly under {root}")
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


class PackageExporter:
    """Exports a run's checkpoint package once ``checkpoint_best`` is ready.

    yp-spot writes ``checkpoint_best.pt`` + ``.json`` shortly after printing
    "New best epoch!", so each export waits (up to a minute) for the files to
    reach the expected epoch before copying. A lock serializes overlapping
    exports; ``schedule`` fire-and-forgets one from a sync parser callback.
    """

    def __init__(self, job_id: str, run_dir: Path, export_fn: Callable[[], dict]):
        self._job_id = job_id
        self._run_dir = run_dir
        self._export_fn = export_fn
        self._lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()

    async def export_once(
        self,
        *,
        expected_epoch: int | None,
        reason: str,
        update_job: bool = True,
    ) -> dict | None:
        for _ in range(120):
            best = load_json_file(self._run_dir / "checkpoint_best.json")
            best_epoch = best.get("epoch") if isinstance(best, dict) else None
            ready = (
                (self._run_dir / "checkpoint_best.pt").exists()
                and isinstance(best_epoch, int)
                and (expected_epoch is None or best_epoch == expected_epoch)
            )
            if ready:
                async with self._lock:
                    summary = await asyncio.to_thread(self._export_fn)
                if update_job:
                    job = job_manager.get_job(self._job_id)
                    await job_manager.update_job(
                        self._job_id,
                        params={
                            **(job.params if job else {}),
                            "checkpoint_package": summary,
                            "checkpoint_package_reason": reason,
                        },
                    )
                return summary
            await asyncio.sleep(0.5)

        log.warning(
            "Timed out waiting to export checkpoint package for %s "
            "(expected_epoch=%s, run_dir=%s)",
            reason,
            expected_epoch,
            self._run_dir,
        )
        return None

    def schedule(self, expected_epoch: int | None, reason: str) -> None:
        task = asyncio.create_task(
            self.export_once(expected_epoch=expected_epoch, reason=reason)
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)


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


def performance_payload(checkpoints_dir: Path, run: str | None = None) -> dict:
    """Per-epoch validation metrics (lr, mAP, per-class, per-video) for a run.

    Reads ``metrics.jsonl`` (falling back to the legacy ``loss.json``) from a
    checkpoint package. Defaults to the most recently modified run; pass
    ``run`` to select one by name. ``runs`` lists the runs (newest first).
    """
    if not checkpoints_dir.exists():
        return {"entries": [], "runs": []}

    def has_metrics(d: Path) -> bool:
        return (d / "metrics.jsonl").exists() or (d / "loss.json").exists()

    runs = sorted(
        (d for d in checkpoints_dir.iterdir() if d.is_dir() and has_metrics(d)),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        return {"entries": [], "runs": []}

    run_dir = (checkpoints_dir / run) if run else runs[0]
    if not has_metrics(run_dir):
        raise HTTPException(404, f"No metrics for run {run_dir.name!r}")

    meta, entries = _read_run_metrics(_freshest_metrics_dir(run_dir))
    best = load_json_file(run_dir / "checkpoint_best.json")
    return {
        "run": run_dir.name,
        "meta": meta,
        "best": best if isinstance(best, dict) else None,
        "entries": entries,
        "runs": [d.name for d in runs],
    }
