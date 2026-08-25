"""The web-facing half of SPOT run plumbing.

Every SPOT run speaks the same stdout protocol; parsing it into live job
params (``TrainProgress`` / ``make_train_parsers``) and exporting checkpoint
packages mid-job (``PackageExporter``) genuinely need the job manager, so
they live here. Everything a run leaves on disk — discovery, packaging,
metrics — is ``yp_video.action.spot_runs``, re-exported below so routers
import one name; this module only translates its errors into HTTP ones.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException

from yp_video.action import spot_runs as _runs
from yp_video.action.spot_runs import (  # noqa: F401 — re-exported for routers
    actor_task_metrics,
    checkpoint_package_options,
    dedupe_run_name,
    export_checkpoint_package,
    load_json_file,
    spot_run_name,
)
from yp_video.web.job_helpers import ProgressParser
from yp_video.web.jobs import job_manager

log = logging.getLogger(__name__)


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
    latest_task_metrics: dict | None = None
    best_epoch: int | None = None
    best_value: float | None = None
    best_breakdown: dict | None = None
    best_task_metrics: dict | None = None


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
                "latest_task_metrics": ctx.latest_task_metrics,
                "best_epoch": ctx.best_epoch,
                "best_value": ctx.best_value,
                "best_breakdown": ctx.best_breakdown,
                "best_task_metrics": ctx.best_task_metrics,
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
        ctx.latest_train_loss = float(match.group(1).split()[-1])
        return {"params": training_params()}

    def on_val_loss(match: re.Match) -> dict:
        ctx.latest_val_loss = float(match.group(1).split()[-1])
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

    def on_task_metrics(match: re.Match) -> dict | None:
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        ctx.latest_task_metrics = payload
        return {"params": training_params()}

    def on_new_best_line(_match: re.Match) -> dict:
        ctx.best_epoch = ctx.current_epoch
        ctx.best_value = (
            ctx.latest_val_map if criterion == "map" else ctx.latest_val_loss
        )
        ctx.best_breakdown = ctx.latest_val_breakdown
        ctx.best_task_metrics = ctx.latest_task_metrics
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
            r"Train loss\s+((?:[0-9.]+\s*)+)",
            on_train_loss,
        ),
        ProgressParser(
            r"Val loss\s+((?:[0-9.]+\s*)+)",
            on_val_loss,
        ),
        ProgressParser(headline_pattern, on_val_map),
        ProgressParser(r"SPOT_METRICS (\{.*\})", on_val_metrics),
        ProgressParser(r"SPOT_TASK_METRICS (\{.*\})", on_task_metrics),
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
            or "SPOT_TASK_METRICS" in line
            or "Train loss" in line
            or "Val loss" in line
        )

    return parsers, is_key_line



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




def validate_checkpoint_dir(path: Path, *, root: Path) -> Path:
    try:
        return _runs.validate_checkpoint_dir(path, root=root)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def performance_payload(
    checkpoints_dir: Path,
    run: str | None = None,
    *,
    package_types: tuple[str, ...] | None = None,
) -> dict:
    try:
        return _runs.performance_payload(
            checkpoints_dir, run, package_types=package_types
        )
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
