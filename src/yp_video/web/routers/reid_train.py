"""ReID Train router — dataset export + embedder evaluation.

Two halves, both grounded in the same labeled crops:

- ``/performance`` scores every registered embedder on the labels we already
  have, per recording session (see reid/evaluate.py). This is the baseline a
  future fine-tune has to beat, and it also calibrates the clustering
  threshold the Label page's slider uses.
- ``/export`` + ``/start`` turn those labels into a yp-reid training dataset
  (Contract A, see reid/dataset.py), which is what yp-reid training consumes.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field

from yp_video.config import (
    REID_CHECKPOINTS_DIR,
    REID_DATASETS_DIR,
    REID_PKG_DIR,
    REID_PYTHON,
    REID_TRAIN_MODULE,
)
from yp_video.contracts.reid import REID_CONTRACT_VERSION, REID_CONTRACT_VERSION_ENV, REID_PROGRESS_PREFIX
from yp_video.core.cache import StatCache
from yp_video.reid import checkpoints, dataset, evaluate, sessions, store
from yp_video.reid.checkpoints import reid_engine_available
from yp_video.reid.embedder import build_embedders, threshold_calibration
from yp_video.web.job_helpers import ProgressParser, fail_job_from_exc, stream_subprocess
from yp_video.web.jobs import JobStatus, job_manager

log = logging.getLogger(__name__)
router = APIRouter()

JOB_TYPE = "reid_dataset_export"
TRAIN_JOB_TYPE = "reid_train"

# Evaluation is a few numpy ops over matrices that are already on disk, so it
# stays a plain request — but it re-runs on every page load and tab switch,
# so it caches on the files it derives from.
_eval_cache: StatCache = StatCache()


def _active_job() -> dict | None:
    for job in job_manager.jobs.values():
        if job.type in (JOB_TYPE, TRAIN_JOB_TYPE) and job.status in (JobStatus.PENDING, JobStatus.RUNNING):
            return job.to_dict()
    return None


def _session_payload(group: sessions.SessionGroup) -> dict:
    return {
        "id": group.id,
        "stems": list(group.stems),
        "players": list(group.players),
        "counts": group.counts,
        "shared": group.shared,
        "n_assigned": group.n_assigned,
        "is_isolated": group.is_isolated,
        "models": {stem: store.embedded_models(stem) for stem in group.stems},
    }


@router.get("/status")
def status() -> dict:
    """Session grouping, per-model coverage, existing exports, active job."""
    groups = sessions.build_sessions()
    labeled = [stem for g in groups for stem in g.stems]
    registry = build_embedders()
    return {
        "sessions": [_session_payload(g) for g in groups],
        "models": [
            {
                "name": name,
                "labeled_videos": sum(1 for s in labeled if name in store.embedded_models(s)),
                "threshold": threshold_calibration(name),
            }
            for name in registry
        ],
        "totals": {
            "labeled_videos": len(labeled),
            "assigned_events": sum(g.n_assigned for g in groups),
            "identities": sum(len(g.players) for g in groups),
            "sessions": len(groups),
        },
        "datasets": dataset.list_datasets(),
        "split_modes": list(dataset.SPLIT_MODES),
        "reid_engine_available": reid_engine_available(),
        "runs": checkpoints.list_checkpoints(),
        "active_job": _active_job(),
    }


@router.get("/runs")
def runs() -> dict:
    """Checkpoint packages on disk, best first — what the embedder binds to."""
    return {"runs": checkpoints.list_checkpoints()}


def _eval_sources(groups) -> list[Path]:
    """Every file the evaluation derives from — the cache's invalidation key."""
    paths: list[Path] = []
    for g in groups:
        for stem in g.stems:
            paths.append(store.players_path(stem))
            paths.append(store.reid_path(stem))
            for model in store.embedded_models(stem):
                paths.append(store.embedding_path(stem, model))
    return paths


@router.get("/performance")
async def performance(model: str | None = None) -> dict:
    """Per-model, per-session separability plus threshold suggestions.

    Note this takes ``model=`` rather than the other train routers' ``run=``:
    there are no training runs here yet, there are embedders.
    """
    groups = sessions.build_sessions()
    if not groups:
        return {"models": [], "evaluated_at": time.time()}
    registry = list(build_embedders())
    if model is not None:
        if model not in registry:
            raise HTTPException(400, f"Unknown embedder: {model} (have: {', '.join(registry)})")
        registry = [model]

    def compute() -> dict:
        return {**evaluate.evaluate_models(groups, registry), "evaluated_at": time.time()}

    try:
        # Off the event loop: numpy holds the GIL for the matmuls.
        return await asyncio.to_thread(
            _eval_cache.get, (tuple(g.id for g in groups), tuple(registry)), _eval_sources(groups), compute
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


def _plan(split_mode: str, test_ratio: float, seed: int, masked: bool):
    groups = sessions.build_sessions()
    if not groups:
        raise HTTPException(400, "No labeled videos — assign players on the ReID Label page first")
    try:
        return dataset.plan_export(
            groups, split_mode=split_mode, test_ratio=test_ratio, seed=seed, masked=masked
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/export")
def export_plan(
    split_mode: str = "auto",
    test_ratio: float = 0.25,
    seed: int = 42,
    masked: bool = False,
) -> Response:
    """The export plan as ndjson — inspect it before writing anything."""
    plan = _plan(split_mode, test_ratio, seed, masked)
    return Response(
        dataset.export_manifest_jsonl(plan),
        media_type="application/x-ndjson",
        headers={"Content-Disposition": 'attachment; filename="reid_dataset_plan.jsonl"'},
    )


class ReidExportRequest(BaseModel):
    name: str | None = None
    split_mode: str = Field(default="auto", pattern="^(auto|session|crops|all_train)$")
    test_ratio: float = Field(default=0.25, gt=0, lt=1)
    seed: int = 42
    #: Reference the background-suppressed crops the masked embedders saw.
    masked: bool = False
    overwrite: bool = False


@router.post("/start")
async def start(req: ReidExportRequest) -> dict:
    """Write the dataset to REID_DATASETS_DIR as a cancellable job."""
    plan = _plan(req.split_mode, req.test_ratio, req.seed, req.masked)
    if not plan.samples:
        raise HTTPException(400, "Nothing to export — every labeled event was dropped (see the plan)")

    name = req.name or f"reid_{plan.config['split_mode']}_{time.strftime('%Y%m%d-%H%M%S')}"
    if "/" in name or name.startswith("."):
        raise HTTPException(400, f"Invalid dataset name: {name}")
    root = REID_DATASETS_DIR / name
    if root.exists() and not req.overwrite:
        raise HTTPException(409, f"Dataset {name} already exists (enable overwrite)")

    job = job_manager.create_job(
        JOB_TYPE,
        {"name": name, "counts": plan.counts, "config": plan.config, "dropped": plan.dropped},
        name=f"ReID dataset ({plan.counts['n_samples']} crops)",
    )

    async def run_job() -> None:
        try:
            await job_manager.update_job(job.id, status=JobStatus.RUNNING, message="Writing dataset...")
            # Two small files; still off the event loop for the fs round trips.
            result = await asyncio.to_thread(dataset.write_export, plan, root)
            await job_manager.update_job(
                job.id,
                status=JobStatus.COMPLETED,
                progress=1.0,
                message=f"{result['n_samples']} samples -> {name}",
                params={**job.params, **result},
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — surfaced onto the job
            log.exception("ReID dataset export failed")
            await fail_job_from_exc(job.id, exc)

    job_manager.attach_task(job, asyncio.create_task(run_job()))
    return job.to_dict()


# ── training ─────────────────────────────────────────────────────────────


def _train_parsers() -> list[ProgressParser]:
    """REID_PROGRESS JSON lines → job progress/params (Contract C).

    One parser, defensive body handling — unlike SPOT there is no zoo of
    print formats to regex over.
    """
    import json as _json

    def handle(match) -> dict | None:
        try:
            data = _json.loads(match.group(1))
        except ValueError:
            return None
        phase = data.get("phase")
        if phase == "train":
            epoch, epochs = data.get("epoch", 1), max(1, data.get("epochs", 1))
            step, steps = data.get("step", 0), max(1, data.get("steps", 1))
            return {
                "progress": min(1.0, ((epoch - 1) + step / steps) / epochs),
                "message": f"epoch {epoch}/{epochs} · step {step}/{steps} · loss {data.get('loss')}",
            }
        if phase == "eval":
            return {
                "message": f"epoch {data.get('epoch')} · mAP {data.get('m_ap')} · rank1 {data.get('rank1')}",
                "params": {"last_eval": data},
            }
        if phase == "best":
            return {
                "message": f"new best · epoch {data.get('epoch')} · {data.get('value')}",
                "params": {"best": data},
            }
        return None

    return [ProgressParser(rf"{REID_PROGRESS_PREFIX}(\{{.*\}})", handle)]


class ReidTrainRequest(BaseModel):
    dataset: str
    run_name: str | None = None
    epochs: int = Field(default=4, ge=1)
    batch_size: int = Field(default=16, ge=2)
    lr: float = Field(default=4e-5, gt=0)
    #: Checkpoint package ref to fine-tune from (reid/checkpoints/<run>).
    init_checkpoint: str | None = None
    overwrite: bool = False


@router.post("/train")
async def train(req: ReidTrainRequest) -> dict:
    """Fine-tune yp-reid on an exported dataset, as a GPU-locked job.

    The subprocess writes the checkpoint package itself on every new best
    (Contract B), so a kill mid-run still leaves the best-so-far usable and
    the embedder registry picks it up without a restart.
    """
    dataset_root = REID_DATASETS_DIR / req.dataset
    if "/" in req.dataset or req.dataset.startswith(".") or not dataset_root.is_dir():
        raise HTTPException(404, f"Unknown dataset: {req.dataset}")
    run_name = req.run_name or f"reid_{time.strftime('%Y%m%d-%H%M%S')}"
    if "/" in run_name or run_name.startswith("."):
        raise HTTPException(400, f"Invalid run name: {run_name}")
    export_dir = REID_CHECKPOINTS_DIR / run_name
    if export_dir.exists() and not req.overwrite:
        raise HTTPException(409, f"Checkpoint package {run_name} already exists (enable overwrite)")
    init_package: Path | None = None
    if req.init_checkpoint:
        try:
            init_package = checkpoints.resolve_checkpoint(req.init_checkpoint)
        except (ValueError, FileNotFoundError, KeyError) as exc:
            raise HTTPException(400, f"Bad init_checkpoint: {exc}") from exc
    if not REID_PYTHON.exists():
        raise HTTPException(503, f"yp-reid venv missing: {REID_PYTHON} (run `uv sync` in yp-reid)")

    cmd = [
        str(REID_PYTHON), "-m", REID_TRAIN_MODULE,
        "--dataset", str(dataset_root),
        "--run-name", run_name,
        "--export-dir", str(export_dir),
        "--epochs", str(req.epochs),
        "--batch-size", str(req.batch_size),
        "--lr", str(req.lr),
    ]
    if init_package is not None:
        cmd += ["--init-checkpoint", str(init_package)]

    job = job_manager.create_job(
        TRAIN_JOB_TYPE,
        {"dataset": req.dataset, "run_name": run_name, "epochs": req.epochs,
         "batch_size": req.batch_size, "lr": req.lr, "init_checkpoint": req.init_checkpoint},
        name=f"ReID train ({req.dataset} → {run_name})",
    )

    async def run_job() -> None:
        try:
            await job_manager.update_job(job.id, status=JobStatus.RUNNING, message="Waiting for GPU...")
            async with job_manager.gpu_lock:
                rc, last_line = await stream_subprocess(
                    job.id,
                    cmd,
                    cwd=REID_PKG_DIR,
                    env={**os.environ, "PYTHONUNBUFFERED": "1",
                         REID_CONTRACT_VERSION_ENV: REID_CONTRACT_VERSION},
                    parsers=_train_parsers(),
                    is_key_line=lambda t: '"phase":"eval"' in t or '"phase":"best"' in t,
                    tee_to_terminal=True,
                    log_path=REID_PKG_DIR / "exp" / run_name / "terminal.log",
                )
            if rc != 0:
                raise RuntimeError(last_line or f"yp-reid train exited with code {rc}")
            manifest = checkpoints.read_manifest(export_dir)  # proves the package landed
            await job_manager.update_job(
                job.id,
                status=JobStatus.COMPLETED,
                progress=1.0,
                message=f"Training complete: {run_name} (best {manifest.get('best')})",
                params={**job.params, "package": checkpoints.checkpoint_ref(export_dir),
                        "best": manifest.get("best"), "metrics": manifest.get("metrics")},
            )
        except asyncio.CancelledError:
            await job_manager.update_job(job.id, status=JobStatus.CANCELLED, message="Training cancelled")
            raise
        except Exception as exc:  # noqa: BLE001 — surfaced onto the job
            log.exception("ReID training failed")
            await fail_job_from_exc(job.id, exc)

    job_manager.attach_task(job, asyncio.create_task(run_job()))
    return job.to_dict()
