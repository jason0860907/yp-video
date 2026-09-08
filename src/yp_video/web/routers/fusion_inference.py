"""Inference: one fusion checkpoint, every stage, for a batch of videos.

The single-stage predict pages remain for re-running one stage with its own
knobs; this page is the "give me everything this model knows" button. The
stage work itself lives router-free in ``web/fusion_inference.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import Field

from yp_video.action import prelabel
from yp_video.action.spot_pass import RallyOptions, SpotOptions
from yp_video.config import SPOT_CHECKPOINTS_DIR, SPOT_DIR, cut_kind_of
from yp_video.extraction.prerequisites import prerequisites
from yp_video.tracklets.store import tracks_current
from yp_video.web import fusion_inference
from yp_video.web.action_annotations import pre_annotation_path
from yp_video.web.job_helpers import init_batch_items, spawn_batch_video_job
from yp_video.web.jobs import JobSummary, JobType, job_manager
from yp_video.web.r2_client import all_cut_paths, resolve_cut, sync_to_r2
from yp_video.web.schemas import StrictModel

log = logging.getLogger(__name__)
router = APIRouter()


class InferenceRequest(StrictModel):
    videos: list[str] = Field(min_length=1)
    checkpoint: str = ""
    rally_min_score: float = Field(default=0.5, ge=0.0, le=1.0)
    max_gap_s: float = Field(default=2.0, ge=0.0, le=30.0)
    min_duration_s: float = Field(default=4.0, ge=0.0, le=60.0)
    action_min_score: float = Field(default=0.15, ge=0.0, le=1.0)
    batch_size: int = Field(default=16, ge=1, le=128)
    clip_len: int = Field(default=64, ge=8, le=256)
    #: ffmpeg decode threads; 0 lets ffmpeg pick.
    num_workers: int = Field(default=0, ge=0, le=32)
    use_amp: bool = True
    #: Redo stages whose machine output already exists. Human labels are
    #: never touched either way.
    overwrite: bool = False
    stop_vllm: bool = False


@router.get("/videos")
def list_videos() -> list[dict]:
    """Every cut, with which stage outputs it already has."""
    rows = []
    for path in sorted(all_cut_paths(), key=lambda p: p.name):
        stem = path.stem
        rows.append({
            "name": path.name,
            "kind": cut_kind_of(path),
            "has_rally_spot": fusion_inference.rally_spot_pre_annotation_path(stem).exists(),
            "has_action_pre": pre_annotation_path(stem).exists(),
            "tracks_current": tracks_current(stem),
            "pipeline": prerequisites(stem).payload(),
        })
    return rows


@router.get("/spot")
def spot_info() -> dict:
    available = prelabel.spot_available()
    info: dict = {"available": available, "spot_dir": str(SPOT_DIR)}
    if not available:
        info["error"] = f"yp-spot not found at {SPOT_DIR}"
        return info
    checkpoints = fusion_inference.list_checkpoints()
    info["checkpoints"] = checkpoints
    info["default_checkpoint"] = fusion_inference.default_checkpoint()
    if not checkpoints:
        info["error"] = (
            f"No package under {SPOT_CHECKPOINTS_DIR} serves rally, action and "
            "actor together; train an Action + Rally + Winner recipe first."
        )
    return info


@router.post("/start", response_model=JobSummary)
async def start(req: InferenceRequest) -> dict:
    try:
        checkpoint = fusion_inference.resolve_checkpoint(req.checkpoint)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    video_paths: list[Path] = []
    for name in req.videos:
        path = resolve_cut(Path(name).name)
        if path is None:
            raise HTTPException(404, f"Video not found: {name}")
        if path not in video_paths:
            video_paths.append(path)

    rally = RallyOptions(
        min_score=req.rally_min_score,
        max_gap_s=req.max_gap_s,
        min_duration_s=req.min_duration_s,
    )
    spot = SpotOptions(
        batch_size=req.batch_size,
        num_workers=req.num_workers,
        clip_len=req.clip_len,
        use_amp=req.use_amp,
    )
    job = job_manager.create_job(
        JobType.FUSION_INFERENCE,
        {
            "videos": [p.name for p in video_paths],
            "checkpoint": prelabel.checkpoint_ref(checkpoint),
            "overwrite": req.overwrite,
            "items": init_batch_items([p.name for p in video_paths]),
        },
        name=f"Inference ({len(video_paths)} videos)",
    )

    def mirror(_path: Path, result: fusion_inference.VideoResult) -> None:
        for written, category in result.written:
            sync_to_r2(written, category)

    spawn_batch_video_job(
        job,
        video_paths,
        stop_vllm=req.stop_vllm,
        work=lambda path, cb: fusion_inference.run_video(
            video=path,
            checkpoint=checkpoint,
            rally=rally,
            action_min_score=req.action_min_score,
            spot=spot,
            overwrite=req.overwrite,
            on_progress=cb,
        ),
        done_message=fusion_inference.summarize,
        start_message="fetching video",
        on_done=mirror,
    )
    return job.to_dict()
