"""TAD inference router."""

import asyncio
import logging
import traceback

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

log = logging.getLogger(__name__)

from yp_video.config import (
    ANNOTATIONS_DIR,
    CUT_R2_CATEGORIES,
    FEATURES_DIR,
    PROJECT_ROOT,
    PREDICTIONS_DIR,
    PRE_ANNOTATIONS_DIR,
    VIDEOS_DIR,
    TAD_CONFIGS_DIR,
    TAD_CHECKPOINTS_DIR,
    cut_kind_of,
    find_cut,
    iter_all_cuts,
)
from yp_video.tad.extract_features import MODEL_CONFIGS
from yp_video.web.jobs import job_manager, JobStatus
from yp_video.web.r2_client import serve_video_or_r2_redirect

router = APIRouter()


class PredictRequest(BaseModel):
    videos: list[str]
    checkpoint: str
    threshold: float = 0.3
    device: str = "cuda"
    cut_rallies: bool = False
    model: str = "base"
    stop_vllm: bool = False


@router.get("/videos")
def list_videos() -> list[dict]:
    """List videos available for prediction.

    `features` maps each V-JEPA size to whether the corresponding feature
    file has already been extracted — lets the UI grey-out videos that
    would otherwise silently fail at inference time.
    """
    feat_stems: dict[str, set[str]] = {
        name: {p.stem for p in (FEATURES_DIR / cfg.dir_suffix).glob("*.npy")}
        if (FEATURES_DIR / cfg.dir_suffix).exists() else set()
        for name, cfg in MODEL_CONFIGS.items()
    }

    results = []
    for f in sorted(iter_all_cuts(), key=lambda p: p.name):
        stem = f.stem
        results.append({
            "name": f.name,
            "kind":               cut_kind_of(f),
            "has_prediction":     (PREDICTIONS_DIR / f"{stem}_annotations.jsonl").exists(),
            "has_annotation":     (ANNOTATIONS_DIR / f"{stem}_annotations.jsonl").exists(),
            "has_pre_annotation": (PRE_ANNOTATIONS_DIR / f"{stem}_annotations.jsonl").exists(),
            "features":           {name: stem in stems for name, stems in feat_stems.items()},
        })
    return results


@router.get("/results")
def list_results() -> list[str]:
    """List prediction result files."""
    if not PREDICTIONS_DIR.exists():
        return []
    return sorted(f.name for f in PREDICTIONS_DIR.glob("*.jsonl"))


@router.get("/results/{name}")
def get_result(name: str) -> dict:
    """Get prediction result contents."""
    from yp_video.core.jsonl import read_jsonl

    path = PREDICTIONS_DIR / name
    if not path.exists():
        raise HTTPException(404, "Result not found")

    meta, records = read_jsonl(path)
    meta["results"] = records
    return meta


@router.get("/video/{video_name:path}")
def stream_video(video_name: str):
    """Serve a video file for playback."""
    from urllib.parse import unquote

    decoded = unquote(video_name)
    video_path = find_cut(decoded)
    if video_path is None:
        # Fall through to R2 lookup using a placeholder path so the redirect
        # still works for files that exist remotely but not locally.
        from yp_video.config import CUTS_BROADCAST_DIR
        video_path = CUTS_BROADCAST_DIR / decoded
    response = serve_video_or_r2_redirect(video_path, CUT_R2_CATEGORIES)
    if response:
        return response
    raise HTTPException(404, f"Video not found: {decoded}")


@router.post("/start")
async def start_prediction(req: PredictRequest):
    """Start a single TAD prediction job that processes all selected videos."""
    checkpoint_path = PROJECT_ROOT / req.checkpoint
    if not checkpoint_path.exists():
        raise HTTPException(404, f"Checkpoint not found: {req.checkpoint}")

    for video_name in req.videos:
        if find_cut(video_name) is None:
            raise HTTPException(404, f"Video not found: {video_name}")

    total = len(req.videos)
    job = job_manager.create_job("infer", {
        "videos": req.videos,
        "checkpoint": req.checkpoint,
    }, name=f"Predict ({total} videos)")

    async def run_all():
        from yp_video.tad.infer import run_inference

        await job_manager.update_job(job.id, status="running", message="Waiting for GPU...")
        vllm_was_stopped = False
        if req.stop_vllm and job_manager.vllm_using_gpu:
            from yp_video.web.vllm_manager import vllm_manager
            await job_manager.update_job(job.id, message="Stopping vLLM to free VRAM...")
            await vllm_manager.stop()
            vllm_was_stopped = True
        try:
            async with job_manager.gpu_lock:
                loop = asyncio.get_event_loop()
                config_path = TAD_CONFIGS_DIR / "volleyball_actionformer.yaml"
                failed = 0

                for i, video_name in enumerate(req.videos):
                    video_path = find_cut(video_name)
                    if video_path is None:
                        raise HTTPException(404, f"Video not found: {video_name}")
                    prefix = f"({i + 1}/{total})"

                    def _make_cbs(pfx, jid):
                        def msg_cb(text):
                            loop.call_soon_threadsafe(
                                lambda t=text: asyncio.ensure_future(
                                    job_manager.update_job(jid, message=f"{pfx} {t}")
                                )
                            )
                        def prog_cb(frac):
                            loop.call_soon_threadsafe(
                                lambda f=frac: asyncio.ensure_future(
                                    job_manager.update_job(jid, progress=f)
                                )
                            )
                        return msg_cb, prog_cb

                    msg_cb, prog_cb = _make_cbs(prefix, job.id)

                    try:
                        await job_manager.update_job(
                            job.id, progress=0.0,
                            name=f"Predict ({i + 1}/{total}) — {video_name}",
                            message=f"{prefix} Starting {video_name}...",
                        )

                        output_path = PREDICTIONS_DIR / f"{video_path.stem}_annotations.jsonl"
                        cut_dir = None
                        if req.cut_rallies:
                            cut_dir = VIDEOS_DIR / "rally_clips" / video_path.stem

                        await loop.run_in_executor(
                            None,
                            lambda vp=video_path, op=output_path, cd=cut_dir, mc=msg_cb, pc=prog_cb: run_inference(
                                vp, checkpoint_path, config_path,
                                op, req.device, req.threshold, cd,
                                model_name=req.model,
                                on_message=mc,
                                on_progress=pc,
                            ),
                        )
                    except asyncio.CancelledError:
                        await job_manager.update_job(job.id, status="cancelled", message="Cancelled")
                        return
                    except Exception as e:
                        failed += 1
                        tb = traceback.format_exc()
                        # Ensure traceback lands in stderr so the tmux/api log shows it
                        print(f"\n[predict] Prediction failed for {video_name}:\n{tb}", flush=True)
                        log.error("Prediction failed for %s:\n%s", video_name, tb)
                        err_type = type(e).__name__
                        err_msg = str(e) or "<no message>"
                        job_obj = job_manager.get_job(job.id)
                        if job_obj:
                            job_obj.logs.append(f"[{video_name}] {err_type}: {err_msg}")
                            for line in tb.splitlines():
                                job_obj.logs.append(line)
                        await job_manager.update_job(
                            job.id,
                            message=f"{prefix} Failed: {video_name} — {err_type}: {err_msg}",
                            error=f"{err_type}: {err_msg}",
                        )

                final_name = f"Predict ({total} videos)"
                if failed == 0:
                    await job_manager.update_job(
                        job.id, status="completed", progress=1.0,
                        name=final_name,
                        message=f"All {total} videos complete",
                    )
                elif failed == total:
                    await job_manager.update_job(
                        job.id, status="failed", progress=1.0,
                        name=final_name,
                        message=f"All {total} videos failed — see logs",
                    )
                else:
                    await job_manager.update_job(
                        job.id, status="completed", progress=1.0,
                        name=final_name,
                        message=f"{total - failed}/{total} completed, {failed} failed",
                    )
        finally:
            if vllm_was_stopped:
                from yp_video.web.vllm_manager import vllm_manager
                log.info("Auto-restarting vLLM after prediction job %s", job.id)
                asyncio.create_task(vllm_manager.start())

    task = asyncio.create_task(run_all())
    job_manager.attach_task([job], task)

    return job.to_dict()
