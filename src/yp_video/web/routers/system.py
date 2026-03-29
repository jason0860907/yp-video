"""System router - vLLM control and system info."""

from fastapi import APIRouter

from yp_video.config import (
    ANNOTATIONS_DIR,
    CUTS_DIR,
    PREDICTIONS_DIR,
    PRE_ANNOTATIONS_DIR,
    SEG_ANNOTATIONS_DIR,
    TAD_FEATURES_DIR,
    VIDEOS_DIR,
    count_files,
)
from yp_video.web.jobs import job_manager
from yp_video.web.vllm_manager import vllm_manager

router = APIRouter()


@router.get("/vllm/status")
async def vllm_status():
    """Get vLLM server status with live health check."""
    await vllm_manager.sync_status()
    return vllm_manager.get_status_dict()


@router.post("/vllm/start")
async def vllm_start():
    """Start vLLM server."""
    return await vllm_manager.start()


@router.post("/vllm/stop")
async def vllm_stop():
    """Stop vLLM server."""
    return await vllm_manager.stop()


@router.get("/vllm/health")
async def vllm_health():
    """Check vLLM health."""
    healthy = await vllm_manager.check_health()
    return {"healthy": healthy}


@router.get("/videos")
def list_videos() -> list[dict]:
    """List cut videos with full pipeline status."""
    results = []
    if CUTS_DIR.exists():
        for f in sorted(CUTS_DIR.glob("*.mp4")):
            stem = f.stem
            results.append({
                "name": f.name,
                "has_detection": (SEG_ANNOTATIONS_DIR / f"{stem}.jsonl").exists(),
                "has_pre_annotation": (PRE_ANNOTATIONS_DIR / f"{stem}_annotations.jsonl").exists(),
                "has_annotation": (ANNOTATIONS_DIR / f"{stem}_annotations.jsonl").exists(),
                "has_features": (TAD_FEATURES_DIR / f"{stem}.npy").exists(),
            })
    return results


@router.get("/stats")
def get_stats():
    """Get pipeline statistics."""
    return {
        "videos": count_files(VIDEOS_DIR, "*.mp4"),
        "cuts": count_files(CUTS_DIR, "*.mp4"),
        "detections": count_files(SEG_ANNOTATIONS_DIR, "*.jsonl"),
        "pre_annotations": count_files(PRE_ANNOTATIONS_DIR, "*.jsonl"),
        "annotations": count_files(ANNOTATIONS_DIR, "*.jsonl"),
        "predictions": count_files(PREDICTIONS_DIR, "*.jsonl"),
        "active_jobs": job_manager.active_count(),
    }
