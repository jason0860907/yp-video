"""System router - vLLM control and system info."""

from fastapi import APIRouter

from yp_video.config import (
    ANNOTATIONS_DIR,
    CUTS_DIR,
    PREDICTIONS_DIR,
    PRE_ANNOTATIONS_DIR,
    SEG_ANNOTATIONS_DIR,
    VIDEOS_DIR,
)
from yp_video.web.jobs import job_manager
from yp_video.web.vllm_manager import vllm_manager

router = APIRouter()


@router.get("/vllm/status")
async def vllm_status():
    """Get vLLM server status."""
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


@router.get("/stats")
def get_stats():
    """Get pipeline statistics."""
    return {
        "videos": len(list(VIDEOS_DIR.glob("*.mp4"))) if VIDEOS_DIR.exists() else 0,
        "cuts": len(list(CUTS_DIR.glob("*.mp4"))) if CUTS_DIR.exists() else 0,
        "detections": len(list(SEG_ANNOTATIONS_DIR.glob("*.jsonl"))) if SEG_ANNOTATIONS_DIR.exists() else 0,
        "pre_annotations": len(list(PRE_ANNOTATIONS_DIR.glob("*.jsonl"))) if PRE_ANNOTATIONS_DIR.exists() else 0,
        "annotations": len(list(ANNOTATIONS_DIR.glob("*.jsonl"))) if ANNOTATIONS_DIR.exists() else 0,
        "predictions": len(list(PREDICTIONS_DIR.glob("*.jsonl"))) if PREDICTIONS_DIR.exists() else 0,
        "active_jobs": job_manager.active_count(),
    }
