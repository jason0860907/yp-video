"""System router - vLLM control and system info."""

from pathlib import Path

from fastapi import APIRouter

from app.jobs import job_manager
from app.vllm_manager import vllm_manager

router = APIRouter()

VIDEOS_DIR = Path.home() / "videos"


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
    cuts_dir = VIDEOS_DIR / "cuts"
    seg_dir = VIDEOS_DIR / "seg-annotations"
    pre_ann_dir = VIDEOS_DIR / "rally-pre-annotations"
    ann_dir = VIDEOS_DIR / "rally-annotations"
    pred_dir = VIDEOS_DIR / "tad-predictions"

    return {
        "videos": len(list(VIDEOS_DIR.glob("*.mp4"))) if VIDEOS_DIR.exists() else 0,
        "cuts": len(list(cuts_dir.glob("*.mp4"))) if cuts_dir.exists() else 0,
        "detections": len(list(seg_dir.glob("*.jsonl"))) if seg_dir.exists() else 0,
        "pre_annotations": len(list(pre_ann_dir.glob("*.jsonl"))) if pre_ann_dir.exists() else 0,
        "annotations": len(list(ann_dir.glob("*.jsonl"))) if ann_dir.exists() else 0,
        "predictions": len(list(pred_dir.glob("*.jsonl"))) if pred_dir.exists() else 0,
        "active_jobs": job_manager.active_count(),
    }
