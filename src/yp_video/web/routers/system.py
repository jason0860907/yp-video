"""System router - vLLM control and system info."""

import time

from fastapi import APIRouter
from pydantic import BaseModel, Field

from yp_video.config import (
    RALLY_ANNOTATIONS_DIR,
    ACTION_ANNOTATIONS_DIR,
    ACTION_PRE_ANNOTATIONS_DIR,
    RALLY_PRE_ANNOTATIONS_DIR,
    SEG_ANNOTATIONS_DIR,
    RAW_VIDEOS_DIR,
    count_files,
    cut_kind_of,
    iter_all_cuts,
)
from yp_video.web.jobs import job_manager
from yp_video.web.vllm_manager import vllm_manager

router = APIRouter()

# ── Presence (who has the page open right now) ────────────────────
# Each browser sends a heartbeat with a persistent random id every ~30 s;
# a client counts as online while its last beat is younger than the TTL, and
# as active while its latest beat says the user recently interacted (the
# idle threshold lives client-side). In-memory on purpose: a restart
# repopulates within one heartbeat.
_PRESENCE_TTL_S = 75.0
_presence: dict[str, tuple[float, bool]] = {}  # client_id -> (last_seen, is_active)


class PresenceBeat(BaseModel):
    client_id: str = Field(min_length=8, max_length=64)
    # False once the user has gone idle (no input past the client threshold).
    active: bool = True


@router.post("/presence")
def presence(beat: PresenceBeat) -> dict:
    """Record one heartbeat and return online/active client counts."""
    now = time.monotonic()
    _presence[beat.client_id] = (now, beat.active)
    for cid in [c for c, (seen, _a) in _presence.items() if now - seen > _PRESENCE_TTL_S]:
        del _presence[cid]
    return {
        "online": len(_presence),
        "active": sum(1 for _seen, is_active in _presence.values() if is_active),
    }


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
    for f in sorted(iter_all_cuts(), key=lambda p: p.name):
        stem = f.stem
        results.append({
            "name": f.name,
            "kind": cut_kind_of(f),
            # "Detected" means the whole video finished: rally-pre-annotations
            # is only written after detection + convert-to-rally completes.
            # seg-annotations is written incrementally, so a partial/aborted
            # run would leave a file there and falsely look done.
            "has_detection": (RALLY_PRE_ANNOTATIONS_DIR / f"{stem}_annotations.jsonl").exists(),
            "has_pre_annotation": (RALLY_PRE_ANNOTATIONS_DIR / f"{stem}_annotations.jsonl").exists(),
            "has_annotation": (RALLY_ANNOTATIONS_DIR / f"{stem}_annotations.jsonl").exists(),
        })
    return results


@router.get("/stats")
def get_stats():
    """Get pipeline statistics."""
    return {
        "videos": count_files(RAW_VIDEOS_DIR, "*.mp4"),
        "cuts": sum(1 for _ in iter_all_cuts()),
        "detections": count_files(SEG_ANNOTATIONS_DIR, "*.jsonl"),
        "pre_annotations": count_files(RALLY_PRE_ANNOTATIONS_DIR, "*.jsonl"),
        "annotations": count_files(RALLY_ANNOTATIONS_DIR, "*.jsonl"),
        "action_pre_annotations": count_files(ACTION_PRE_ANNOTATIONS_DIR, "*.jsonl"),
        "actions": count_files(ACTION_ANNOTATIONS_DIR, "*.jsonl"),
        "active_jobs": job_manager.active_count(),
    }
