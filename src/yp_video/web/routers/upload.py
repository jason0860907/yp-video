"""R2 cloud storage upload/download router."""

import asyncio
from collections.abc import Callable
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from yp_video.config import R2_CATEGORIES
from yp_video.web.jobs import job_manager, make_progress_callback
from yp_video.web.r2_client import r2_client

router = APIRouter()


class UploadRequest(BaseModel):
    category: str
    files: list[str]


class DownloadRequest(BaseModel):
    category: str
    files: list[str]  # R2 keys relative to category prefix


class DeleteLocalRequest(BaseModel):
    category: str
    files: list[str]
    force: bool = False  # If False, only delete files that exist on R2

# Categories that skip R2 upload (local-only)
LOCAL_ONLY_CATEGORIES = {"videos"}


@router.get("/status")
def get_status():
    """Check R2 configuration status."""
    return {
        "configured": r2_client.configured,
        "bucket": r2_client.bucket if r2_client.configured else None,
    }


@router.get("/files")
def list_local_files(category: str = "cuts") -> list[dict]:
    """List local files with R2 sync status."""
    base_dir = _get_base_dir(category)
    if not base_dir.exists():
        return []

    # Build set of existing R2 keys for fast lookup (skip for local-only categories)
    r2_keys: set[str] = set()
    is_local_only = category in LOCAL_ONLY_CATEGORIES
    if not is_local_only and r2_client.configured:
        try:
            for obj in r2_client.list_objects(prefix=f"{category}/"):
                r2_keys.add(obj["key"])
        except Exception:
            pass  # R2 unavailable, show all as un-synced

    files = []
    if category == "tad-checkpoints":
        # Nested: actionformer/{model}/{date}/{file} — only sync key files
        SYNC_FILES = {"best.pth.tar", "config.txt", "train_log.jsonl", "train_log.json"}
        for f in sorted(base_dir.rglob("*")):
            if f.is_file() and f.name in SYNC_FILES:
                rel = str(f.relative_to(base_dir))
                r2_key = f"{category}/{rel}"
                group = str(f.parent.relative_to(base_dir))
                files.append({
                    "name": f.name,
                    "path": rel,
                    "group": group,
                    "size": f.stat().st_size,
                    "r2_key": r2_key,
                    "uploaded": r2_key in r2_keys,
                })
    elif category == "rally_clips":
        # Nested: rally_clips/{video_stem}/clip.mp4
        for video_dir in sorted(base_dir.iterdir()):
            if video_dir.is_dir():
                for f in sorted(video_dir.glob("*.mp4")):
                    rel = str(f.relative_to(base_dir))
                    r2_key = f"{category}/{rel}"
                    files.append({
                        "name": f.name,
                        "path": rel,
                        "group": video_dir.name,
                        "size": f.stat().st_size,
                        "r2_key": r2_key,
                        "uploaded": r2_key in r2_keys,
                    })
    else:
        pattern = R2_CATEGORIES[category].glob_pattern
        # For videos dir, only list .mp4 at top level (not subdirectories)
        glob_results = sorted(base_dir.glob(pattern))
        if category == "videos":
            glob_results = [f for f in glob_results if f.parent == base_dir]
        for f in glob_results:
            rel = f.name
            r2_key = f"{category}/{rel}"
            files.append({
                "name": f.name,
                "path": rel,
                "group": None,
                "size": f.stat().st_size,
                "r2_key": r2_key,
                "uploaded": r2_key in r2_keys,
            })

    return files


@router.get("/r2-files")
def list_r2_files(category: str = "cuts") -> list[dict]:
    """List files on R2 for a category."""
    if not r2_client.configured:
        raise HTTPException(400, "R2 not configured")

    if category not in R2_CATEGORIES:
        raise HTTPException(400, f"Unknown category: {category}")

    objects = r2_client.list_objects(prefix=f"{category}/")
    # Check which ones exist locally
    base_dir = R2_CATEGORIES[category].local_dir
    result = []
    for obj in objects:
        # Strip category prefix to get relative path
        rel = obj["key"][len(category) + 1:]
        local_path = base_dir / rel
        entry: dict = {
            "name": Path(rel).name,
            "path": rel,
            "r2_key": obj["key"],
            "size": obj["size"],
            "local": local_path.exists(),
        }
        # Add group for nested categories
        rel_path = Path(rel)
        if len(rel_path.parts) > 1:
            entry["group"] = str(rel_path.parent)
        result.append(entry)
    return result


async def _run_transfer_jobs(
    jobs: list,
    base_dir: Path,
    transfer_fn: Callable[[Path, str, Callable | None], None],
    verb: str,
):
    """Run file transfer jobs (upload or download) sequentially with progress."""
    loop = asyncio.get_running_loop()

    for job in jobs:
        file_path = base_dir / job.params["file"]
        r2_key = job.params["r2_key"]

        try:
            await job_manager.update_job(
                job.id, status="running",
                message=f"{verb}ing {file_path.name}...",
            )

            progress_cb = make_progress_callback(
                job.id, loop, f"{verb}ing {{done}}/{{total}} bytes",
            )
            await loop.run_in_executor(
                None,
                lambda fp=file_path, k=r2_key, cb=progress_cb:
                    transfer_fn(fp, k, cb),
            )

            await job_manager.update_job(
                job.id, status="completed", progress=1.0,
                message=f"{verb}ed {file_path.name}",
            )
        except asyncio.CancelledError:
            await job_manager.update_job(job.id, status="cancelled", message="Cancelled")
            for remaining in jobs[jobs.index(job) + 1:]:
                await job_manager.update_job(
                    remaining.id, status="cancelled", message="Cancelled"
                )
            return
        except Exception as e:
            await job_manager.update_job(
                job.id, status="failed", error=str(e),
            )


@router.post("/start")
async def start_upload(req: UploadRequest):
    """Start upload jobs — one job per file."""
    if not r2_client.configured:
        raise HTTPException(400, "R2 not configured. Fill in r2.env file.")

    base_dir = _get_base_dir(req.category)

    jobs = []
    for file_path in req.files:
        local_path = base_dir / file_path
        if not local_path.exists():
            continue

        r2_key = f"{req.category}/{file_path}"
        job = job_manager.create_job("r2_upload", {
            "file": file_path,
            "category": req.category,
            "r2_key": r2_key,
            "size": local_path.stat().st_size,
        }, name=local_path.name)
        jobs.append(job)

    if not jobs:
        raise HTTPException(400, "No valid files to upload")

    def transfer(local_path, r2_key, cb):
        r2_client.upload_file(local_path, r2_key, on_progress=cb)

    task = asyncio.create_task(_run_transfer_jobs(jobs, base_dir, transfer, "Upload"))
    job_manager.attach_task(jobs, task)

    return [job.to_dict() for job in jobs]


@router.post("/download")
async def start_download(req: DownloadRequest):
    """Download files from R2 to local — one job per file."""
    if not r2_client.configured:
        raise HTTPException(400, "R2 not configured")

    base_dir = _get_base_dir(req.category)

    jobs = []
    for file_path in req.files:
        r2_key = f"{req.category}/{file_path}"
        job = job_manager.create_job("r2_download", {
            "file": file_path,
            "category": req.category,
            "r2_key": r2_key,
        }, name=Path(file_path).name)
        jobs.append(job)

    if not jobs:
        raise HTTPException(400, "No files to download")

    def transfer(local_path, r2_key, cb):
        r2_client.download_file(r2_key, local_path, on_progress=cb)

    task = asyncio.create_task(_run_transfer_jobs(jobs, base_dir, transfer, "Download"))
    job_manager.attach_task(jobs, task)

    return [job.to_dict() for job in jobs]


@router.post("/delete-local")
def delete_local_files(req: DeleteLocalRequest):
    """Delete local files. By default only deletes files already on R2.

    For local-only categories (e.g. videos), always allows deletion.
    """
    base_dir = _get_base_dir(req.category)
    is_local_only = req.category in LOCAL_ONLY_CATEGORIES

    deleted = []
    skipped = []

    # Pre-fetch R2 keys once instead of checking each file individually (N+1 → 1)
    r2_keys: set[str] = set()
    needs_r2_check = not req.force and not is_local_only and r2_client.configured
    if needs_r2_check:
        try:
            for obj in r2_client.list_objects(prefix=f"{req.category}/"):
                r2_keys.add(obj["key"])
        except Exception:
            pass

    for file_path in req.files:
        local_path = base_dir / file_path
        if not local_path.exists():
            continue

        # Safety check: only delete if already on R2 (unless force or local-only category)
        if needs_r2_check:
            r2_key = f"{req.category}/{file_path}"
            if r2_key not in r2_keys:
                skipped.append(file_path)
                continue

        local_path.unlink()
        deleted.append(file_path)

        # Clean up empty parent directories (for rally_clips)
        parent = local_path.parent
        if parent != base_dir and parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()

    return {"deleted": deleted, "skipped": skipped}


@router.get("/presign")
def get_presigned_url(key: str, expires: int = 3600):
    """Generate a presigned URL for an R2 object."""
    if not r2_client.configured:
        raise HTTPException(400, "R2 not configured")

    url = r2_client.generate_presigned_url(key, expires)
    return {"url": url, "expires": expires}


def _get_base_dir(category: str) -> Path:
    if category not in R2_CATEGORIES:
        raise HTTPException(400, f"Unknown category: {category}")
    return R2_CATEGORIES[category].local_dir
