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


class DeleteR2Request(BaseModel):
    category: str
    files: list[str]  # paths relative to the category prefix

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
    elif category == "tad-features":
        # Nested: tad-features/{model}/*.npy
        for f in sorted(base_dir.glob("**/*.npy")):
            if f.is_file():
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


async def _run_batch_transfer(
    job,
    files: list[str],
    category: str,
    base_dir: Path,
    transfer_fn: Callable[[Path, str, Callable | None], None],
    verb: str,
):
    """Run file transfers as a single job with per-file progress."""
    loop = asyncio.get_running_loop()
    total = len(files)
    failed = 0

    for i, file_path in enumerate(files):
        local_path = base_dir / file_path
        r2_key = f"{category}/{file_path}"
        prefix = f"({i + 1}/{total})"

        try:
            await job_manager.update_job(
                job.id, status="running", progress=0.0,
                message=f"{prefix} {verb}ing {Path(file_path).name}...",
            )

            progress_cb = make_progress_callback(
                job.id, loop, prefix + f" {verb}ing {{done}}/{{total}} bytes",
            )
            await loop.run_in_executor(
                None,
                lambda fp=local_path, k=r2_key, cb=progress_cb:
                    transfer_fn(fp, k, cb),
            )
        except asyncio.CancelledError:
            await job_manager.update_job(job.id, status="cancelled", message="Cancelled")
            return
        except Exception as e:
            failed += 1
            await job_manager.update_job(
                job.id, message=f"{prefix} Failed: {Path(file_path).name} — {e}",
            )

    if failed == 0:
        await job_manager.update_job(
            job.id, status="completed", progress=1.0,
            message=f"{verb}ed all {total} files",
        )
    else:
        await job_manager.update_job(
            job.id, status="completed", progress=1.0,
            message=f"{total - failed}/{total} {verb.lower()}ed, {failed} failed",
        )


@router.post("/start")
async def start_upload(req: UploadRequest):
    """Start a single upload job for all selected files."""
    if not r2_client.configured:
        raise HTTPException(400, "R2 not configured. Fill in r2.env file.")

    base_dir = _get_base_dir(req.category)
    valid = [f for f in req.files if (base_dir / f).exists()]
    if not valid:
        raise HTTPException(400, "No valid files to upload")

    job = job_manager.create_job("r2_upload", {
        "category": req.category,
        "count": len(valid),
    }, name=f"Upload ({len(valid)} files)")

    def transfer(local_path, r2_key, cb):
        r2_client.upload_file(local_path, r2_key, on_progress=cb)

    task = asyncio.create_task(
        _run_batch_transfer(job, valid, req.category, base_dir, transfer, "Upload")
    )
    job_manager.attach_task([job], task)

    return job.to_dict()


@router.post("/download")
async def start_download(req: DownloadRequest):
    """Start a single download job for all selected files."""
    if not r2_client.configured:
        raise HTTPException(400, "R2 not configured")

    if not req.files:
        raise HTTPException(400, "No files to download")

    base_dir = _get_base_dir(req.category)
    job = job_manager.create_job("r2_download", {
        "category": req.category,
        "count": len(req.files),
    }, name=f"Download ({len(req.files)} files)")

    def transfer(local_path, r2_key, cb):
        r2_client.download_file(r2_key, local_path, on_progress=cb)

    task = asyncio.create_task(
        _run_batch_transfer(job, req.files, req.category, base_dir, transfer, "Download")
    )
    job_manager.attach_task([job], task)

    return job.to_dict()


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


@router.post("/delete-r2")
def delete_r2_files(req: DeleteR2Request):
    """Delete selected objects from R2. Does not touch local files."""
    if not r2_client.configured:
        raise HTTPException(400, "R2 not configured")
    if req.category not in R2_CATEGORIES:
        raise HTTPException(400, f"Unknown category: {req.category}")
    if req.category in LOCAL_ONLY_CATEGORIES:
        raise HTTPException(400, f"Category {req.category} has no R2 prefix")
    if not req.files:
        return {"deleted": 0}

    client = r2_client._get_client()
    deleted = 0
    # S3 delete_objects accepts up to 1000 keys per call
    CHUNK = 1000
    for i in range(0, len(req.files), CHUNK):
        batch = req.files[i:i + CHUNK]
        keys = [{"Key": f"{req.category}/{p}"} for p in batch]
        resp = client.delete_objects(Bucket=r2_client.bucket, Delete={"Objects": keys})
        deleted += len(resp.get("Deleted", []))
    return {"deleted": deleted}


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
