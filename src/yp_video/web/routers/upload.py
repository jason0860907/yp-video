"""R2 cloud storage upload/download router."""

import asyncio
import time
from collections.abc import Callable
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from yp_video.config import R2_CATEGORIES
from yp_video.web.job_helpers import (
    batch_message,
    fail_job_from_exc,
    finalize_batch_job,
    terminal_prefix,
)
from yp_video.web.jobs import job_manager, threadsafe_update
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

# terminal.log is intentionally excluded: it's a bulky training-process record,
# kept locally in the package dir but neither listed nor uploaded to R2.
CHECKPOINT_PACKAGE_FILES = {
    "checkpoint_best.pt",
    "checkpoint_best.json",
    "config.json",
    "loss.json",
    "manifest.json",
    "metrics.jsonl",
}

# Nested checkpoint-package categories: {run}/<package file> plus the training
# label snapshot under {run}/labels/<subdir>/*.jsonl. The subdir names are
# baked into existing checkpoint packages (local and on R2) — they are package
# structure, not category names, and deliberately did not follow the category
# rename to family paths.
CHECKPOINT_CATEGORIES = {
    "action/checkpoints": "action-annotations",
    "rally-spot/checkpoints": "rally-annotations",
}


def _is_checkpoint_package_file(path: Path, base_dir: Path, labels_subdir: str) -> bool:
    rel = path.relative_to(base_dir)
    if len(rel.parts) == 2 and rel.name in CHECKPOINT_PACKAGE_FILES:
        return True
    return (
        len(rel.parts) == 4
        and rel.parts[1] == "labels"
        and rel.parts[2] == labels_subdir
        and rel.name.endswith(".jsonl")
    )


@router.get("/status")
def get_status():
    """Check R2 configuration status."""
    return {
        "configured": r2_client.configured,
        "bucket": r2_client.bucket if r2_client.configured else None,
    }


@router.get("/categories")
def list_categories() -> list[dict]:
    """The Storage page's category tabs, rendered from config.R2_CATEGORIES
    verbatim (order included) — adding a category never touches the frontend."""
    return [
        {"key": key, "label": spec.label, "local_only": spec.local_only}
        for key, spec in R2_CATEGORIES.items()
    ]


@router.get("/files")
def list_local_files(category: str = "cuts-broadcast") -> list[dict]:
    """List local files with R2 sync status."""
    base_dir = _get_base_dir(category)
    if not base_dir.exists():
        return []

    # Build set of existing R2 keys for fast lookup (skip for local-only categories)
    r2_keys: set[str] = set()
    is_local_only = R2_CATEGORIES[category].local_only
    if not is_local_only and r2_client.configured:
        try:
            for obj in r2_client.list_objects(prefix=f"{category}/"):
                r2_keys.add(obj["key"])
        except Exception:
            pass  # R2 unavailable, show all as un-synced

    files = []
    if category in CHECKPOINT_CATEGORIES:
        # Nested: {run}/checkpoint_best.pt + metadata/log + label snapshot.
        labels_subdir = CHECKPOINT_CATEGORIES[category]
        for f in sorted(base_dir.rglob("*")):
            if f.is_file() and _is_checkpoint_package_file(f, base_dir, labels_subdir):
                rel = str(f.relative_to(base_dir))
                r2_key = f"{category}/{rel}"
                files.append({
                    "name": f.name,
                    "path": rel,
                    "group": f.relative_to(base_dir).parts[0],
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
def list_r2_files(category: str = "cuts-broadcast") -> list[dict]:
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
        # Add group for nested categories. Checkpoint packages group by the
        # run dir only (first part), matching list_local_files so the frontend
        # renders the nested label/metadata files as folders under the run.
        rel_path = Path(rel)
        if category in CHECKPOINT_CATEGORIES:
            entry["group"] = rel_path.parts[0]
        elif len(rel_path.parts) > 1:
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
    """Run file transfers as a single job with per-file byte-level progress.

    Emits ``bytes_done / bytes_total / speed / eta`` into ``job.params`` so the
    frontend can render live speed and ETA without an extra endpoint.

    Wrapped so any unhandled exception (e.g. an API contract drift on
    ``job_manager.update_job``) surfaces as a failed job instead of a silent
    "Task exception was never retrieved" warning on a job that just sits in
    pending forever.
    """
    try:
        await _run_batch_transfer_inner(
            job, files, category, base_dir, transfer_fn, verb,
        )
    except Exception as e:
        print(f"{terminal_prefix(job)}batch transfer crashed: {type(e).__name__}: {e}", flush=True)
        await fail_job_from_exc(job.id, e)


async def _run_batch_transfer_inner(
    job,
    files: list[str],
    category: str,
    base_dir: Path,
    transfer_fn: Callable[[Path, str, Callable | None], None],
    verb: str,
):
    """Inner body of _run_batch_transfer; see wrapper for failure handling."""
    loop = asyncio.get_running_loop()
    total = len(files)
    failed = 0
    batch_started = time.monotonic()
    bytes_done_so_far = 0  # cumulative across files for batch-level speed/ETA

    # Best-effort total: for downloads we don't have the local file yet, so
    # this stays 0 and the UI just won't show ETA until per-file progress lands.
    bytes_total_estimate = 0
    for f in files:
        p = base_dir / f
        try:
            if p.exists():
                bytes_total_estimate += p.stat().st_size
        except OSError:
            pass

    update = threadsafe_update(job.id, loop)

    for i, file_path in enumerate(files):
        local_path = base_dir / file_path
        r2_key = f"{category}/{file_path}"
        name = Path(file_path).name

        await job_manager.update_job(
            job.id, status="running", progress=i / total,
            message=batch_message(i, total, name, f"{verb.lower()}ing..."),
            params={
                **job.params,
                "current_file": name,
                "bytes_done": bytes_done_so_far,
                "bytes_total": bytes_total_estimate,
            },
        )

        last_emit = 0.0
        prev_done_in_file = 0

        def make_cb(file_idx=i, file_name=name):
            def on_bytes(done: int, total_bytes: int):
                nonlocal last_emit, prev_done_in_file, bytes_done_so_far
                now = time.monotonic()
                # Throttle to ~4Hz, but always emit the final byte
                if now - last_emit < 0.25 and done < total_bytes:
                    return
                last_emit = now

                delta = done - prev_done_in_file
                prev_done_in_file = done
                bytes_done_so_far += delta

                # For downloads bytes_total_estimate may be 0 — fall back to
                # the in-flight file's known total.
                btotal = max(bytes_total_estimate, bytes_done_so_far + max(total_bytes - done, 0))
                elapsed = max(now - batch_started, 0.001)
                speed = bytes_done_so_far / elapsed
                remaining = max(btotal - bytes_done_so_far, 0)
                eta = int(remaining / speed) if speed > 0 else 0
                file_pct = done / total_bytes if total_bytes else 0
                overall = (file_idx + file_pct) / total

                snapshot = {
                    **job.params,
                    "current_file": file_name,
                    "bytes_done": bytes_done_so_far,
                    "bytes_total": btotal,
                    "speed": speed,
                    "eta": eta,
                }
                pct_str = done * 100 // max(total_bytes, 1)
                update(
                    progress=overall,
                    message=batch_message(
                        file_idx, total, file_name, f"{verb.lower()}ing · {pct_str}%",
                    ),
                    params=snapshot,
                )
            return on_bytes

        try:
            await loop.run_in_executor(
                None,
                lambda fp=local_path, k=r2_key, cb=make_cb():
                    transfer_fn(fp, k, cb),
            )
            # Reconcile after each file so cumulative bytes stay correct even
            # when callbacks were throttled near the tail.
            try:
                actual_size = local_path.stat().st_size
                bytes_done_so_far += max(actual_size - prev_done_in_file, 0)
            except OSError:
                pass
        except asyncio.CancelledError:
            await job_manager.update_job(job.id, status="cancelled", message="Cancelled")
            return
        except Exception as e:
            failed += 1
            await job_manager.update_job(
                job.id, message=batch_message(i, total, name, f"failed — {e}"),
            )

    await finalize_batch_job(job.id, total, failed, noun="files")


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
    is_local_only = R2_CATEGORIES[req.category].local_only

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

        # Clean up empty parent directories (for nested categories like action-checkpoints)
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
    if R2_CATEGORIES[req.category].local_only:
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
