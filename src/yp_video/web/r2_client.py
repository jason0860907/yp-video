"""Cloudflare R2 storage client (S3-compatible)."""

import asyncio
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

from yp_video.config import load_r2_env

log = logging.getLogger(__name__)


class R2Client:
    """Cloudflare R2 object storage client."""

    def __init__(self):
        self._client = None
        self._config: dict[str, str] = {}
        self._loaded = False

    def _ensure_config(self):
        """Load config from disk if not yet loaded."""
        if not self._loaded:
            self.reload()

    def reload(self):
        """Reload configuration from r2.env and reset the boto3 client."""
        self._client = None
        self._config = load_r2_env()
        self._loaded = True

    @property
    def configured(self) -> bool:
        self._ensure_config()
        return bool(
            self._config.get("R2_ACCESS_KEY_ID")
            and self._config.get("R2_SECRET_ACCESS_KEY")
            and self._config.get("R2_BUCKET_NAME")
        )

    @property
    def bucket(self) -> str:
        self._ensure_config()
        return self._config.get("R2_BUCKET_NAME", "")

    def _get_client(self):
        if self._client is None:
            import boto3
            from botocore.config import Config as BotoConfig

            self._ensure_config()
            account_id = self._config.get("R2_ACCOUNT_ID", "")
            endpoint = f"https://{account_id}.r2.cloudflarestorage.com"

            self._client = boto3.client(
                "s3",
                endpoint_url=endpoint,
                aws_access_key_id=self._config["R2_ACCESS_KEY_ID"],
                aws_secret_access_key=self._config["R2_SECRET_ACCESS_KEY"],
                region_name="auto",
                config=BotoConfig(
                    s3={"addressing_style": "path"},
                    retries={"max_attempts": 3, "mode": "standard"},
                ),
            )
        return self._client

    def reset(self):
        """Reset client and force config reload on next access."""
        self._client = None
        self._config = {}
        self._loaded = False

    def list_objects(self, prefix: str = "") -> list[dict]:
        """List objects in bucket with optional prefix."""
        client = self._get_client()
        objects = []
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                objects.append({
                    "key": obj["Key"],
                    "size": obj["Size"],
                    "last_modified": obj["LastModified"].isoformat(),
                })
        return objects

    def object_exists(self, key: str) -> bool:
        """Check if an object exists in R2."""
        from botocore.exceptions import ClientError

        try:
            self._get_client().head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

    def upload_file(
        self,
        local_path: Path,
        key: str,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> dict:
        """Upload a file to R2 with multipart + progress callback."""
        import boto3.s3.transfer

        client = self._get_client()
        file_size = local_path.stat().st_size

        callback = None
        if on_progress:
            uploaded = 0

            def callback(bytes_amount):
                nonlocal uploaded
                uploaded += bytes_amount
                on_progress(uploaded, file_size)

        # Guess content type
        suffix = local_path.suffix.lower()
        content_types = {
            ".mp4": "video/mp4",
            ".jsonl": "application/jsonl",
            ".json": "application/json",
            ".npy": "application/octet-stream",
        }
        content_type = content_types.get(suffix, "application/octet-stream")

        client.upload_file(
            str(local_path),
            self.bucket,
            key,
            Callback=callback,
            ExtraArgs={"ContentType": content_type},
            Config=boto3.s3.transfer.TransferConfig(
                multipart_threshold=8 * 1024 * 1024,
                multipart_chunksize=8 * 1024 * 1024,
                max_concurrency=4,
            ),
        )

        return {"key": key, "size": file_size}

    def download_file(
        self,
        key: str,
        local_path: Path,
        on_progress: Callable[[int, int], None] | None = None,
    ):
        """Download a file from R2 with progress callback."""
        import boto3.s3.transfer

        client = self._get_client()

        # Get file size for progress
        head = client.head_object(Bucket=self.bucket, Key=key)
        file_size = head["ContentLength"]

        callback = None
        if on_progress:
            downloaded = 0

            def callback(bytes_amount):
                nonlocal downloaded
                downloaded += bytes_amount
                on_progress(downloaded, file_size)

        local_path.parent.mkdir(parents=True, exist_ok=True)

        client.download_file(
            self.bucket,
            key,
            str(local_path),
            Callback=callback,
            Config=boto3.s3.transfer.TransferConfig(
                multipart_threshold=8 * 1024 * 1024,
                multipart_chunksize=8 * 1024 * 1024,
                max_concurrency=4,
            ),
        )

    def generate_presigned_url(self, key: str, expires: int = 3600) -> str:
        """Generate a presigned URL for temporary access."""
        return self._get_client().generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires,
        )

    def delete_object(self, key: str):
        """Delete an object from R2."""
        self._get_client().delete_object(Bucket=self.bucket, Key=key)


# Module-level instance
r2_client = R2Client()


def serve_video_or_r2_redirect(
    local_path: Path,
    r2_categories: Sequence[str] = ("cuts", "videos"),
):
    """Serve a video via R2 presigned URL (preferred) or local file fallback.

    Prefers R2 when configured so that users close to the R2 region get
    lower latency than streaming through the VM.

    Returns a FastAPI response if the file is found on R2 or locally,
    or None if not found anywhere.
    """
    from fastapi.responses import FileResponse, RedirectResponse

    # Prefer R2 presigned URL — video is served directly from the edge
    if r2_client.configured:
        for category in r2_categories:
            r2_key = f"{category}/{local_path.name}"
            if r2_client.object_exists(r2_key):
                url = r2_client.generate_presigned_url(r2_key)
                return RedirectResponse(url)

    # Fallback: serve from local disk
    if local_path.exists() and local_path.is_file():
        return FileResponse(local_path, media_type="video/mp4")
    return None


def sync_to_r2(local_path: Path, category: str) -> None:
    """Fire-and-forget background upload of a file to R2.

    Safe to call from sync or async context — silently skips
    if R2 is not configured.
    """
    if not r2_client.configured:
        return

    r2_key = f"{category}/{local_path.name}"

    async def _upload():
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: r2_client.upload_file(local_path, r2_key),
            )
            log.info("R2 sync: %s -> %s", local_path.name, r2_key)
        except Exception as e:
            log.warning("R2 sync failed for %s: %s", local_path.name, e)

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_upload())
    except RuntimeError:
        log.debug("sync_to_r2 skipped (no running event loop) for %s", local_path.name)


def sync_to_r2_nested(local_path: Path, category: str, base_dir: Path) -> None:
    """Fire-and-forget upload preserving directory structure relative to base_dir.

    Example: sync_to_r2_nested(
        .../tad-checkpoints/actionformer/vjepa-b/2026-0401/best.pth.tar,
        "tad-checkpoints",
        .../tad-checkpoints/
    ) → R2 key: tad-checkpoints/actionformer/vjepa-b/2026-0401/best.pth.tar
    """
    if not r2_client.configured:
        return

    rel = local_path.relative_to(base_dir)
    r2_key = f"{category}/{rel}"

    async def _upload():
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: r2_client.upload_file(local_path, r2_key),
            )
            log.info("R2 sync: %s -> %s", rel, r2_key)
        except Exception as e:
            log.warning("R2 sync failed for %s: %s", rel, e)

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_upload())
    except RuntimeError:
        log.debug("sync_to_r2_nested skipped (no running event loop) for %s", rel)


def sync_directory_to_r2(directory: Path, category: str, pattern: str = "*.jsonl") -> None:
    """Fire-and-forget background upload of all matching files in a directory."""
    if not r2_client.configured or not directory.exists():
        return

    for f in directory.glob(pattern):
        sync_to_r2(f, category)
