"""Cloudflare R2 storage client (S3-compatible)."""

import asyncio
import logging
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from yp_video.config import CUT_KINDS, CutKind, find_cut, iter_all_cuts, load_env

log = logging.getLogger(__name__)


class R2Client:
    """Cloudflare R2 object storage client."""

    def __init__(self):
        self._client = None
        self._config: dict[str, str] = {}
        self._loaded = False
        # prefix -> (monotonic deadline, objects). Writes through this client
        # invalidate the affected prefixes; cross-machine writes surface once
        # the TTL lapses.
        self._list_cache: dict[str, tuple[float, list[dict]]] = {}
        self._list_lock = threading.Lock()

    def _ensure_config(self):
        """Load config from disk if not yet loaded."""
        if not self._loaded:
            self.reload()

    def reload(self):
        """Reload configuration from the workspace .env and reset the client."""
        self._client = None
        self._config = load_env()
        self._loaded = True

    @property
    def configured(self) -> bool:
        self._ensure_config()
        return bool(
            self._config.get("R2_ACCESS_KEY_ID")
            and self._config.get("R2_SECRET_ACCESS_KEY")
            and self._config.get("R2_BUCKET_PIPELINE")
        )

    @property
    def bucket(self) -> str:
        self._ensure_config()
        return self._config.get("R2_BUCKET_PIPELINE", "")

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
        with self._list_lock:
            self._list_cache.clear()

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

    def list_objects_cached(self, prefix: str = "", ttl: float = 30.0) -> list[dict]:
        """``list_objects`` behind a per-prefix TTL.

        For list endpoints hit on every page load — a full paginated listing
        per request is seconds of latency. Mutations through this client drop
        the affected prefixes immediately, so only writes from other machines
        wait out the TTL.
        """
        now = time.monotonic()
        with self._list_lock:
            hit = self._list_cache.get(prefix)
            if hit and hit[0] > now:
                return hit[1]
        objects = self.list_objects(prefix)  # network I/O outside the lock
        with self._list_lock:
            self._list_cache[prefix] = (now + ttl, objects)
        return objects

    def _invalidate_listings(self, key: str) -> None:
        with self._list_lock:
            for prefix in [p for p in self._list_cache if key.startswith(p)]:
                del self._list_cache[prefix]

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

        self._invalidate_listings(key)
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
        self._invalidate_listings(key)

    def delete_objects(self, keys: list[str]) -> int:
        """Bulk delete, chunked at the S3 limit of 1000 keys per call."""
        client = self._get_client()
        deleted = 0
        chunk = 1000
        for i in range(0, len(keys), chunk):
            batch = keys[i:i + chunk]
            resp = client.delete_objects(
                Bucket=self.bucket,
                Delete={"Objects": [{"Key": k} for k in batch]},
            )
            deleted += len(resp.get("Deleted", []))
            for k in batch:
                self._invalidate_listings(k)
        return deleted


# Module-level instance
r2_client = R2Client()

# asyncio only weak-refs Tasks: a fire-and-forget sync task with no strong
# reference can be garbage-collected mid-upload, silently leaving R2 stale.
_sync_tasks: set[asyncio.Task] = set()


_MIRROR_LOCK = threading.Lock()


def mirror_file(path: Path, key: str) -> None:
    """Push ``path`` to R2 as ``key`` on a background thread.

    For files a human just edited (the label-done ledger): the request must
    not wait on the network, and a failed upload must not undo the save —
    it is logged, and the next write tries again with the whole file.
    Uploads are serialized so the last write on disk is the last one in R2.
    """
    if not r2_client.configured:
        return

    def run() -> None:
        with _MIRROR_LOCK:
            try:
                r2_client.upload_file(path, key)
            except Exception:  # noqa: BLE001
                log.warning("R2 mirror of %s failed", key, exc_info=True)

    threading.Thread(target=run, name=f"r2-mirror:{path.name}", daemon=True).start()


def _remote_cut_entry(name: str) -> tuple[CutKind, dict] | None:
    """The R2 listing entry of a cut — its kind plus the object row — or None."""
    if not r2_client.configured:
        return None
    for kind in CUT_KINDS.values():
        key = f"{kind.r2_category}/{name}"
        for obj in r2_client.list_objects_cached(f"{kind.r2_category}/", ttl=300.0):
            if obj["key"] == key:
                return kind, obj
    return None


def _remote_cut_path(name: str) -> Path | None:
    """Canonical local path of a cut whose bytes live only in R2. The parent
    directory encodes the camera view, so ``cut_kind_of`` works on the
    returned path even though the file is absent."""
    entry = _remote_cut_entry(name)
    return entry[0].local_dir / name if entry else None


def resolve_cut(name: str) -> Path | None:
    """The web layer's ``CutResolver``: canonical path of a cut whose bytes
    are local or in R2. The returned path may not exist on disk — every SPOT
    training source (rally and action alike) reads frame caches, never the
    mp4, so this is what they are handed."""
    return find_cut(name) or _remote_cut_path(name)


def all_cut_paths() -> list[Path]:
    """Every cut video, local and R2-only, as canonical local paths.

    The video universe the work lists iterate — labeling must not depend on
    which machine holds the bytes. Local files win on name collisions; an
    R2-only cut resolves under its kind's local dir (see ``_remote_cut_path``).
    An R2 outage degrades to the local list rather than taking the page down.
    """
    cuts = list(iter_all_cuts())
    seen = {p.name for p in cuts}
    if r2_client.configured:
        for kind in CUT_KINDS.values():
            try:
                objects = r2_client.list_objects_cached(f"{kind.r2_category}/", ttl=300.0)
            except Exception:  # noqa: BLE001
                log.warning("R2 listing failed; remote cuts will look absent")
                break
            for obj in objects:
                name = Path(obj["key"]).name
                if name.endswith(".mp4") and name not in seen:
                    seen.add(name)
                    cuts.append(kind.local_dir / name)
    return cuts


def cut_media_source(path: Path) -> str | None:
    """An ffmpeg/ffprobe-readable source for a cut: the local file when it
    exists, else a presigned URL to its R2 copy, else None.

    24h expiry for the same reason as ``serve_video_or_r2_redirect`` — one
    URL must survive a whole labeling session.
    """
    if path.exists():
        return str(path)
    entry = _remote_cut_entry(path.name)
    if entry is None:
        return None
    return r2_client.generate_presigned_url(
        f"{entry[0].r2_category}/{path.name}", expires=24 * 3600
    )


def serve_video_or_r2_redirect(
    local_path: Path,
    r2_categories: Sequence[str] = ("cuts-broadcast", "cuts-sideline", "videos"),
):
    """Serve a video, preferring Cloudflare's edge over this machine's uplink.

    A presigned R2 redirect streams straight from the edge instead of riding
    the tunnel out and back; local disk answers whatever R2 does not hold —
    raw videos are ``local_only``, so the Cut page always lands there.
    FileResponse handles Range; the explicit identity encoding keeps
    GZipMiddleware off those 206s (it would gzip the chunks and drop
    Content-Length). Returns None if the video exists nowhere.

    This used to branch on the Host header, serving local disk to LAN clients.
    Direct access to this machine is closed now — everything arrives through
    the tunnel with a Cloudflare Access identity — so the branch had exactly
    one live side left.

    24h presign expiry: a <video> element holds one URL for the whole
    labeling session, and an expired signature kills it mid-seek.
    """
    from fastapi.responses import FileResponse, RedirectResponse

    def from_disk():
        if local_path.exists() and local_path.is_file():
            return FileResponse(
                local_path, media_type="video/mp4", headers={"Content-Encoding": "identity"}
            )
        return None

    def from_r2():
        if not r2_client.configured:
            return None
        for category in r2_categories:
            r2_key = f"{category}/{local_path.name}"
            if r2_client.object_exists(r2_key):
                url = r2_client.generate_presigned_url(r2_key, expires=24 * 3600)
                return RedirectResponse(url)
        return None

    return from_r2() or from_disk()


def sync_to_r2(local_path: Path, category: str, *, base_dir: Path | None = None) -> None:
    """Fire-and-forget background upload of a file to R2.

    Safe to call from sync or async context — silently skips if R2 is not
    configured or there is no running event loop. The R2 key is built from
    ``category`` plus either:

    - ``local_path.name`` (default — flat layout), or
    - ``local_path.relative_to(base_dir)`` when ``base_dir`` is given,
      preserving the nested directory structure under ``category``.

    Example with ``base_dir``::

        sync_to_r2(
            .../spot/checkpoints/<run>/checkpoint_best.pt,
            "spot/checkpoints",
            base_dir=.../spot/checkpoints,
        )
        # → R2 key: spot/checkpoints/<run>/checkpoint_best.pt
    """
    if not r2_client.configured:
        return

    rel = local_path.relative_to(base_dir) if base_dir is not None else Path(local_path.name)
    r2_key = f"{category}/{rel}"
    # A save that empties a sidecar removes the file (JsonSidecar.write(None));
    # the mirror must follow, or a restore would resurrect cleared labels.
    exists = local_path.exists()

    async def _upload():
        loop = asyncio.get_running_loop()
        try:
            if exists:
                await loop.run_in_executor(
                    None, lambda: r2_client.upload_file(local_path, r2_key)
                )
                log.info("R2 sync: %s -> %s", rel, r2_key)
            else:
                await loop.run_in_executor(None, lambda: r2_client.delete_object(r2_key))
                log.info("R2 sync: removed %s", r2_key)
        except Exception as e:
            log.warning("R2 sync failed for %s: %s", rel, e)

    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(_upload())
        _sync_tasks.add(task)
        task.add_done_callback(_sync_tasks.discard)
    except RuntimeError:
        log.debug("sync_to_r2 skipped (no running event loop) for %s", rel)


def sync_directory_to_r2(directory: Path, category: str, pattern: str = "*.jsonl") -> None:
    """Fire-and-forget background upload of all matching files in a directory."""
    if not r2_client.configured or not directory.exists():
        return

    for f in directory.glob(pattern):
        sync_to_r2(f, category)
