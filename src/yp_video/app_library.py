"""Shared helpers for publishing curated matches to the VolleyIQ app library."""

from __future__ import annotations

import json
import os
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from yp_video.config import PROJECT_ROOT, load_tokens_env

# Stable Match/Rally UUIDs are cached here so re-publishing after annotation
# tweaks updates existing iOS rows instead of duplicating them.
ID_CACHE_PATH = PROJECT_ROOT / ".export_to_app_ids.json"
ANGLE_FOR_SIDELINE = "phoneSideline"
TOOL_LIBRARY_VIDEO_UPLOAD_PATH = "/tools/library/video-upload-url"
TOOL_LIBRARY_MANIFEST_UPLOAD_PATH = "/tools/library/manifest-upload-url"
VIDEO_CONTENT_TYPE = "video/mp4"
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


class AppLibraryExportError(Exception):
    """A user-correctable failure during app-library export."""


@dataclass(frozen=True)
class AppLibraryConfig:
    endpoint: str
    token: str
    user_id: str


def resolve_config(*, include_tokens_env: bool, required: bool) -> AppLibraryConfig:
    """Resolve app-library config from env, optionally falling back to tokens.env."""
    tokens = load_tokens_env() if include_tokens_env else {}
    endpoint = (
        os.environ.get("UPLOAD_ENDPOINT")
        or tokens.get("WORKER_URL")
        or ""
    ).rstrip("/")
    token = os.environ.get("UPLOAD_TOKEN") or tokens.get("AUTH_TOKEN") or ""
    user_id = os.environ.get("LIBRARY_USER_ID") or tokens.get("LIBRARY_USER_ID") or ""

    missing = [
        name
        for name, value in (
            ("UPLOAD_ENDPOINT/WORKER_URL", endpoint),
            ("UPLOAD_TOKEN/AUTH_TOKEN", token),
            ("LIBRARY_USER_ID", user_id),
        )
        if not value
    ]
    if missing and required:
        raise AppLibraryExportError(f"missing app export config: {', '.join(missing)}")
    if user_id and not UUID_RE.match(user_id):
        raise AppLibraryExportError("LIBRARY_USER_ID must be a lowercase UUID")
    return AppLibraryConfig(endpoint=endpoint, token=token, user_id=user_id)


def load_id_cache(path: Path = ID_CACHE_PATH) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_id_cache(cache: dict, path: Path = ID_CACHE_PATH) -> None:
    path.write_text(
        json.dumps(cache, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def cache_entry_for(cache: dict, basename: str) -> dict:
    return dict(cache.get(basename) or {"match_id": str(uuid.uuid4())})


def parse_rally_annotations(
    path: Path,
    *,
    require_rallies: bool,
) -> tuple[float, list[dict]]:
    """Return (duration_seconds, [{start, end}, ...]) from rally JSONL."""
    duration: float | None = None
    rallies: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("_meta"):
                duration = float(row.get("duration", 0.0))
                continue
            if row.get("label") != "rally":
                continue
            rallies.append({"start": float(row["start"]), "end": float(row["end"])})
    if duration is None:
        raise AppLibraryExportError(f"{path.name} has no _meta line with a duration.")
    if require_rallies and not rallies:
        raise AppLibraryExportError(f"{path.name} has no rally annotations to export.")
    return duration, rallies


def parse_match_date(basename: str) -> str | None:
    """Pull a YYYY-MM-DD from the leading 8 digits of a basename, if any."""
    match = re.match(r"^(\d{8})", basename)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d").date().isoformat()
    except ValueError:
        return None


def rally_cache_key(start: float, end: float) -> str:
    """Stable key for a rally from its start/end, rounded to 0.1s."""
    return f"{round(start, 1)}|{round(end, 1)}"


def build_match_entry(
    *,
    basename: str,
    cache_entry: dict,
    public_url: str,
    duration: float,
    rallies: list[dict],
) -> tuple[dict, dict]:
    """Return (manifest_match_entry, updated_cache_entry)."""
    rally_ids = dict(cache_entry.get("rally_ids", {}))
    rally_entries = []
    for idx, rally in enumerate(rallies, start=1):
        key = rally_cache_key(rally["start"], rally["end"])
        rally_id = rally_ids.get(key) or str(uuid.uuid4())
        rally_ids[key] = rally_id
        rally_entries.append(
            {
                "id": rally_id,
                "index": idx,
                "set_number": 1,
                "start_seconds": rally["start"],
                "end_seconds": rally["end"],
            }
        )

    match_entry = {
        "id": cache_entry["match_id"],
        "title": basename,
        "angle": ANGLE_FOR_SIDELINE,
        "video_url": public_url,
        "duration_seconds": duration,
        "match_date": parse_match_date(basename),
        "rallies": rally_entries,
    }
    return match_entry, {"match_id": cache_entry["match_id"], "rally_ids": rally_ids}


def build_manifest(*, user_id: str, matches: list[dict]) -> dict:
    return {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "user_id": user_id,
        "matches": matches,
    }


def encode_manifest(manifest: dict) -> bytes:
    return json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")


def fetch_video_upload_url(
    config: AppLibraryConfig,
    *,
    match_id: str,
    content_type: str = VIDEO_CONTENT_TYPE,
) -> dict:
    return post_json(
        f"{config.endpoint}{TOOL_LIBRARY_VIDEO_UPLOAD_PATH}",
        config.token,
        {
            "user_id": config.user_id,
            "match_id": match_id,
            "content_type": content_type,
        },
    )


def fetch_manifest_upload_url(
    config: AppLibraryConfig,
    *,
    name: str | None = None,
) -> dict:
    payload = {"name": name} if name else None
    return post_json(
        f"{config.endpoint}{TOOL_LIBRARY_MANIFEST_UPLOAD_PATH}",
        config.token,
        payload,
    )


def post_json(url: str, token: str, payload: dict | None) -> dict:
    import requests

    headers = {"Authorization": f"Bearer {token}"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    response = requests.post(url, json=payload, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


class SizedFileReader:
    """File wrapper exposing __len__ so requests sends Content-Length to R2."""

    def __init__(
        self,
        file,
        size: int,
        on_progress: Callable[[int], None] | None = None,
    ):
        self._file = file
        self._size = size
        self._on_progress = on_progress

    def __len__(self) -> int:
        return self._size

    def read(self, n: int = -1) -> bytes:
        chunk = self._file.read(n)
        if chunk and self._on_progress:
            self._on_progress(len(chunk))
        return chunk


def put_file(
    url: str,
    path: Path,
    content_type: str,
    *,
    on_progress: Callable[[int], None] | None = None,
) -> None:
    import requests

    size = path.stat().st_size
    with path.open("rb") as f:
        response = requests.put(
            url,
            data=SizedFileReader(f, size, on_progress),
            headers={"Content-Type": content_type},
            timeout=None,
        )
        response.raise_for_status()


def put_bytes(url: str, data: bytes, content_type: str) -> None:
    import requests

    response = requests.put(
        url,
        data=data,
        headers={"Content-Type": content_type},
        timeout=60,
    )
    response.raise_for_status()
