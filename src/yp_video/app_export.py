"""Push one hand-corrected match into the iOS app's R2 library.

The Annotate page calls `export_one_match()` after every save: it uploads
the cut video to R2 (skipped when the file is unchanged) and writes a
single-match manifest the iOS app imports by URL.

Each match gets its OWN manifest file (`library/<match_id>.json`) rather
than one shared catalog. The iOS importer upserts per match and never
deletes matches absent from a manifest, so a per-match file lets the user
import exactly the matches they want — one URL at a time.

The bulk CLI (`scripts/export_to_app.py`) is independent and unaffected;
it still writes the shared `library/manifest.json`.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests

from yp_video.config import ANNOTATIONS_DIR, PROJECT_ROOT, find_cut, load_tokens_env

# Stable Match/Rally UUIDs are cached here so re-publishing a match after a
# small annotation tweak reuses the same iOS rows instead of duplicating.
ID_CACHE_PATH = PROJECT_ROOT / ".export_to_app_ids.json"
ANGLE_FOR_SIDELINE = "phoneSideline"  # matches the iOS CameraAngle enum
TOOL_LIBRARY_VIDEO_UPLOAD_PATH = "/tools/library/video-upload-url"
TOOL_LIBRARY_MANIFEST_UPLOAD_PATH = "/tools/library/manifest-upload-url"
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


class AppExportError(Exception):
    """A user-correctable failure during app export (bad config, missing
    files, no annotations). The web layer maps this to an HTTP 400."""


def export_one_match(basename: str) -> dict:
    """Upload one match's cut video + a single-match manifest to the app's
    R2 bucket. Returns ``{"manifest_url", "match_id", "rally_count"}``.

    `basename` is the cut-video stem, e.g. ``20260426-小窩-01``.
    """
    endpoint, token, user_id = _resolve_config()

    ann_path = ANNOTATIONS_DIR / f"{basename}_annotations.jsonl"
    if not ann_path.exists():
        raise AppExportError(
            f"No saved rally annotations for {basename} — save first."
        )
    mp4 = find_cut(f"{basename}.mp4")
    if mp4 is None or not mp4.exists():
        raise AppExportError(f"Cut video not found for {basename}.")

    duration, rallies = _parse_rally_annotations(ann_path)

    cache = _load_id_cache()
    entry_cache = cache.get(basename) or {"match_id": str(uuid.uuid4())}
    match_id = entry_cache["match_id"]

    # 1. Upload the cut video to R2 — but skip it when the file is unchanged
    #    since the last push (same size + mtime). The video is the slow part;
    #    a re-publish after an annotation tweak then only re-sends the tiny
    #    manifest. The first push for a match always uploads.
    mp4_stat = mp4.stat()
    video_size = mp4_stat.st_size
    video_mtime = int(mp4_stat.st_mtime)
    cached_url = entry_cache.get("video_url")
    if (
        cached_url
        and entry_cache.get("video_size") == video_size
        and entry_cache.get("video_mtime") == video_mtime
    ):
        public_video_url = cached_url
        video_uploaded = False
    else:
        signed = _post_json(
            f"{endpoint}{TOOL_LIBRARY_VIDEO_UPLOAD_PATH}",
            token,
            {"user_id": user_id, "match_id": match_id, "content_type": "video/mp4"},
        )
        _put_file(signed["upload_url"], mp4, "video/mp4")
        public_video_url = signed["public_url"]
        video_uploaded = True

    # 2. Build this match's manifest entry. Rally UUIDs are keyed by
    #    (start, end) so a re-publish after a sub-0.1s tweak updates the
    #    same iOS rows; a bigger edit gets a fresh UUID (delete + insert).
    rally_ids = dict(entry_cache.get("rally_ids", {}))
    rally_entries = []
    for idx, r in enumerate(rallies, start=1):
        key = _rally_cache_key(r["start"], r["end"])
        rid = rally_ids.get(key) or str(uuid.uuid4())
        rally_ids[key] = rid
        rally_entries.append(
            {
                "id": rid,
                "index": idx,
                "set_number": 1,
                "start_seconds": r["start"],
                "end_seconds": r["end"],
            }
        )
    cache[basename] = {
        "match_id": match_id,
        "rally_ids": rally_ids,
        # Recorded so the next push can skip the video upload if unchanged.
        "video_size": video_size,
        "video_mtime": video_mtime,
        "video_url": public_video_url,
    }
    _save_id_cache(cache)

    manifest = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "user_id": user_id,
        "matches": [
            {
                "id": match_id,
                "title": basename,
                "angle": ANGLE_FOR_SIDELINE,
                "video_url": public_video_url,
                "duration_seconds": duration,
                "match_date": _parse_match_date(basename),
                "rallies": rally_entries,
            }
        ],
    }

    # 3. Upload a per-match manifest (library/<match_id>.json) so the user
    #    imports exactly this match by its own URL.
    msigned = _post_json(
        f"{endpoint}{TOOL_LIBRARY_MANIFEST_UPLOAD_PATH}",
        token,
        {"name": match_id},
    )
    resp = requests.put(
        msigned["upload_url"],
        data=json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        timeout=60,
    )
    resp.raise_for_status()

    return {
        "manifest_url": msigned["public_url"],
        "match_id": match_id,
        "rally_count": len(rallies),
        "video_uploaded": video_uploaded,
    }


# ── Config ────────────────────────────────────────────────────────────


def _resolve_config() -> tuple[str, str, str]:
    """Resolve (endpoint, token, user_id), preferring process env then
    tokens.env. Accepts both the CLI's UPLOAD_* names and tokens.env's
    WORKER_URL / AUTH_TOKEN so one config serves both entry points."""
    tokens = load_tokens_env()
    endpoint = os.environ.get("UPLOAD_ENDPOINT") or tokens.get("WORKER_URL", "")
    token = os.environ.get("UPLOAD_TOKEN") or tokens.get("AUTH_TOKEN", "")
    user_id = os.environ.get("LIBRARY_USER_ID") or tokens.get("LIBRARY_USER_ID", "")

    missing = [
        name
        for name, value in (
            ("WORKER_URL", endpoint),
            ("AUTH_TOKEN", token),
            ("LIBRARY_USER_ID", user_id),
        )
        if not value
    ]
    if missing:
        raise AppExportError(f"tokens.env is missing: {', '.join(missing)}")
    if not _UUID_RE.match(user_id):
        raise AppExportError("LIBRARY_USER_ID in tokens.env must be a lowercase UUID")
    return endpoint.rstrip("/"), token, user_id


# ── Annotation parsing ────────────────────────────────────────────────


def _parse_rally_annotations(path: Path) -> tuple[float, list[dict]]:
    """Return (duration_seconds, [{start, end}, ...]) from a rally-
    annotations JSONL, keeping only rows labelled ``rally``."""
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
        raise AppExportError(f"{path.name} has no _meta line with a duration.")
    if not rallies:
        raise AppExportError(f"{path.name} has no rally annotations to export.")
    return duration, rallies


def _parse_match_date(basename: str) -> str | None:
    """Pull a YYYY-MM-DD from the leading 8 digits of a basename, if any."""
    m = re.match(r"^(\d{8})", basename)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d").date().isoformat()
    except ValueError:
        return None


# ── UUID cache ────────────────────────────────────────────────────────


def _load_id_cache() -> dict:
    if ID_CACHE_PATH.exists():
        return json.loads(ID_CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_id_cache(cache: dict) -> None:
    ID_CACHE_PATH.write_text(
        json.dumps(cache, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def _rally_cache_key(start: float, end: float) -> str:
    """Stable key for a rally from its start/end, rounded to 0.1s."""
    return f"{round(start, 1)}|{round(end, 1)}"


# ── HTTP ──────────────────────────────────────────────────────────────


def _post_json(url: str, token: str, payload: dict) -> dict:
    resp = requests.post(
        url,
        json=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


class _SizedReader:
    """File wrapper exposing ``__len__`` so requests sends a real Content-
    Length instead of Transfer-Encoding: chunked — R2 rejects chunked
    bodies on presigned PUT URLs (the TCP connection drops mid-handshake)."""

    def __init__(self, file, size: int):
        self._file = file
        self._size = size

    def __len__(self) -> int:
        return self._size

    def read(self, n: int = -1) -> bytes:
        return self._file.read(n)


def _put_file(url: str, path: Path, content_type: str) -> None:
    size = path.stat().st_size
    with path.open("rb") as f:
        resp = requests.put(
            url,
            data=_SizedReader(f, size),
            headers={"Content-Type": content_type},
            timeout=None,
        )
        resp.raise_for_status()
