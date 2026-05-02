"""Push hand-corrected match videos + rally annotations into the iOS app.

Reads from:
  ~/videos/cuts-sideline/<basename>.mp4
  ~/videos/rally-annotations/<basename>_annotations.jsonl   (REQUIRED — never falls
                                                              back to tad-predictions
                                                              per project policy)

For each basename:
  1. Resolves a stable Match UUID (cached in `.export_to_app_ids.json`)
  2. Asks the iOS Cloudflare Worker for a signed PUT URL
  3. PUTs the mp4 to R2 at videos/<LIBRARY_USER_ID>/<matchId>/source.mp4
  4. Parses rally-annotations into manifest entries (UUIDs cached by
     (start, end) tuple so re-runs reuse the same Rally rows on iOS)

After all videos upload, PUTs a single manifest.json to library/manifest.json.
The script prints the public URL the user pastes into Settings → Library
manifest URL on iOS.

Env vars (required):
  UPLOAD_ENDPOINT   e.g. https://volleyiq-upload-service.<sub>.workers.dev
  UPLOAD_TOKEN      shared AUTH_TOKEN with the Worker
  LIBRARY_USER_ID   fixed UUID for the "library owner" namespace in R2 keys

Usage:
  python -m scripts.export_to_app                 # default 12 names
  python -m scripts.export_to_app --basenames 20260426-小窩-01 20260426-小窩-05
  python -m scripts.export_to_app --dry-run       # parse + plan, no uploads
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests
from tqdm import tqdm

VIDEOS_DIR = Path.home() / "videos"
CUTS_SIDELINE_DIR = VIDEOS_DIR / "cuts-sideline"
RALLY_ANNOTATIONS_DIR = VIDEOS_DIR / "rally-annotations"
ID_CACHE_PATH = Path(__file__).resolve().parent.parent / ".export_to_app_ids.json"

DEFAULT_BASENAMES = [f"20260426-小窩-{i:02d}" for i in range(1, 13)]
ANGLE_FOR_SIDELINE = "phoneSideline"  # matches CameraAngle enum on iOS


def parse_match_date(basename: str) -> str | None:
    """Pull a YYYY-MM-DD from the leading 8 digits of a filename if present."""
    m = re.match(r"^(\d{8})", basename)
    if not m:
        return None
    raw = m.group(1)
    try:
        return datetime.strptime(raw, "%Y%m%d").date().isoformat()
    except ValueError:
        return None


def load_id_cache() -> dict:
    if ID_CACHE_PATH.exists():
        return json.loads(ID_CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_id_cache(cache: dict) -> None:
    ID_CACHE_PATH.write_text(
        json.dumps(cache, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def rally_cache_key(start: float, end: float) -> str:
    """Stable key for a rally based on its start/end (1 decimal place).

    Re-running the export after a tiny annotation tweak (< 0.1s) reuses the
    same Rally UUID; bigger edits get a fresh UUID, which iOS treats as
    delete-old + insert-new.
    """
    return f"{round(start, 1)}|{round(end, 1)}"


def parse_rally_annotations(path: Path) -> tuple[float, list[dict]]:
    """Returns (duration_seconds, [rally rows]).

    Rally rows are filtered to label == 'rally' only. Times are passed
    through as-is (already in seconds).
    """
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
            rallies.append(
                {"start": float(row["start"]), "end": float(row["end"])}
            )
    if duration is None:
        raise ValueError(f"{path} has no _meta line with duration")
    return duration, rallies


def fetch_signed_url(endpoint: str, token: str, payload: dict) -> dict:
    r = requests.post(
        f"{endpoint.rstrip('/')}/upload-url",
        json=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def fetch_manifest_signed_url(endpoint: str, token: str) -> dict:
    r = requests.post(
        f"{endpoint.rstrip('/')}/manifest-url",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def put_with_progress(url: str, path: Path, content_type: str) -> None:
    size = path.stat().st_size
    with path.open("rb") as f, tqdm(
        total=size,
        unit="B",
        unit_scale=True,
        desc=path.name,
        leave=False,
    ) as bar:

        def chunked():
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                bar.update(len(chunk))
                yield chunk

        r = requests.put(
            url,
            data=chunked(),
            headers={"Content-Type": content_type, "Content-Length": str(size)},
            timeout=None,
        )
        r.raise_for_status()


def build_match_entry(
    *,
    basename: str,
    cache_entry: dict,
    user_id: str,
    public_url: str,
    duration: float,
    rallies: list[dict],
) -> tuple[dict, dict]:
    """Returns (manifest_entry, updated_cache_entry)."""
    rally_ids = dict(cache_entry.get("rally_ids", {}))
    rally_entries = []
    for idx, r in enumerate(rallies, start=1):
        key = rally_cache_key(r["start"], r["end"])
        rally_id = rally_ids.get(key) or str(uuid.uuid4())
        rally_ids[key] = rally_id
        rally_entries.append({
            "id": rally_id,
            "index": idx,
            "set_number": 1,
            "start_seconds": r["start"],
            "end_seconds": r["end"],
        })

    manifest_entry = {
        "id": cache_entry["match_id"],
        "title": basename,
        "angle": ANGLE_FOR_SIDELINE,
        "video_url": public_url,
        "duration_seconds": duration,
        "match_date": parse_match_date(basename),
        "rallies": rally_entries,
    }
    updated_cache = {
        "match_id": cache_entry["match_id"],
        "rally_ids": rally_ids,
    }
    return manifest_entry, updated_cache


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--basenames",
        nargs="+",
        default=DEFAULT_BASENAMES,
        help="basename list (no extension); default = 20260426-小窩-01..12",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="parse rally-annotations and resolve UUIDs but skip all HTTP calls",
    )
    args = ap.parse_args()

    endpoint = os.environ.get("UPLOAD_ENDPOINT")
    token = os.environ.get("UPLOAD_TOKEN")
    user_id = os.environ.get("LIBRARY_USER_ID")
    missing = [k for k, v in (
        ("UPLOAD_ENDPOINT", endpoint),
        ("UPLOAD_TOKEN", token),
        ("LIBRARY_USER_ID", user_id),
    ) if not v]
    if missing and not args.dry_run:
        print(f"missing env vars: {', '.join(missing)}", file=sys.stderr)
        return 2
    if user_id and not re.match(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        user_id,
    ):
        print("LIBRARY_USER_ID must be a lowercase UUID", file=sys.stderr)
        return 2

    cache = load_id_cache()
    matches: list[dict] = []

    for basename in args.basenames:
        mp4 = CUTS_SIDELINE_DIR / f"{basename}.mp4"
        ann = RALLY_ANNOTATIONS_DIR / f"{basename}_annotations.jsonl"
        if not mp4.exists():
            print(f"skip {basename}: missing {mp4}", file=sys.stderr)
            continue
        if not ann.exists():
            # NEVER fall back to tad-predictions — those are raw model output.
            print(
                f"skip {basename}: rally-annotations missing "
                f"(refusing to use tad-predictions as a fallback)",
                file=sys.stderr,
            )
            continue

        duration, rallies = parse_rally_annotations(ann)
        cache_entry = cache.get(basename) or {"match_id": str(uuid.uuid4())}
        match_id = cache_entry["match_id"]

        if args.dry_run:
            public_url = f"<dry-run>/videos/{user_id or '<USER_ID>'}/{match_id}/source.mp4"
            print(f"[dry] {basename}  match_id={match_id}  rallies={len(rallies)}  duration={duration:.1f}s")
        else:
            print(f"upload {basename}  match_id={match_id}  rallies={len(rallies)}")
            signed = fetch_signed_url(
                endpoint, token,
                {
                    "user_id": user_id,
                    "match_id": match_id,
                    "content_type": "video/mp4",
                },
            )
            put_with_progress(signed["upload_url"], mp4, "video/mp4")
            public_url = signed["public_url"]

        entry, updated = build_match_entry(
            basename=basename,
            cache_entry=cache_entry,
            user_id=user_id or "",
            public_url=public_url,
            duration=duration,
            rallies=rallies,
        )
        matches.append(entry)
        cache[basename] = updated
        if not args.dry_run:
            save_id_cache(cache)  # persist after each video so a crash is recoverable

    manifest = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "user_id": user_id or "",
        "matches": matches,
    }
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")

    if args.dry_run:
        print("\n----- manifest preview -----")
        print(manifest_bytes.decode("utf-8"))
        return 0

    print(f"\nuploading manifest ({len(matches)} matches, {len(manifest_bytes)} bytes)")
    signed = fetch_manifest_signed_url(endpoint, token)
    r = requests.put(
        signed["upload_url"],
        data=manifest_bytes,
        headers={"Content-Type": "application/json"},
        timeout=60,
    )
    r.raise_for_status()

    print("\nDone.")
    print(f"Public manifest URL (paste this into iOS Settings → Library manifest URL):")
    print(f"  {signed['public_url']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
