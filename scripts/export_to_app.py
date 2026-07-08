"""Bulk-publish hand-corrected matches into the VolleyIQ app library.

Reads from:
  ~/videos/cuts-sideline/<basename>.mp4
  ~/videos/rally-annotations/<basename>_annotations.jsonl

After all videos upload, PUTs one shared manifest to `library/manifest.json`.
The Annotate web page uses `yp_video.app_export` for single-match manifests;
both paths share the same app-library helpers.

Env/config:
  UPLOAD_ENDPOINT or tokens.env WORKER_URL
  UPLOAD_TOKEN or tokens.env AUTH_TOKEN
  LIBRARY_USER_ID

Usage:
  python -m scripts.export_to_app
  python -m scripts.export_to_app --basenames 20260426-小窩-01 20260426-小窩-05
  python -m scripts.export_to_app --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from yp_video.app_library import (
    VIDEO_CONTENT_TYPE,
    AppLibraryExportError,
    build_manifest,
    build_match_entry,
    cache_entry_for,
    encode_manifest,
    fetch_manifest_upload_url,
    fetch_video_upload_url,
    load_id_cache,
    parse_rally_annotations,
    put_bytes,
    put_file,
    resolve_config,
    save_id_cache,
)
from yp_video.config import ANNOTATIONS_DIR, CUTS_SIDELINE_DIR

DEFAULT_BASENAMES = [f"20260426-小窩-{i:02d}" for i in range(1, 13)]


def put_with_progress(url: str, path: Path, content_type: str) -> None:
    from tqdm import tqdm

    with tqdm(
        total=path.stat().st_size,
        unit="B",
        unit_scale=True,
        desc=path.name,
        leave=False,
    ) as bar:
        put_file(url, path, content_type, on_progress=bar.update)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--basenames",
        nargs="+",
        default=DEFAULT_BASENAMES,
        help="basename list (no extension); default = 20260426-小窩-01..12",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="parse rally annotations and resolve UUIDs but skip all HTTP calls",
    )
    args = parser.parse_args()

    try:
        config = resolve_config(include_tokens_env=True, required=not args.dry_run)
    except AppLibraryExportError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    cache = load_id_cache()
    matches: list[dict] = []

    for basename in args.basenames:
        mp4 = CUTS_SIDELINE_DIR / f"{basename}.mp4"
        ann = ANNOTATIONS_DIR / f"{basename}_annotations.jsonl"
        if not mp4.exists():
            print(f"skip {basename}: missing {mp4}", file=sys.stderr)
            continue
        if not ann.exists():
            # Only reviewed rally-annotations qualify; never raw model output.
            print(f"skip {basename}: rally-annotations missing", file=sys.stderr)
            continue

        try:
            duration, rallies = parse_rally_annotations(ann, require_rallies=False)
        except AppLibraryExportError as exc:
            print(f"skip {basename}: {exc}", file=sys.stderr)
            continue

        cache_entry = cache_entry_for(cache, basename)
        match_id = cache_entry["match_id"]

        if args.dry_run:
            public_url = (
                f"<dry-run>/videos/{config.user_id or '<USER_ID>'}/"
                f"{match_id}/source.mp4"
            )
            print(
                f"[dry] {basename}  match_id={match_id}  "
                f"rallies={len(rallies)}  duration={duration:.1f}s"
            )
        else:
            print(f"upload {basename}  match_id={match_id}  rallies={len(rallies)}")
            signed = fetch_video_upload_url(
                config,
                match_id=match_id,
                content_type=VIDEO_CONTENT_TYPE,
            )
            put_with_progress(signed["upload_url"], mp4, VIDEO_CONTENT_TYPE)
            public_url = signed["public_url"]

        entry, updated_cache = build_match_entry(
            basename=basename,
            cache_entry=cache_entry,
            public_url=public_url,
            duration=duration,
            rallies=rallies,
        )
        matches.append(entry)
        cache[basename] = updated_cache
        if not args.dry_run:
            save_id_cache(cache)

    manifest = build_manifest(user_id=config.user_id, matches=matches)
    manifest_bytes = encode_manifest(manifest)

    if args.dry_run:
        print("\n----- manifest preview -----")
        print(manifest_bytes.decode("utf-8"))
        return 0

    print(f"\nuploading manifest ({len(matches)} matches, {len(manifest_bytes)} bytes)")
    signed = fetch_manifest_upload_url(config)
    put_bytes(signed["upload_url"], manifest_bytes, "application/json")

    print("\nDone.")
    print("Public manifest URL (paste this into iOS Settings -> Library manifest URL):")
    print(f"  {signed['public_url']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
