"""Push one hand-corrected match into the iOS app's R2 library.

The Annotate page calls `export_one_match()` after every save: it uploads
the cut video to R2 (skipped when the file is unchanged) and writes a
single-match manifest the iOS app imports by URL.

Each match gets its OWN manifest file (`library/<match_id>.json`) rather
than one shared catalog. The iOS importer upserts per match and never
deletes matches absent from a manifest, so a per-match file lets the user
import exactly the matches they want — one URL at a time.

The bulk CLI (`scripts/export_to_app.py`) uses the same app-library helpers;
it still writes the shared `library/manifest.json`.
"""

from __future__ import annotations

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
from yp_video.config import ANNOTATIONS_DIR, find_cut


AppExportError = AppLibraryExportError


def export_one_match(basename: str) -> dict:
    """Upload one match's cut video + a single-match manifest to the app's
    R2 bucket. Returns ``{"manifest_url", "match_id", "rally_count"}``.

    `basename` is the cut-video stem, e.g. ``20260426-小窩-01``.
    """
    config = resolve_config(include_tokens_env=True, required=True)

    ann_path = ANNOTATIONS_DIR / f"{basename}_annotations.jsonl"
    if not ann_path.exists():
        raise AppExportError(
            f"No saved rally annotations for {basename} — save first."
        )
    mp4 = find_cut(f"{basename}.mp4")
    if mp4 is None or not mp4.exists():
        raise AppExportError(f"Cut video not found for {basename}.")

    duration, rallies = parse_rally_annotations(ann_path, require_rallies=True)

    cache = load_id_cache()
    entry_cache = cache_entry_for(cache, basename)
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
        signed = fetch_video_upload_url(
            config,
            match_id=match_id,
            content_type=VIDEO_CONTENT_TYPE,
        )
        put_file(signed["upload_url"], mp4, VIDEO_CONTENT_TYPE)
        public_video_url = signed["public_url"]
        video_uploaded = True

    # 2. Build this match's manifest entry. Rally UUIDs are keyed by
    #    (start, end) so a re-publish after a sub-0.1s tweak updates the
    #    same iOS rows; a bigger edit gets a fresh UUID (delete + insert).
    match_entry, updated_cache = build_match_entry(
        basename=basename,
        cache_entry=entry_cache,
        public_url=public_video_url,
        duration=duration,
        rallies=rallies,
    )
    cache[basename] = {
        **updated_cache,
        # Recorded so the next push can skip the video upload if unchanged.
        "video_size": video_size,
        "video_mtime": video_mtime,
        "video_url": public_video_url,
    }
    save_id_cache(cache)

    manifest = build_manifest(user_id=config.user_id, matches=[match_entry])

    # 3. Upload a per-match manifest (library/<match_id>.json) so the user
    #    imports exactly this match by its own URL.
    msigned = fetch_manifest_upload_url(config, name=match_id)
    put_bytes(
        msigned["upload_url"],
        encode_manifest(manifest),
        "application/json",
    )

    return {
        "manifest_url": msigned["public_url"],
        "match_id": match_id,
        "rally_count": len(rallies),
        "video_uploaded": video_uploaded,
    }
