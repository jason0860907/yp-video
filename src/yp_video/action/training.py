"""What SPOT action training reads and writes, minus the web.

The routers (action-train, fusion-model, actor-association) all orchestrate
the same corpus: the annotation directory, the hand-editable validation-set
file, the run-local label snapshot and its train/val split. Those rules live
here once, importable without FastAPI; the routers keep only request
validation and job plumbing.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable
from pathlib import Path

from yp_video.config import (
    ACTION_ANNOTATIONS_DIR,
    ACTION_CHECKPOINTS_DIR,
    ACTION_FRAMES_DIR,
    ACTION_VAL_SET_FILE,
    SPOT_DIR,
    cut_kind_of,
    find_cut,
)
from yp_video.core.jsonl import read_jsonl

log = logging.getLogger(__name__)

#: Resolves a cut filename whose bytes are not on local disk to its canonical
#: path (parent dir encodes the camera view). The web layer passes an R2-backed
#: resolver; ``None`` keeps this module import-clean of storage clients.
CutResolver = Callable[[str], Path | None]


def _locate_cut(name: str, resolve_missing: CutResolver | None) -> Path | None:
    return find_cut(name) or (resolve_missing(name) if resolve_missing else None)

#: Seconds of slack added on each side of the match window so clips straddling
#: the first/last rally boundary are not clipped too tightly.
RALLY_SAMPLE_MARGIN_S = 2.0


def count_jsonl_records(path: Path) -> tuple[int, int]:
    """(videos, events) for one label JSONL, or (0, 0) when absent."""
    if not path.exists():
        return 0, 0
    meta, records = read_jsonl(path)
    return len(records), int(meta.get("num_events") or sum(len(r.get("events", [])) for r in records))


def read_val_set_file() -> list[str]:
    """Validation video names from ACTION_VAL_SET_FILE, ignoring blanks/comments.

    The file is the hand-editable source of truth for holdout mode: one video per
    line, ``#`` starts a comment. Absent or all-comments → empty list (the caller
    turns that into a clear "populate the file" error).
    """
    if not ACTION_VAL_SET_FILE.exists():
        return []
    names: list[str] = []
    for line in ACTION_VAL_SET_FILE.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        # Only whole-line comments: video filenames legitimately contain '#'
        # (e.g. "#獅子王 vs. #屏東台電"), so an inline-'#' rule would truncate them.
        if not entry or entry.startswith("#"):
            continue
        names.append(entry)
    return names


def annotation_stats(resolve_missing: CutResolver | None = None) -> dict:
    ACTION_ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    # Totals plus a per-camera-view breakdown so the UI can reflect the selected
    # view. A video's view is its cut kind (broadcast / sideline).
    by_view: dict[str, dict[str, int]] = {
        "broadcast": {"videos": 0, "events": 0, "frames": 0},
        "sideline": {"videos": 0, "events": 0, "frames": 0},
    }
    val_names = {Path(entry).name for entry in read_val_set_file()}
    per_video: list[dict] = []
    videos = 0
    events = 0
    frames = 0
    for path in sorted(ACTION_ANNOTATIONS_DIR.glob("*_actions.jsonl")):
        try:
            meta, records = read_jsonl(path)
        except (OSError, json.JSONDecodeError):
            continue
        n_events = len(records)
        n_frames = int(meta.get("num_frames") or 0)
        videos += 1
        events += n_events
        frames += n_frames
        stem = str(meta.get("video") or path.stem.removesuffix("_actions"))
        video_path = _locate_cut(f"{stem}.mp4", resolve_missing)
        view = cut_kind_of(video_path) if video_path else None
        if view in by_view:
            by_view[view]["videos"] += 1
            by_view[view]["events"] += n_events
            by_view[view]["frames"] += n_frames
        per_video.append({
            "video": stem,
            "events": n_events,
            "frames": n_frames,
            "view": view or "unknown",
            "is_val": path.name in val_names,
        })
    return {
        "label_dir": str(ACTION_ANNOTATIONS_DIR),
        "frame_dir": str(ACTION_FRAMES_DIR),
        "checkpoint_dir": str(ACTION_CHECKPOINTS_DIR),
        "videos": videos,
        "events": events,
        "frames": frames,
        "by_view": by_view,
        "per_video": per_video,
        "exists": ACTION_ANNOTATIONS_DIR.exists(),
    }


def label_items(
    resolve_missing: CutResolver | None = None,
) -> list[tuple[Path, Path]]:
    """One ``(label_file, cut_video)`` pair per annotated video.

    Snapshotted once per training job and shared by the frame-cache and
    label-preparation phases — annotations saved while a job is already
    running land in the *next* run instead of desyncing the two phases
    (label prep would otherwise see a video the cache phase never built).

    ``resolve_missing`` may map a cut that is absent locally to its canonical
    path (e.g. because its bytes live only in R2). Such a video trains fine
    off its existing frame cache; the cache phase raises if the cache would
    need a rebuild.
    """
    items: list[tuple[Path, Path]] = []
    missing: list[str] = []
    for path in sorted(ACTION_ANNOTATIONS_DIR.glob("*_actions.jsonl")):
        try:
            meta, _records = read_jsonl(path)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot read action labels: {path.name}") from exc

        stem = str(meta.get("video") or path.stem.removesuffix("_actions"))
        video_path = _locate_cut(f"{stem}.mp4", resolve_missing)
        if video_path is None:
            missing.append(f"{stem}.mp4")
            continue

        items.append((path, video_path))

    if missing:
        sample = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f" and {len(missing) - 5} more"
        raise RuntimeError(f"Missing source video(s) for action labels: {sample}{suffix}")
    return items


def rally_match_span(meta: dict, num_frames: int) -> tuple[int, int] | None:
    """Frame span ``[first_rally_start, last_rally_end]`` (± margin) for sampling.

    Restricting training clips to this match window keeps the in-rally actions
    *and* the genuine dead time between rallies (real background), while
    excluding the warm-up / post-match regions whose real-but-unlabelled actions
    would otherwise be sampled as background and confuse the model. Returns
    ``None`` when the video has no rallies, so non-rally datasets fall back to
    whole-video sampling.
    """
    rallies = meta.get("rallies") or []
    fps = float(meta.get("fps") or 30.0)
    starts = [float(r["start"]) for r in rallies if r.get("start") is not None]
    ends = [float(r["end"]) for r in rallies if r.get("end") is not None]
    if not starts or not ends:
        return None
    start = max(0, int(round((min(starts) - RALLY_SAMPLE_MARGIN_S) * fps)))
    end = min(num_frames, int(round((max(ends) + RALLY_SAMPLE_MARGIN_S) * fps)))
    if end <= start:
        return None
    return start, end


def materialize_holdout_split(label_dir: Path, holdout_videos: list[str]) -> dict:
    """Split the flat label snapshot into ``train/`` and ``val/`` by filename.

    The chosen videos become validation; every other labelled video trains.
    The val-set file is one list mixing camera views: entries whose label
    exists in the source annotations but not in the (camera_view-filtered)
    snapshot simply aren't part of this run and are skipped, so a broadcast
    run validates on the list's broadcast videos and a sideline run on its
    sideline ones. Unknown names (typos) still fail loud (``ValueError``), as
    does an empty side after filtering: a silent mis-split is worse than a
    stopped job. Symlinks (not copies) keep the flat snapshot — and the audio
    precompute that globs it — intact.
    """
    files = sorted(label_dir.glob("*_actions.jsonl"))
    by_name = {path.name: path for path in files}
    # Entries are label-file paths (or bare filenames); match on the basename so
    # the val-set file can point straight at action-annotations/<video>.jsonl.
    wanted = {Path(entry).name for entry in holdout_videos if entry.strip()}
    if not wanted:
        raise ValueError("holdout mode needs at least one validation video")

    unknown = sorted(
        name for name in wanted
        if name not in by_name and not (ACTION_ANNOTATIONS_DIR / name).exists()
    )
    if unknown:
        raise ValueError(
            f"Validation label file(s) not found in {ACTION_ANNOTATIONS_DIR}: "
            f"{'; '.join(unknown)}"
        )

    skipped = sorted(name for name in wanted if name not in by_name)
    if skipped:
        log.info(
            "holdout: skipping %d val entr(ies) outside this camera view: %s",
            len(skipped), ", ".join(skipped),
        )
        wanted -= set(skipped)
    if not wanted:
        raise ValueError(
            "holdout mode: none of the validation videos match this camera view. "
            f"Add a matching video to {ACTION_VAL_SET_FILE}"
        )

    train_dir = label_dir.parent / "train"
    val_dir = label_dir.parent / "val"
    for target in (train_dir, val_dir):
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)

    train_videos: list[str] = []
    val_videos: list[str] = []
    for name, path in by_name.items():
        is_val = name in wanted
        (val_dir if is_val else train_dir).joinpath(name).symlink_to(path)
        stem = name.removesuffix("_actions.jsonl")
        (val_videos if is_val else train_videos).append(stem)

    if not train_videos:
        raise ValueError("holdout mode left no training videos; hold out fewer")

    out = {
        "train_label_dir": str(train_dir),
        "val_label_dir": str(val_dir),
        "train_videos": sorted(train_videos),
        "val_videos": sorted(val_videos),
    }
    if skipped:
        out["val_skipped_other_view"] = skipped
    return out


def vnl_stats() -> dict:
    base = SPOT_DIR / "data" / "vnl_1.5"
    train_path = base / "train.jsonl"
    val_path = base / "val.jsonl"
    test_path = base / "test.jsonl"
    train_videos, train_events = count_jsonl_records(train_path)
    val_videos, val_events = count_jsonl_records(val_path)
    test_videos, test_events = count_jsonl_records(test_path)
    frame_dir = base / "frames_224p"
    return {
        "dataset": "vnl_1.5",
        "base_dir": str(base),
        "frame_dir": str(frame_dir),
        "frame_dir_exists": frame_dir.exists(),
        "train_jsonl": str(train_path),
        "val_jsonl": str(val_path),
        "train_videos": train_videos,
        "train_events": train_events,
        "val_videos": val_videos,
        "val_events": val_events,
        "test_videos": test_videos,
        "test_events": test_events,
        "ready": train_path.exists() and val_path.exists() and frame_dir.exists(),
    }


def checkpoint_stats() -> dict:
    count = 0
    if ACTION_CHECKPOINTS_DIR.exists():
        count = sum(1 for path in ACTION_CHECKPOINTS_DIR.glob("*/checkpoint_best.pt") if path.is_file())
    return {
        "dir": str(ACTION_CHECKPOINTS_DIR),
        "runs": count,
        "exists": ACTION_CHECKPOINTS_DIR.exists(),
    }
