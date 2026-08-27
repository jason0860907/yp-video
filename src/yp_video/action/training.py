"""What SPOT action training reads and writes, minus the web.

Fusion Train and Association Train orchestrate the same corpus: the
annotation directory, the hand-editable validation-set file, the run-local
label snapshot and its train/val split. Those rules live here once,
importable without FastAPI; the routers keep only request validation and job
plumbing.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable
from pathlib import Path

from yp_video.config import (
    ACTION_ANNOTATIONS_DIR,
    ACTION_FRAMES_DIR,
    ACTION_PRE_ANNOTATIONS_DIR,
    ACTION_VAL_SET_FILE,
    SPOT_CHECKPOINTS_DIR,
    cut_kind_of,
)
from yp_video.contracts.action import LABEL_FILE_GLOB, RALLY_LABEL_FILE_GLOB
from yp_video.core.jsonl import read_jsonl
from yp_video.core.rallies import load_rallies

log = logging.getLogger(__name__)

#: Resolves a cut filename to its canonical path, whose parent dir encodes the
#: camera view. The path may not exist on disk: training reads a video's frame
#: cache, never the mp4, so a cut whose bytes live only in R2 still trains.
#: The web layer passes ``r2_client.resolve_cut``; this module stays
#: import-clean of storage clients.
CutResolver = Callable[[str], Path | None]

#: Seconds of slack added on each side of the match window so clips straddling
#: the first/last rally boundary are not clipped too tightly.
RALLY_SAMPLE_MARGIN_S = 2.0


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


def annotation_stats(resolve: CutResolver) -> dict:
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
        video_path = resolve(f"{stem}.mp4")
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
        "checkpoint_dir": str(SPOT_CHECKPOINTS_DIR),
        "videos": videos,
        "events": events,
        "frames": frames,
        "by_view": by_view,
        "per_video": per_video,
        "exists": ACTION_ANNOTATIONS_DIR.exists(),
    }


def label_items(
    resolve: CutResolver,
    *,
    include_predictions: bool = False,
) -> list[tuple[Path, Path]]:
    """One ``(label_file, cut_video)`` pair per annotated video.

    Snapshotted once per training job and shared by the frame-cache and
    label-preparation phases — annotations saved while a job is already
    running land in the *next* run instead of desyncing the two phases
    (label prep would otherwise see a video the cache phase never built).

    ``include_predictions`` also picks up SPOT pre-annotations for videos
    that have no human label file yet — pseudo-labels for training only;
    the holdout guard in the launcher keeps them out of validation. A human
    label of the same name always wins over the prediction.

    ``resolve`` maps each cut filename to its canonical path; a cut whose
    bytes live only in R2 trains fine off its existing frame cache, and the
    cache phase raises if that cache would need a rebuild.
    """
    label_files = sorted(ACTION_ANNOTATIONS_DIR.glob("*_actions.jsonl"))
    if include_predictions:
        annotated = {path.name for path in label_files}
        label_files += sorted(
            path
            for path in ACTION_PRE_ANNOTATIONS_DIR.glob("*_actions.jsonl")
            if path.name not in annotated
        )

    items: list[tuple[Path, Path]] = []
    missing: list[str] = []
    for path in label_files:
        try:
            meta, _records = read_jsonl(path)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot read action labels: {path.name}") from exc

        stem = str(meta.get("video") or path.stem.removesuffix("_actions"))
        video_path = resolve(f"{stem}.mp4")
        if video_path is None:
            missing.append(f"{stem}.mp4")
            continue

        items.append((path, video_path))

    if missing:
        sample = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f" and {len(missing) - 5} more"
        raise RuntimeError(f"Missing source video(s) for action labels: {sample}{suffix}")
    return items


def prediction_label_stems(items: list[tuple[Path, Path]]) -> set[str]:
    """Video stems in ``items`` whose labels are SPOT predictions, not human work."""
    return {
        path.stem.removesuffix("_actions")
        for path, _video in items
        if path.parent == ACTION_PRE_ANNOTATIONS_DIR
    }


def rally_match_span(stem: str, *, fps: float, num_frames: int) -> tuple[int, int] | None:
    """Frame span ``[first_rally_start, last_rally_end]`` (± margin) for sampling.

    Restricting training clips to this match window keeps the in-rally actions
    *and* the genuine dead time between rallies (real background), while
    excluding the warm-up / post-match regions whose real-but-unlabelled actions
    would otherwise be sampled as background and confuse the model. Returns
    ``None`` when the video has no rallies, so non-rally datasets fall back to
    whole-video sampling.

    Reads the LIVE rally store (core/rallies) — the action label file carries
    no rally copy, so a rally re-edit reaches the next training run without
    anyone re-saving the action annotation.
    """
    rallies = load_rallies(stem)
    starts = [float(r["start"]) for r in rallies if r.get("start") is not None]
    ends = [float(r["end"]) for r in rallies if r.get("end") is not None]
    if not starts or not ends:
        return None
    start = max(0, int(round((min(starts) - RALLY_SAMPLE_MARGIN_S) * fps)))
    end = min(num_frames, int(round((max(ends) + RALLY_SAMPLE_MARGIN_S) * fps)))
    if end <= start:
        return None
    return start, end


def materialize_holdout_split(
    label_dir: Path, holdout_stems: set[str], *, known_stems: set[str]
) -> dict:
    """Split the flat label snapshot into ``train/`` and ``val/`` by video stem.

    The chosen videos become validation; every other labelled video trains.
    Works for action (``*_actions.jsonl``) and rally (``*_rally.jsonl``)
    snapshots alike. ``known_stems`` is every annotated video before
    camera-view filtering: a wanted stem that is annotated but not in this
    snapshot simply isn't part of the run and is skipped (a broadcast run
    validates on the list's broadcast videos); an unknown stem (a typo) fails
    loud, as does an empty side — a silent mis-split is worse than a stopped
    job. Symlinks (not copies) keep the flat snapshot — and the audio
    precompute that globs it — intact.
    """
    files = sorted(
        path for glob in (LABEL_FILE_GLOB, RALLY_LABEL_FILE_GLOB)
        for path in label_dir.glob(glob)
    )
    by_stem = {_label_stem(path.name): path for path in files}
    wanted = {stem for stem in holdout_stems if stem}
    if not wanted:
        raise ValueError("manual validation needs at least one validation video")

    unknown = sorted(stem for stem in wanted if stem not in by_stem and stem not in known_stems)
    if unknown:
        raise ValueError(f"Validation video(s) not annotated: {'; '.join(unknown)}")

    skipped = sorted(stem for stem in wanted if stem not in by_stem)
    if skipped:
        log.info(
            "holdout: skipping %d val entr(ies) outside this run: %s",
            len(skipped), ", ".join(skipped),
        )
        wanted -= set(skipped)
    if not wanted:
        raise ValueError(
            "manual validation: none of the validation videos are in this run "
            "(camera view / scope filtered them all out)"
        )

    train_dir = label_dir.parent / "train"
    val_dir = label_dir.parent / "val"
    for target in (train_dir, val_dir):
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)

    train_videos: list[str] = []
    val_videos: list[str] = []
    for stem, path in by_stem.items():
        is_val = stem in wanted
        (val_dir if is_val else train_dir).joinpath(path.name).symlink_to(path)
        (val_videos if is_val else train_videos).append(stem)

    if not train_videos:
        raise ValueError("manual validation left no training videos; hold out fewer")

    out = {
        "train_label_dir": str(train_dir),
        "val_label_dir": str(val_dir),
        "train_videos": sorted(train_videos),
        "val_videos": sorted(val_videos),
    }
    if skipped:
        out["val_skipped_other_view"] = skipped
    return out


def _label_stem(filename: str) -> str:
    for glob in (LABEL_FILE_GLOB, RALLY_LABEL_FILE_GLOB):
        filename = filename.removesuffix(glob.removeprefix("*"))
    return filename


def checkpoint_stats() -> dict:
    count = 0
    if SPOT_CHECKPOINTS_DIR.exists():
        count = sum(1 for path in SPOT_CHECKPOINTS_DIR.glob("*/checkpoint_best.pt") if path.is_file())
    return {
        "dir": str(SPOT_CHECKPOINTS_DIR),
        "runs": count,
        "exists": SPOT_CHECKPOINTS_DIR.exists(),
    }
