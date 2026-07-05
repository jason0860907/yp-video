"""SPOT action-label training router."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from yp_video.config import (
    ACTION_ANNOTATIONS_DIR,
    ACTION_AUDIO_DIR,
    ACTION_CHECKPOINTS_DIR,
    ACTION_FRAMES_DIR,
    ACTION_VAL_SET_FILE,
    CUTS_DIRS,
    SPOT_AUDIO_PRECOMPUTE_MODULE,
    SPOT_DIR,
    SPOT_PYTHON,
    SPOT_TRAIN_MODULE,
    cut_kind_of,
    find_cut,
)
from yp_video.contracts.action import (
    ACTION_CONTRACT_VERSION,
    ACTION_CONTRACT_VERSION_ENV,
)
from yp_video.action.frames import ensure_action_frame_caches, inspect_action_frame_cache
from yp_video.action.prelabel import resolve_checkpoint_path
from yp_video.core.jsonl import read_jsonl, write_jsonl
from yp_video.web.job_helpers import ProgressParser, stop_vllm_for_job, stream_subprocess
from yp_video.web.jobs import JobStatus, job_manager

log = logging.getLogger(__name__)
router = APIRouter()


@dataclass(slots=True)
class _TrainProgress:
    """Mutable running state for a SPOT training job's progress parsers.

    A dataclass (not a dict) so a mis-typed field raises AttributeError instead
    of silently creating a dead key — the parsers below all mutate this from
    different regex callbacks.
    """

    epochs: int
    completed_epoch: int = -1
    current_epoch: int = 0
    train_total: int = 0
    latest_train_loss: float | None = None
    latest_val_loss: float | None = None
    latest_val_map: float | None = None
    latest_val_breakdown: dict | None = None
    best_epoch: int | None = None
    best_value: float | None = None
    best_breakdown: dict | None = None


class ActionTrainRequest(BaseModel):
    source: str = Field(default="vnl_1_5", pattern="^(vnl_1_5|action_annotations)$")
    training_mode: str = Field(default="split", pattern="^(split|all|holdout)$")
    dataset: str | None = None
    frame_dir: str | None = None
    save_dir: str | None = None
    checkpoint_dir: str | None = None
    # None / "" → train from scratch; an explicit path → that checkpoint
    # (selected from ACTION_CHECKPOINTS_DIR).
    init_checkpoint: str | None = None
    # Continue an interrupted run: restore weights + optimizer/scheduler/history
    # from `save_dir` and keep training toward num_epochs. Requires `save_dir`
    # to point at an existing run with optimizer state; `init_checkpoint` is
    # ignored (SPOT loads from the checkpoint instead).
    resume: bool = False
    gpu: int = Field(default=0, ge=0)
    # "logmel" → late-fusion audio (precomputed before training); "none" →
    # pure-visual model (no audio, no precompute). Must match at inference.
    audio_backend: str = Field(default="logmel", pattern="^(logmel|none)$")
    feature_arch: str = "rny008_gsm"
    temporal_arch: str = "gru"
    pred_loc_arch: str = "mlp"
    clip_len: int = Field(default=64, ge=8, le=256)
    # Per-video frame stride targeting this sampling rate, so clip_len spans
    # the same wall-clock time on 30fps and 60fps sources. 0 = every frame.
    sample_fps: float = Field(default=30.0, ge=0, le=120)
    batch_size: int = Field(default=8, ge=1, le=64)
    num_epochs: int = Field(default=50, ge=1, le=1000)
    warm_up_epochs: int = Field(default=3, ge=0, le=100)
    learning_rate: float = Field(default=0.0003, gt=0)
    num_workers: int = Field(default=4, ge=0, le=32)
    criterion: str = Field(default="map", pattern="^(map|loss)$")
    start_val_epoch: int = Field(default=0, ge=0)
    epoch_num_frames: int | None = Field(default=None, ge=1)
    val_ratio: float = Field(default=0.2, gt=0, lt=1)
    split_seed: int = 42
    # holdout mode: the exact videos to hold out as the validation set; every
    # other labelled video trains. Entries may be the raw stem, `<stem>.mp4`, or
    # `<stem>_actions.jsonl` — matched against the run-local label snapshot.
    holdout_videos: list[str] = Field(default_factory=list)
    # "all" trains every view together; "broadcast"/"sideline" restrict to one
    # camera view (labels carry a camera_view tag from _prepare_action_training_labels).
    camera_view: str = Field(default="all", pattern="^(all|broadcast|sideline)$")
    predict_location: bool = True
    stop_vllm: bool = False


def _spot_path(path: str | Path) -> Path:
    p = Path(os.path.expanduser(str(path)))
    if not p.is_absolute():
        p = SPOT_DIR / p
    return p


def _default_frame_dir(source: str) -> str:
    if source == "vnl_1_5":
        return "data/vnl_1.5/frames_224p"
    return str(ACTION_FRAMES_DIR)


def _default_dataset(source: str) -> str:
    return "vnl_1.5" if source == "vnl_1_5" else "yp_actions"


def _safe_run_name(dataset: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset).strip("._") or "actions"


def _audio_tag(req: ActionTrainRequest) -> str:
    """Run-name fragment marking the modality: 'visual' or 'fusion'.

    Makes a run dir self-describing (e.g. yp_actions_fusion_<stamp> vs
    yp_actions_visual_<stamp>) so visual-only and audio late-fusion runs are
    distinguishable at a glance in exp/ and action-checkpoints/.
    """
    return "visual" if req.audio_backend == "none" else "fusion"


def _resolve_save_dir(req: ActionTrainRequest, dataset: str | None = None) -> Path:
    dataset = dataset or req.dataset or _default_dataset(req.source)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{_safe_run_name(dataset)}_{req.camera_view}_{_audio_tag(req)}_{stamp}"
    return _spot_path(req.save_dir or (Path("exp") / name))


def _action_checkpoint_path(path: str | Path) -> Path:
    return _validate_action_checkpoint_dir(resolve_checkpoint_path(path))


def _validate_action_checkpoint_dir(path: Path) -> Path:
    root = ACTION_CHECKPOINTS_DIR.resolve()
    resolved = path.expanduser().resolve()
    if resolved.parent != root:
        raise HTTPException(
            400,
            f"Checkpoint dir must be directly under {ACTION_CHECKPOINTS_DIR}",
        )
    return resolved


def _resolve_checkpoint_dir(req: ActionTrainRequest, *, save_dir: Path) -> Path:
    if req.checkpoint_dir:
        return _action_checkpoint_path(req.checkpoint_dir)
    return _validate_action_checkpoint_dir(ACTION_CHECKPOINTS_DIR / save_dir.name)


def _count_jsonl_records(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    meta, records = read_jsonl(path)
    return len(records), int(meta.get("num_events") or sum(len(r.get("events", [])) for r in records))


def _action_annotation_stats() -> dict:
    ACTION_ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    # Totals plus a per-camera-view breakdown so the UI can reflect the selected
    # view. A video's view is its cut kind (broadcast / sideline).
    by_view: dict[str, dict[str, int]] = {
        "broadcast": {"videos": 0, "events": 0, "frames": 0},
        "sideline": {"videos": 0, "events": 0, "frames": 0},
    }
    val_names = {Path(entry).name for entry in _read_val_set_file()}
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
        video_path = find_cut(f"{stem}.mp4")
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


def _action_frame_items() -> list[tuple[Path, int | None]]:
    items: list[tuple[Path, int | None]] = []
    missing: list[str] = []
    for path in sorted(ACTION_ANNOTATIONS_DIR.glob("*_actions.jsonl")):
        try:
            meta, _records = read_jsonl(path)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot read action labels: {path.name}") from exc

        stem = str(meta.get("video") or path.stem.removesuffix("_actions"))
        video_path = find_cut(f"{stem}.mp4")
        if video_path is None:
            missing.append(f"{stem}.mp4")
            continue

        # Action JSONL metadata can inherit an over-reported MP4 frame count.
        # The training labels are normalized against the extracted cache later.
        items.append((video_path, None))

    if missing:
        sample = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f" and {len(missing) - 5} more"
        raise RuntimeError(f"Missing source video(s) for action labels: {sample}{suffix}")
    return items


# Seconds of slack added on each side of the match window so clips straddling
# the first/last rally boundary are not clipped too tightly.
RALLY_SAMPLE_MARGIN_S = 2.0


def _rally_match_span(meta: dict, num_frames: int) -> tuple[int, int] | None:
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


def _prepare_action_training_labels(
    *, frame_dir: Path, save_dir: Path, camera_view: str = "all"
) -> dict:
    """Write run-local label copies whose frame counts match the SPOT cache.

    When ``camera_view`` restricts to a single view, only matching videos are
    written, so the saved label snapshot equals what training actually used.
    """

    label_files = sorted(ACTION_ANNOTATIONS_DIR.glob("*_actions.jsonl"))
    if not label_files:
        raise RuntimeError(f"No action JSONL labels found in {ACTION_ANNOTATIONS_DIR}")

    label_dir = save_dir / "labels" / "action-annotations"
    label_dir.mkdir(parents=True, exist_ok=True)
    for stale in label_dir.glob("*_actions.jsonl"):
        stale.unlink()

    videos = 0
    events = 0
    total_frames = 0
    span_frames = 0
    adjusted: list[dict] = []
    for path in label_files:
        try:
            meta, records = read_jsonl(path)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot read action labels: {path.name}") from exc

        stem = str(meta.get("video") or path.stem.removesuffix("_actions"))
        video_path = find_cut(f"{stem}.mp4")
        if video_path is None:
            raise RuntimeError(f"Missing source video for action labels: {stem}.mp4")

        view = cut_kind_of(video_path)
        if camera_view != "all" and view != camera_view:
            continue

        cache = inspect_action_frame_cache(video_path, cache_root=frame_dir)
        cache_frames = int(cache.get("frame_count") or 0)
        if cache_frames <= 0:
            raise RuntimeError(f"Missing action frame cache for {stem}")

        # An event past the extracted frame cache (usually a frame or two lost at
        # the video tail) can't be sampled — drop it instead of failing the run.
        kept = [
            event for event in records
            if int(round(float(event.get("frame", 0) or 0))) < cache_frames
        ]
        dropped = len(records) - len(kept)
        if dropped:
            log.warning(
                "%s: dropped %d action event(s) beyond the %d-frame cache",
                path.name, dropped, cache_frames,
            )
        records = kept

        original_frames = int(meta.get("num_frames") or 0)
        training_meta = {
            **meta,
            "num_frames": cache_frames,
            "training_num_frames_source": "action_frame_cache",
            "camera_view": view,
        }
        if original_frames and original_frames != cache_frames:
            training_meta["source_num_frames"] = original_frames
            adjusted.append({
                "video": stem,
                "source_num_frames": original_frames,
                "training_num_frames": cache_frames,
            })

        match_span = _rally_match_span(meta, cache_frames)
        if match_span is not None:
            training_meta["sample_spans"] = [list(match_span)]
            span_frames += match_span[1] - match_span[0]
        else:
            span_frames += cache_frames

        write_jsonl(label_dir / path.name, training_meta, records)
        videos += 1
        events += len(records)
        total_frames += cache_frames

    if videos == 0:
        raise RuntimeError(
            f"No '{camera_view}' action labels found in {ACTION_ANNOTATIONS_DIR}"
        )

    return {
        "label_dir": str(label_dir),
        "source_label_dir": str(ACTION_ANNOTATIONS_DIR),
        "videos": videos,
        "events": events,
        "frames": total_frames,
        "sample_frames": span_frames,
        "adjusted": adjusted,
    }


def _read_val_set_file() -> list[str]:
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


def _resolve_holdout_videos(req: ActionTrainRequest) -> list[str]:
    """Explicit request list wins; otherwise fall back to the val-set file."""
    names = req.holdout_videos or _read_val_set_file()
    if not names:
        raise HTTPException(
            400,
            "holdout mode needs a validation set. Add one video filename per line "
            f"to {ACTION_VAL_SET_FILE}",
        )
    return names


def _materialize_holdout_split(label_dir: Path, holdout_videos: list[str]) -> dict:
    """Split the flat label snapshot into ``train/`` and ``val/`` by filename.

    The chosen videos become validation; every other labelled video trains.
    The val-set file is one list mixing camera views: entries whose label
    exists in the source annotations but not in the (camera_view-filtered)
    snapshot simply aren't part of this run and are skipped, so a broadcast
    run validates on the list's broadcast videos and a sideline run on its
    sideline ones. Unknown names (typos) still fail loud, as does an empty
    side after filtering: a silent mis-split is worse than a stopped job.
    Symlinks (not copies) keep the flat snapshot — and the audio precompute
    that globs it — intact.
    """
    files = sorted(label_dir.glob("*_actions.jsonl"))
    by_name = {path.name: path for path in files}
    # Entries are label-file paths (or bare filenames); match on the basename so
    # the val-set file can point straight at action-annotations/<video>.jsonl.
    wanted = {Path(entry).name for entry in holdout_videos if entry.strip()}
    if not wanted:
        raise HTTPException(400, "holdout mode needs at least one validation video")

    unknown = sorted(
        name for name in wanted
        if name not in by_name and not (ACTION_ANNOTATIONS_DIR / name).exists()
    )
    if unknown:
        raise HTTPException(
            400,
            f"Validation label file(s) not found in {ACTION_ANNOTATIONS_DIR}: "
            f"{'; '.join(unknown)}",
        )

    skipped = sorted(name for name in wanted if name not in by_name)
    if skipped:
        log.info(
            "holdout: skipping %d val entr(ies) outside this camera view: %s",
            len(skipped), ", ".join(skipped),
        )
        wanted -= set(skipped)
    if not wanted:
        raise HTTPException(
            400,
            "holdout mode: none of the validation videos match this camera view. "
            f"Add a matching video to {ACTION_VAL_SET_FILE}",
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
        raise HTTPException(400, "holdout mode left no training videos; hold out fewer")

    out = {
        "train_label_dir": str(train_dir),
        "val_label_dir": str(val_dir),
        "train_videos": sorted(train_videos),
        "val_videos": sorted(val_videos),
    }
    if skipped:
        out["val_skipped_other_view"] = skipped
    return out


def _resolve_audio_dir(req: ActionTrainRequest, *, frame_dir: Path) -> Path | None:
    """Per-frame audio feature dir for this run's backend, or None for visual-only.

    Action labels precompute into a managed per-backend cache (built here, see
    ``_audio_precompute_command``). The VNL dataset's source videos aren't local,
    so its features must be precomputed offline next to the frame dir — fail loud
    if absent rather than silently training without audio.
    """
    if req.audio_backend == "none":
        return None
    if req.source == "vnl_1_5":
        audio_dir = frame_dir.parent / f"audio_{req.audio_backend}"
        if not audio_dir.exists():
            raise RuntimeError(
                f"VNL audio features not found: {audio_dir}. Precompute them with "
                f"`python -m yp_spot.audio.precompute --label-file data/vnl_1.5/*.jsonl "
                f"--video-root <vnl videos> --out {audio_dir} --backend {req.audio_backend}`, "
                "or set the audio backend to none for a visual-only model."
            )
        return audio_dir
    return ACTION_AUDIO_DIR / req.audio_backend


def _audio_precompute_command(
    req: ActionTrainRequest, *, label_dir: Path, audio_dir: Path
) -> list[str]:
    """Build the ``yp_spot.audio.precompute`` command for the run-local labels.

    Features are keyed by video name and reused across runs (precompute skips
    already-cached videos), so re-training the same set is cheap.
    """
    label_files = sorted(label_dir.glob("*_actions.jsonl"))
    if not label_files:
        raise RuntimeError(f"No action labels to precompute audio from in {label_dir}")
    return [
        str(SPOT_PYTHON),
        "-m",
        SPOT_AUDIO_PRECOMPUTE_MODULE,
        "--label-file",
        *(str(p) for p in label_files),
        "--video-root",
        *(str(d) for d in CUTS_DIRS),
        "--out",
        str(audio_dir),
        "--backend",
        req.audio_backend,
    ]


def _load_json_file(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _reset_checkpoint_package_dir(package_dir: Path) -> None:
    package_dir = _validate_action_checkpoint_dir(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    for child in package_dir.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink(missing_ok=True)


def _export_action_checkpoint_package(
    *,
    run_dir: Path,
    package_dir: Path,
    req: ActionTrainRequest,
    cmd: list[str],
    label_summary: dict | None,
) -> dict:
    best_checkpoint = run_dir / "checkpoint_best.pt"
    if not best_checkpoint.exists():
        raise RuntimeError(f"checkpoint_best.pt was not found in {run_dir}")

    _reset_checkpoint_package_dir(package_dir)

    copied: list[str] = []
    for name in (
        "checkpoint_best.pt",
        "checkpoint_best.json",
        "config.json",
        "metrics.jsonl",
        "loss.json",
        "terminal.log",
    ):
        src = run_dir / name
        if src.exists():
            dst = package_dir / name
            shutil.copy2(src, dst)
            copied.append(name)

    src_label_dir = run_dir / "labels" / "action-annotations"
    if src_label_dir.exists():
        dst_label_dir = package_dir / "labels" / "action-annotations"
        dst_label_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_label_dir, dst_label_dir)
        copied.extend(
            str(path.relative_to(package_dir))
            for path in sorted(dst_label_dir.glob("*_actions.jsonl"))
        )

    best = _load_json_file(run_dir / "checkpoint_best.json")
    config = _load_json_file(run_dir / "config.json")
    manifest = {
        "type": "yp-video-action-checkpoint",
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": package_dir.name,
        "source_run_dir": str(run_dir),
        "package_dir": str(package_dir),
        "checkpoint": "checkpoint_best.pt",
        "best": best if isinstance(best, dict) else None,
        "config": config if isinstance(config, dict) else None,
        "training": {
            "source": req.source,
            "training_mode": req.training_mode,
            "dataset": req.dataset or _default_dataset(req.source),
            "frame_dir": str(_spot_path(req.frame_dir or _default_frame_dir(req.source))),
            "init_checkpoint": req.init_checkpoint or "",
            "label_summary": label_summary,
        },
        "command": cmd,
        "files": copied,
        "omitted": [
            "checkpoint_*.pt",
            "optim_*.pt",
            "pred-val.*",
            "*.recall.json.gz",
        ],
    }
    manifest_path = package_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    copied.append("manifest.json")

    return {
        "dir": str(package_dir),
        "checkpoint": str(package_dir / "checkpoint_best.pt"),
        "files": copied,
        "best": manifest["best"],
    }


def _vnl_stats() -> dict:
    base = SPOT_DIR / "data" / "vnl_1.5"
    train_path = base / "train.jsonl"
    val_path = base / "val.jsonl"
    test_path = base / "test.jsonl"
    train_videos, train_events = _count_jsonl_records(train_path)
    val_videos, val_events = _count_jsonl_records(val_path)
    test_videos, test_events = _count_jsonl_records(test_path)
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


def _init_checkpoint_options() -> list[dict]:
    """Selectable init-checkpoint options: packaged action runs.

    Action checkpoints live under ACTION_CHECKPOINTS_DIR/<run>/checkpoint_best.pt
    and are returned as absolute paths (passed through _spot_path unchanged).
    """
    options: list[dict] = []
    if ACTION_CHECKPOINTS_DIR.exists():
        for run_dir in sorted(ACTION_CHECKPOINTS_DIR.iterdir(), reverse=True):
            ckpt = run_dir / "checkpoint_best.pt"
            if not run_dir.is_dir() or not ckpt.is_file():
                continue
            best = _load_json_file(run_dir / "checkpoint_best.json")
            value = best.get("value") if isinstance(best, dict) else None
            label = run_dir.name
            if isinstance(value, (int, float)):
                label = f"{run_dir.name} (mAP {value:.3f})"
            options.append({"label": label, "value": str(ckpt)})
    return options


def _last_resumable_epoch(run_dir: Path) -> int | None:
    """Latest epoch with optimizer state in ``run_dir``, or None if not resumable.

    Mirrors SPOT's ``get_last_epoch`` (globs ``optim_*.pt``): ``--resume`` needs
    the optimizer/scheduler snapshot, and SPOT prunes all but the latest one.
    """
    epochs = [
        int(m.group(1))
        for p in run_dir.glob("optim_*.pt")
        if (m := re.fullmatch(r"optim_(\d+)", p.stem))
    ]
    return max(epochs) if epochs else None


def _resumable_run_options() -> list[dict]:
    """Runs under ``exp/`` that ``--resume`` can continue (have optimizer state)."""
    exp_dir = SPOT_DIR / "exp"
    if not exp_dir.exists():
        return []
    options: list[dict] = []
    for run_dir in sorted(exp_dir.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        last_epoch = _last_resumable_epoch(run_dir)
        if last_epoch is None:
            continue
        best = _load_json_file(run_dir / "checkpoint_best.json")
        best_value = best.get("value") if isinstance(best, dict) else None
        label = f"{run_dir.name} (E{last_epoch + 1}"
        if isinstance(best_value, (int, float)):
            label += f", best {best_value:.3f}"
        label += ")"
        options.append({"label": label, "value": str(run_dir)})
    return options


def _action_checkpoint_stats() -> dict:
    count = 0
    if ACTION_CHECKPOINTS_DIR.exists():
        count = sum(1 for path in ACTION_CHECKPOINTS_DIR.glob("*/checkpoint_best.pt") if path.is_file())
    return {
        "dir": str(ACTION_CHECKPOINTS_DIR),
        "runs": count,
        "exists": ACTION_CHECKPOINTS_DIR.exists(),
    }


def _active_job() -> dict | None:
    for job in job_manager.jobs.values():
        if job.type == "action_train" and job.status == JobStatus.RUNNING:
            return job.to_dict()
    return None


@router.get("/status")
def status() -> dict:
    return {
        "spot_available": SPOT_DIR.exists() and SPOT_PYTHON.exists(),
        "spot_dir": str(SPOT_DIR),
        "spot_python": str(SPOT_PYTHON),
        "init_checkpoints": _init_checkpoint_options(),
        "resumable_runs": _resumable_run_options(),
        "vnl_1_5": _vnl_stats(),
        "action_annotations": _action_annotation_stats(),
        "action_checkpoints": _action_checkpoint_stats(),
        "active_job": _active_job(),
    }


def _normalize_metrics_entry(rec: dict) -> dict:
    """Flatten one epoch record into the flat shape the UI reads.

    Handles both the new ``metrics.jsonl`` schema (nested ``mAP``/``loss`` +
    ``lr``/``per_class``) and the legacy ``loss.json`` schema (flat ``val_mAP*``).
    """
    if "mAP" in rec:  # new metrics.jsonl schema
        m = rec.get("mAP") or {}
        loss = rec.get("loss") or {}
        return {
            "epoch": rec.get("epoch"),
            "lr": rec.get("lr"),
            "val_mAP": m.get("harmonic", 0),
            "val_mAP_temporal": m.get("temporal", 0),
            "val_mAP_spatial": m.get("spatial", 0),
            "train_loss": loss.get("train"),
            "val_loss": loss.get("val"),
            "per_class": rec.get("per_class") or {},
            "val_per_video": rec.get("per_video") or [],
        }
    return {  # legacy loss.json schema
        "epoch": rec.get("epoch"),
        "lr": rec.get("lr"),
        "val_mAP": rec.get("val_mAP", 0),
        "val_mAP_temporal": rec.get("val_mAP_temporal", 0),
        "val_mAP_spatial": rec.get("val_mAP_spatial", 0),
        "train_loss": rec.get("train"),
        "val_loss": rec.get("val"),
        "per_class": rec.get("per_class") or {},
        "val_per_video": rec.get("val_per_video") or [],
    }


def _read_run_metrics(run_dir: Path) -> tuple[dict | None, list[dict]]:
    """Read a run's per-epoch metrics, preferring metrics.jsonl over loss.json.

    Returns ``(meta, entries)`` where entries are normalized to the flat UI shape.
    """
    jsonl = run_dir / "metrics.jsonl"
    if jsonl.exists():
        meta: dict | None = None
        entries: list[dict] = []
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("_meta"):
                meta = rec
            else:
                entries.append(_normalize_metrics_entry(rec))
        return meta, entries

    loss = _load_json_file(run_dir / "loss.json")
    if isinstance(loss, list):
        return None, [_normalize_metrics_entry(r) for r in loss]
    return None, []


@router.get("/performance")
def performance(run: str | None = None) -> dict:
    """Per-epoch validation metrics (lr, mAP, per-class, per-video) for a run.

    Reads ``metrics.jsonl`` (falling back to the legacy ``loss.json``) from an
    action-checkpoints package. Defaults to the most recently modified run; pass
    ``run`` to select one by name. ``runs`` lists the runs (newest first).
    """
    base = ACTION_CHECKPOINTS_DIR
    if not base.exists():
        return {"entries": [], "runs": []}

    def has_metrics(d: Path) -> bool:
        return (d / "metrics.jsonl").exists() or (d / "loss.json").exists()

    runs = sorted(
        (d for d in base.iterdir() if d.is_dir() and has_metrics(d)),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        return {"entries": [], "runs": []}

    run_dir = (base / run) if run else runs[0]
    if not has_metrics(run_dir):
        raise HTTPException(404, f"No metrics for run {run_dir.name!r}")

    meta, entries = _read_run_metrics(run_dir)
    best = _load_json_file(run_dir / "checkpoint_best.json")
    return {
        "run": run_dir.name,
        "meta": meta,
        "best": best if isinstance(best, dict) else None,
        "entries": entries,
        "runs": [d.name for d in runs],
    }


def _build_command(
    req: ActionTrainRequest,
    *,
    save_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
    action_label_dir: Path | None = None,
    audio_dir: Path | None = None,
) -> tuple[list[str], Path, dict]:
    if not SPOT_DIR.exists():
        raise HTTPException(503, "SPOT is not available at ~/yp-spot")
    if not SPOT_PYTHON.exists():
        raise HTTPException(503, f"SPOT python not found: {SPOT_PYTHON}")

    if req.training_mode == "holdout" and req.source != "action_annotations":
        raise HTTPException(
            400, "holdout mode is only supported for action_annotations labels"
        )

    dataset = req.dataset or _default_dataset(req.source)
    frame_dir_value = req.frame_dir or _default_frame_dir(req.source)
    frame_dir = _spot_path(frame_dir_value)
    if not frame_dir.exists():
        raise HTTPException(400, f"Frame directory not found: {frame_dir}")

    if req.source == "vnl_1_5":
        for rel in ("data/vnl_1.5/train.jsonl", "data/vnl_1.5/val.jsonl"):
            if not (SPOT_DIR / rel).exists():
                raise HTTPException(400, f"Missing VNL JSONL labels: {SPOT_DIR / rel}")
    if req.init_checkpoint:
        init_checkpoint = _spot_path(req.init_checkpoint)
        if not init_checkpoint.exists():
            raise HTTPException(400, f"Init checkpoint not found: {init_checkpoint}")
    else:
        init_checkpoint = None

    save_dir = save_dir or _resolve_save_dir(req, dataset)
    checkpoint_dir = checkpoint_dir or _resolve_checkpoint_dir(req, save_dir=save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(SPOT_PYTHON),
        "-m",
        SPOT_TRAIN_MODULE,
        dataset,
        str(frame_dir),
        # Second -m is yp_spot.train's own feature-arch flag, not python's.
        "-m",
        req.feature_arch,
        "-t",
        req.temporal_arch,
        "-p",
        req.pred_loc_arch,
        "--clip_len",
        str(req.clip_len),
        "--sample_fps",
        str(req.sample_fps),
        "--batch_size",
        str(req.batch_size),
        "--num_epochs",
        str(req.num_epochs),
        "--warm_up_epochs",
        str(req.warm_up_epochs),
        "--learning_rate",
        str(req.learning_rate),
        "--num_workers",
        str(req.num_workers),
        "--criterion",
        req.criterion,
        "--start_val_epoch",
        str(req.start_val_epoch),
        "-s",
        str(save_dir),
    ]
    cmd.extend(["--audio_backend", req.audio_backend])
    if req.audio_backend != "none":
        if audio_dir is None:
            raise HTTPException(400, "Audio features missing for late-fusion training")
        cmd.extend(["--audio_dir", str(audio_dir)])
    if req.camera_view != "all":
        if req.source != "action_annotations":
            raise HTTPException(
                400,
                f"camera_view='{req.camera_view}' is only supported for "
                "action_annotations labels; VNL labels carry no camera_view tag.",
            )
        cmd.extend(["--camera_view", req.camera_view])
    if req.predict_location:
        cmd.append("--predict_location")
    if req.resume:
        if _last_resumable_epoch(save_dir) is None:
            raise HTTPException(
                400,
                f"Cannot resume: no optimizer checkpoint (optim_*.pt) in {save_dir}",
            )
        cmd.append("--resume")
    elif init_checkpoint is not None:
        cmd.extend(["--init_checkpoint", str(init_checkpoint)])
    if req.epoch_num_frames is not None:
        cmd.extend(["--epoch_num_frames", str(req.epoch_num_frames)])
    if req.source == "action_annotations":
        label_dir = action_label_dir or ACTION_ANNOTATIONS_DIR
        if not any(label_dir.glob("*_actions.jsonl")):
            raise HTTPException(400, f"No action JSONL labels found in {label_dir}")
        if req.training_mode == "all":
            cmd.extend([
                "--train_labels",
                str(label_dir),
                "--val_labels",
                str(label_dir),
            ])
        elif req.training_mode == "holdout":
            # train/ and val/ are symlink dirs materialized next to the flat
            # snapshot by _materialize_holdout_split before training starts.
            cmd.extend([
                "--train_labels",
                str(label_dir.parent / "train"),
                "--val_labels",
                str(label_dir.parent / "val"),
            ])
        else:
            cmd.extend([
                "--label_dir",
                str(label_dir),
                "--val_ratio",
                str(req.val_ratio),
                "--split_seed",
                str(req.split_seed),
            ])

    params = {
        "source": req.source,
        "dataset": dataset,
        "frame_dir": str(frame_dir),
        "save_dir": str(save_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "init_checkpoint": str(init_checkpoint) if init_checkpoint else "",
        "resume": req.resume,
        "gpu": req.gpu,
        "epochs": req.num_epochs,
        "feature_arch": req.feature_arch,
        "criterion": req.criterion,
        "audio_backend": req.audio_backend,
    }
    if audio_dir is not None:
        params["audio_dir"] = str(audio_dir)
    if req.source == "action_annotations":
        params["label_dir"] = str(action_label_dir or ACTION_ANNOTATIONS_DIR)
        params["training_mode"] = req.training_mode
        params["camera_view"] = req.camera_view
        if req.training_mode == "split":
            params["val_ratio"] = req.val_ratio
            params["split_seed"] = req.split_seed
        elif req.training_mode == "holdout":
            # Resolved val list lands in training_labels.val_videos; record the
            # source here so the manifest shows where the split came from.
            params["holdout_videos"] = req.holdout_videos
            params["val_set_file"] = str(ACTION_VAL_SET_FILE)
    return cmd, save_dir, params


@router.post("/start")
async def start(req: ActionTrainRequest) -> dict:
    dataset = req.dataset or _default_dataset(req.source)
    save_dir = _resolve_save_dir(req, dataset)
    checkpoint_dir = _resolve_checkpoint_dir(req, save_dir=save_dir)
    initial_params = {
        "source": req.source,
        "dataset": dataset,
        "frame_dir": str(_spot_path(req.frame_dir or _default_frame_dir(req.source))),
        "save_dir": str(save_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "gpu": req.gpu,
        "epochs": req.num_epochs,
        "feature_arch": req.feature_arch,
        "criterion": req.criterion,
    }
    if req.source == "action_annotations":
        initial_params["training_mode"] = req.training_mode
    job = job_manager.create_job(
        "action_train",
        initial_params,
        name=f"SPOT action training ({dataset})",
    )

    async def run_job() -> None:
        checkpoint_exporter: Callable[..., Awaitable[dict | None]] | None = None
        try:
            await job_manager.update_job(job.id, status="running", message="Preparing action training...")
            frame_dir = _spot_path(req.frame_dir or _default_frame_dir(req.source))
            action_label_dir = None
            label_summary = None
            if req.source == "action_annotations":
                items = await asyncio.to_thread(_action_frame_items)
                if not items:
                    raise RuntimeError(f"No action JSONL labels found in {ACTION_ANNOTATIONS_DIR}")

                loop = asyncio.get_running_loop()

                def frame_progress(done: int, total: int, message: str) -> None:
                    progress = 0.02 + (0.16 * done / total if total else 0.0)
                    loop.call_soon_threadsafe(
                        lambda progress=progress, message=message: asyncio.ensure_future(
                            job_manager.update_job(
                                job.id,
                                progress=progress,
                                message=message,
                            )
                        )
                    )

                summary = await asyncio.to_thread(
                    ensure_action_frame_caches,
                    items,
                    cache_root=frame_dir,
                    progress=frame_progress,
                )
                await job_manager.update_job(
                    job.id,
                    progress=0.18,
                    message="Frame cache ready.",
                    params={**job.params, "frame_cache": summary},
                )
                label_summary = await asyncio.to_thread(
                    _prepare_action_training_labels,
                    frame_dir=frame_dir,
                    save_dir=save_dir,
                    camera_view=req.camera_view,
                )
                action_label_dir = Path(label_summary["label_dir"])
                if req.training_mode == "holdout":
                    holdout_videos = _resolve_holdout_videos(req)
                    split = await asyncio.to_thread(
                        _materialize_holdout_split, action_label_dir, holdout_videos
                    )
                    label_summary = {**label_summary, **split}
                await job_manager.update_job(
                    job.id,
                    progress=0.2,
                    message="Training labels validated.",
                    params={**job.params, "training_labels": label_summary},
                )

            # Resolve / build audio features for late fusion (no-op visual-only).
            audio_dir = await asyncio.to_thread(
                _resolve_audio_dir, req, frame_dir=frame_dir
            )
            if audio_dir is not None and req.source == "action_annotations":
                audio_dir.mkdir(parents=True, exist_ok=True)
                pre_cmd = _audio_precompute_command(
                    req, label_dir=action_label_dir, audio_dir=audio_dir
                )
                await job_manager.update_job(
                    job.id,
                    message=f"Precomputing {req.audio_backend} audio features...",
                )
                rc, last_line = await stream_subprocess(job.id, pre_cmd, cwd=SPOT_DIR)
                if rc != 0:
                    raise RuntimeError(
                        f"Audio precompute failed (rc={rc}): {last_line}"
                    )

            cmd, resolved_save_dir, params = _build_command(
                req,
                save_dir=save_dir,
                checkpoint_dir=checkpoint_dir,
                action_label_dir=action_label_dir,
                audio_dir=audio_dir,
            )
            await job_manager.update_job(
                job.id,
                params={**job.params, **params},
                message="Waiting for GPU...",
            )
            async with stop_vllm_for_job(job.id, when=req.stop_vllm):
                async with job_manager.gpu_lock:
                    await job_manager.update_job(job.id, message="Starting SPOT training...")
                    ctx = _TrainProgress(epochs=req.num_epochs)
                    checkpoint_export_lock = asyncio.Lock()
                    checkpoint_export_tasks: set[asyncio.Task] = set()

                    def training_params(**extra) -> dict:
                        return {
                            "action_train_progress": {
                                "epoch": ctx.current_epoch,
                                "epoch_display": ctx.current_epoch + 1,
                                "epochs": max(1, ctx.epochs),
                                "completed_epoch": ctx.completed_epoch,
                                "latest_train_loss": ctx.latest_train_loss,
                                "latest_val_loss": ctx.latest_val_loss,
                                "latest_val_map": ctx.latest_val_map,
                                "latest_val_breakdown": ctx.latest_val_breakdown,
                                "best_epoch": ctx.best_epoch,
                                "best_value": ctx.best_value,
                                "best_breakdown": ctx.best_breakdown,
                                **extra,
                            }
                        }

                    def phase_progress(epoch: int, phase: str, step: int, total: int) -> float:
                        phase_offsets = {"train": 0.0, "val": 0.78, "map": 0.94}
                        phase_weights = {"train": 0.78, "val": 0.16, "map": 0.06}
                        frac = step / max(1, total)
                        epoch_frac = phase_offsets[phase] + phase_weights[phase] * frac
                        total_epochs = max(1, ctx.epochs)
                        return min(0.99, 0.2 + 0.79 * ((epoch + epoch_frac) / total_epochs))

                    def on_epoch(match: re.Match) -> dict:
                        epoch = int(match.group(1))
                        ctx.completed_epoch = max(ctx.completed_epoch, epoch)
                        ctx.current_epoch = epoch
                        return {
                            "params": training_params(
                                phase="summary",
                                phase_label="Epoch summary",
                            ),
                        }

                    def on_config_epochs(match: re.Match) -> dict | None:
                        ctx.epochs = int(match.group(1))
                        return None

                    def on_tqdm(match: re.Match) -> dict:
                        step = int(match.group("step"))
                        total = int(match.group("total"))
                        tail = match.group("tail") or ""
                        if "sum=" in tail:
                            if total >= int(ctx.train_total or 0):
                                ctx.train_total = total
                                phase = "train"
                                epoch = max(0, int(ctx.completed_epoch) + 1)
                            else:
                                phase = "val"
                                epoch = max(0, int(ctx.current_epoch))
                        else:
                            phase = "map"
                            epoch = max(0, int(ctx.current_epoch))

                        ctx.current_epoch = epoch
                        phase_label = {
                            "train": "Training",
                            "val": "Validation loss",
                            "map": "mAP evaluation",
                        }[phase]
                        loss_match = re.search(r"sum=([0-9.]+)", tail)
                        current_loss = float(loss_match.group(1)) if loss_match else None
                        pct = int(step * 100 / max(1, total))
                        total_epochs = max(1, ctx.epochs)
                        return {
                            "progress": phase_progress(epoch, phase, step, total),
                            "message": (
                                f"Epoch {epoch + 1}/{total_epochs} - "
                                f"{phase_label} {step}/{total} ({pct}%)"
                            ),
                            "params": training_params(
                                phase=phase,
                                phase_label=phase_label,
                                step=step,
                                total=total,
                                phase_progress=step / max(1, total),
                                current_loss=current_loss,
                            ),
                        }

                    def on_train_loss(match: re.Match) -> dict:
                        ctx.latest_train_loss = float(match.group(4))
                        return {"params": training_params()}

                    def on_val_loss(match: re.Match) -> dict:
                        ctx.latest_val_loss = float(match.group(4))
                        return {"params": training_params()}

                    def on_val_map(match: re.Match) -> dict:
                        ctx.latest_val_map = float(match.group(1)) / 100.0
                        return {"params": training_params()}

                    def on_val_metrics(match: re.Match) -> dict | None:
                        try:
                            ctx.latest_val_breakdown = json.loads(match.group(1))
                        except json.JSONDecodeError:
                            return None
                        return {"params": training_params()}

                    async def export_checkpoint_package_once(
                        *,
                        expected_epoch: int | None,
                        reason: str,
                        update_job: bool = True,
                    ) -> dict | None:
                        for _ in range(120):
                            best = _load_json_file(resolved_save_dir / "checkpoint_best.json")
                            best_epoch = best.get("epoch") if isinstance(best, dict) else None
                            ready = (
                                (resolved_save_dir / "checkpoint_best.pt").exists()
                                and isinstance(best_epoch, int)
                                and (expected_epoch is None or best_epoch == expected_epoch)
                            )
                            if ready:
                                async with checkpoint_export_lock:
                                    summary = await asyncio.to_thread(
                                        _export_action_checkpoint_package,
                                        run_dir=resolved_save_dir,
                                        package_dir=checkpoint_dir,
                                        req=req,
                                        cmd=cmd,
                                        label_summary=label_summary,
                                    )
                                if update_job:
                                    await job_manager.update_job(
                                        job.id,
                                        params={
                                            **job.params,
                                            "checkpoint_package": summary,
                                            "checkpoint_package_reason": reason,
                                        },
                                    )
                                return summary
                            await asyncio.sleep(0.5)

                        log.warning(
                            "Timed out waiting to export action checkpoint package "
                            "for %s (expected_epoch=%s, run_dir=%s)",
                            reason,
                            expected_epoch,
                            resolved_save_dir,
                        )
                        return None

                    checkpoint_exporter = export_checkpoint_package_once

                    def schedule_checkpoint_export(expected_epoch: int | None, reason: str) -> None:
                        task = asyncio.create_task(
                            export_checkpoint_package_once(
                                expected_epoch=expected_epoch,
                                reason=reason,
                            )
                        )
                        checkpoint_export_tasks.add(task)
                        task.add_done_callback(checkpoint_export_tasks.discard)

                    def on_new_best(_match: re.Match) -> dict:
                        ctx.best_epoch = ctx.current_epoch
                        ctx.best_value = (
                            ctx.latest_val_map
                            if req.criterion == "map"
                            else ctx.latest_val_loss
                        )
                        ctx.best_breakdown = ctx.latest_val_breakdown
                        schedule_checkpoint_export(ctx.best_epoch, "new_best")
                        return {"params": training_params()}

                    env = {
                        **os.environ,
                        "PYTHONUNBUFFERED": "1",
                        "PYTHONPATH": (
                            f"{SPOT_DIR}{os.pathsep}{os.environ['PYTHONPATH']}"
                            if os.environ.get("PYTHONPATH")
                            else str(SPOT_DIR)
                        ),
                        "CUDA_VISIBLE_DEVICES": str(req.gpu),
                        ACTION_CONTRACT_VERSION_ENV: ACTION_CONTRACT_VERSION,
                    }
                    rc, last_line = await stream_subprocess(
                        job.id,
                        cmd,
                        cwd=SPOT_DIR,
                        env=env,
                        parsers=[
                            ProgressParser(r'"num_epochs":\s*(\d+)', on_config_epochs),
                            ProgressParser(
                                r"(?P<pct>\d+)%\|.*?\|\s*(?P<step>\d+)/(?P<total>\d+)\s*\[[^\]]+\](?P<tail>.*)",
                                on_tqdm,
                            ),
                            ProgressParser(r"Epoch:\s*(\d+)", on_epoch),
                            ProgressParser(
                                r"Train loss\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)",
                                on_train_loss,
                            ),
                            ProgressParser(
                                r"Val loss\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)",
                                on_val_loss,
                            ),
                            ProgressParser(
                                r"Harmonic mean \(temporal and spatial mAPs\):\s*([0-9.]+)%",
                                on_val_map,
                            ),
                            ProgressParser(r"SPOT_METRICS (\{.*\})", on_val_metrics),
                            ProgressParser(r"New best epoch!", on_new_best),
                        ],
                        is_key_line=lambda line: (
                            "Epoch:" in line
                            or "Best epoch" in line
                            or "New best epoch" in line
                            or "Harmonic mean" in line
                            or "SPOT_METRICS" in line
                            or "Train loss" in line
                            or "Val loss" in line
                        ),
                        tee_to_terminal=True,
                        terminal_prefix="[action-train] ",
                        log_path=resolved_save_dir / "terminal.log",
                    )
            if rc == 0:
                if checkpoint_exporter is None:
                    raise RuntimeError("Checkpoint package exporter was not initialized")
                checkpoint_summary = await checkpoint_exporter(
                    expected_epoch=None,
                    reason="completed",
                    update_job=False,
                )
                if checkpoint_summary is None:
                    raise RuntimeError(f"Training finished but no checkpoint package was exported to {checkpoint_dir}")
                await job_manager.update_job(
                    job.id,
                    status="completed",
                    progress=1.0,
                    message=f"Training complete: {checkpoint_dir}",
                    params={**job.params, "checkpoint_package": checkpoint_summary},
                )
            else:
                raise RuntimeError(last_line or f"SPOT training exited with code {rc}")
        except asyncio.CancelledError:
            checkpoint_summary = None
            if checkpoint_exporter is not None:
                try:
                    checkpoint_summary = await checkpoint_exporter(
                        expected_epoch=None,
                        reason="cancelled",
                        update_job=False,
                    )
                except Exception:  # noqa: BLE001
                    log.exception("Failed to export action checkpoint package after cancellation")
            await job_manager.update_job(
                job.id,
                status="cancelled",
                message="Training cancelled",
                params={
                    **job.params,
                    **({"checkpoint_package": checkpoint_summary} if checkpoint_summary else {}),
                },
            )
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            print(f"\n[action-train] Failed:\n{tb}", flush=True)
            log.error("Action training failed:\n%s", tb)
            job_obj = job_manager.get_job(job.id)
            checkpoint_summary = None
            if checkpoint_exporter is not None:
                try:
                    checkpoint_summary = await checkpoint_exporter(
                        expected_epoch=None,
                        reason="failed",
                        update_job=False,
                    )
                except Exception:  # noqa: BLE001
                    log.exception("Failed to export action checkpoint package after failure")
            if job_obj:
                job_obj.logs.append(f"{type(exc).__name__}: {exc}")
                job_obj.logs.extend(tb.splitlines())
            await job_manager.update_job(
                job.id,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                message="SPOT action training failed",
                params={
                    **(job_obj.params if job_obj else job.params),
                    **({"checkpoint_package": checkpoint_summary} if checkpoint_summary else {}),
                },
            )

    task = asyncio.create_task(run_job())
    job_manager.attach_task(job, task)
    return job.to_dict()
