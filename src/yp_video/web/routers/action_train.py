"""SPOT action-label training router."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
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
from yp_video.web.job_helpers import (
    fail_job_from_exc,
    stop_vllm_for_job,
    stream_subprocess,
    terminal_prefix,
)
from yp_video.web.jobs import JobStatus, job_manager
from yp_video.web.spot_runs import (
    PackageExporter,
    TrainProgress,
    checkpoint_package_options,
    export_checkpoint_package,
    last_resumable_epoch,
    make_train_parsers,
    performance_payload,
    resumable_run_options,
    validate_checkpoint_dir,
)

log = logging.getLogger(__name__)
router = APIRouter()


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
    return validate_checkpoint_dir(
        resolve_checkpoint_path(path), root=ACTION_CHECKPOINTS_DIR
    )


def _resolve_checkpoint_dir(req: ActionTrainRequest, *, save_dir: Path) -> Path:
    if req.checkpoint_dir:
        return _action_checkpoint_path(req.checkpoint_dir)
    return validate_checkpoint_dir(
        ACTION_CHECKPOINTS_DIR / save_dir.name, root=ACTION_CHECKPOINTS_DIR
    )


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


def _action_label_items() -> list[tuple[Path, Path]]:
    """One ``(label_file, cut_video)`` pair per annotated video.

    Snapshotted once per training job and shared by the frame-cache and
    label-preparation phases — annotations saved while a job is already
    running land in the *next* run instead of desyncing the two phases
    (label prep would otherwise see a video the cache phase never built).
    """
    items: list[tuple[Path, Path]] = []
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

        items.append((path, video_path))

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
    *, items: list[tuple[Path, Path]], frame_dir: Path, save_dir: Path, camera_view: str = "all"
) -> dict:
    """Write run-local label copies whose frame counts match the SPOT cache.

    ``items`` is the job's label snapshot (see ``_action_label_items``) — the
    same list the frame-cache phase ran on, so every video here has a cache.
    When ``camera_view`` restricts to a single view, only matching videos are
    written, so the saved label snapshot equals what training actually used.
    """

    label_dir = save_dir / "labels" / "action-annotations"
    label_dir.mkdir(parents=True, exist_ok=True)
    for stale in label_dir.glob("*_actions.jsonl"):
        stale.unlink()

    videos = 0
    events = 0
    total_frames = 0
    span_frames = 0
    adjusted: list[dict] = []
    for path, video_path in items:
        try:
            meta, records = read_jsonl(path)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot read action labels: {path.name}") from exc

        stem = str(meta.get("video") or path.stem.removesuffix("_actions"))
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


def _export_action_checkpoint_package(
    *,
    run_dir: Path,
    package_dir: Path,
    req: ActionTrainRequest,
    cmd: list[str],
    label_summary: dict | None,
) -> dict:
    return export_checkpoint_package(
        run_dir=run_dir,
        package_dir=package_dir,
        checkpoints_root=ACTION_CHECKPOINTS_DIR,
        package_type="yp-video-action-checkpoint",
        label_subdir="action-annotations",
        label_glob="*_actions.jsonl",
        training={
            "source": req.source,
            "training_mode": req.training_mode,
            "dataset": req.dataset or _default_dataset(req.source),
            "frame_dir": str(_spot_path(req.frame_dir or _default_frame_dir(req.source))),
            "init_checkpoint": req.init_checkpoint or "",
            "label_summary": label_summary,
        },
        cmd=cmd,
    )


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
        "init_checkpoints": checkpoint_package_options(ACTION_CHECKPOINTS_DIR),
        "resumable_runs": resumable_run_options(),
        "vnl_1_5": _vnl_stats(),
        "action_annotations": _action_annotation_stats(),
        "action_checkpoints": _action_checkpoint_stats(),
        "active_job": _active_job(),
    }


@router.get("/performance")
def performance(run: str | None = None) -> dict:
    """Per-epoch validation metrics for an action-checkpoints run."""
    return performance_payload(ACTION_CHECKPOINTS_DIR, run)


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
        if last_resumable_epoch(save_dir) is None:
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
        name=f"Action Train ({dataset})",
    )

    async def run_job() -> None:
        exporter: PackageExporter | None = None
        try:
            await job_manager.update_job(job.id, status="running", message="Preparing action training...")
            frame_dir = _spot_path(req.frame_dir or _default_frame_dir(req.source))
            action_label_dir = None
            label_summary = None
            if req.source == "action_annotations":
                items = await asyncio.to_thread(_action_label_items)
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

                # Action JSONL metadata can inherit an over-reported MP4 frame
                # count, so expected_frames is None — the training labels are
                # normalized against the extracted cache in the next step.
                summary = await asyncio.to_thread(
                    ensure_action_frame_caches,
                    [(video_path, None) for _label, video_path in items],
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
                    items=items,
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
                    ctx = TrainProgress(epochs=req.num_epochs)
                    exporter = PackageExporter(
                        job.id,
                        resolved_save_dir,
                        lambda: _export_action_checkpoint_package(
                            run_dir=resolved_save_dir,
                            package_dir=checkpoint_dir,
                            req=req,
                            cmd=cmd,
                            label_summary=label_summary,
                        ),
                    )

                    parsers, is_key_line = make_train_parsers(
                        ctx,
                        params_key="action_train_progress",
                        criterion=req.criterion,
                        headline_pattern=(
                            r"Harmonic mean \(temporal and spatial mAPs\):\s*([0-9.]+)%"
                        ),
                        on_new_best=lambda: exporter.schedule(
                            ctx.best_epoch, "new_best"
                        ),
                    )

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
                        parsers=parsers,
                        is_key_line=is_key_line,
                        tee_to_terminal=True,
                        log_path=resolved_save_dir / "terminal.log",
                    )
            if rc == 0:
                if exporter is None:
                    raise RuntimeError("Checkpoint package exporter was not initialized")
                checkpoint_summary = await exporter.export_once(
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
            if exporter is not None:
                try:
                    checkpoint_summary = await exporter.export_once(
                        expected_epoch=None,
                        reason="cancelled",
                        update_job=False,
                    )
                except Exception:  # noqa: BLE001
                    log.exception("Failed to export action checkpoint package after cancellation")
            await job_manager.update_job(
                job.id,
                status="cancelled",
                message="Cancelled",
                params={
                    **job.params,
                    **({"checkpoint_package": checkpoint_summary} if checkpoint_summary else {}),
                },
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{terminal_prefix(job)}Failed: {type(exc).__name__}: {exc}", flush=True)
            log.exception("Action training failed")
            checkpoint_summary = None
            if exporter is not None:
                try:
                    checkpoint_summary = await exporter.export_once(
                        expected_epoch=None,
                        reason="failed",
                        update_job=False,
                    )
                except Exception:  # noqa: BLE001
                    log.exception("Failed to export action checkpoint package after failure")
            if checkpoint_summary:
                job_obj = job_manager.get_job(job.id)
                await job_manager.update_job(
                    job.id,
                    params={
                        **(job_obj.params if job_obj else job.params),
                        "checkpoint_package": checkpoint_summary,
                    },
                )
            await fail_job_from_exc(job.id, exc)

    task = asyncio.create_task(run_job())
    job_manager.attach_task(job, task)
    return job.to_dict()
