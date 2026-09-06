"""One fusion checkpoint, every answer it gives for a video.

Stages per video, in pipeline order: rally (the spans, and who won each)
→ action (events, kept inside those spans) → association (who acted,
chosen among the tracklets). Rally and action share one decode pass of the
video (``run_spot_stages``). Each stage writes the same machine store the
single-stage predict page writes, so the label editors, the work lists and
the pipeline chips see no difference between the two routes.

Router-free and synchronous: the job runs one video's stages in an executor
thread (``job_helpers.spawn_batch_video_job``). Mirroring to R2 needs the
event loop, so a stage reports what it wrote and the router mirrors it.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from yp_video.action import prelabel
from yp_video.action.predict import run_spot_inference
from yp_video.action.rally import events_to_rally_segments
from yp_video.action.segments import filter_events_to_spans, pad_and_merge_spans
from yp_video.actor import policy as actor_policy
from yp_video.actor import spot_associate
from yp_video.config import RALLY_SPOT_PRE_ANNOTATIONS_DIR, SPOT_CHECKPOINTS_DIR
from yp_video.core.ffmpeg import probe_video_metadata
from yp_video.core.jsonl import write_jsonl
from yp_video.core.rallies import annotation_name, load_rallies, number_rallies
from yp_video.extraction import reassociate
from yp_video.extraction.store import records_path
from yp_video.tracklets.store import tracks_path
from yp_video.web.action_annotations import (
    pre_annotation_path,
    save_spot_pre_annotation,
)

#: The heads one checkpoint must carry to answer every stage.
REQUIRED_TASKS = ("rally", "action", "actor")

#: Action inference scans each rally span with this much slack on both
#: sides — the same cascade Action Predict and the selfhost worker run.
RALLY_PAD_S = 2.0

STAGES = ("rally", "action", "association")
_UNITS_PER_STAGE = 100

#: ``(done, total, message)`` — the batch job's per-item progress.
BatchProgress = Callable[[int, int, str], None]
#: ``(fraction, message)`` within one stage.
StageProgress = Callable[[float, str], None]


@dataclass(frozen=True)
class RallyOptions:
    min_score: float
    max_gap_s: float
    min_duration_s: float


@dataclass(frozen=True)
class SpotOptions:
    batch_size: int
    num_workers: int
    clip_len: int
    use_amp: bool = True


# ---------------------------------------------------------------- checkpoints


def package_tasks(checkpoint: Path) -> list[str]:
    """The heads a checkpoint package declares (manifest.json ``tasks``)."""
    manifest = checkpoint.parent / "manifest.json"
    try:
        return list(json.loads(manifest.read_text(encoding="utf-8")).get("tasks") or [])
    except (OSError, ValueError):
        return []


def list_checkpoints(root: Path = SPOT_CHECKPOINTS_DIR) -> list[dict]:
    """SPOT packages that serve every stage, newest first."""
    return [
        row for row in prelabel.list_checkpoints(root)
        if set(REQUIRED_TASKS) <= set(row["tasks"])
    ]


def default_checkpoint(root: Path = SPOT_CHECKPOINTS_DIR) -> str:
    rows = list_checkpoints(root)
    return rows[0]["path"] if rows else ""


def resolve_checkpoint(value: str) -> Path:
    """The checkpoint file behind ``value`` (the newest fusion package when
    empty), verified to carry every head the stages need."""
    ref = value or default_checkpoint()
    if not ref:
        raise FileNotFoundError(
            "No SPOT package serves rally, action and actor together; "
            "train an Action + Rally + Winner recipe on the Train page first"
        )
    checkpoint = prelabel.resolve_checkpoint(ref)
    tasks = package_tasks(checkpoint)
    missing = [task for task in REQUIRED_TASKS if task not in tasks]
    if missing:
        raise ValueError(
            f"{checkpoint.parent.name} serves {tasks or 'no tasks'}; "
            f"Inference needs {', '.join(missing)} as well"
        )
    reason = spot_associate.rejection(checkpoint)
    if reason is not None:
        raise ValueError(reason)
    return checkpoint


# ------------------------------------------------------------- spot stages


def rally_spot_pre_annotation_path(stem: str) -> Path:
    return RALLY_SPOT_PRE_ANNOTATIONS_DIR / annotation_name(stem)


def save_rally_pre_annotation(
    *,
    video: Path,
    source: str,
    events: list[dict],
    checkpoint: Path,
    options: RallyOptions,
) -> tuple[Path, int]:
    """Merge yp-spot's per-frame rally events into numbered spans and write
    them as the video's SPOT rally pre-annotation. Returns the file and the
    number of rallies."""
    metadata = probe_video_metadata(source)
    segments, max_rally_id = number_rallies(
        events_to_rally_segments(
            events,
            native_fps=float(metadata["fps"]),
            min_score=options.min_score,
            max_gap_s=options.max_gap_s,
            min_duration_s=options.min_duration_s,
        )
    )
    path = rally_spot_pre_annotation_path(video.stem)
    write_jsonl(
        path,
        {
            "video": str(video),
            "duration": float(metadata["duration"]),
            "max_rally_id": max_rally_id,
            "source": {
                "type": "rally-spot",
                "checkpoint": prelabel.checkpoint_ref(checkpoint),
                "min_score": options.min_score,
                "max_gap_s": options.max_gap_s,
                "min_duration_s": options.min_duration_s,
            },
        },
        segments,
    )
    return path, len(segments)


def _action_scan_spans(stem: str, duration_s: float) -> list[tuple[float, float]] | None:
    """Where the action scan looks: the padded rally spans — dead time only
    contributes false positives. None when the video has no rally source, in
    which case the whole video counts."""
    rallies = load_rallies(stem)
    if not rallies:
        return None
    return pad_and_merge_spans(rallies, pad_s=RALLY_PAD_S, duration_s=duration_s)


def run_spot_stages(
    *,
    video: Path,
    source: str,
    checkpoint: Path,
    tasks: Sequence[str],
    rally: RallyOptions,
    action_min_score: float,
    spot: SpotOptions,
    on_progress: StageProgress,
) -> dict:
    """The rally and/or action head, in ONE decode pass of the video.

    Both heads read the same frames, so asking yp-spot for them together
    decodes the video once instead of once per head — and decoding, not the
    model, is where the time goes. When rally runs in this pass the action
    scan covers the whole video and its events are cut to the padded rally
    spans afterwards; when only action runs, the existing rallies restrict
    the decode up front.
    """
    tasks = tuple(tasks)
    meta = probe_video_metadata(source)
    duration_s = float(meta["duration"])
    segments = _action_scan_spans(video.stem, duration_s) if tasks == ("action",) else None
    predictions = run_spot_inference(
        source,
        checkpoint=checkpoint,
        tasks=tasks,
        batch_size=spot.batch_size,
        num_workers=spot.num_workers,
        clip_len=spot.clip_len,
        use_amp=spot.use_amp,
        segments=segments,
        on_progress=lambda fraction: on_progress(fraction, "inference"),
    )
    out: dict = {"written": []}
    if "rally" in tasks:
        on_progress(1.0, "merging rallies")
        records = predictions["rally"]
        events = (records[0].get("events") or []) if records else []
        path, count = save_rally_pre_annotation(
            video=video, source=source, events=events, checkpoint=checkpoint, options=rally,
        )
        out["rallies"] = count
        out["written"].append((path, "rally-spot/pre-annotations"))
    if "action" in tasks:
        on_progress(1.0, "saving events")
        records = predictions["action"]
        if segments is None:
            spans = _action_scan_spans(video.stem, duration_s)
            if spans is not None:
                records = filter_events_to_spans(records, spans, fps=float(meta["fps"]))
        data = save_spot_pre_annotation(
            video=video, meta=meta, predictions=records, checkpoint=checkpoint, min_score=action_min_score,
        )
        out["events"] = data["num_events"]
        out["written"].append((pre_annotation_path(video.name), "action/pre-annotations"))
    return out


# ---------------------------------------------------------- association stage


def association_blocker(stem: str) -> str | None:
    """Why association cannot run for this video yet, or None.

    The actor head picks among tracklets and writes its pick into the
    extraction records, so both upstream stages must exist. Neither is this
    job's to produce: tracking and player detection have their own pages.
    """
    if not tracks_path(stem).exists():
        return "run Rally Tracking first"
    if not records_path(stem).exists():
        return "run Player Detection first"
    return None


def run_association_stage(
    *, video: Path, checkpoint: Path, on_progress: StageProgress
) -> dict:
    def progress(done: int, total: int, message: str) -> None:
        on_progress(done / total if total else 0.0, message)

    plan = actor_policy.SpotPlan(checkpoint)
    counts = reassociate.reassociate_video(
        video, plan.build(video, progress), on_progress=progress
    )
    return {"counts": counts, "written": []}


# ------------------------------------------------------------------ per video


@dataclass(frozen=True)
class StagePlan:
    """Per stage: None to run it, else why it is skipped."""

    rally: str | None
    action: str | None
    association: str | None


def plan_stages(stem: str, *, overwrite: bool) -> StagePlan:
    """Without ``overwrite`` a stage whose machine output already exists is
    kept, so a run fills the gaps; association runs whenever it can, since
    it only re-decides automatic picks and keeps every human verdict."""
    return StagePlan(
        rally=(
            None if overwrite or not rally_spot_pre_annotation_path(stem).exists()
            else "kept existing rallies"
        ),
        action=(
            None if overwrite or not pre_annotation_path(stem).exists()
            else "kept existing actions"
        ),
        association=association_blocker(stem),
    )


@dataclass
class VideoResult:
    rallies: int | None = None
    events: int | None = None
    association: dict | None = None
    skipped: dict[str, str] = field(default_factory=dict)
    #: ``(path, r2_category)`` pairs the caller mirrors from the event loop.
    written: list[tuple[Path, str]] = field(default_factory=list)


def run_video(
    *,
    video: Path,
    source: str,
    checkpoint: Path,
    rally: RallyOptions,
    action_min_score: float,
    spot: SpotOptions,
    overwrite: bool,
    on_progress: BatchProgress,
) -> VideoResult:
    """Every stage for one video, in order. ``source`` is what yp-spot decodes
    — the local file or a presigned R2 URL — while ``video`` is the cut's
    canonical path that names every output."""
    total = _UNITS_PER_STAGE * len(STAGES)

    def stage_progress(first: int, count: int = 1) -> StageProgress:
        label = "+".join(STAGES[first:first + count])

        def report(fraction: float, message: str) -> None:
            span = _UNITS_PER_STAGE * count
            done = first * _UNITS_PER_STAGE + int(max(0.0, min(1.0, fraction)) * span)
            on_progress(done, total, f"{label}: {message}")
        return report

    result = VideoResult()
    plan = plan_stages(video.stem, overwrite=overwrite)

    on_progress(0, total, "rally: starting")
    spot_plan = (("rally", plan.rally), ("action", plan.action))
    for task, skip in spot_plan:
        if skip is not None:
            result.skipped[task] = skip
    spot_tasks = tuple(task for task, skip in spot_plan if skip is None)
    if spot_tasks:
        out = run_spot_stages(
            video=video, source=source, checkpoint=checkpoint, tasks=spot_tasks,
            rally=rally, action_min_score=action_min_score, spot=spot,
            on_progress=stage_progress(STAGES.index(spot_tasks[0]), len(spot_tasks)),
        )
        if "rallies" in out:
            result.rallies = out["rallies"]
        if "events" in out:
            result.events = out["events"]
        result.written += out["written"]

    stage_progress(2)(0.0, "starting")
    if plan.association is not None:
        result.skipped["association"] = plan.association
    else:
        out = run_association_stage(
            video=video, checkpoint=checkpoint, on_progress=stage_progress(2),
        )
        result.association = out["counts"]

    on_progress(total, total, "done")
    return result


def summarize(result: VideoResult) -> str:
    """The one line the job card shows per video."""
    parts = []
    parts.append(
        f"{result.rallies} rallies" if result.rallies is not None
        else f"rally: {result.skipped.get('rally', 'skipped')}"
    )
    parts.append(
        f"{result.events} actions" if result.events is not None
        else f"action: {result.skipped.get('action', 'skipped')}"
    )
    counts = result.association
    if counts is not None:
        parts.append(
            f"association: {counts.get('changed', 0)} moved · "
            f"{counts.get('unchanged', 0)} unchanged · {counts.get('labeled', 0)} labeled kept"
        )
    else:
        parts.append(f"association: {result.skipped.get('association', 'skipped')}")
    return " · ".join(parts)
