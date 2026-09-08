"""One fusion checkpoint, every answer it gives for a video.

Stages per video, in pipeline order: rally (the spans, and who won each)
→ action (events, kept inside those spans) → tracking (who is on court over
each rally) → detection (everyone on each event frame) → association (who
acted, chosen among the tracklets). Rally and action share one decode pass
of the video (``run_spot_stages``). Tracking and detection are not the
fusion model's — the person detector is — but the actor head answers by
naming a tracklet and writes its pick into the detection records, so
without them the checkpoint's third head has nothing to choose from and
nowhere to put its answer. Running them here, in this order, is also what
makes detection nearly free: the dense tracking pass keeps its raw
detections for the event frames it just learned about.

Each stage writes the same machine store the single-stage page writes, so
the label editors, the work lists and the pipeline chips see no difference
between the two routes.

Router-free and synchronous: the job runs one video's stages in an executor
thread (``job_helpers.spawn_batch_video_job``). The cut's bytes are fetched
into the layout for the duration of the video (``r2_client.materialized_cut``).
Mirroring to R2 needs the event loop, so a stage reports what it wrote and
the router mirrors it.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from yp_video.action import prelabel
from yp_video.action.spot_pass import RallyOptions, SpotOptions, run_spot_pass
from yp_video.actor import policy as actor_policy
from yp_video.actor import spot_associate
from yp_video.config import RALLY_SPOT_PRE_ANNOTATIONS_DIR, SPOT_CHECKPOINTS_DIR
from yp_video.core.jsonl import write_jsonl
from yp_video.core.rallies import annotation_name, load_rallies, number_rallies
from yp_video.extraction import reassociate
from yp_video.extraction.pipeline import detect_video, detections_current, load_events
from yp_video.extraction.store import records_path
from yp_video.tracklets import tracking
from yp_video.tracklets.store import tracks_current, tracks_path
from yp_video.web.action_annotations import (
    pre_annotation_path,
    save_spot_pre_annotation,
)
from yp_video.web.r2_client import materialized_cut

#: The heads one checkpoint must carry to answer every stage.
REQUIRED_TASKS = ("rally", "action", "actor")

#: Action inference scans each rally span with this much slack on both
#: sides — the same cascade Action Predict and the selfhost worker run.
RALLY_PAD_S = 2.0

STAGES = ("rally", "action", "tracking", "detection", "association")
_UNITS_PER_STAGE = 100

#: ``(done, total, message)`` — the batch job's per-item progress.
BatchProgress = Callable[[int, int, str], None]
#: ``(fraction, message)`` within one stage.
StageProgress = Callable[[float, str], None]


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
    duration_s: float,
    segments: list[dict],
    checkpoint: Path,
    options: RallyOptions,
) -> tuple[Path, int]:
    """Number the merged rally segments and write them as the video's SPOT
    rally pre-annotation. Returns the file and the number of rallies."""
    segments, max_rally_id = number_rallies(segments)
    path = rally_spot_pre_annotation_path(video.stem)
    write_jsonl(
        path,
        {
            "video": str(video),
            "duration": duration_s,
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
    """The rally and/or action head, in ONE decode pass of the video
    (``spot_pass.run_spot_pass``), each written to its machine store.

    When only action runs, the video's existing rallies (any source) bound
    the scan — dead time only contributes false positives; a video without
    any rally source scans in full.
    """
    tasks = tuple(tasks)
    known_rallies = load_rallies(video.stem) if "rally" not in tasks else None
    result = run_spot_pass(
        source,
        checkpoint=checkpoint,
        tasks=tasks,
        rally=rally,
        spot=spot,
        rally_pad_s=RALLY_PAD_S,
        action_min_score=action_min_score,
        rallies=known_rallies or None,
        on_progress=lambda fraction: on_progress(fraction, "inference"),
    )
    out: dict = {"written": []}
    if result.rallies is not None:
        on_progress(1.0, "saving rallies")
        path, count = save_rally_pre_annotation(
            video=video, duration_s=result.duration_s, segments=result.rallies,
            checkpoint=checkpoint, options=rally,
        )
        out["rallies"] = count
        out["written"].append((path, "rally-spot/pre-annotations"))
    if result.actions is not None:
        on_progress(1.0, "saving events")
        data = save_spot_pre_annotation(
            video=video,
            meta={"fps": result.fps, "num_frames": result.num_frames},
            predictions=result.actions,
            checkpoint=checkpoint,
            min_score=action_min_score,
        )
        out["events"] = data["num_events"]
        out["written"].append((pre_annotation_path(video.name), "action/pre-annotations"))
    return out


# ----------------------------------------------------- perception stages


def tracking_skip(stem: str, *, overwrite: bool, rallies: bool) -> str | None:
    """Why tracking does not run for this video, or None to run it."""
    if not rallies:
        return "no rallies"
    if not overwrite and tracks_current(stem):
        return "kept existing tracks"
    return None


def detection_skip(
    stem: str, *, overwrite: bool, events: bool, action_ran: bool
) -> str | None:
    """Why detection does not run, or None. Records list one row per event,
    so a fresh action output always refreshes them; picks already made
    survive that (see pipeline.detect_video)."""
    if not events:
        return "no action events"
    if not overwrite and not action_ran and detections_current(stem):
        return "kept existing detections"
    return None


def association_skip(stem: str, *, events: bool) -> str | None:
    """Why association cannot run, or None. Association always re-decides
    when it can: it only touches automatic picks and keeps every verdict."""
    if not events:
        return "no action events"
    if not tracks_path(stem).exists():
        return "no tracks"
    if not records_path(stem).exists():
        return "no detections"
    return None


def run_tracking_stage(
    *, video: Path, events: Sequence[dict], on_progress: StageProgress
) -> dict:
    """Dense per-rally detection + ByteTrack. The event frames ride along so
    their raw detections persist for the detection stage right after."""
    return tracking.track_video(
        video,
        stride=1,
        event_frames={int(event["frame"]) for event in events},
        on_progress=_fractional(on_progress),
    )


def run_detection_stage(*, video: Path, on_progress: StageProgress) -> dict:
    return detect_video(video, on_progress=_fractional(on_progress))


def run_association_stage(
    *, video: Path, checkpoint: Path, on_progress: StageProgress
) -> dict:
    progress = _fractional(on_progress)
    plan = actor_policy.SpotPlan(checkpoint)
    return reassociate.reassociate_video(
        video, plan.build(video, progress), on_progress=progress
    )


def _fractional(on_progress: StageProgress):
    """A stage's ``(done, total, message)`` as this module's ``(fraction, message)``."""

    def progress(done: int, total: int, message: str) -> None:
        on_progress(done / total if total else 0.0, message)

    return progress


# ------------------------------------------------------------------ per video


def plan_spot_stages(stem: str, *, overwrite: bool) -> dict[str, str | None]:
    """Per SPOT stage: None to run it, else why it is kept. Without
    ``overwrite`` a stage whose machine output exists is kept, so a run fills
    the gaps."""
    return {
        "rally": (
            None if overwrite or not rally_spot_pre_annotation_path(stem).exists()
            else "kept existing rallies"
        ),
        "action": (
            None if overwrite or not pre_annotation_path(stem).exists()
            else "kept existing actions"
        ),
    }


@dataclass
class VideoResult:
    rallies: int | None = None
    events: int | None = None
    tracklets: int | None = None
    detections: int | None = None
    association: dict | None = None
    skipped: dict[str, str] = field(default_factory=dict)
    #: ``(path, r2_category)`` pairs the caller mirrors from the event loop.
    written: list[tuple[Path, str]] = field(default_factory=list)


def run_video(
    *,
    video: Path,
    checkpoint: Path,
    rally: RallyOptions,
    action_min_score: float,
    spot: SpotOptions,
    overwrite: bool,
    on_progress: BatchProgress,
) -> VideoResult:
    """Every stage for one video, in order. ``video`` is the cut's canonical
    path; its bytes are materialized there for the duration."""
    total = _UNITS_PER_STAGE * len(STAGES)

    def stage_progress(first: int, count: int = 1) -> StageProgress:
        label = "+".join(STAGES[first:first + count])

        def report(fraction: float, message: str) -> None:
            span = _UNITS_PER_STAGE * count
            done = first * _UNITS_PER_STAGE + int(max(0.0, min(1.0, fraction)) * span)
            on_progress(done, total, f"{label}: {message}")
        return report

    def fetch_progress(done: int, size: int) -> None:
        # One unit in, not zero: the batch job lets every done=0 report
        # through unthrottled, and boto3 calls this per 8 MB chunk.
        on_progress(1, total, f"fetching video: {done / size:.0%}" if size else "fetching video")

    stem = video.stem
    result = VideoResult()
    on_progress(0, total, "fetching video")
    with materialized_cut(video, on_progress=fetch_progress):
        spot_plan = plan_spot_stages(stem, overwrite=overwrite)
        result.skipped.update({task: why for task, why in spot_plan.items() if why})
        spot_tasks = tuple(task for task, why in spot_plan.items() if why is None)
        if spot_tasks:
            out = run_spot_stages(
                video=video, source=str(video), checkpoint=checkpoint,
                tasks=spot_tasks, rally=rally, action_min_score=action_min_score,
                spot=spot,
                on_progress=stage_progress(STAGES.index(spot_tasks[0]), len(spot_tasks)),
            )
            result.rallies = out.get("rallies")
            result.events = out.get("events")
            result.written += out["written"]

        rallies = load_rallies(stem)
        events = load_events(stem)

        stage_progress(2)(0.0, "starting")
        skip = tracking_skip(stem, overwrite=overwrite, rallies=bool(rallies))
        if skip is not None:
            result.skipped["tracking"] = skip
        else:
            counts = run_tracking_stage(
                video=video, events=events, on_progress=stage_progress(2),
            )
            result.tracklets = counts["tracklets"]

        stage_progress(3)(0.0, "starting")
        skip = detection_skip(
            stem, overwrite=overwrite, events=bool(events),
            action_ran="action" in spot_tasks,
        )
        if skip is not None:
            result.skipped["detection"] = skip
        else:
            counts = run_detection_stage(video=video, on_progress=stage_progress(3))
            result.detections = counts["detections"]

        stage_progress(4)(0.0, "starting")
        skip = association_skip(stem, events=bool(events))
        if skip is not None:
            result.skipped["association"] = skip
        else:
            result.association = run_association_stage(
                video=video, checkpoint=checkpoint, on_progress=stage_progress(4),
            )

    on_progress(total, total, "done")
    return result


def summarize(result: VideoResult) -> str:
    """The one line the job card shows per video."""
    skipped = result.skipped
    parts = [
        f"{result.rallies} rallies" if result.rallies is not None
        else f"rally: {skipped.get('rally', 'skipped')}",
        f"{result.events} actions" if result.events is not None
        else f"action: {skipped.get('action', 'skipped')}",
        f"{result.tracklets} tracklets" if result.tracklets is not None
        else f"tracking: {skipped.get('tracking', 'skipped')}",
        f"{result.detections} people detected" if result.detections is not None
        else f"detection: {skipped.get('detection', 'skipped')}",
    ]
    counts = result.association
    if counts is not None:
        parts.append(
            f"association: {counts.get('changed', 0)} moved · "
            f"{counts.get('unchanged', 0)} unchanged · {counts.get('labeled', 0)} labeled kept"
        )
    else:
        parts.append(f"association: {skipped.get('association', 'skipped')}")
    return " · ".join(parts)
