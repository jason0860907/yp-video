"""Write the run-local label snapshot SPOT trains on, actor sidecar included.

The snapshot joins two corpora: the action labels (when/where, normalized
against the extracted frame cache) and the actor-candidate supervision this
package builds from tracking + human verdicts. That join is actor-side
knowledge — ``action`` must stay importable without this package — so the
exporter lives here, next to ``candidates``, and the training routers call
down into it.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

from yp_video.action.frames import inspect_action_frame_cache
from yp_video.action.training import rally_match_span
from yp_video.actor import candidates
from yp_video.config import ACTION_ANNOTATIONS_DIR, cut_kind_of
from yp_video.contracts.action import ACTOR_FILE_SUFFIX, TASKS
from yp_video.core.jsonl import read_jsonl, write_jsonl

log = logging.getLogger(__name__)


def prepare_action_training_labels(
    *,
    items: list[tuple[Path, Path]],
    frame_dir: Path,
    save_dir: Path,
    tasks: Sequence[str],
    camera_view: str = "all",
    require_actor_targets: bool = False,
) -> dict:
    """Write run-local label copies whose frame counts match the SPOT cache.

    ``items`` is the job's label snapshot (see ``action.training.label_items``)
    — the same list the frame-cache phase ran on, so every video here has a
    cache. When ``camera_view`` restricts to a single view, only matching
    videos are written, so the saved label snapshot equals what training
    actually used. The actor-candidate sidecar is written only when
    ``tasks`` trains the actor head, so the snapshot equals what the run read.
    """
    action, actor = TASKS["action"], TASKS["actor"]
    label_dir = save_dir / "labels" / action.label_subdir
    label_dir.mkdir(parents=True, exist_ok=True)
    for stale in label_dir.glob(action.label_glob):
        stale.unlink()
    actor_dir = save_dir / "labels" / actor.label_subdir
    if actor_dir.exists():
        for stale in actor_dir.glob(actor.label_glob):
            stale.unlink()
    write_actors = actor.name in tasks

    videos = 0
    events = 0
    total_frames = 0
    span_frames = 0
    actor_targets = {"track": 0, "occluded": 0, "untracked": 0, "unresolved_box": 0}
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
        # Explicit field pick, not {**meta}: the snapshot must carry only
        # what yp-spot's dataset contract reads (video/fps/num_frames plus
        # our own additions) — never a stale copy of the rally spans.
        training_meta = {
            "video": stem,
            "fps": meta.get("fps"),
            "source": meta.get("source"),
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

        match_span = rally_match_span(
            stem, fps=float(meta.get("fps") or 30.0), num_frames=cache_frames
        )
        if match_span is not None:
            training_meta["sample_spans"] = [list(match_span)]
            span_frames += match_span[1] - match_span[0]
        else:
            span_frames += cache_frames

        # Who acted, where the video can say so. Written to its OWN file: only
        # a handful of videos carry actor work, and the action labels are read
        # by every spotting run over every video.
        actor_rows, tally = (
            candidates.build(stem, records) if write_actors else ([], dict.fromkeys(actor_targets, 0))
        )
        if require_actor_targets and not actor_rows:
            raise RuntimeError(
                f"{stem} was selected for joint Association + Action training "
                "but produced no usable actor targets"
            )
        if actor_rows:
            actor_dir.mkdir(parents=True, exist_ok=True)
            write_jsonl(
                actor_dir / f"{stem}{ACTOR_FILE_SUFFIX}",
                {"video": stem, "num_events": len(actor_rows)},
                actor_rows,
            )
        for key in actor_targets:
            actor_targets[key] += tally[key]

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
        # How much actor supervision this run actually had. Reported so a run
        # that silently exported none is visible on the job card rather than
        # discovered when the actor head refuses to learn.
        "actor_dir": str(actor_dir),
        "actor_targets": actor_targets,
        "adjusted": adjusted,
    }
