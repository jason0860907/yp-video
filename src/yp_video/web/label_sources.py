"""How each recipe family turns its annotation corpus into a run's label snapshot.

A SPOT run reads one label directory (``--label_dir`` / ``--train_labels``)
plus optional task sidecars (``--actor_dir``). Rally recipes draw from the
rally annotations at a reduced extraction fps; action recipes from the action
annotations at native fps, plus the actor-candidate sidecar when the actor
head is trained. Both end in the same ``PreparedLabels`` so the launcher
(``spot_training``) does not branch on recipe.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from yp_video.action import rally as rally_spot
from yp_video.action import training
from yp_video.action.frames import ensure_action_frame_caches
from yp_video.actor import labels as association_labels
from yp_video.actor.training_labels import prepare_action_training_labels
from yp_video.config import ACTION_FRAMES_DIR
from yp_video.contracts.action import (
    LABEL_FILE_SUFFIX,
    RALLY_LABEL_FILE_SUFFIX,
    TASKS,
    Recipe,
    label_subdirs,
    spotting_task,
)
from yp_video.web.r2_client import resolve_cut
from yp_video.web.train_requests import FusionTrainRequest

Progress = Callable[[int, int, str], None]


@dataclass
class PreparedLabels:
    #: The directory the spotting task reads; holdout splits are built next to it.
    label_dir: Path
    #: Every ``labels/<subdir>`` written — what the package snapshots.
    label_subdirs: tuple[str, ...]
    frame_dir: Path
    #: yp-spot's positional dataset name (class list / default paths).
    dataset: str
    #: Extra ``yp_spot.train`` arguments the tasks need (``--actor_dir``).
    extra_args: list[str] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    #: Videos whose labels are SPOT predictions, not human work.
    prediction_stems: set[str] = field(default_factory=set)
    #: Every annotated stem before camera-view filtering — what a manual
    #: validation list may legitimately name.
    all_stems: set[str] = field(default_factory=set)


class LabelSource(Protocol):
    def prepare(
        self, req: FusionTrainRequest, recipe: Recipe, *, save_dir: Path, progress: Progress
    ) -> PreparedLabels: ...


class RallySource:
    """Rally spans (and their winners) in the shared native frame cache."""

    def prepare(
        self, req: FusionTrainRequest, recipe: Recipe, *, save_dir: Path, progress: Progress
    ) -> PreparedLabels:
        items, missing = rally_spot.select_training_items(resolve_cut, req.video_limit)
        if not items:
            raise RuntimeError("No rally annotations with a cut video (local or R2)")
        ensure_action_frame_caches(
            [(video_path, None) for _ann, video_path in items],
            cache_root=ACTION_FRAMES_DIR,
            progress=progress,
        )
        label_dir = save_dir / "labels" / TASKS["rally"].label_subdir
        summary = rally_spot.write_training_labels(items, label_dir=label_dir)
        return PreparedLabels(
            label_dir=label_dir,
            label_subdirs=label_subdirs(recipe.tasks),
            frame_dir=ACTION_FRAMES_DIR,
            dataset="yp_rally",
            summary={**summary, "missing_videos": missing},
            all_stems={video_path.stem for _ann, video_path in items},
        )


class ActionSource:
    """Action events at native fps, plus the actor-candidate sidecar."""

    def prepare(
        self, req: FusionTrainRequest, recipe: Recipe, *, save_dir: Path, progress: Progress
    ) -> PreparedLabels:
        items = training.label_items(resolve_cut, include_predictions=req.include_predictions)
        all_stems = {label.stem.removesuffix("_actions") for label, _video in items}
        joint_only = "actor" in recipe.tasks and req.dataset_scope == "joint_only"
        if joint_only:
            reviewed = set(association_labels.labeled_stems())
            items = [item for item in items if item[0].stem.removesuffix("_actions") in reviewed]
            if not items:
                raise RuntimeError(
                    "Joint-only scope found no videos carrying both Action and "
                    "Association labels"
                )
        if not items:
            raise RuntimeError("No action labels selected for training")

        # Action JSONL metadata can inherit an over-reported MP4 frame count,
        # so expected_frames is None — the labels are normalized against the
        # extracted cache in the next step.
        ensure_action_frame_caches(
            [(video_path, None) for _label, video_path in items],
            cache_root=ACTION_FRAMES_DIR,
            progress=progress,
        )
        summary = prepare_action_training_labels(
            items=items,
            frame_dir=ACTION_FRAMES_DIR,
            save_dir=save_dir,
            tasks=recipe.tasks,
            camera_view=req.camera_view,
            require_actor_targets=joint_only,
        )
        extra_args: list[str] = []
        if "actor" in recipe.tasks:
            extra_args = ["--actor_dir", summary["actor_dir"]]
        return PreparedLabels(
            label_dir=Path(summary["label_dir"]),
            label_subdirs=label_subdirs(recipe.tasks),
            frame_dir=ACTION_FRAMES_DIR,
            dataset="yp_actions",
            extra_args=extra_args,
            summary=summary,
            prediction_stems=training.prediction_label_stems(items),
            all_stems=all_stems,
        )


def source_for(recipe: Recipe) -> LabelSource:
    return RallySource() if spotting_task(recipe.tasks) == "rally" else ActionSource()


def check_task_supervision(recipe: Recipe, prepared: PreparedLabels) -> None:
    """Refuse a head with nothing to learn from — before the GPU is taken.

    A run that quietly trains an unsupervised head costs hours and exports a
    package claiming a task it never learned. yp-spot repeats the check on
    the loaded labels; this one names the corpus fix instead.
    """
    summary = prepared.summary
    present = {
        "winner": bool(summary.get("rallies_with_winner")),
        "actor": bool((summary.get("actor_targets") or {}).get("track")),
    }
    hints = {
        "winner": "annotate the winning side in the rally editor first",
        "actor": "review actors in Association Label first",
    }
    for task in recipe.tasks:
        if task in present and not present[task]:
            fields = "/".join(TASKS[task].event_fields) or "actor targets"
            raise RuntimeError(
                f"Recipe {recipe.id} trains the {task} head but the label "
                f"snapshot carries no {fields} — {hints[task]}"
            )


def label_stem(entry: str) -> str:
    """A validation-list entry (stem, ``<stem>.mp4``, or a label filename) → stem."""
    name = Path(entry.strip()).name
    for suffix in (LABEL_FILE_SUFFIX, RALLY_LABEL_FILE_SUFFIX, "_annotations.jsonl", ".mp4"):
        name = name.removesuffix(suffix)
    return name
