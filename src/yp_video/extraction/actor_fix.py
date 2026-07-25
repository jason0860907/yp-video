"""Application service for the actor-fix use case.

Re-pointing one event at a different person touches four stores at once: the
durable actor label, the identity assignment it invalidates, the derived
extraction record, and every embedding sidecar. This module is their sole
coordinator — it owns the ordering, the locks and the rollback. Transport
validation stays in the router; each store keeps its own writes.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

from yp_video.actor import labels as actor_labels
from yp_video.actor.labels import ActorLabel, ActorVerdict
from yp_video.tracklets.geometry import TrackRef
from yp_video.extraction import pipeline, store as extraction_store
from yp_video.reid import identity, store
from yp_video.reid.embedder import base_embedder_name

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PickActor:
    mode: Literal["pick"]
    event_id: str
    box: tuple[float, float, float, float]
    #: The tracklet the user clicked, when they clicked one. Then the box is
    #: only the anchor that can re-derive it (see actor/labels.py).
    track: TrackRef | None = None
    frame: int | None = None
    snap: bool = True

    @property
    def label(self) -> ActorLabel:
        return ActorLabel(
            ActorVerdict.MANUAL,
            track=self.track,
            box=self.box,
            frame=self.frame,
            snap=self.snap,
        )


@dataclass(frozen=True)
class MarkOccluded:
    mode: Literal["occluded"]
    event_id: str

    @property
    def label(self) -> ActorLabel:
        return ActorLabel(ActorVerdict.OCCLUDED)


@dataclass(frozen=True)
class RevertActor:
    mode: Literal["auto"]
    event_id: str

    @property
    def label(self) -> None:
        """Reverting states nothing about the actor — it withdraws the claim."""
        return None


#: Each command carries the label it stands for, so applying one is the same
#: three writes regardless of which it is.
ActorFixCommand = PickActor | MarkOccluded | RevertActor


@dataclass(frozen=True)
class ActorFixResult:
    record: dict
    refreshing_models: tuple[str, ...]
    actor_revision: int


@dataclass(frozen=True)
class _FileSnapshot:
    path: Path
    data: bytes | None


_transaction_lock = threading.Lock()


def _validate(command: ActorFixCommand) -> None:
    if not command.event_id.strip():
        raise ValueError("event_id must not be empty")
    if isinstance(command, PickActor):
        x0, y0, x1, y1 = command.box
        if not all(math.isfinite(v) for v in command.box):
            raise ValueError("Actor box coordinates must be finite")
        if x1 <= x0 or y1 <= y0:
            raise ValueError("Actor box must have positive width and height")
        if command.frame is not None and command.frame < 0:
            raise ValueError("Actor frame must be non-negative")


def _snapshot(paths: list[Path]) -> list[_FileSnapshot]:
    return [_FileSnapshot(path, path.read_bytes() if path.exists() else None) for path in paths]


def _restore(snapshot: _FileSnapshot) -> None:
    path = snapshot.path
    if snapshot.data is None:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f"{path.name}.", suffix=".rollback", delete=False) as f:
        try:
            f.write(snapshot.data)
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            os.unlink(f.name)
            raise
    os.replace(f.name, path)


def _directory_files(path: Path) -> set[Path]:
    return set(path.iterdir()) if path.exists() else set()


def apply(
    video_path: Path, command: ActorFixCommand, *, active_model: str
) -> ActorFixResult:
    """Apply one actor fix and synchronously refresh the active weight family."""
    _validate(command)
    stem = video_path.stem
    record_file = extraction_store.records_path(stem)
    if not record_file.exists():
        raise FileNotFoundError(f"No extraction records for {stem}")

    with (
        _transaction_lock,
        store.embedding_write_transaction(),
        actor_labels.write_transaction(),
        identity.players_write_transaction(),
    ):
        embedded_models = store.embedded_models(stem)
        if active_model not in embedded_models:
            raise FileNotFoundError(
                f"No {active_model} embeddings for {stem} — backfill the model first"
            )
        active_family = base_embedder_name(active_model)
        synchronous_models = [
            model
            for model in embedded_models
            if base_embedder_name(model) == active_family
        ]
        deferred_models = tuple(
            model
            for model in embedded_models
            if model not in synchronous_models
        )
        model_files = [
            store.embedding_path(stem, model) for model in embedded_models
        ]
        snapshots = _snapshot(
            [
                record_file,
                actor_labels.actors_path(stem),
                store.players_path(stem),
                store.embedding_refresh_path(stem),
                *model_files,
            ]
        )
        crop_dirs = (
            extraction_store.crop_dir(stem),
            extraction_store.masked_crop_dir(stem),
        )
        existing_crops = {
            directory: _directory_files(directory) for directory in crop_dirs
        }
        try:
            # Derived record first: it is the only step that can fail on the
            # video itself, and a failed fix must not leave a label behind.
            record = pipeline.apply_actor_fix(
                video_path,
                command.event_id,
                command.label,
                models=synchronous_models,
            )
            actor_labels.save(stem, command.event_id, command.label)
            # The crop now shows a different person (or nobody), so whatever
            # name was attached to the old one is no longer evidence.
            identity.drop_assignment(stem, command.event_id)
            return ActorFixResult(
                record=record,
                refreshing_models=deferred_models,
                actor_revision=int(record["actor_revision"]),
            )
        except BaseException:
            for item in snapshots:
                _restore(item)
            # Crop filenames are cache-busted per pick. Rollback removes only
            # files that did not exist before this transaction.
            for directory, before in existing_crops.items():
                for created in _directory_files(directory) - before:
                    if created.is_file():
                        created.unlink(missing_ok=True)
            raise


def refresh_deferred(
    stem: str,
    event_id: str,
    *,
    models: tuple[str, ...],
    expected_revision: int,
) -> None:
    """Best-effort background refresh for matrices not visible during the fix."""
    if not models:
        return
    try:
        pipeline.refresh_actor_embeddings(
            stem,
            event_id,
            models=list(models),
            expected_revision=expected_revision,
        )
    except Exception:
        # Their event ids remain in the refresh sidecar, so stale reads are
        # rejected and a later full backfill can safely recover.
        log.exception(
            "Deferred actor embedding refresh failed for %s/%s (%s)",
            stem,
            event_id,
            ", ".join(models),
        )
