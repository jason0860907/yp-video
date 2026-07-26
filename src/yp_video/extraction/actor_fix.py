"""Application service for the actor-fix use case.

Re-pointing one event at a different person touches four stores at once: the
durable actor label, the identity assignment it invalidates, the derived
extraction record, and every embedding sidecar. This module is their sole
coordinator — it owns the ordering, the locks and the rollback. Transport
validation stays in the router; each store keeps its own writes.

LOCK ORDER. One fix acquires six locks across four modules, and nested in
this order every time:

    1. actor_fix._transaction_lock        one fix at a time
    2. reid.store._embedding_write_lock   any matrix or sidecar commit
    3. actor.labels._lock                 the actor verdict file
    4. reid.store._players_lock           the player-name file
    5. pipeline._embedding_locks[stem, m] one model's matrix
    6. pipeline._actor_fix_lock           the extraction record jsonl

A path may skip levels — the background refresh enters at 2 — but must never
invert them. Two of these are taken in modules that know nothing of each
other (the fix endpoint holds 2 while pipeline takes 5 and 6), so the order
is not visible from any single file; ``tests/test_actor_fix_locking.py``
is what makes a new lock, or a new caller, fail loudly instead of deadlocking
under a second concurrent click.
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
from yp_video.reid import store
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
    video_path: Path, command: ActorFixCommand, *, active_model: str | None
) -> ActorFixResult:
    """Apply one actor fix and synchronously refresh the active weight family.

    ``active_model`` is None when the video has not been embedded yet, which
    is the normal case: actor review comes BEFORE embedding (see
    extraction/pipeline.py). Then there is no matrix to keep in step and the
    fix is three writes and a crop. A fix that arrives later — spotted on the
    ReID board, after the vectors exist — still refreshes them.
    """
    _validate(command)
    stem = video_path.stem
    record_file = extraction_store.records_path(stem)
    if not record_file.exists():
        raise FileNotFoundError(f"No extraction records for {stem}")

    with (
        _transaction_lock,
        store.embedding_write_transaction(),
        actor_labels.write_transaction(),
        store.players_write_transaction(),
    ):
        embedded_models = store.embedded_models(stem)
        if active_model is not None and active_model not in embedded_models:
            raise FileNotFoundError(
                f"No {active_model} embeddings for {stem} — backfill the model first"
            )
        active_family = (
            base_embedder_name(active_model) if active_model is not None else None
        )
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
        # Only the matrices this transaction WRITES are snapshotted. A
        # deferred model's matrix is untouched until the background refresh,
        # and each matrix is ~1 MB — snapshotting every registered model on
        # every click was several MB of pure read per fix. What protects the
        # deferred ones is the refresh sidecar, which is snapshotted here.
        snapshots = _snapshot(
            [
                record_file,
                actor_labels.actors_path(stem),
                store.players_path(stem),
                store.embedding_refresh_path(stem),
                *(store.embedding_path(stem, m) for m in synchronous_models),
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
            store.drop_assignment(stem, command.event_id)
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
