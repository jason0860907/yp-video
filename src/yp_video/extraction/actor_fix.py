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
from yp_video.core.jsonl import read_jsonl_cached
from yp_video.extraction import pipeline
from yp_video.extraction import store as extraction_store
from yp_video.reid import store

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PickActor:
    mode: Literal["pick"]
    event_id: str
    #: Pixels on the event's own frame — ``apply`` stamps the label with it.
    box: tuple[float, float, float, float]

    def label_on(self, frame: int) -> ActorLabel:
        return ActorLabel(ActorVerdict.MANUAL, frame=frame, box=self.box)


@dataclass(frozen=True)
class MarkOccluded:
    mode: Literal["occluded"]
    event_id: str

    def label_on(self, frame: int) -> ActorLabel:
        return ActorLabel(ActorVerdict.OCCLUDED)


@dataclass(frozen=True)
class RevertActor:
    mode: Literal["auto"]
    event_id: str

    def label_on(self, frame: int) -> None:
        """Reverting states nothing about the actor — it withdraws the claim."""
        return None


#: Each command names the label it stands for on the event's frame, so
#: applying one is the same three writes regardless of which it is.
ActorFixCommand = PickActor | MarkOccluded | RevertActor


@dataclass(frozen=True)
class ActorFixResult:
    record: dict
    label: ActorLabel | None
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


def apply(stem: str, frame_source: str, command: ActorFixCommand) -> ActorFixResult:
    """Apply one actor fix: the label, the derived record and its crop.

    ``frame_source`` is what OpenCV opens to cut the crop — the local mp4, or
    a presigned R2 URL when the cut isn't on this machine (one seek, so
    streaming beats downloading the whole file). Every embedding matrix is
    refreshed afterwards by ``refresh_deferred``: re-embedding inline cold-
    loads the ReID engine in a subprocess, several seconds per click. Until
    the refresh lands, the refresh sidecar marks the row stale so no reader
    takes the old vector as current.
    """
    _validate(command)
    record_file = extraction_store.records_path(stem)
    if not record_file.exists():
        raise FileNotFoundError(f"No extraction records for {stem}")

    with (
        _transaction_lock,
        store.embedding_write_transaction(),
        actor_labels.write_transaction(),
        store.players_write_transaction(),
    ):
        refreshing_models = tuple(store.embedded_models(stem))
        # No matrix is written here, so none is snapshotted; what keeps them
        # honest until the background refresh is the sidecar, snapshotted
        # with the files this transaction does write.
        snapshots = _snapshot(
            [
                record_file,
                actor_labels.actors_path(stem),
                store.players_path(stem),
                store.embedding_refresh_path(stem),
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
            label = command.label_on(_event_frame(stem, command.event_id))
            record = pipeline.apply_actor_fix(
                stem, frame_source, command.event_id, label
            )
            actor_labels.save(stem, command.event_id, label)
            # The crop now shows a different person (or nobody), so whatever
            # name was attached to the old one is no longer evidence.
            store.drop_assignment(stem, command.event_id)
            return ActorFixResult(
                record=record,
                label=label,
                refreshing_models=refreshing_models,
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


def _event_frame(stem: str, event_id: str) -> int:
    """The event's current frame — the action annotation's, joined by id."""
    _meta, records = read_jsonl_cached(extraction_store.records_path(stem))
    for record in extraction_store.with_current_actions(records, stem):
        if record["id"] == event_id:
            return int(record["frame"])
    raise KeyError(f"No current action event {event_id}")


def refresh_deferred(
    stem: str,
    event_id: str,
    *,
    models: tuple[str, ...],
    expected_revision: int,
) -> None:
    """Best-effort background refresh of every matrix a fix made stale."""
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
