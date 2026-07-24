"""Application service for the actor-fix use case.

The durable human verdict, player assignment, derived ReID record and every
embedding sidecar form one logical change. This module is their sole
coordinator; transport validation stays in the router and store-specific work
stays in identity/pipeline.
"""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

from yp_video.reid import identity, pipeline, store


@dataclass(frozen=True)
class PickActor:
    mode: Literal["pick"]
    event_id: str
    box: tuple[float, float, float, float]
    frame: int | None = None
    snap: bool = True


@dataclass(frozen=True)
class MarkOccluded:
    mode: Literal["occluded"]
    event_id: str


@dataclass(frozen=True)
class RevertActor:
    mode: Literal["auto"]
    event_id: str


ActorFixCommand = PickActor | MarkOccluded | RevertActor


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


def apply(video_path: Path, command: ActorFixCommand) -> dict:
    """Apply one actor fix or restore every authoritative file on failure."""
    _validate(command)
    stem = video_path.stem
    reid_file = store.reid_path(stem)
    if not reid_file.exists():
        raise FileNotFoundError(f"No ReID results for {stem}")

    with _transaction_lock, identity.players_write_transaction():
        model_files = [
            store.embedding_path(stem, model)
            for model in store.embedded_models(stem)
        ]
        snapshots = _snapshot([reid_file, store.players_path(stem), *model_files])
        crop_dirs = (store.crop_dir(stem), store.masked_crop_dir(stem))
        existing_crops = {
            directory: _directory_files(directory) for directory in crop_dirs
        }
        try:
            if isinstance(command, PickActor):
                record = pipeline.apply_actor_fix(
                    video_path,
                    command.event_id,
                    list(command.box),
                    frame=command.frame,
                    snap=command.snap,
                )
                identity.apply_actor_fix_annotation(
                    stem,
                    command.event_id,
                    mode="pick",
                    box=list(command.box),
                    frame=command.frame,
                    snap=command.snap,
                )
            elif isinstance(command, MarkOccluded):
                record = pipeline.apply_actor_fix(
                    video_path, command.event_id, None, none=True
                )
                identity.apply_actor_fix_annotation(
                    stem, command.event_id, mode="occluded"
                )
            else:
                record = pipeline.apply_actor_fix(
                    video_path, command.event_id, None
                )
                identity.apply_actor_fix_annotation(
                    stem, command.event_id, mode="auto"
                )
            return record
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
