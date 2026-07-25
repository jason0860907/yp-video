"""Where ReID data lives on disk — paths, IO and freshness policy.

The lowest layer of the reid package: identity (matching) and the web router
depend on this module, never on each other, for where files live. The two
things it does NOT own are deliberate — event records and crops belong to
extraction (extraction/store.py), tracklets to their own package
(tracklets/store.py) — because both are read by actor as well as reid.

Layout under videos/reid/ — annotations/ is the hand-made part, the rest is
recomputable derived data:
    annotations/<stem>_players.json  player assignments + the done flag
    embeddings/<stem>.<model>.npy    float32 (n_records, dim) embedding
                                     matrix, row i ↔ record i, NaN = none

Embeddings are a pure numeric matrix, so they live as npy sidecars, not JSON:
records stay small enough to serve raw, matrices load in milliseconds, and a
one-row update (actor fix) never rewrites the record file. Which weights a
model name stands for is answered by embedder.weights_id() at runtime.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from yp_video.config import REID_ANNOTATIONS_DIR, REID_DIR
from yp_video.core.cache import StatCache
from yp_video.core.jsonl import atomic_write
from yp_video.extraction.store import records_path

EMBEDDINGS_DIR = REID_DIR / "embeddings"

#: The name this package owns inside the shared annotations directory, where
#: actor labels live under a suffix of their own. Public so a caller can count
#: or list player-labelled videos without re-spelling it.
PLAYERS_SUFFIX = "_players.json"


def embedding_path(stem: str, model: str) -> Path:
    return EMBEDDINGS_DIR / f"{stem}.{model}.npy"


def embedding_refresh_path(stem: str) -> Path:
    """Pending per-event matrix refreshes after actor fixes."""
    return EMBEDDINGS_DIR / f"{stem}_embedding-refresh.json"


_embedding_write_lock = threading.RLock()


@contextmanager
def embedding_write_transaction() -> Iterator[None]:
    """Serialize matrix/state commits without serializing model inference."""
    with _embedding_write_lock:
        yield


def _load_embedding_refreshes(stem: str) -> dict[str, set[str]]:
    path = embedding_refresh_path(stem)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {
        str(model): {str(event_id) for event_id in event_ids}
        for model, event_ids in data.items()
    }


def _save_embedding_refreshes(
    stem: str, refreshes: dict[str, set[str]]
) -> None:
    path = embedding_refresh_path(stem)
    data = {
        model: sorted(event_ids)
        for model, event_ids in sorted(refreshes.items())
        if event_ids
    }
    if not data:
        path.unlink(missing_ok=True)
        return
    with atomic_write(path) as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


def mark_actor_embedding_stale(
    stem: str, models: list[str], event_id: str
) -> None:
    """Record the exact event each matrix still needs to incorporate."""
    with _embedding_write_lock:
        refreshes = _load_embedding_refreshes(stem)
        for model in models:
            refreshes.setdefault(model, set()).add(event_id)
        _save_embedding_refreshes(stem, refreshes)


def mark_actor_embedding_refreshed(
    stem: str, model: str, event_id: str
) -> None:
    with _embedding_write_lock:
        refreshes = _load_embedding_refreshes(stem)
        pending = refreshes.get(model)
        if pending is not None:
            pending.discard(event_id)
            if not pending:
                refreshes.pop(model)
        _save_embedding_refreshes(stem, refreshes)


def clear_embedding_refreshes(stem: str, model: str) -> None:
    """A full model backfill supersedes every pending one-row refresh."""
    with _embedding_write_lock:
        refreshes = _load_embedding_refreshes(stem)
        refreshes.pop(model, None)
        _save_embedding_refreshes(stem, refreshes)


# One dir scan serves every embedded_models call (the video list asks per
# cut); any matrix create/delete churns the directory entry via temp+rename,
# so the dir's own stat is a correct invalidation key.
_models_cache: StatCache = StatCache()


def embedded_models(stem: str) -> list[str]:
    """Models that have an embedding matrix for this video."""
    if not EMBEDDINGS_DIR.exists():
        return []
    return _models_map().get(stem, [])


def stale_embedding_models(stem: str) -> list[str]:
    """Matrices with pending actor events or older than their source records.

    The event-level sidecar is the authoritative state. The mtime fallback
    recognizes stale matrices created before that sidecar was introduced and
    also keeps a crashed partial write safely unavailable.
    """
    source = records_path(stem)
    if not source.exists():
        return []
    source_mtime = source.stat().st_mtime_ns
    with _embedding_write_lock:
        pending = _load_embedding_refreshes(stem)
    return [
        model
        for model in embedded_models(stem)
        if pending.get(model)
        or embedding_path(stem, model).stat().st_mtime_ns < source_mtime
    ]


def embedding_is_fresh(stem: str, model: str) -> bool:
    path = embedding_path(stem, model)
    source = records_path(stem)
    return (
        path.exists()
        and source.exists()
        and model not in stale_embedding_models(stem)
    )


def _models_map() -> dict[str, list[str]]:
    def scan() -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for p in EMBEDDINGS_DIR.glob("*.npy"):
            stem, _, model = p.name[: -len(".npy")].rpartition(".")
            if stem:
                out.setdefault(stem, []).append(model)
        return {stem: sorted(models) for stem, models in out.items()}

    return _models_cache.get("map", [EMBEDDINGS_DIR], scan)


def require_embedding_path(stem: str, model: str) -> Path:
    """The matrix path, or an actionable FileNotFoundError when it's absent."""
    path = embedding_path(stem, model)
    if not path.exists():
        raise FileNotFoundError(
            f"No {model} embeddings for {stem} — run extraction or backfill embeddings"
        )
    return path


def load_embedding_matrix(stem: str, model: str) -> np.ndarray:
    """The (n_records, dim) matrix for one model; NaN rows = no embedding."""
    return np.load(require_embedding_path(stem, model))


def save_embedding_matrix(stem: str, model: str, matrix: np.ndarray) -> None:
    """Atomic replace, mirroring jsonl.atomic_write: readers see old or new."""
    path = embedding_path(stem, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _embedding_write_lock:
        with NamedTemporaryFile(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp", delete=False) as f:
            try:
                np.save(f, matrix.astype(np.float32, copy=False))
            except BaseException:
                os.unlink(f.name)
                raise
        os.replace(f.name, path)


def players_path(stem: str) -> Path:
    return REID_ANNOTATIONS_DIR / f"{stem}{PLAYERS_SUFFIX}"

