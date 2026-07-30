"""The lock order an actor fix depends on, made a thing that fails loudly.

Six locks across four modules coordinate one click, and no single file shows
more than two of them: the fix endpoint holds the embedding-write lock while
``pipeline`` reaches for a per-model lock and then the record file. The order
was correct by discipline alone, which lasts until somebody adds a lock — and
the symptom of getting it wrong is a deadlock under a second concurrent fix,
which no ordinary unit test would reproduce.

So the locks are wrapped in rank-checking proxies and the real code paths are
driven through them. Any inversion raises here instead of hanging in
production.
"""

from __future__ import annotations

import tempfile
import threading
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np

from yp_video.actor import labels as actor_labels
from yp_video.extraction import actor_fix, pipeline
from yp_video.reid import store as reid_store

#: The documented hierarchy (see extraction/actor_fix.py). Lower is outer.
RANKS = {
    "transaction": 1,
    "embedding-write": 2,
    "actor-labels": 3,
    "players": 4,
    "model-matrix": 5,
    "record-file": 6,
}


class LockOrderViolation(AssertionError):
    pass


class _RankedLock:
    """A lock that refuses to be acquired inside a lock ranked below it."""

    _held = threading.local()
    #: Every lock taken since the last reset — so a test can prove the code
    #: reached the locks at all, rather than passing because it took none.
    seen: set[str] = set()

    def __init__(self, name: str, inner):
        self._name = name
        self._rank = RANKS[name]
        self._inner = inner

    @classmethod
    def stack(cls) -> list[str]:
        if not hasattr(cls._held, "names"):
            cls._held.names = []
        return cls._held.names

    def __enter__(self):
        held = self.stack()
        # Re-entering a lock already held is fine — several of these are
        # RLocks precisely so a nested caller can take them again.
        outer = [name for name in held if name != self._name]
        if outer and RANKS[outer[-1]] > self._rank:
            raise LockOrderViolation(
                f"{self._name} (rank {self._rank}) acquired inside "
                f"{outer[-1]} (rank {RANKS[outer[-1]]})"
            )
        self._inner.__enter__()
        held.append(self._name)
        _RankedLock.seen.add(self._name)
        return self

    def __exit__(self, *exc):
        self.stack().pop()
        return self._inner.__exit__(*exc)


@contextmanager
def instrumented(*extra):
    """Every lock in the hierarchy replaced by its ranked proxy."""
    patches = (
        patch.object(
            actor_fix, "_transaction_lock", _RankedLock("transaction", threading.Lock())
        ),
        patch.object(
            reid_store,
            "_embedding_write_lock",
            _RankedLock("embedding-write", threading.RLock()),
        ),
        patch.object(
            actor_labels._store, "_lock", _RankedLock("actor-labels", threading.RLock())
        ),
        patch.object(
            reid_store._players_store, "_lock", _RankedLock("players", threading.RLock())
        ),
        patch.object(
            pipeline, "_actor_fix_lock", _RankedLock("record-file", threading.RLock())
        ),
        patch.object(
            pipeline,
            "_embedding_lock",
            lambda _stem, _model: _RankedLock("model-matrix", threading.Lock()),
        ),
    )
    _RankedLock.seen = set()
    with ExitStack() as stack:
        for patcher in (*patches, *extra):
            stack.enter_context(patcher)
        yield _RankedLock.seen


class LockOrderTests(unittest.TestCase):
    def tearDown(self) -> None:
        self.assertEqual(_RankedLock.stack(), [], "a lock was left held")

    def test_the_proxy_would_actually_catch_an_inversion(self) -> None:
        """A guard that cannot fail is decoration."""
        outer = _RankedLock("record-file", threading.RLock())
        inner = _RankedLock("embedding-write", threading.RLock())
        with self.assertRaises(LockOrderViolation), outer:
            with inner:
                pass

    def test_reentering_the_same_lock_is_allowed(self) -> None:
        """Several of these are RLocks so a nested caller can take them again."""
        lock = _RankedLock("embedding-write", threading.RLock())
        with lock, lock:
            pass

    def test_applying_a_fix_respects_the_hierarchy(self) -> None:
        models = ["clip-reident", "clip-reident-masked"]
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            (root / "match_reid.jsonl").write_bytes(b"reid")
            for model in models:
                (root / f"match.{model}.npy").write_bytes(b"embedding")

            with instrumented(
                patch.object(
                    actor_fix.extraction_store,
                    "records_path",
                    return_value=root / "match_reid.jsonl",
                ),
                patch.object(
                    actor_fix.actor_labels,
                    "actors_path",
                    return_value=root / "actors.json",
                ),
                patch.object(
                    actor_fix.store,
                    "players_path",
                    return_value=root / "players.json",
                ),
                patch.object(
                    actor_fix.store,
                    "embedding_refresh_path",
                    return_value=root / "refresh.json",
                ),
                patch.object(
                    actor_fix.store, "embedded_models", return_value=models
                ),
                patch.object(
                    actor_fix.store,
                    "embedding_path",
                    side_effect=lambda _s, model: root / f"match.{model}.npy",
                ),
                patch.object(
                    actor_fix.extraction_store,
                    "crop_dir",
                    return_value=root / "crops",
                ),
                patch.object(
                    actor_fix.extraction_store,
                    "masked_crop_dir",
                    return_value=root / "masked",
                ),
                patch.object(
                    actor_fix.pipeline,
                    "apply_actor_fix",
                    return_value={"id": "e1", "actor_revision": 1},
                ),
                patch.object(actor_fix.actor_labels, "save"),
                patch.object(actor_fix.store, "drop_assignment"),
            ) as seen:
                actor_fix.apply(
                    root / "match.mp4",
                    actor_fix.MarkOccluded(mode="occluded", event_id="e1"),
                    active_model="clip-reident-masked",
                )

        # Proof the code reached the locks: a fix that took none would
        # otherwise satisfy an order check trivially.
        self.assertEqual(
            seen, {"transaction", "embedding-write", "actor-labels", "players"}
        )

    def test_patching_one_matrix_row_respects_the_hierarchy(self) -> None:
        """The nesting no single file shows: embedding-write is held while
        this takes the per-model lock and then the record file."""

        class FakeEmbedder:
            def embed_paths(self, paths):
                return np.ones((len(paths), 2), dtype=np.float32)

        record = {"id": "e1", "crop": "e1.jpg"}
        with instrumented(
            patch.object(pipeline, "embedded_models", return_value=["clip-reident"]),
            patch.object(pipeline, "_record_revision_is_current", return_value=True),
            patch.object(
                pipeline,
                "build_embedders",
                return_value={"clip-reident": FakeEmbedder()},
            ),
            patch.object(pipeline, "crop_dir", return_value=Path("/nonexistent")),
            patch.object(
                pipeline,
                "load_embedding_matrix",
                side_effect=lambda _s, _m: np.zeros((1, 2), dtype=np.float32),
            ),
            patch.object(pipeline, "save_embedding_matrix"),
            patch.object(pipeline, "mark_actor_embedding_refreshed"),
        ) as seen:
            updated = pipeline._patch_embedding_row(
                "match",
                record,
                0,
                object(),
                models=["clip-reident"],
                expected_revision=1,
            )

        self.assertEqual(updated, ["clip-reident"])
        self.assertEqual(
            seen, {"embedding-write", "model-matrix", "record-file"}
        )


if __name__ == "__main__":
    unittest.main()
