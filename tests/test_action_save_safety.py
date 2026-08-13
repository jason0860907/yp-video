"""The two rules that keep hand-made action labels safe on disk.

Both exist because a real revert happened: a stale writer (a leftover browser
draft, a second tab, an unload flush racing a later save) blind-overwrote a
newer annotation file, and a prelabel run with overwrite deleted the human
store outright.

1. Saving is optimistic-concurrency-checked: every save carries the store
   revision (the file's mtime_ns) it was based on; a mismatch is a 409, never
   an overwrite.
2. Prelabel writes machine output only. The human store neither gates a
   re-run nor is ever deleted by one.
"""

from __future__ import annotations

import contextlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from yp_video.web import action_annotations
from yp_video.web.routers import action_annotate


@contextlib.contextmanager
def scratch_stores():
    """Route both annotation stores and the cut lookup into a temp dir."""
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        ann_dir = root / "annotations"
        pre_dir = root / "pre-annotations"
        video = root / "cuts" / "match.mp4"
        video.parent.mkdir(parents=True)
        video.touch()
        with (
            patch.object(action_annotate, "resolve_cut", return_value=video),
            patch.object(action_annotate, "find_cut", return_value=video),
            patch.object(action_annotate, "ACTION_ANNOTATIONS_DIR", ann_dir),
            patch.object(action_annotations, "ACTION_ANNOTATIONS_DIR", ann_dir),
            patch.object(action_annotations, "ACTION_PRE_ANNOTATIONS_DIR", pre_dir),
            patch.object(action_annotate, "_load_rallies", return_value=[]),
            patch.object(action_annotate, "sync_to_r2", lambda *a, **k: None),
        ):
            yield ann_dir, pre_dir


def save_request(revision: str | None) -> action_annotate.SaveActionAnnotationsRequest:
    return action_annotate.SaveActionAnnotationsRequest(
        video="match.mp4", fps=30.0, num_frames=100, events=[], revision=revision
    )


class SaveRevisionTests(unittest.IsolatedAsyncioTestCase):
    async def test_stale_revision_is_rejected_not_overwritten(self) -> None:
        with scratch_stores() as (ann_dir, _):
            first = await action_annotate.save_annotations(save_request(None))
            self.assertIsNotNone(first["revision"])
            self.assertTrue((ann_dir / "match_actions.jsonl").exists())

            # A writer still claiming "no file yet" lost the race: 409.
            with self.assertRaises(HTTPException) as ctx:
                await action_annotate.save_annotations(save_request(None))
            self.assertEqual(ctx.exception.status_code, 409)

            # The current revision saves, and the token advances with the file.
            second = await action_annotate.save_annotations(
                save_request(first["revision"])
            )
            self.assertIsNotNone(second["revision"])

            # The superseded token is now stale too.
            with self.assertRaises(HTTPException) as ctx:
                await action_annotate.save_annotations(save_request(first["revision"]))
            self.assertEqual(ctx.exception.status_code, 409)


class PrelabelHumanStoreTests(unittest.TestCase):
    def test_human_store_neither_gates_nor_blocks_prelabel(self) -> None:
        with scratch_stores() as (ann_dir, pre_dir):
            ann_dir.mkdir(parents=True)
            (ann_dir / "match_actions.jsonl").write_text('{"_meta": true}\n')

            # A human-labeled video is prelabel-able without overwrite …
            entries = action_annotate._resolve_prelabel_entries(
                ["match.mp4"], overwrite=False
            )
            self.assertEqual(entries[0][1], pre_dir / "match_actions.jsonl")

            # … only an existing pre file gates the re-run …
            pre_dir.mkdir(parents=True)
            (pre_dir / "match_actions.jsonl").write_text('{"_meta": true}\n')
            with self.assertRaises(HTTPException) as ctx:
                action_annotate._resolve_prelabel_entries(["match.mp4"], overwrite=False)
            self.assertEqual(ctx.exception.status_code, 409)

            # … and overwrite rebuilds it while the human file stays put.
            entries = action_annotate._resolve_prelabel_entries(
                ["match.mp4"], overwrite=True
            )
            self.assertEqual(entries[0][1], pre_dir / "match_actions.jsonl")
            self.assertTrue((ann_dir / "match_actions.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
