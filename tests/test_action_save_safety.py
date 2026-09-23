"""The rules that keep hand-made action labels safe and single-source on disk.

1. Prelabel writes machine output only: the human store neither gates a
   re-run nor is ever deleted by one (an overwrite run once deleted it).
2. The annotation file persists only the human's facts. Rally spans and the
   fields derived from them (rally_id / relative_frame / time) are joined
   from the live rally store on every read — a stored copy is exactly the
   stale data that once left the Association board navigating by old spans.
3. A machine pre-annotation additionally keeps each event's model ``score``;
   the human store never does.
"""

from __future__ import annotations

import contextlib
import json
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
            patch.object(action_annotate, "ACTION_ANNOTATIONS_DIR", ann_dir),
            patch.object(action_annotations, "ACTION_ANNOTATIONS_DIR", ann_dir),
            patch.object(action_annotations, "ACTION_PRE_ANNOTATIONS_DIR", pre_dir),
            patch.object(action_annotate, "_load_rallies", return_value=[]),
            patch.object(action_annotate, "sync_to_r2", lambda *a, **k: None),
        ):
            yield ann_dir, pre_dir


class PersistedShapeTests(unittest.IsolatedAsyncioTestCase):
    async def test_save_persists_facts_only_never_rally_copies(self) -> None:
        with scratch_stores() as (ann_dir, _):
            req = action_annotate.SaveActionAnnotationsRequest(
                video="match.mp4",
                fps=30.0,
                num_frames=100,
                events=[
                    action_annotate.ActionEvent(
                        frame=10,
                        label="spike",
                        xy=(0.5, 0.5),
                        # UI state the frontend sends along — must not persist.
                        rally_id=3,
                        time=0.3333,
                        relative_frame=4,
                    )
                ],
            )
            await action_annotate.save_annotations(req)

            lines = (ann_dir / "match_actions.jsonl").read_text().splitlines()
            meta, event = json.loads(lines[0]), json.loads(lines[1])
            self.assertNotIn("rallies", meta)
            self.assertEqual(
                set(event), {"id", "frame", "label", "xy", "visible"}
            )
            self.assertEqual(event["frame"], 10)
            self.assertEqual(event["label"], "spike")


class PreAnnotationScoreTests(unittest.TestCase):
    def test_pre_annotation_keeps_every_event_with_its_score(self) -> None:
        with scratch_stores() as (_, pre_dir):
            data = action_annotations.save_spot_pre_annotation(
                video=Path("match.mp4"),
                meta={"fps": 30.0, "num_frames": 100},
                predictions=[{"video": "match", "events": [
                    {"frame": 10, "label": "spike", "score": 0.91234, "xy": [0.2, 0.3]},
                    {"frame": 40, "label": "block", "score": 0.02},
                ]}],
                checkpoint=Path("ckpt.pt"),
            )
            lines = (pre_dir / "match_actions.jsonl").read_text().splitlines()
            meta = json.loads(lines[0])
            events = [json.loads(line) for line in lines[1:]]

            self.assertNotIn("min_score", meta["source"])
            self.assertEqual(data["num_events"], 2)
            self.assertEqual([e["score"] for e in events], [0.9123, 0.02])
            self.assertEqual(
                set(events[0]), {"id", "frame", "label", "xy", "visible", "score"}
            )


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
