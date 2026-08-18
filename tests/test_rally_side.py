"""Rally side: which court side won, from annotation to aggregated segment.

Covers the producer-side path: the editor's save persists ``side`` (and only
when set), ``load_rallies`` carries it to every consumer, training labels
attach it to the segment events, and ``events_to_rally_segments`` turns
per-frame ``side_probs`` into one winning side per rally — averaged over the
segment's final ``SIDE_TAIL_S`` seconds only, since earlier frames cannot
know the outcome.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from yp_video.action.rally import events_to_rally_segments
from yp_video.contracts.action import SIDE_TAIL_S
from yp_video.core import rallies as core_rallies
from yp_video.core.jsonl import read_jsonl
from yp_video.web.routers import annotate


def _write(tmp: Path, annotations: list[annotate.Annotation]) -> Path:
    out = tmp / "vid_annotations.jsonl"
    annotate._write_annotations_atomic(out, "vid.mp4", 100.0, annotations)
    return out


class SavePersistsSideTest(unittest.TestCase):
    def test_side_saved_only_when_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp), [
                annotate.Annotation(start=1.0, end=10.0, label="rally", side="left"),
                annotate.Annotation(start=20.0, end=30.0, label="rally"),
            ])
            _meta, rows = read_jsonl(path)
        self.assertEqual(rows[0]["side"], "left")
        self.assertNotIn("side", rows[1])

    def test_unknown_side_rejected(self):
        with self.assertRaises(ValidationError):
            annotate.Annotation(start=1.0, end=2.0, label="rally", side="up")


class LoadRalliesCarriesSideTest(unittest.TestCase):
    def test_side_reaches_every_consumer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp), [
                annotate.Annotation(start=1.0, end=10.0, label="rally", side="near"),
                annotate.Annotation(start=20.0, end=30.0, label="rally"),
            ])
            with patch.object(core_rallies, "rally_annotation_path", return_value=path):
                rallies = core_rallies.load_rallies("vid")
        self.assertEqual(rallies[0]["side"], "near")
        self.assertIsNone(rallies[1]["side"])


class SegmentAggregationTest(unittest.TestCase):
    def _tick(self, t: float, probs=None) -> dict:
        event = {"frame": int(t * 30), "score": 0.9}
        if probs is not None:
            event["side_probs"] = probs
        return event

    def test_tail_decides_the_side(self):
        # Early frames vote right, the SIDE_TAIL_S tail votes left: the tail wins.
        events = [
            self._tick(
                t / 2,
                [0.8, 0.1, 0.05, 0.05]
                if t / 2 >= 10.0 - SIDE_TAIL_S
                else [0.1, 0.8, 0.05, 0.05],
            )
            for t in range(0, 21)
        ]
        (seg,) = events_to_rally_segments(
            events, native_fps=30, min_score=0.5, max_gap_s=2.0, min_duration_s=4.0
        )
        self.assertEqual(seg["side"], "left")
        self.assertEqual(seg["side_score"], 0.8)

    def test_sideless_events_yield_sideless_segment(self):
        events = [self._tick(t / 2) for t in range(0, 21)]
        (seg,) = events_to_rally_segments(
            events, native_fps=30, min_score=0.5, max_gap_s=2.0, min_duration_s=4.0
        )
        self.assertNotIn("side", seg)
        self.assertNotIn("side_score", seg)


class TrainingLabelsCarrySideTest(unittest.TestCase):
    def test_write_training_labels_attaches_side(self):
        from yp_video.action import rally as rally_mod

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ann = _write(tmp_path, [
                annotate.Annotation(start=1.0, end=10.0, label="rally", side="far"),
                annotate.Annotation(start=20.0, end=30.0, label="rally"),
            ])
            video = tmp_path / "vid.mp4"
            video.touch()
            label_dir = tmp_path / "labels"
            with patch.object(
                rally_mod,
                "inspect_action_frame_cache",
                return_value={"ready": True, "frame_count": 200},
            ), patch.object(rally_mod, "cut_kind_of", return_value="sideline"):
                summary = rally_mod.write_training_labels(
                    [(ann, video)],
                    cache_root=tmp_path,
                    extract_fps=2.0,
                    label_dir=label_dir,
                )
            _meta, events = read_jsonl(label_dir / "vid_rally.jsonl")
        self.assertEqual(summary["sided_rallies"], 1)
        self.assertEqual(events[0]["side"], "far")
        self.assertNotIn("side", events[1])


if __name__ == "__main__":
    unittest.main()
