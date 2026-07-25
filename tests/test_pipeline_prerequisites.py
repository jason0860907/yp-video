"""The stage dependencies a new video walks through.

Two things are pinned here. First, that rally spans have exactly ONE source
of truth: the action annotator used to look in two of the three possible
locations and silently miss the SPOT predictor's output, which produced an
action file with no rallies and a failure that only surfaced a stage later.
Second, that tracking depends on rallies and NOT on the action annotation —
it needs to know where the rallies are, not what happened inside them, and
reading the action file made it wait for a stage it does not depend on.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from yp_video.core import rallies as core_rallies
from yp_video.core.jsonl import write_jsonl
from yp_video.extraction import prerequisites as prereq


def _write_rallies(directory: Path, stem: str, spans: list[tuple[float, float]]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / core_rallies.annotation_name(stem)
    write_jsonl(
        path,
        {"video": stem},
        [{"start": s, "end": e, "label": "rally"} for s, e in spans],
    )
    return path


@contextmanager
def _sources(root: Path):
    """Point the three rally locations at scratch directories."""
    dirs = {
        "annotation": root / "manual",
        "spot-pre-annotation": root / "spot",
        "pre-annotation": root / "vlm",
    }
    table = tuple(
        core_rallies.RallySource(tag, dirs[tag], src.r2_category)
        for tag, src in ((s.tag, s) for s in core_rallies.RALLY_SOURCES)
    )
    with patch.object(core_rallies, "RALLY_SOURCES", table):
        yield dirs


class RallySourceTests(unittest.TestCase):
    def test_priority_is_human_then_spot_then_vlm(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            with _sources(root) as dirs:
                _write_rallies(dirs["pre-annotation"], "m", [(0.0, 1.0)])
                self.assertEqual(core_rallies.rally_sources("m"), ["pre-annotation"])

                # The trained model outranks the VLM bootstrap. This is the
                # source that used to be invisible to the action annotator.
                _write_rallies(dirs["spot-pre-annotation"], "m", [(2.0, 3.0)])
                self.assertEqual(core_rallies.load_rallies("m")[0]["start"], 2.0)

                # A human outranks both.
                _write_rallies(dirs["annotation"], "m", [(4.0, 5.0)])
                self.assertEqual(core_rallies.load_rallies("m")[0]["start"], 4.0)
                self.assertEqual(
                    core_rallies.rally_sources("m"),
                    ["annotation", "spot-pre-annotation", "pre-annotation"],
                )

    def test_rally_ids_are_positional_over_sorted_spans(self) -> None:
        """Ids come from order, so the order has to be defined in one place."""
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            with _sources(root) as dirs:
                _write_rallies(
                    dirs["annotation"], "m", [(30.0, 40.0), (10.0, 20.0), (50.0, 60.0)]
                )
                spans = core_rallies.load_rallies("m")

            self.assertEqual([r["rally_id"] for r in spans], [1, 2, 3])
            self.assertEqual([r["start"] for r in spans], [10.0, 30.0, 50.0])

    def test_identical_spans_do_not_crash_the_sort(self) -> None:
        """A plain sort would fall through to comparing the record dicts."""
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            with _sources(root) as dirs:
                _write_rallies(dirs["annotation"], "m", [(1.0, 2.0), (1.0, 2.0)])
                self.assertEqual(len(core_rallies.load_rallies("m")), 2)

    def test_fingerprint_tracks_the_spans_not_the_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            with _sources(root) as dirs:
                self.assertIsNone(core_rallies.rally_fingerprint("m"))
                _write_rallies(dirs["annotation"], "m", [(1.0, 2.0)])
                first = core_rallies.rally_fingerprint("m")
                _write_rallies(dirs["annotation"], "m", [(1.0, 2.0)])
                self.assertEqual(core_rallies.rally_fingerprint("m"), first)
                # Moving a span renumbers every track key downstream.
                _write_rallies(dirs["annotation"], "m", [(1.0, 2.0), (5.0, 6.0)])
                self.assertNotEqual(core_rallies.rally_fingerprint("m"), first)


class TrackingDependencyTests(unittest.TestCase):
    def test_tracking_reads_rallies_not_the_action_file(self) -> None:
        """The whole point of the decoupling: no action annotation needed."""
        from yp_video.tracklets import tracking

        source = Path(tracking.__file__).read_text(encoding="utf-8")
        self.assertNotIn("action_annotation_path", source)

        with (
            patch.object(tracking, "load_rallies", return_value=[]) as load,
            self.assertRaisesRegex(ValueError, "No rally spans"),
        ):
            tracking.track_video(Path("/nonexistent/match.mp4"))
        load.assert_called_once_with("match")


class PrerequisiteTests(unittest.TestCase):
    def _state(self, **overrides) -> prereq.Prerequisites:
        base = dict(
            rally_sources=["annotation"],
            has_action=True,
            has_tracks=True,
            has_masks=True,
            tracks_stale=False,
            has_records=True,
        )
        return prereq.Prerequisites(**{**base, **overrides})

    def test_blocked_on_reports_the_first_unmet_stage(self) -> None:
        self.assertIsNone(self._state().blocked_on)
        self.assertEqual(self._state(rally_sources=[]).blocked_on, "rallies")
        self.assertEqual(self._state(has_action=False).blocked_on, "action")
        self.assertEqual(self._state(has_tracks=False).blocked_on, "tracks")
        self.assertEqual(self._state(has_records=False).blocked_on, "records")

    def test_the_earliest_gap_wins(self) -> None:
        """Telling someone to run tracking when they have no rallies yet is
        a wrong answer, not merely an unhelpful one."""
        state = self._state(rally_sources=[], has_action=False, has_tracks=False)
        self.assertEqual(state.blocked_on, "rallies")

    def test_the_header_is_read_without_parsing_the_tracklets(self) -> None:
        """A tracks jsonl is megabytes and every video list asks all of them."""
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "match_tracks.jsonl"
            write_jsonl(path, {"video": "match", "rallies": {"fingerprint": "x"}}, [])
            # A record line that would explode if it were parsed at all.
            with open(path, "a", encoding="utf-8") as f:
                f.write("{not json\n")
            with (
                patch.object(prereq, "tracks_path", return_value=path),
                patch.object(prereq, "rally_fingerprint", return_value="x"),
            ):
                self.assertFalse(prereq._tracks_stale("match"))

    def test_tracks_without_a_fingerprint_are_unknown_not_stale(self) -> None:
        """Tracks predating the fingerprint must not all flag as stale."""
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "match_tracks.jsonl"
            write_jsonl(path, {"video": "match"}, [])
            with patch.object(prereq, "tracks_path", return_value=path):
                self.assertFalse(prereq._tracks_stale("match"))

    def test_moved_rallies_make_existing_tracks_stale(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "match_tracks.jsonl"
            write_jsonl(
                path,
                {"video": "match", "rallies": {"count": 1, "fingerprint": "rallies_old"}},
                [],
            )
            with (
                patch.object(prereq, "tracks_path", return_value=path),
                patch.object(prereq, "rally_fingerprint", return_value="rallies_new"),
            ):
                self.assertTrue(prereq._tracks_stale("match"))


if __name__ == "__main__":
    unittest.main()
