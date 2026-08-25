"""An audit row means somebody changed something.

The rally and action editors autosave two seconds after editing stops and
flush again when the tab closes, so unchanged rewrites arrive routinely. A
trail that logged those would say "edited set1" when nobody edited anything,
and a trail you cannot trust is worse than none.
"""

from __future__ import annotations

import pathlib
import tempfile
import unittest
from pathlib import Path

from yp_video.web import audit
from yp_video.web.routers import annotate


class DiffTests(unittest.TestCase):
    key = staticmethod(lambda r: r["id"])

    def test_identical_revisions_are_empty(self) -> None:
        rows = [{"id": 1, "v": "a"}, {"id": 2, "v": "b"}]
        self.assertEqual(audit.diff(rows, list(rows), key=self.key).counts, {})

    def test_an_empty_diff_is_falsy(self) -> None:
        """The call sites branch on it directly."""
        self.assertFalse(audit.diff([{"id": 1}], [{"id": 1}], key=self.key))

    def test_counts_additions_removals_and_edits_separately(self) -> None:
        before = [{"id": 1, "v": "a"}, {"id": 2, "v": "b"}, {"id": 3, "v": "c"}]
        after = [{"id": 1, "v": "a"}, {"id": 2, "v": "CHANGED"}, {"id": 4, "v": "d"}]
        self.assertEqual(
            audit.diff(before, after, key=self.key).counts,
            {"added": 1, "removed": 1, "edited": 1},
        )

    def test_zero_counts_are_dropped(self) -> None:
        """So the summary reads "+2" rather than "+2 -0 ~0"."""
        self.assertEqual(
            audit.diff([], [{"id": 1}, {"id": 2}], key=self.key).counts, {"added": 2}
        )

    def test_reordering_alone_is_not_a_change(self) -> None:
        """Sort order is presentation; identity follows the row."""
        before = [{"id": 1, "v": "a"}, {"id": 2, "v": "b"}]
        self.assertEqual(audit.diff(before, list(reversed(before)), key=self.key).counts, {})

    def test_first_save_of_a_video_is_all_additions(self) -> None:
        self.assertEqual(audit.diff([], [{"id": 1}], key=self.key).counts, {"added": 1})


class RallySaveTests(unittest.TestCase):
    """The rally writer hands back what was on disk, so the handler can diff."""

    @staticmethod
    def _delta(out, video, duration, anns):
        rows, before = annotate._write_annotations_atomic(out, video, duration, anns)
        return rows, audit.diff(before, rows, key=lambda r: r["rally_id"])

    def _ann(self, start, end, rally_id=None, winner=None):
        return annotate.Annotation(
            start=start, end=end, label="rally", rally_id=rally_id, winner=winner
        )

    def test_the_first_save_counts_every_rally_as_new(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            out = Path(raw_dir) / "m_annotations.jsonl"
            _rows, delta = self._delta(
                out, "m", 60.0, [self._ann(1, 2), self._ann(3, 4)]
            )
            self.assertEqual(delta.counts, {"added": 2})

    def test_an_unchanged_rewrite_reports_nothing(self) -> None:
        """This is the autosave case: the timer fires, the content is the same."""
        with tempfile.TemporaryDirectory() as raw_dir:
            out = Path(raw_dir) / "m_annotations.jsonl"
            annotate._write_annotations_atomic(
                out, "m", 60.0, [self._ann(1, 2), self._ann(3, 4)]
            )
            _rows, delta = self._delta(
                out, "m", 60.0,
                [self._ann(1, 2, rally_id=1), self._ann(3, 4, rally_id=2)],
            )
            self.assertEqual(delta.counts, {})

    def test_moving_a_boundary_is_an_edit(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            out = Path(raw_dir) / "m_annotations.jsonl"
            annotate._write_annotations_atomic(out, "m", 60.0, [self._ann(1, 2)])
            _rows, delta = self._delta(
                out, "m", 60.0, [self._ann(1, 2.5, rally_id=1)]
            )
            self.assertEqual(delta.counts, {"edited": 1})

    def test_setting_the_winning_side_is_an_edit(self) -> None:
        """`winner` is written only when set, so it must not read as unchanged."""
        with tempfile.TemporaryDirectory() as raw_dir:
            out = Path(raw_dir) / "m_annotations.jsonl"
            annotate._write_annotations_atomic(out, "m", 60.0, [self._ann(1, 2)])
            _rows, delta = self._delta(
                out, "m", 60.0, [self._ann(1, 2, rally_id=1, winner="left")]
            )
            self.assertEqual(delta.counts, {"edited": 1})

    def test_deleting_and_adding_are_counted_apart(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            out = Path(raw_dir) / "m_annotations.jsonl"
            annotate._write_annotations_atomic(
                out, "m", 60.0, [self._ann(1, 2), self._ann(3, 4)]
            )
            _rows, delta = self._delta(
                out, "m", 60.0, [self._ann(1, 2, rally_id=1), self._ann(9, 10)]
            )
            self.assertEqual(delta.counts, {"added": 1, "removed": 1})


class ChangeDetailTests(unittest.TestCase):
    """"Somebody edited this video" is not an answer to "what did they change"."""

    key = staticmethod(lambda r: r["id"])

    def test_an_edit_records_only_the_fields_that_moved(self) -> None:
        before = [{"id": 3, "start": 40.0, "end": 45.2, "label": "rally"}]
        after = [{"id": 3, "start": 40.0, "end": 47.8, "label": "rally"}]
        (change,) = audit.diff(before, after, key=self.key).changes
        self.assertEqual(change["op"], "edited")
        self.assertEqual(change["id"], 3)
        # start and label are unchanged and must not appear as noise.
        self.assertEqual(change["fields"], {"end": [45.2, 47.8]})

    def test_an_addition_records_the_item(self) -> None:
        (change,) = audit.diff([], [{"id": 40, "start": 83.0}], key=self.key).changes
        self.assertEqual(change["op"], "added")
        self.assertEqual(change["item"], {"id": 40, "start": 83.0})

    def test_a_removal_keeps_what_was_deleted(self) -> None:
        """The deleted row is gone from disk; this is the only record of it."""
        (change,) = audit.diff([{"id": 12, "start": 12.0}], [], key=self.key).changes
        self.assertEqual(change["op"], "removed")
        self.assertEqual(change["item"], {"id": 12, "start": 12.0})

    def test_a_field_appearing_records_none_as_the_before(self) -> None:
        """Setting the winning side is an edit from nothing to something."""
        before = [{"id": 1, "start": 1.0}]
        after = [{"id": 1, "start": 1.0, "winner": "left"}]
        (change,) = audit.diff(before, after, key=self.key).changes
        self.assertEqual(change["fields"], {"winner": [None, "left"]})

    def test_an_unchanged_save_records_no_changes(self) -> None:
        rows = [{"id": 1, "start": 1.0}]
        self.assertEqual(audit.diff(rows, list(rows), key=self.key).changes, [])

    def test_a_huge_edit_is_bounded_but_says_how_much_it_dropped(self) -> None:
        """Silent truncation would hide exactly what this table is for."""
        after = [{"id": i, "start": float(i)} for i in range(audit._MAX_CHANGES_PER_SAVE + 25)]
        changes = audit.diff([], after, key=self.key).changes
        self.assertEqual(len(changes), audit._MAX_CHANGES_PER_SAVE + 1)
        self.assertEqual(changes[-1], {"op": "truncated", "count": 25})


class CoalescingContractTests(unittest.TestCase):
    """Every action folded into a work session must file a real diff.

    Folding an action means three promises at once: its unchanged saves leave
    no row, its row carries a tally, and its ×N badge expands into the actual
    edits. `audit.record_diff` is what keeps those together — ReID once sat in
    the fold set with none of them, so its no-op saves were billed as work.

    This walks the live route table rather than a hand-kept list, so a new
    autosaving editor cannot join the set and quietly skip the contract.
    """

    def test_every_coalescing_action_records_a_diff(self) -> None:
        import inspect

        from yp_video.web.app import app

        by_action = {}
        for route in app.routes:
            path = getattr(route, "path", None)
            endpoint = getattr(route, "endpoint", None)
            if not path or endpoint is None:
                continue
            for method in getattr(route, "methods", ()) or ():
                by_action[f"{method} {path}"] = endpoint

        offenders = []
        for action in sorted(audit._COALESCING):
            endpoint = by_action.get(action)
            if endpoint is None:
                offenders.append(f"{action} → no such route")
                continue
            if "record_diff" not in inspect.getsource(endpoint):
                offenders.append(f"{action} → does not call audit.record_diff")
        self.assertEqual(offenders, [], "\n".join(offenders))

    def test_every_coalescing_action_has_a_display_name(self) -> None:
        """The Audit page shows the raw route when a name is missing, which
        reads as a bug next to its three named neighbours."""
        labels = (
            pathlib.Path(__file__).resolve().parents[1]
            / "src/yp_video/web/frontend/src/lib/auditLabels.ts"
        ).read_text(encoding="utf-8")
        missing = [a for a in sorted(audit._COALESCING) if f"'{a}'" not in labels]
        self.assertEqual(missing, [])

    def test_the_rule_would_catch_a_violation(self) -> None:
        """A contract test that cannot fail is decoration."""
        import inspect

        def endpoint_without_diff():
            audit.detail(target="x")

        self.assertNotIn("record_diff", inspect.getsource(endpoint_without_diff))


class SkipTests(unittest.TestCase):
    def test_skip_outside_a_request_is_a_no_op(self) -> None:
        audit.skip()  # must not raise

    def test_skip_marks_the_current_request(self) -> None:
        collected = {"target": None, "summary": {}, "skip": False}
        token = audit._detail.set(collected)
        try:
            audit.skip()
        finally:
            audit._detail.reset(token)
        self.assertTrue(collected["skip"])


if __name__ == "__main__":
    unittest.main()
