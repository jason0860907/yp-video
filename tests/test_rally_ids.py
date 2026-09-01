"""Stable rally ids: the file is the ledger, position is presentation.

Covers the two moments an id exists: birth (the editor's save mints above
the high-water mark; a model pass numbers its own file) and reading (stored
ids verified, never recomputed).
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from fastapi import HTTPException

from yp_video.core.jsonl import read_jsonl
from yp_video.core.rallies import number_rallies, resolve_rally_ids
from yp_video.web import rally_annotations
from yp_video.web.routers import annotate


class ResolveTests(unittest.TestCase):
    def test_stored_ids_are_the_answer(self) -> None:
        records = [{"rally_id": 7}, {"rally_id": 1}, {"rally_id": 3}]
        self.assertEqual(resolve_rally_ids(records), [7, 1, 3])

    def test_a_record_without_an_id_is_refused(self) -> None:
        """No silent positional fallback: inventing ids from sort order is
        exactly the renumbering this scheme exists to end."""
        for bad in ({}, {"rally_id": None}, {"rally_id": 0}, {"rally_id": -2},
                    {"rally_id": "3"}, {"rally_id": True}):
            with self.assertRaisesRegex(ValueError, "valid rally_id"):
                resolve_rally_ids([{"rally_id": 1}, bad])

    def test_duplicate_ids_are_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, r"Duplicate rally_id\(s\): \[4\]"):
            resolve_rally_ids([{"rally_id": 4}, {"rally_id": 4}, {"rally_id": 1}])


class NumberRalliesTests(unittest.TestCase):
    def test_a_model_pass_numbers_itself_in_start_order(self) -> None:
        rows, max_id = number_rallies(
            [
                {"start": 9.0, "end": 10.0, "label": "rally", "score": 0.9},
                {"start": 1.0, "end": 2.0, "label": "rally", "score": 0.8},
            ]
        )
        self.assertEqual(max_id, 2)
        self.assertEqual([r["rally_id"] for r in rows], [1, 2])
        self.assertEqual([r["start"] for r in rows], [1.0, 9.0])
        # Extra fields (score) ride along untouched.
        self.assertEqual(rows[0]["score"], 0.8)


class SaveTests(unittest.TestCase):
    def _annotation(self, start: float, end: float, rally_id: int | None = None):
        return rally_annotations.Annotation(start=start, end=end, label="rally", rally_id=rally_id)

    def test_ids_follow_rows_and_new_rows_mint_above_high_water(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            out = Path(raw_dir) / "m_annotations.jsonl"
            rows, _ = rally_annotations.write_annotations_atomic(
                out, "m", 60.0,
                [self._annotation(10, 20), self._annotation(5, 8)],
            )
            self.assertEqual([r["rally_id"] for r in rows], [1, 2])
            self.assertEqual([r["start"] for r in rows], [5, 10])

            # Move a span and insert one BEFORE it: identity follows the row.
            rows, _ = rally_annotations.write_annotations_atomic(
                out, "m", 60.0,
                [
                    self._annotation(9, 21, rally_id=2),
                    self._annotation(5, 8, rally_id=1),
                    self._annotation(1, 3),  # new
                ],
            )
            self.assertEqual(
                [(r["start"], r["rally_id"]) for r in rows],
                [(1, 3), (5, 1), (9, 2)],
            )
            meta, records = read_jsonl(out)
            self.assertEqual(meta["max_rally_id"], 3)
            self.assertEqual(records, rows)

    def test_a_deleted_id_is_never_reused(self) -> None:
        """max(present)+1 would re-issue a deleted id and every stored
        tracklet key "<id>:<track>" would silently re-attach."""
        with tempfile.TemporaryDirectory() as raw_dir:
            out = Path(raw_dir) / "m_annotations.jsonl"
            rally_annotations.write_annotations_atomic(
                out, "m", 60.0,
                [self._annotation(1, 2), self._annotation(3, 4)],
            )
            # Delete rally 2, then add a new one.
            rows, _ = rally_annotations.write_annotations_atomic(
                out, "m", 60.0,
                [self._annotation(1, 2, rally_id=1), self._annotation(5, 6)],
            )
            self.assertEqual([r["rally_id"] for r in rows], [1, 3])

    def test_duplicate_client_ids_are_a_400(self) -> None:
        request = annotate.SaveAnnotationsRequest(
            video="m.mp4",
            duration=60.0,
            annotations=[
                self._annotation(1, 2, rally_id=5),
                self._annotation(3, 4, rally_id=5),
            ],
        )
        with self.assertRaises(HTTPException) as caught:
            asyncio.run(annotate.save_annotations(request))
        self.assertEqual(caught.exception.status_code, 400)
        self.assertIn("5", str(caught.exception.detail))


if __name__ == "__main__":
    unittest.main()
