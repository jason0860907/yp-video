"""Stable rally ids: the file is the ledger, position is presentation.

Covers the three moments an id exists: birth (the editor's save mints above
the high-water mark; a model pass numbers its own file), reading (stored ids
verified, never recomputed), and the freeze migration (positional numbering
stamped in without moving a fingerprint).
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from yp_video.core import rallies as core_rallies
from yp_video.core.jsonl import read_jsonl, write_jsonl
from yp_video.core.rallies import number_rallies, resolve_rally_ids
from yp_video.web.routers import annotate


def _load_freeze_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "freeze_rally_ids", root / "scripts" / "freeze_rally_ids.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: the script defines a dataclass, and dataclasses
    # resolve annotations through sys.modules[cls.__module__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
        return annotate.Annotation(start=start, end=end, label="rally", rally_id=rally_id)

    def test_ids_follow_rows_and_new_rows_mint_above_high_water(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            out = Path(raw_dir) / "m_annotations.jsonl"
            rows = annotate._write_annotations_atomic(
                out, "m", 60.0,
                [self._annotation(10, 20), self._annotation(5, 8)],
            )
            self.assertEqual([r["rally_id"] for r in rows], [1, 2])
            self.assertEqual([r["start"] for r in rows], [5, 10])

            # Move a span and insert one BEFORE it: identity follows the row.
            rows = annotate._write_annotations_atomic(
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
            annotate._write_annotations_atomic(
                out, "m", 60.0,
                [self._annotation(1, 2), self._annotation(3, 4)],
            )
            # Delete rally 2, then add a new one.
            rows = annotate._write_annotations_atomic(
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


class FreezeTests(unittest.TestCase):
    @contextmanager
    def _sources(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            table = tuple(
                core_rallies.RallySource(s.tag, root / s.tag, s.r2_category)
                for s in core_rallies.RALLY_SOURCES
            )
            with patch.object(core_rallies, "RALLY_SOURCES", table):
                yield {s.tag: s.directory for s in table}

    def test_freeze_stamps_positional_ids_without_moving_the_fingerprint(self) -> None:
        freeze = _load_freeze_module()
        with self._sources() as dirs:
            directory = dirs["annotation"]
            directory.mkdir(parents=True)
            path = directory / core_rallies.annotation_name("m")
            # Legacy file: no ids, deliberately out of start order.
            write_jsonl(
                path,
                {"video": "m"},
                [
                    {"start": 30.0, "end": 40.0, "label": "rally"},
                    {"start": 10.0, "end": 20.0, "label": "rally"},
                ],
            )
            plan = freeze.plan_file(path, "annotation", "rally-spot/annotations")
            self.assertEqual(plan.action, "stamp-ids")
            self.assertIsNone(plan.refused)

            with patch.object(freeze, "rally_annotation_path", core_rallies.rally_annotation_path), \
                 patch.object(freeze, "rally_fingerprint", core_rallies.rally_fingerprint):
                freeze.apply_plan(plan)

            spans = core_rallies.load_rallies("m")
            self.assertEqual(
                [(r["start"], r["rally_id"]) for r in spans], [(10.0, 1), (30.0, 2)]
            )
            self.assertEqual(
                core_rallies.rally_fingerprint("m"), plan.legacy_fingerprint
            )
            meta, _records = read_jsonl(path)
            self.assertEqual(meta["max_rally_id"], 2)

    def test_freeze_skips_and_preserves_already_stamped_files(self) -> None:
        freeze = _load_freeze_module()
        with self._sources() as dirs:
            directory = dirs["annotation"]
            directory.mkdir(parents=True)
            path = directory / core_rallies.annotation_name("m")
            write_jsonl(
                path,
                {"video": "m", "max_rally_id": 9},
                [{"start": 1.0, "end": 2.0, "label": "rally", "rally_id": 9}],
            )
            plan = freeze.plan_file(path, "annotation", "rally-spot/annotations")
            self.assertEqual(plan.action, "skip")

    def test_freeze_refuses_a_mixed_file(self) -> None:
        freeze = _load_freeze_module()
        with self._sources() as dirs:
            directory = dirs["annotation"]
            directory.mkdir(parents=True)
            path = directory / core_rallies.annotation_name("m")
            write_jsonl(
                path,
                {"video": "m"},
                [
                    {"start": 1.0, "end": 2.0, "label": "rally", "rally_id": 1},
                    {"start": 3.0, "end": 4.0, "label": "rally"},
                ],
            )
            plan = freeze.plan_file(path, "annotation", "rally-spot/annotations")
            self.assertEqual(plan.refused, "mixed: some records have rally_id, some not")


if __name__ == "__main__":
    unittest.main()
