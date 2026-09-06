"""The Inference page's stage logic: which package can serve every stage,
which stages a run performs, and the one-line summary per video."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from yp_video.contracts.action import SPOT_PACKAGE_TYPE
from yp_video.web import fusion_inference as fi


def _make_package(root: Path, name: str, tasks: list[str]) -> Path:
    package = root / name
    package.mkdir()
    (package / "checkpoint_best.pt").write_bytes(b"")
    (package / "manifest.json").write_text(
        json.dumps({"type": SPOT_PACKAGE_TYPE, "tasks": tasks, "best": {"epoch": 1}}),
        encoding="utf-8",
    )
    return package / "checkpoint_best.pt"


class CheckpointTests(unittest.TestCase):
    def test_only_packages_with_every_required_head_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_package(root, "fusion", ["action", "location", "actor", "rally", "winner"])
            _make_package(root, "rally_only", ["rally", "winner"])
            _make_package(root, "assoc_only", ["action", "location", "actor"])
            rows = fi.list_checkpoints(root)
            self.assertEqual([row["experiment"] for row in rows], ["fusion"])
            self.assertTrue(fi.default_checkpoint(root).endswith("fusion/checkpoint_best.pt"))

    def test_package_tasks_come_from_the_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = _make_package(Path(tmp), "fusion", ["rally", "action", "actor"])
            self.assertEqual(fi.package_tasks(checkpoint), ["rally", "action", "actor"])
            self.assertEqual(fi.package_tasks(Path(tmp) / "nowhere" / "x.pt"), [])

    def test_resolve_rejects_a_package_missing_a_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = _make_package(Path(tmp), "rally_only", ["rally", "winner"])
            with (
                patch.object(fi.prelabel, "resolve_checkpoint", return_value=checkpoint),
                self.assertRaises(ValueError) as ctx,
            ):
                fi.resolve_checkpoint("rally_only/checkpoint_best.pt")
            self.assertIn("action", str(ctx.exception))
            self.assertIn("actor", str(ctx.exception))

    def test_resolve_without_any_fusion_package_is_not_found(self):
        with (
            patch.object(fi, "default_checkpoint", return_value=""),
            self.assertRaises(FileNotFoundError),
        ):
            fi.resolve_checkpoint("")


class StagePlanTests(unittest.TestCase):
    def _plan(self, *, rally_exists, action_exists, tracks, records, overwrite):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            present = root / "present"
            present.touch()
            absent = root / "absent"
            with (
                patch.object(fi, "rally_spot_pre_annotation_path", return_value=present if rally_exists else absent),
                patch.object(fi, "pre_annotation_path", return_value=present if action_exists else absent),
                patch.object(fi, "tracks_path", return_value=present if tracks else absent),
                patch.object(fi, "records_path", return_value=present if records else absent),
            ):
                return fi.plan_stages("match", overwrite=overwrite)

    def test_fresh_video_runs_rally_and_action_and_waits_for_tracking(self):
        plan = self._plan(rally_exists=False, action_exists=False, tracks=False, records=False, overwrite=False)
        self.assertIsNone(plan.rally)
        self.assertIsNone(plan.action)
        self.assertEqual(plan.association, "run Rally Tracking first")

    def test_existing_outputs_are_kept_unless_overwriting(self):
        kept = self._plan(rally_exists=True, action_exists=True, tracks=True, records=True, overwrite=False)
        self.assertEqual(kept.rally, "kept existing rallies")
        self.assertEqual(kept.action, "kept existing actions")
        self.assertIsNone(kept.association)
        redo = self._plan(rally_exists=True, action_exists=True, tracks=True, records=True, overwrite=True)
        self.assertIsNone(redo.rally)
        self.assertIsNone(redo.action)

    def test_tracked_but_undetected_video_names_player_detection(self):
        plan = self._plan(rally_exists=False, action_exists=False, tracks=True, records=False, overwrite=False)
        self.assertEqual(plan.association, "run Player Detection first")


class SummaryTests(unittest.TestCase):
    def test_counts_and_skip_reasons_read_as_one_line(self):
        result = fi.VideoResult(
            rallies=12, events=340,
            association={"changed": 3, "unchanged": 300, "labeled": 7},
        )
        self.assertEqual(
            fi.summarize(result),
            "12 rallies · 340 actions · association: 3 moved · 300 unchanged · 7 labeled kept",
        )
        skipped = fi.VideoResult(skipped={
            "rally": "kept existing rallies",
            "action": "kept existing actions",
            "association": "run Rally Tracking first",
        })
        self.assertEqual(
            fi.summarize(skipped),
            "rally: kept existing rallies · action: kept existing actions · association: run Rally Tracking first",
        )


if __name__ == "__main__":
    unittest.main()
