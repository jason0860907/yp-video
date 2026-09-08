"""The Inference page's stage logic: which package can serve every stage,
which stages a run performs, and the one-line summary per video."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from yp_video.contracts.action import SPOT_PACKAGE_TYPE
from yp_video.tracklets import store as tracks_store
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
    def _paths(self, *, rally_exists, action_exists):
        present = Path(tempfile.mkdtemp()) / "present"
        present.touch()
        absent = present.parent / "absent"
        return (
            patch.object(fi, "rally_spot_pre_annotation_path", return_value=present if rally_exists else absent),
            patch.object(fi, "pre_annotation_path", return_value=present if action_exists else absent),
        )

    def test_fresh_video_runs_both_spot_heads(self):
        a, b = self._paths(rally_exists=False, action_exists=False)
        with a, b:
            plan = fi.plan_spot_stages("match", overwrite=False)
        self.assertEqual(plan, {"rally": None, "action": None})

    def test_existing_spot_outputs_are_kept_unless_overwriting(self):
        a, b = self._paths(rally_exists=True, action_exists=True)
        with a, b:
            kept = fi.plan_spot_stages("match", overwrite=False)
            redo = fi.plan_spot_stages("match", overwrite=True)
        self.assertEqual(kept, {"rally": "kept existing rallies", "action": "kept existing actions"})
        self.assertEqual(redo, {"rally": None, "action": None})


class PerceptionPlanTests(unittest.TestCase):
    """Tracking only survives when it provably matches today's rallies;
    detection follows the action output; association runs whenever both
    inputs exist."""

    def test_tracking_needs_rallies_and_keeps_only_matching_tracks(self):
        self.assertEqual(fi.tracking_skip("m", overwrite=False, rallies=False), "no rallies")
        with patch.object(fi, "tracks_current", return_value=True):
            self.assertEqual(fi.tracking_skip("m", overwrite=False, rallies=True), "kept existing tracks")
            self.assertIsNone(fi.tracking_skip("m", overwrite=True, rallies=True))
        with patch.object(fi, "tracks_current", return_value=False):
            self.assertIsNone(fi.tracking_skip("m", overwrite=False, rallies=True))

    def test_tracks_are_current_only_with_masks_and_a_matching_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracks = Path(tmp) / "m.jsonl"
            masks = Path(tmp) / "m_masks.npz"
            masks.touch()

            def check(header: dict, *, fingerprint: str, masks_path=masks) -> bool:
                tracks.write_text(json.dumps({"video": "m", **header}) + "\n", encoding="utf-8")
                with (
                    patch.object(tracks_store, "tracks_path", return_value=tracks),
                    patch.object(tracks_store, "tracks_masks_path", return_value=masks_path),
                    patch.object(tracks_store, "rally_fingerprint", return_value=fingerprint),
                ):
                    return tracks_store.tracks_current("m")

            self.assertFalse(check({"rallies": {"count": 3}}, fingerprint="abc"))
            self.assertTrue(check({"rallies": {"fingerprint": "abc"}}, fingerprint="abc"))
            self.assertFalse(check({"rallies": {"fingerprint": "abc"}}, fingerprint="moved"))
            self.assertFalse(
                check({"rallies": {"fingerprint": "abc"}}, fingerprint="abc", masks_path=Path(tmp) / "none.npz")
            )

    def test_detection_refreshes_after_a_new_action_output(self):
        self.assertEqual(fi.detection_skip("m", overwrite=False, events=False, action_ran=True), "no action events")
        with patch.object(fi, "detections_current", return_value=True):
            self.assertEqual(
                fi.detection_skip("m", overwrite=False, events=True, action_ran=False),
                "kept existing detections",
            )
            self.assertIsNone(fi.detection_skip("m", overwrite=False, events=True, action_ran=True))
            self.assertIsNone(fi.detection_skip("m", overwrite=True, events=True, action_ran=False))
        with patch.object(fi, "detections_current", return_value=False):
            self.assertIsNone(fi.detection_skip("m", overwrite=False, events=True, action_ran=False))

    def test_association_names_the_missing_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            present = Path(tmp) / "present"
            present.touch()
            absent = Path(tmp) / "absent"
            self.assertEqual(fi.association_skip("m", events=False), "no action events")
            with (
                patch.object(fi, "tracks_path", return_value=absent),
                patch.object(fi, "records_path", return_value=present),
            ):
                self.assertEqual(fi.association_skip("m", events=True), "no tracks")
            with (
                patch.object(fi, "tracks_path", return_value=present),
                patch.object(fi, "records_path", return_value=absent),
            ):
                self.assertEqual(fi.association_skip("m", events=True), "no detections")
            with (
                patch.object(fi, "tracks_path", return_value=present),
                patch.object(fi, "records_path", return_value=present),
            ):
                self.assertIsNone(fi.association_skip("m", events=True))


class SummaryTests(unittest.TestCase):
    def test_counts_and_skip_reasons_read_as_one_line(self):
        result = fi.VideoResult(
            rallies=12, events=340, tracklets=800, detections=2100,
            association={"changed": 3, "unchanged": 300, "labeled": 7},
        )
        self.assertEqual(
            fi.summarize(result),
            "12 rallies · 340 actions · 800 tracklets · 2100 people detected · "
            "association: 3 moved · 300 unchanged · 7 labeled kept",
        )
        skipped = fi.VideoResult(skipped={
            "rally": "kept existing rallies",
            "action": "kept existing actions",
            "tracking": "kept existing tracks",
            "detection": "no action events",
            "association": "no action events",
        })
        self.assertEqual(
            fi.summarize(skipped),
            "rally: kept existing rallies · action: kept existing actions · "
            "tracking: kept existing tracks · detection: no action events · "
            "association: no action events",
        )


if __name__ == "__main__":
    unittest.main()
