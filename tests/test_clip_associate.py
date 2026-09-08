"""Association by the clip classifier: the decision rule and the package.

The rule is a comparison against the event's label, so the failure modes
worth pinning are the silent ones — naming the blocker for a spike because
"someone" was acting, or naming nobody when a candidate's own best class
disagrees with the label.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from yp_video.actor import clip_associate as ca
from yp_video.contracts.action import CLIP_PACKAGE_TYPE


def _scored(track: str, **probs: float) -> dict:
    return {"event_id": "e1", "track": track, "probs": probs, "contact_px": [10.0, 20.0]}


class DecisionTests(unittest.TestCase):
    def test_picks_the_candidate_most_likely_to_have_done_the_label(self) -> None:
        rows = [
            _scored("1:1", none=0.1, spike=0.8, block=0.1),
            _scored("1:2", none=0.1, spike=0.2, block=0.7),
        ]
        answer = ca.decide_event("spike", rows)
        self.assertEqual(answer.track.key, "1:1")
        self.assertAlmostEqual(answer.confidence, 0.8)
        self.assertEqual(answer.contact_px, (10.0, 20.0))
        self.assertEqual(ca.decide_event("block", rows).track.key, "1:2")

    def test_abstains_when_the_best_candidate_disagrees_with_the_label(self) -> None:
        rows = [
            _scored("1:1", none=0.6, spike=0.3, block=0.1),
            _scored("1:2", none=0.7, spike=0.2, block=0.1),
        ]
        self.assertIsNone(ca.decide_event("spike", rows))

    def test_unknown_label_is_nobody(self) -> None:
        self.assertIsNone(ca.decide_event("score", [_scored("1:1", none=0.5, spike=0.5)]))

    def test_decide_groups_by_event(self) -> None:
        events = [{"id": "e1", "frame": 10, "label": "spike"}, {"id": "e2", "frame": 20, "label": "set"}]
        scored = [_scored("1:1", none=0.1, spike=0.9), {**_scored("1:3", none=0.2, set=0.8), "event_id": "e2"}]
        answers = ca.decide(events, scored)
        self.assertEqual({k: v.track.key for k, v in answers.items()}, {"e1": "1:1", "e2": "1:3"})


class PackageTests(unittest.TestCase):
    def test_package_run_writes_a_listable_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "exp" / "20260909_clips_rny008_tv_gsm"
            run.mkdir(parents=True)
            (run / "checkpoint_best.pt").write_bytes(b"x")
            (run / "config.json").write_text(json.dumps({"classes": ["none", "spike"], "feature_arch": "rny008_tv_gsm"}))
            (run / "metrics.jsonl").write_text(
                "\n".join(json.dumps({"epoch": e, "val": {"macro_f1_actions": f}}) for e, f in [(0, 0.5), (1, 0.7), (2, 0.6)])
            )
            packages = root / "checkpoints"
            package = ca.package_run(run, packages)
            manifest = json.loads((package / "manifest.json").read_text())
            self.assertEqual(manifest["type"], CLIP_PACKAGE_TYPE)
            self.assertEqual(manifest["best"], {"epoch": 1, "metric": "macro_f1_actions", "value": 0.7})
            rows = ca.list_checkpoints(packages)
            self.assertEqual([r["experiment"] for r in rows], [run.name])
            self.assertEqual(rows[0]["epoch"], 1)
            self.assertTrue(ca.default_checkpoint(packages).endswith("checkpoint_best.pt"))

    def test_spot_packages_are_not_listed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "fusion"
            package.mkdir()
            (package / "checkpoint_best.pt").write_bytes(b"x")
            (package / "manifest.json").write_text(json.dumps({"type": "yp-video-spot-checkpoint"}))
            self.assertEqual(ca.list_checkpoints(Path(tmp)), [])
            self.assertEqual(ca.default_checkpoint(Path(tmp)), "")


if __name__ == "__main__":
    unittest.main()
