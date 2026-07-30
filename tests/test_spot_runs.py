import json
import tempfile
import unittest
from pathlib import Path

from fastapi import HTTPException

from yp_video.web.spot_runs import (
    TrainProgress,
    make_train_parsers,
    performance_payload,
)


def _feed(parsers, line: str):
    updates = []
    for parser in parsers:
        match = parser.pattern.search(line)
        if match:
            update = parser.handler(match)
            if update:
                updates.append(update)
    return updates


class SpotTaskMetricsProgressTest(unittest.TestCase):
    def test_parses_common_task_metrics_and_snapshots_best(self):
        ctx = TrainProgress(epochs=10)
        parsers, is_key_line = make_train_parsers(
            ctx,
            params_key="fusion_model_train_progress",
            criterion="map",
            headline_pattern=r"Val mAP:\s*([0-9.]+)%",
        )
        tasks = {
            "actor": {
                "primary_metric": "player_top1",
                "train": {
                    "loss": 1.0,
                    "metrics": {"player_top1": 0.7},
                    "counts": {"player_events": 10},
                },
                "validation": {
                    "loss": 0.8,
                    "metrics": {"player_top1": 0.75},
                    "counts": {"player_events": 20},
                },
            }
        }
        line = "SPOT_TASK_METRICS " + json.dumps(tasks)

        self.assertTrue(is_key_line(line))
        _feed(parsers, line)
        self.assertEqual(ctx.latest_task_metrics, tasks)

        _feed(parsers, "Val mAP: 80.0%")
        _feed(parsers, "New best epoch!")
        self.assertEqual(ctx.best_task_metrics, tasks)

    def test_loss_parser_accepts_legacy_and_task_aware_tables(self):
        ctx = TrainProgress(epochs=10)
        parsers, _ = make_train_parsers(
            ctx,
            params_key="progress",
            criterion="map",
            headline_pattern=r"Val mAP:\s*([0-9.]+)%",
        )

        _feed(parsers, "Train loss  0.10  0.20  0.00  1.30")
        self.assertEqual(ctx.latest_train_loss, 1.3)

        _feed(parsers, "Val loss  0.10  0.20  0.90  0.00  1.20")
        self.assertEqual(ctx.latest_val_loss, 1.2)

    def test_performance_can_filter_a_shared_checkpoint_root(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            for name in ("yp_fusion_one", "yp_action_one"):
                run = root / name
                run.mkdir()
                (run / "metrics.jsonl").write_text(
                    json.dumps(
                        {
                            "epoch": 0,
                            "mAP": {"harmonic": 0.5},
                            "tasks": {"actor": {"primary_metric": "player_top1"}},
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )

            payload = performance_payload(
                root,
                run_prefixes=("yp_fusion_",),
            )

            self.assertEqual(payload["runs"], ["yp_fusion_one"])
            self.assertIn("actor", payload["entries"][0]["tasks"])
            with self.assertRaises(HTTPException):
                performance_payload(
                    root,
                    "yp_action_one",
                    run_prefixes=("yp_fusion_",),
                )


if __name__ == "__main__":
    unittest.main()
