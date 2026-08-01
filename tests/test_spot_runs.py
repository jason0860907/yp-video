import json
import tempfile
import unittest
from pathlib import Path

from fastapi import HTTPException

from yp_video.action.spot_runs import (
    best_epochs_per_task,
    checkpoint_package_options,
    export_checkpoint_package,
)
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
            for name, package_type in (
                ("yp_fusion_one", "actor-association-spot"),
                ("yp_action_one", "yp-video-action-checkpoint"),
            ):
                run = root / name
                run.mkdir()
                (run / "manifest.json").write_text(
                    json.dumps({"type": package_type}), encoding="utf-8"
                )
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
                package_types=("actor-association-spot",),
            )

            self.assertEqual(payload["runs"], ["yp_fusion_one"])
            self.assertIn("actor", payload["entries"][0]["tasks"])
            with self.assertRaises(HTTPException):
                performance_payload(
                    root,
                    "yp_action_one",
                    package_types=("actor-association-spot",),
                )


def _write_run(root: Path, epochs: list[dict], definitions: dict) -> Path:
    """A run dir the way yp-spot leaves one: task declarations live in the
    ``{"_meta": true}`` header of metrics.jsonl, NOT in config.json."""
    run_dir = root / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text("{}", encoding="utf-8")
    records = [{"_meta": True, "task_definitions": definitions}, *epochs]
    (run_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return run_dir


def _epoch(epoch: int, harmonic: float, player: float) -> dict:
    return {
        "epoch": epoch,
        "tasks": {
            "action": {"validation": {"metrics": {"harmonic_mAP": harmonic}}},
            "actor": {"validation": {"metrics": {"player_top1": player}}},
        },
    }


FUSION_TASKS = {
    "action": {"primary_metric": "harmonic_mAP"},
    "actor": {"primary_metric": "player_top1"},
}


class BestEpochsPerTaskTest(unittest.TestCase):
    """Each task's best epoch is its own — one selection cannot serve all."""

    def test_tasks_peak_on_different_epochs(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            run_dir = _write_run(
                Path(raw_dir),
                [_epoch(0, 0.1, 0.6), _epoch(1, 0.3, 0.4), _epoch(2, 0.2, 0.5)],
                FUSION_TASKS,
            )
            best = best_epochs_per_task(run_dir)

        self.assertEqual(best["action"]["epoch"], 1)
        self.assertEqual(best["actor"]["epoch"], 0)
        # The winning epoch's full metrics ride along, so pickers never have
        # to re-read the metrics file to describe a package.
        self.assertEqual(best["actor"]["metrics"], {"player_top1": 0.6})

    def test_an_undeclared_or_unvalidated_task_is_absent(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            run_dir = _write_run(
                Path(raw_dir),
                [_epoch(0, 0.1, 0.6)],
                {"rally": {"primary_metric": "mAP"}},
            )
            self.assertEqual(best_epochs_per_task(run_dir), {})

    def test_a_run_without_task_definitions_selects_nothing(self):
        """The independent association trainer's config has no task
        definitions — the mechanism must be a no-op there, not a crash."""
        with tempfile.TemporaryDirectory() as raw_dir:
            run_dir = _write_run(Path(raw_dir), [_epoch(0, 0.1, 0.6)], {})
            self.assertEqual(best_epochs_per_task(run_dir), {})


class ExportBestPerTaskTest(unittest.TestCase):
    """Only serveable tasks earn a weights file, and the headline epoch's
    file is shared rather than duplicated."""

    def _export(self, root: Path) -> tuple[Path, dict]:
        run_dir = _write_run(
            root,
            [_epoch(0, 0.1, 0.6), _epoch(1, 0.3, 0.4)],
            FUSION_TASKS,
        )
        for name in ("checkpoint_000.pt", "checkpoint_001.pt"):
            (run_dir / name).write_bytes(name.encode())
        (run_dir / "checkpoint_best.pt").write_bytes(b"checkpoint_001.pt")
        (run_dir / "checkpoint_best.json").write_text(
            json.dumps({"epoch": 1, "metric": "val_mAP", "value": 0.3}),
            encoding="utf-8",
        )
        package_dir = root / "checkpoints" / "run"
        summary = export_checkpoint_package(
            run_dir=run_dir,
            package_dir=package_dir,
            checkpoints_root=root / "checkpoints",
            package_type="actor-association-spot",
            label_subdir="action-annotations",
            label_glob="*_actions.jsonl",
            training={},
            cmd=[],
            serveable_tasks=("action", "actor"),
        )
        return package_dir, summary

    def test_actor_best_gets_its_own_file_action_shares_the_headline(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            package_dir, summary = self._export(Path(raw_dir))
            per_task = summary["best_per_task"]

            self.assertEqual(per_task["action"]["file"], "checkpoint_best.pt")
            self.assertEqual(
                per_task["actor"]["file"], "checkpoint_best_actor.pt"
            )
            # The actor file IS epoch 0's weights, not another copy of best.
            self.assertEqual(
                (package_dir / "checkpoint_best_actor.pt").read_bytes(),
                b"checkpoint_000.pt",
            )
            manifest = json.loads(
                (package_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["best_per_task"], per_task)

    def test_picker_label_names_every_serveable_best(self):
        """A label showing only the headline metric misdescribes every other
        task — the actor head's quality is not the action mAP."""
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            self._export(root)
            [option] = checkpoint_package_options(
                root / "checkpoints",
                package_types=("actor-association-spot",),
            )
            self.assertEqual(
                option["label"], "run (action mAP 0.300 · actor Top-1 0.600)"
            )


if __name__ == "__main__":
    unittest.main()
