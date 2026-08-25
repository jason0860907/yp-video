import json
import tempfile
import unittest
from pathlib import Path

from yp_video.action import prelabel
from yp_video.contracts.action import ASSOCIATION_PACKAGE_TYPE, SPOT_PACKAGE_TYPE


def _make_package(root: Path, name: str, manifest: dict | None, files=("checkpoint_best.pt",)) -> None:
    package = root / name
    package.mkdir()
    for file in files:
        (package / file).write_bytes(b"")
    if manifest is not None:
        (package / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class ListCheckpointsByTaskTest(unittest.TestCase):
    def test_rows_follow_the_manifest_task_list_and_per_task_best_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_package(
                root, "fusion",
                {
                    "type": SPOT_PACKAGE_TYPE,
                    "tasks": ["action", "location", "actor"],
                    "best": {"epoch": 5, "metric": "val_mAP", "value": 0.3},
                    "best_per_task": {
                        "action": {"epoch": 5, "file": "checkpoint_best.pt"},
                        "actor": {"epoch": 2, "file": "checkpoint_best_actor.pt", "metric": "player_top1", "value": 0.6},
                    },
                },
                files=("checkpoint_best.pt", "checkpoint_best_actor.pt"),
            )
            _make_package(root, "rally", {"type": SPOT_PACKAGE_TYPE, "tasks": ["rally", "winner"], "best": {"epoch": 1}})
            _make_package(root, "independent", {"type": ASSOCIATION_PACKAGE_TYPE})
            _make_package(root, "legacy", None)

            action = {row["experiment"]: row for row in prelabel.list_checkpoints(root, task="action")}
            actor = {row["experiment"]: row for row in prelabel.list_checkpoints(root, task="actor")}
            rally = {row["experiment"]: row for row in prelabel.list_checkpoints(root, task="rally")}
            independent = prelabel.list_checkpoints(root, package_type=ASSOCIATION_PACKAGE_TYPE)

        self.assertEqual(set(action), {"fusion"})
        self.assertEqual(action["fusion"]["name"], "fusion/checkpoint_best.pt")
        self.assertEqual(action["fusion"]["tasks"], ["action", "location", "actor"])
        # The actor row points at the actor-best weights, not the headline.
        self.assertEqual(actor["fusion"]["name"], "fusion/checkpoint_best_actor.pt")
        self.assertEqual(actor["fusion"]["epoch"], 2)
        self.assertEqual(set(rally), {"rally"})
        self.assertEqual([row["experiment"] for row in independent], ["independent"])


if __name__ == "__main__":
    unittest.main()
