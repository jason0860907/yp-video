import json
import tempfile
import unittest
from pathlib import Path

from yp_video.action import prelabel


def _make_package(root: Path, name: str, config: dict | None) -> None:
    package = root / name
    package.mkdir()
    (package / "checkpoint_best.pt").write_bytes(b"")
    if config is not None:
        (package / "config.json").write_text(json.dumps(config), encoding="utf-8")


class ListCheckpointsActorHeadTest(unittest.TestCase):
    def test_predicts_actor_reflects_the_package_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_package(root, "yp_fusion_run", {"predict_actor": True})
            _make_package(root, "yp_action_run", {"predict_actor": False})
            _make_package(root, "legacy_run", None)

            by_experiment = {
                row["experiment"]: row["predicts_actor"]
                for row in prelabel.list_checkpoints(root)
            }

        self.assertEqual(by_experiment, {
            "yp_fusion_run": True,
            "yp_action_run": False,
            "legacy_run": False,
        })


if __name__ == "__main__":
    unittest.main()
