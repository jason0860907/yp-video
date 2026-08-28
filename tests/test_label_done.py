import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from yp_video.core import label_done


class LedgerTest(unittest.TestCase):
    def test_flags_round_trip_in_one_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "label-done.jsonl"
            with patch.object(label_done, "ledger", label_done.Ledger(path)):
                self.assertFalse(label_done.is_done("a", "rally"))
                self.assertEqual(label_done.set_done("a", "rally", True)["rally"], True)
                label_done.set_done("b", "reid", True)
                label_done.set_done("a", "action", True)
                self.assertEqual(label_done.load("a"), {"rally": True, "action": True, "association": False, "reid": False})
                self.assertTrue(label_done.is_done("b", "reid"))
                # Unsetting the last flag drops the line; the file stays one ledger.
                label_done.set_done("b", "reid", False)
                self.assertNotIn('"b"', path.read_text())
                self.assertEqual(path.read_text().count("\n"), 2)  # meta + "a"
                with self.assertRaises(ValueError):
                    label_done.set_done("a", "bogus", True)


if __name__ == "__main__":
    unittest.main()


class OnWriteHookTest(unittest.TestCase):
    def test_hook_fires_after_each_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            seen = []
            ledger = label_done.Ledger(Path(tmp) / "label-done.jsonl", on_write=seen.append)
            ledger.set("a", "rally", True)
            ledger.set("a", "rally", False)
            self.assertEqual(seen, [ledger.path, ledger.path])
