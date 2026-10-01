"""Player names are stored per event, so a re-track cannot move them."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from yp_video.reid import identity, store


class ReidNamesTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "match_players.json"
        self._patch = patch.object(store, "players_path", return_value=self.path)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()

    def test_save_writes_only_event_names(self):
        store.save_players("match", {"e1": " A ", "e2": "", "e3": "B"})

        self.assertEqual(
            json.loads(self.path.read_text()),
            {"version": store.PLAYERS_SCHEMA_VERSION, "assignments": {"e1": "A", "e3": "B"}},
        )
        self.assertEqual(identity.load_assignments("match"), {"e1": "A", "e3": "B"})

    def test_names_survive_a_retrack(self):
        store.save_players("match", {"e1": "A", "e2": "A", "e3": "B"})
        records = [{"id": "e1"}, {"id": "e2"}, {"id": "e3"}, {"id": "e4"}]

        before = identity.build_units(records, {"e1": "1:1", "e2": "1:1", "e3": "1:2"})
        # The new run reuses id 1:1 for B's track and splits A across two.
        after = identity.build_units(records, {"e1": "1:3", "e2": "1:4", "e3": "1:1", "e4": "1:1"})
        assignments = identity.load_assignments("match")

        self.assertEqual(identity.unit_names(before, assignments), {"t:1:1": "A", "t:1:2": "B"})
        self.assertEqual(
            identity.unit_names(after, assignments),
            {"t:1:3": "A", "t:1:4": "A", "t:1:1": "B"},
        )

    def test_disagreeing_events_leave_the_unit_unnamed(self):
        units = identity.build_units([{"id": "e1"}, {"id": "e2"}], {"e1": "1:1", "e2": "1:1"})

        self.assertEqual(identity.unit_names(units, {"e1": "A", "e2": "B"}), {})


if __name__ == "__main__":
    unittest.main()
