"""The dataset export's split has to be honest about what it is measuring.

Crops are frames; a tracklet is a person. Splitting crops at random put two
adjacent frames of one tracklet on opposite sides of train/test — measured on
the real labeled set, 13 of 17 players and 16% of test crops — which makes
rank-1 retrieval "find the neighbouring frame of this crop" and the resulting
mAP a number about nothing. The split is therefore over tracklets.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from yp_video.reid import dataset
from yp_video.reid.sessions import SessionGroup


def _crops(unit: str, n: int, start: int = 0) -> list[dict]:
    return [
        {"id": f"{unit}-{i}", "path": f"crops/{unit}-{i}.jpg", "frame": start + i, "unit": unit}
        for i in range(n)
    ]


class SplitTests(unittest.TestCase):
    GROUP = SessionGroup(id="g0", stems=("match",), players=("A",))

    def _plan(self, crops: list[dict], **kw):
        with patch.object(dataset, "_candidates", return_value=({("g0", "A"): crops}, {})):
            return dataset.plan_export([self.GROUP], split_mode="crops", **kw)

    def _units(self, plan, split):
        by_id = {c["id"]: c["unit"] for c in self._all}
        return {by_id[s.id] for s in plan.samples if s.split == split}

    def test_no_tracklet_lands_on_both_sides(self) -> None:
        self._all = _crops("t1", 6) + _crops("t2", 6, 100) + _crops("t3", 6, 200)
        plan = self._plan(self._all)
        self.assertFalse(self._units(plan, "train") & self._units(plan, "test"))

    def test_query_and_gallery_come_from_different_tracklets(self) -> None:
        self._all = _crops("t1", 4) + _crops("t2", 4, 100) + _crops("t3", 4, 200)
        plan = self._plan(self._all)
        by_id = {c["id"]: c["unit"] for c in self._all}
        queries = {by_id[s.id] for s in plan.samples if s.role == "query"}
        galleries = {by_id[s.id] for s in plan.samples if s.role == "gallery"}
        self.assertTrue(queries)
        self.assertFalse(queries & galleries)

    def test_one_tracklet_cannot_be_split_however_many_crops(self) -> None:
        """20 frames of one person in one rally are one observation."""
        self._all = _crops("t1", 20)
        plan = self._plan(self._all)
        self.assertEqual(self._units(plan, "test"), set())
        self.assertIn("single_unit", plan.dropped)
        # Still worth training on — dropping it would throw away real labels.
        self.assertEqual(len(self._units(plan, "train")), 1)

    def test_two_tracklets_cannot_supply_both_test_roles(self) -> None:
        """One test tracklet is one observation: it can be the query or the
        gallery, not both. Rather than score that, the player trains."""
        self._all = _crops("t1", 4) + _crops("t2", 4, 100)
        plan = self._plan(self._all)
        self.assertEqual(self._units(plan, "test"), set())
        self.assertEqual(len(self._units(plan, "train")), 2)
        self.assertIn("test_single_unit", plan.dropped)

    def test_three_tracklets_are_enough_to_score(self) -> None:
        self._all = _crops("t1", 4) + _crops("t2", 4, 100) + _crops("t3", 4, 200)
        plan = self._plan(self._all)
        self.assertGreaterEqual(len(self._units(plan, "test")), 2)
        self.assertTrue(self._units(plan, "train"))

    def test_the_counts_report_tracklets_not_just_crops(self) -> None:
        self._all = _crops("t1", 4) + _crops("t2", 4, 100) + _crops("t3", 4, 200)
        plan = self._plan(self._all)
        self.assertEqual(plan.counts["n_units"], 3)


if __name__ == "__main__":
    unittest.main()
