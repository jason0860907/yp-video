from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from yp_video.core.cache import StatCache
from yp_video.core.jsonl import write_jsonl
from yp_video.tracklets.store import load_tracklets


class StatCacheBudgetTests(unittest.TestCase):
    def test_budget_evicts_least_recently_used_entry(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            first = root / "first"
            second = root / "second"
            first.write_bytes(b"aaa")
            second.write_bytes(b"bbb")
            cache = StatCache(max_source_bytes=5)
            calls = {"first": 0, "second": 0}

            def compute(name: str) -> str:
                calls[name] += 1
                return name

            self.assertEqual(cache.get("first", [first], lambda: compute("first")), "first")
            self.assertEqual(cache.get("second", [second], lambda: compute("second")), "second")
            self.assertEqual(cache.get("second", [second], lambda: compute("second")), "second")
            self.assertEqual(cache.get("first", [first], lambda: compute("first")), "first")
            self.assertEqual(calls, {"first": 2, "second": 1})

    def test_rewrite_replaces_weight_without_leaking_budget(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "source"
            path.write_bytes(b"aaa")
            cache = StatCache(max_source_bytes=4)
            self.assertEqual(cache.get("key", [path], lambda: 1), 1)
            path.write_bytes(b"bbbb")
            self.assertEqual(cache.get("key", [path], lambda: 2), 2)
            self.assertEqual(cache._source_bytes, 4)
            self.assertEqual(len(cache._entries), 1)

    def test_oversized_entry_remains_usable_and_displaces_others(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            small = root / "small"
            large = root / "large"
            small.write_bytes(b"a")
            large.write_bytes(b"0123456789")
            cache = StatCache(max_source_bytes=4)
            cache.get("small", [small], lambda: "small")
            self.assertEqual(cache.get("large", [large], lambda: "large"), "large")
            self.assertEqual(list(cache._entries), ["large"])
            self.assertEqual(cache.get("large", [large], lambda: "wrong"), "large")

    def test_evicts_before_computing_a_value_that_would_exceed_budget(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            first = root / "first"
            second = root / "second"
            first.write_bytes(b"aaa")
            second.write_bytes(b"bbb")
            cache = StatCache(max_source_bytes=5)
            cache.get("first", [first], lambda: "first")

            def compute_second() -> str:
                self.assertEqual(len(cache._entries), 0)
                return "second"

            self.assertEqual(cache.get("second", [second], compute_second), "second")


class TrackletCacheTests(unittest.TestCase):
    def test_records_and_index_share_one_stat_keyed_entry(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "match_tracks.jsonl"
            write_jsonl(
                path,
                {"stride": 2},
                [{
                    "rally_id": 1,
                    "track_id": 7,
                    "frames": [10],
                    "boxes": [[1, 2, 3, 4]],
                    "scores": [0.9],
                }],
            )
            first = load_tracklets(path)
            second = load_tracklets(path)
            self.assertIs(first, second)
            self.assertIs(first.index.tracklet((1, 7)), first.records[0])
            self.assertEqual(first.meta["stride"], 2)

            write_jsonl(path, {"stride": 1}, [])
            replaced = load_tracklets(path)
            self.assertIsNot(replaced, first)
            self.assertEqual(replaced.meta["stride"], 1)
            self.assertEqual(len(replaced.index), 0)


if __name__ == "__main__":
    unittest.main()
