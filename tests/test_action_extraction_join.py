from __future__ import annotations

import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from yp_video.actor import labels as actor_labels
from yp_video.actor.labels import ActorLabel, ActorVerdict
from yp_video.core.cache import StatCache
from yp_video.core.jsonl import write_jsonl
from yp_video.extraction import store
from yp_video.web import worklists


class CurrentActionJoinTests(unittest.TestCase):
    """Extraction is detector output; Action owns event meaning."""

    def test_current_action_replaces_stale_event_fields(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            action = Path(raw_dir) / "match_actions.jsonl"
            write_jsonl(
                action,
                {"fps": 30},
                [
                    {
                        "id": "keep",
                        "frame": 30,
                        "time": 1.0,
                        "label": "set",
                        "xy": [0.2, 0.3],
                        "visible": False,
                    },
                    {
                        "id": "now-score",
                        "frame": 60,
                        "time": 2.0,
                        "label": "score",
                        "xy": [0.4, 0.5],
                        "visible": False,
                    },
                ],
            )
            stale = [
                {
                    "id": "keep",
                    "frame": 30,
                    "label": "spike",
                    "xy": [0.1, 0.1],
                    "visible": True,
                    "detections": [{"box": [1, 2, 3, 4]}],
                },
                {
                    "id": "now-score",
                    "frame": 60,
                    "label": "spike",
                },
                {"id": "deleted", "frame": 90, "label": "serve"},
            ]

            with (
                patch.object(store, "action_annotation_path", return_value=action),
                patch.object(
                    store,
                    "load_rallies",
                    return_value=[{"start": 0.0, "end": 10.0}],
                ),
            ):
                records = store.labelable(stale, "match", 30.0)

        self.assertEqual([record["id"] for record in records], ["keep"])
        self.assertEqual(records[0]["label"], "set")
        # Rally-derived fields belong to the live rally store, not extraction.
        self.assertNotIn("time", records[0])
        self.assertEqual(records[0]["xy"], [0.2, 0.3])
        self.assertFalse(records[0]["visible"])
        self.assertEqual(
            records[0]["detections"], [{"box": [1, 2, 3, 4]}]
        )

    def test_relabel_is_visible_without_rewriting_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            action = Path(raw_dir) / "match_actions.jsonl"
            stored = [{"id": "event", "frame": 30, "label": "spike"}]
            with (
                patch.object(store, "action_annotation_path", return_value=action),
                patch.object(
                    store,
                    "load_rallies",
                    return_value=[{"start": 0.0, "end": 10.0}],
                ),
            ):
                write_jsonl(
                    action,
                    {"fps": 30},
                    [{"id": "event", "frame": 30, "label": "spike"}],
                )
                self.assertEqual(
                    [record["id"] for record in store.labelable(stored, "match", 30)],
                    ["event"],
                )

                # Only Action is rewritten. The derived extraction row remains
                # byte-for-byte stale and must immediately stop being work.
                write_jsonl(
                    action,
                    {"fps": 30},
                    [{"id": "event", "frame": 30, "label": "score"}],
                )
                self.assertEqual(store.labelable(stored, "match", 30), [])


class AssociationProgressTests(unittest.TestCase):
    def _listing(self, root: Path, video: Path, action: Path, *, dense: bool = True, rallies=("annotation",)):
        dense_file = root / "match_dense.npz"
        if dense:
            dense_file.touch()
        return (
            patch.object(worklists, "all_cut_paths", return_value=[video]),
            patch.object(worklists, "dense_path", return_value=dense_file),
            patch.object(worklists, "rally_sources", return_value=list(rallies)),
            patch.object(store, "action_annotation_path", return_value=action),
            patch.object(store, "load_rallies", return_value=[{"start": 0.0, "end": 10.0}]),
            patch.object(actor_labels, "actors_path", return_value=root / "match_actors.json"),
            patch.object(actor_labels._store, "_cache", StatCache()),
        )

    def test_progress_counts_only_current_labelable_action_ids(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir, ExitStack() as stack:
            root = Path(raw_dir)
            video = root / "match.mp4"
            video.touch()
            action = root / "match_actions.jsonl"
            write_jsonl(
                action,
                {"fps": 30},
                [
                    {"id": "reviewed", "frame": 30, "time": 1.0, "label": "set"},
                    {"id": "pending", "frame": 60, "time": 2.0, "label": "set"},
                    {"id": "now-score", "frame": 90, "time": 3.0, "label": "score"},
                    {"id": "outside", "frame": 900, "time": 30.0, "label": "serve"},
                ],
            )
            for p in self._listing(root, video, action):
                stack.enter_context(p)
            actor_labels.save(
                "match", "reviewed", ActorLabel(ActorVerdict.CONFIRMED, 30, (1, 2, 3, 4))
            )
            # Old verdicts are durable, but do not count toward current
            # progress or its verdict breakdown.
            actor_labels.save("match", "now-score", ActorLabel(ActorVerdict.OCCLUDED))
            result = worklists.association_videos()[0]

        self.assertEqual(result["event_count"], 2)
        self.assertEqual(result["reviewed"], 1)
        self.assertEqual(result["unreviewed"], 1)
        self.assertEqual(result["verdicts"], {"confirmed": 1})

    def test_a_video_missing_an_association_input_is_not_listed(self) -> None:
        """Actions own the events, rallies which are in play, and the dense
        pass the boxes a pick chooses among — without any of them a row would
        offer work that cannot be done here. Player Detection is not one: a
        video it never ran on is listed."""
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            video = root / "match.mp4"
            video.touch()
            action = root / "match_actions.jsonl"
            write_jsonl(action, {"fps": 30}, [{"id": "a", "frame": 30, "label": "set"}])

            def listed(**kwargs) -> list[dict]:
                with ExitStack() as stack:
                    for p in self._listing(root, video, kwargs.pop("action", action), **kwargs):
                        stack.enter_context(p)
                    return worklists.association_videos()

            # No records file anywhere in the fixture, and it is listed.
            self.assertEqual(len(listed()), 1)
            with self.subTest(missing="rallies"):
                self.assertEqual(listed(rallies=()), [])
            with self.subTest(missing="action"):
                self.assertEqual(listed(action=None), [])
            with self.subTest(missing="dense"):
                (root / "match_dense.npz").unlink()
                self.assertEqual(listed(dense=False), [])
