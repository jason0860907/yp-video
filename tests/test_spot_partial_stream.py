"""Progressive SPOT_PARTIAL streaming: payload parsing, the per-task reader
semantics, and normalization parity between the partial path and the final
annotation."""

import sys
import unittest
from pathlib import Path
from unittest import mock

from yp_video.action import predict, prelabel
from yp_video.action.predict import _spot_partial_payload


def _stub_command(lines: list[str], save_dir: Path, tasks: tuple[str, ...]) -> list[str]:
    """A subprocess that prints the given stdout lines and writes an empty
    predictions.json per task where run_spot_inference expects them."""
    body = ["import pathlib"]
    for task in tasks:
        pred_file = Path(save_dir) / task / "predictions.json"
        body.append(f"pathlib.Path({str(pred_file.parent)!r}).mkdir(parents=True, exist_ok=True)")
    body += [f"print({line!r})" for line in lines]
    for task in tasks:
        pred_file = Path(save_dir) / task / "predictions.json"
        body.append(f"pathlib.Path({str(pred_file)!r}).write_text('[]')")
    return [sys.executable, "-c", "\n".join(body)]


class SpotPartialPayloadTests(unittest.TestCase):
    def test_delta_line_parses(self):
        line = 'SPOT_PARTIAL {"task":"rally","cumulative":false,"events":[{"frame":3,"score":0.9}]}'
        self.assertEqual(
            _spot_partial_payload(line), ("rally", False, [{"frame": 3, "score": 0.9}])
        )

    def test_cumulative_line_parses(self):
        line = (
            'SPOT_PARTIAL {"task":"action","cumulative":true,'
            '"events":[{"frame":3,"label":"spike","score":0.9}]}'
        )
        task, cumulative, events = _spot_partial_payload(line)
        self.assertEqual(task, "action")
        self.assertTrue(cumulative)
        self.assertEqual(events, [{"frame": 3, "label": "spike", "score": 0.9}])

    def test_prefix_glued_to_a_tqdm_fragment_still_parses(self):
        line = ' 25%|██ | 1/4 [00:09<00:27]SPOT_PARTIAL {"task":"rally","cumulative":false,"events":[]}'
        self.assertEqual(_spot_partial_payload(line), ("rally", False, []))

    def test_non_partial_line_is_none(self):
        self.assertIsNone(_spot_partial_payload("Timing video=x frames=1"))

    def test_malformed_or_taskless_payload_is_none(self):
        self.assertIsNone(_spot_partial_payload("SPOT_PARTIAL {oops"))
        self.assertIsNone(_spot_partial_payload('SPOT_PARTIAL {"cumulative":false,"events":[]}'))


class SpotPartialReaderTests(unittest.TestCase):
    """run_spot_inference's stdout reader keeps one cumulative list per head:
    deltas accumulate, cumulative payloads replace, and heads never mix."""

    def _run(self, lines: list[str], tasks=("action",)) -> list[tuple[str, list[dict]]]:
        seen: list[tuple[str, list[dict]]] = []

        def fake_build_command(**kwargs):
            return _stub_command(lines, kwargs["save_dir"], tuple(kwargs["tasks"]))

        with (
            mock.patch.object(prelabel, "spot_available", return_value=True),
            mock.patch.object(prelabel, "build_command", side_effect=fake_build_command),
            mock.patch.object(predict, "SPOT_DIR", Path.cwd()),
        ):
            predict.run_spot_inference(
                Path("video.mp4"),
                checkpoint=Path("ckpt.pt"),
                tasks=tasks,
                on_events=lambda task, events: seen.append((task, list(events))),
            )
        return seen

    def test_delta_lines_accumulate(self):
        seen = self._run([
            'SPOT_PARTIAL {"task":"rally","cumulative":false,"events":[{"frame":1,"score":0.9}]}',
            'SPOT_PARTIAL {"task":"rally","cumulative":false,"events":[{"frame":2,"score":0.8}]}',
        ], tasks=("rally",))
        self.assertEqual([(t, [e["frame"] for e in s]) for t, s in seen], [("rally", [1]), ("rally", [1, 2])])

    def test_cumulative_lines_replace(self):
        seen = self._run([
            'SPOT_PARTIAL {"task":"action","cumulative":true,'
            '"events":[{"frame":1,"label":"spike","score":0.9}]}',
            'SPOT_PARTIAL {"task":"action","cumulative":true,'
            '"events":[{"frame":1,"label":"spike","score":0.9},'
            '{"frame":9,"label":"score","score":0.7}]}',
        ])
        self.assertEqual([[e["frame"] for e in s] for _, s in seen], [[1], [1, 9]])

    def test_joint_pass_keeps_heads_apart(self):
        seen = self._run([
            'SPOT_PARTIAL {"task":"rally","cumulative":false,"events":[{"frame":6,"score":0.9}]}',
            'SPOT_PARTIAL {"task":"action","cumulative":true,"events":[{"frame":4,"label":"serve","score":0.9}]}',
            'SPOT_PARTIAL {"task":"rally","cumulative":false,"events":[{"frame":12,"score":0.9}]}',
            'SPOT_PARTIAL {"task":"action","cumulative":true,"events":[{"frame":4,"label":"serve","score":0.9},{"frame":40,"label":"receive","score":0.8}]}',
        ], tasks=("rally", "action"))
        self.assertEqual(
            [(t, [e["frame"] for e in s]) for t, s in seen],
            [("rally", [6]), ("action", [4]), ("rally", [6, 12]), ("action", [4, 40])],
        )


class NormalizeEventParityTests(unittest.TestCase):
    def test_partial_normalization_matches_final_annotation(self):
        raw = [
            {"label": "SPIKE", "frame": 7, "score": 0.9, "xy": [0.2, 1.4]},
            {"label": "not-a-label", "frame": 8, "score": 0.9},
            {"label": "score", "frame": 9, "score": 0.05},
            {"label": "serve", "frame": 999, "score": 0.8, "visible": False},
        ]
        final = prelabel.predictions_to_annotation(
            [{"video": "v", "events": raw}],
            video_path=Path("v.mp4"),
            metadata={"fps": 30.0, "num_frames": 100},
            checkpoint_path=Path("ckpt.pt"),
            min_score=0.1,
        )["events"]
        partial = [
            item
            for item in (
                prelabel.normalize_event(ev, num_frames=100, min_score=0.1)
                for ev in raw
            )
            if item is not None
        ]
        self.assertEqual(final, sorted(partial, key=lambda e: (e["frame"], e["label"])))


if __name__ == "__main__":
    unittest.main()
