"""Progressive SPOT_PARTIAL streaming: payload parsing, reader semantics,
and normalization parity between the partial path and the final JSONL."""

import sys
import unittest
from pathlib import Path
from unittest import mock

from yp_video.action import predict, prelabel
from yp_video.action.predict import _spot_partial_payload


def _stub_command(lines: list[str], save_dir: Path) -> list[str]:
    """A subprocess that prints the given stdout lines and writes an empty
    predictions.json where run_spot_inference expects it."""
    pred_file = str(Path(save_dir) / "predictions.json")
    body = "\n".join(
        ["import pathlib"]
        + [f"print({line!r})" for line in lines]
        + [f"pathlib.Path({pred_file!r}).write_text('[]')"]
    )
    return [sys.executable, "-c", body]


class SpotPartialPayloadTests(unittest.TestCase):
    def test_delta_line_parses(self):
        line = 'SPOT_PARTIAL {"cumulative":false,"events":[{"frame":3,"score":0.9}]}'
        self.assertEqual(
            _spot_partial_payload(line), (False, [{"frame": 3, "score": 0.9}])
        )

    def test_cumulative_line_parses(self):
        line = (
            'SPOT_PARTIAL {"cumulative":true,'
            '"events":[{"frame":3,"label":"spike","score":0.9}]}'
        )
        cumulative, events = _spot_partial_payload(line)
        self.assertTrue(cumulative)
        self.assertEqual(events, [{"frame": 3, "label": "spike", "score": 0.9}])

    def test_non_partial_line_is_none(self):
        self.assertIsNone(_spot_partial_payload("Timing video=x frames=1"))

    def test_malformed_payload_is_empty_delta(self):
        self.assertEqual(_spot_partial_payload("SPOT_PARTIAL {oops"), (False, []))


class SpotPartialReaderTests(unittest.TestCase):
    """run_spot_inference's stdout reader: deltas accumulate, cumulative
    payloads replace wholesale — always handing the callback the full list."""

    def _run(self, lines: list[str]) -> list[list[dict]]:
        seen: list[list[dict]] = []

        def fake_build_command(**kwargs):
            return _stub_command(lines, kwargs["save_dir"])

        with (
            mock.patch.object(prelabel, "spot_available", return_value=True),
            mock.patch.object(prelabel, "build_command", side_effect=fake_build_command),
            mock.patch.object(predict, "SPOT_DIR", Path.cwd()),
        ):
            predict.run_spot_inference(
                Path("video.mp4"),
                checkpoint=Path("ckpt.pt"),
                task="action",
                on_events=lambda events: seen.append(list(events)),
            )
        return seen

    def test_delta_lines_accumulate(self):
        seen = self._run([
            'SPOT_PARTIAL {"cumulative":false,"events":[{"frame":1,"score":0.9}]}',
            'SPOT_PARTIAL {"cumulative":false,"events":[{"frame":2,"score":0.8}]}',
        ])
        self.assertEqual([[e["frame"] for e in s] for s in seen], [[1], [1, 2]])

    def test_cumulative_lines_replace(self):
        seen = self._run([
            'SPOT_PARTIAL {"cumulative":true,'
            '"events":[{"frame":1,"label":"spike","score":0.9}]}',
            'SPOT_PARTIAL {"cumulative":true,'
            '"events":[{"frame":1,"label":"spike","score":0.9},'
            '{"frame":9,"label":"score","score":0.7}]}',
        ])
        self.assertEqual([[e["frame"] for e in s] for s in seen], [[1], [1, 9]])


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
