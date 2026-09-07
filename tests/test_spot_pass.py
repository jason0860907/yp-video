"""One SPOT pass: rallies merged, actions cut to their spans, progressive
callbacks per head, and the actions jsonl round trip — with the yp-spot
subprocess replaced by a fake."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from yp_video.action import spot_pass
from yp_video.action.spot_pass import (
    RallyOptions,
    SpotOptions,
    run_spot_pass,
    write_actions_jsonl,
)
from yp_video.core.jsonl import read_jsonl

FPS = 30.0
META = {"fps": FPS, "num_frames": 3000, "duration": 100.0}
RALLY = RallyOptions(min_score=0.5, max_gap_s=2.0, min_duration_s=4.0)
SPOT = SpotOptions(batch_size=4, num_workers=0, clip_len=64)


def _rally_events(start_s: float, end_s: float, step: int = 6) -> list[dict]:
    return [
        {"frame": f, "score": 0.9}
        for f in range(int(start_s * FPS), int(end_s * FPS) + 1, step)
    ]


# One rally 10–20 s, another 40–50 s (both longer than min_duration_s).
RALLY_RECORDS = [{"video": "v", "events": _rally_events(10, 20) + _rally_events(40, 50)}]
ACTION_RECORDS = [{"video": "v", "events": [
    {"frame": 330, "label": "serve", "score": 0.9},    # 11 s, inside rally 1
    {"frame": 900, "label": "spike", "score": 0.9},    # 30 s, dead time
    {"frame": 1230, "label": "set", "score": 0.9},     # 41 s, inside rally 2
    {"frame": 1545, "label": "score", "score": 0.9},   # 51.5 s, inside the 2 s pad
]}]


def _fake_inference(*, on_events_script=None, predictions=None):
    """A stand-in for run_spot_inference that records its kwargs, replays a
    scripted partial stream, then returns the given predictions."""
    calls: list[dict] = []

    def fake(source, **kwargs):
        calls.append(kwargs)
        if kwargs.get("on_events") is not None:
            for task, events in on_events_script or []:
                kwargs["on_events"](task, events)
        return {task: (predictions or {}).get(task, []) for task in kwargs["tasks"]}

    return fake, calls


class JointPassTests(unittest.TestCase):
    def _run(self, **overrides):
        fake, calls = _fake_inference(
            predictions={"rally": RALLY_RECORDS, "action": ACTION_RECORDS},
            on_events_script=overrides.pop("script", None),
        )
        with (
            mock.patch.object(spot_pass, "run_spot_inference", side_effect=fake),
            mock.patch.object(spot_pass, "probe_video_metadata", return_value=META),
        ):
            result = run_spot_pass(
                "v.mp4", checkpoint=Path("ckpt.pt"), tasks=("rally", "action"),
                rally=RALLY, spot=SPOT, rally_pad_s=2.0, **overrides,
            )
        return result, calls

    def test_rallies_merge_and_actions_are_cut_to_padded_spans(self):
        result, calls = self._run()
        self.assertEqual([(round(r["start"]), round(r["end"])) for r in result.rallies], [(10, 20), (40, 50)])
        self.assertEqual([e["frame"] for e in result.actions[0]["events"]], [330, 1230, 1545])
        # A joint pass decodes the whole video — no segments handed to yp-spot.
        self.assertIsNone(calls[0]["segments"])
        self.assertEqual(calls[0]["tasks"], ("rally", "action"))
        self.assertEqual((result.fps, result.num_frames, result.duration_s), (FPS, 3000, 100.0))

    def test_progressive_callbacks_are_per_head(self):
        rallies_seen, actions_seen = [], []
        script = [
            ("rally", _rally_events(10, 20)),                          # one rally: held back
            ("action", [{"frame": 330, "label": "serve", "score": 0.9},
                        {"frame": 900, "label": "spike", "score": 0.9}]),
            ("rally", _rally_events(10, 20) + _rally_events(40, 50)),  # two: first one settles
        ]
        self._run(script=script, on_rallies=rallies_seen.append, on_action_events=actions_seen.append)
        self.assertEqual([[round(r["start"]) for r in s] for s in rallies_seen], [[10]])
        # The 30 s spike lies outside the only rally known at that moment.
        self.assertEqual(len(actions_seen), 1)
        self.assertEqual([e["frame"] for e in actions_seen[0]], [330])
        self.assertAlmostEqual(actions_seen[0][0]["time"], 11.0)
        self.assertEqual(actions_seen[0][0]["label"], "serve")

    def test_actions_jsonl_round_trips(self):
        result, _ = self._run()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "v_actions.jsonl"
            data = write_actions_jsonl(result, out, video_path=Path("v.mp4"))
            meta, events = read_jsonl(out)
        self.assertEqual(meta["fps"], FPS)
        self.assertEqual(meta["num_frames"], 3000)
        self.assertEqual([e["frame"] for e in events], [330, 1230, 1545])
        self.assertEqual(data["num_events"], 3)


class ActionOnlyPassTests(unittest.TestCase):
    def test_known_rallies_bound_the_decode_and_actions_stay_uncut(self):
        fake, calls = _fake_inference(predictions={"action": ACTION_RECORDS})
        with (
            mock.patch.object(spot_pass, "run_spot_inference", side_effect=fake),
            mock.patch.object(spot_pass, "probe_video_metadata", return_value=META),
        ):
            result = run_spot_pass(
                "v.mp4", checkpoint=Path("ckpt.pt"), tasks=("action",),
                rally=RALLY, spot=SPOT, rally_pad_s=2.0,
                rallies=[{"start": 10.0, "end": 20.0}, {"start": 40.0, "end": 50.0}],
            )
        self.assertEqual(calls[0]["segments"], [(8.0, 22.0), (38.0, 52.0)])
        self.assertIsNone(result.rallies)
        # yp-spot only decoded inside the spans; nothing to cut afterwards.
        self.assertEqual(len(result.actions[0]["events"]), 4)

    def test_invalid_task_sets_are_rejected(self):
        with mock.patch.object(spot_pass, "probe_video_metadata", return_value=META):
            with self.assertRaises(ValueError):
                run_spot_pass("v.mp4", checkpoint=Path("c.pt"), tasks=(), rally=RALLY, spot=SPOT, rally_pad_s=2.0)
            with self.assertRaises(ValueError):
                run_spot_pass("v.mp4", checkpoint=Path("c.pt"), tasks=("winner",), rally=RALLY, spot=SPOT, rally_pad_s=2.0)
            with self.assertRaises(ValueError):
                run_spot_pass(
                    "v.mp4", checkpoint=Path("c.pt"), tasks=("rally", "action"),
                    rally=RALLY, spot=SPOT, rally_pad_s=2.0, rallies=[{"start": 1.0, "end": 9.0}],
                )


if __name__ == "__main__":
    unittest.main()
