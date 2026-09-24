"""Clip framing held to the App's answer.

The fixture lives with the App (``VolleyIQTests/Fixtures/clip_windows.json``)
and ``ClipWindowFixtureTests`` runs it against ``Match+Actions``; this runs the
same file through the Python port, so a framing change on either side fails
one of them until the fixture is regenerated from the App's behaviour.
"""

import json
from pathlib import Path

from yp_video.web.app_review.models import Bundle
from yp_video.web.app_review.projection import project

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "VolleyIQ/VolleyIQTests/Fixtures/clip_windows.json"
)


def fixture_bundle(f: dict) -> Bundle:
    events = [{**e, "frame": round(e["time"] * 30)} for e in f["events"]]
    position = {e["time"]: i for i, e in enumerate(events)}
    indices = sorted({*map(int, f["whistles"]), *map(int, f["deciding"])})
    rallies = [{**r, "set": 1} for r in f["rallies"]]
    return Bundle.model_validate(
        {
            "result": {
                "job_id": "job",
                "user_id": "user",
                "match_id": "match",
                "video_r2_key": "",
                "total_duration": 100,
                "action_events": events,
                "rallies": rallies,
                "attacks": [
                    {"rally_index": int(r), "event_indices": [position[t] for t in a]}
                    for r, attacks in f["attacks"].items()
                    for a in attacks
                ],
                "rally_outcomes": [
                    {
                        "rally_index": i,
                        "serve_point": None,
                        "score_event_index": position.get(f["whistles"].get(str(i))),
                        "deciding_event_indices": [
                            position[t] for t in f["deciding"].get(str(i), [])
                        ],
                    }
                    for i in indices
                ],
            },
            "library_rallies": [{**r, "match_id": "match"} for r in rallies],
        }
    )


def framed(clips: list[dict]) -> list[dict]:
    return [
        {
            "key": c["key"],
            "start": c["start"],
            "end": c["end"],
            "markable": c["event_id"],
            "touches": [t["time"] for t in c["touches"]],
            "loss_reason": c["loss_reason"],
        }
        for c in clips
    ]


def actual(f: dict) -> dict:
    b = fixture_bundle(f)
    return {
        "actions": {
            mode: framed(project(b, mode=mode)["actions"])
            for mode in ("full_play", "to_next", "whole_rally")
        },
        "scores": framed(project(b)["scores"]),
    }


def test_python_projection_frames_clips_like_the_app():
    f = json.loads(FIXTURE.read_text())
    assert actual(f) == f["expected"]
