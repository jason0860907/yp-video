from yp_video.action.segments import build_action_segments, event_timeline


def test_public_events_keep_the_stable_extraction_id() -> None:
    events = [
        {"frame": 30, "label": "receive", "xy": [0.2, 0.7]},
        {"frame": 60, "label": "set", "xy": [0.4, 0.5]},
        {"frame": 90, "label": "spike", "xy": [0.7, 0.3]},
    ]
    rallies = [{"start": 0.0, "end": 4.0}]

    timeline = event_timeline(events, fps=30.0)
    segment = build_action_segments(events, rallies, fps=30.0)[0]

    assert [event["id"] for event in timeline] == ["f30", "f60", "f90"]
    assert timeline[0]["frame"] == 30
    assert segment["anchor"]["id"] == "f90"
    assert [event["id"] for event in segment["chain"]] == ["f30", "f60", "f90"]
