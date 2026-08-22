from yp_video.action.segments import event_timeline


def test_public_events_keep_the_stable_extraction_id() -> None:
    """The id is the join key player identification points back at, so it has
    to survive the frame → seconds projection unchanged."""
    events = [
        {"frame": 30, "label": "receive", "xy": [0.2, 0.7]},
        {"frame": 60, "label": "set", "xy": [0.4, 0.5]},
        {"frame": 90, "label": "spike", "xy": [0.7, 0.3]},
    ]

    timeline = event_timeline(events, fps=30.0)

    assert [event["id"] for event in timeline] == ["f30", "f60", "f90"]
    assert timeline[0]["frame"] == 30
    assert [event["time"] for event in timeline] == [1.0, 2.0, 3.0]


def test_timeline_carries_every_label_and_sorts_by_time() -> None:
    """The timeline is the whole action payload — a client that can only see
    spikes cannot address a set or a receive as a clip."""
    events = [
        {"frame": 90, "label": "spike"},
        {"frame": 30, "label": "serve"},
        {"frame": 120, "label": "score"},
        {"frame": 60, "label": "receive"},
        {"frame": 75, "label": "block"},
    ]

    timeline = event_timeline(events, fps=30.0)

    assert [event["label"] for event in timeline] == [
        "serve", "receive", "block", "spike", "score",
    ]
