from yp_video.action.segments import event_timeline, filter_events_to_spans


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


def test_filter_events_to_spans_keeps_only_in_span_events() -> None:
    predictions = [{
        "video": "v",
        "events": [
            {"frame": 30, "label": "serve", "score": 0.9},   # 1.0 s, inside
            {"frame": 300, "label": "spike", "score": 0.9},  # 10.0 s, gap
            {"frame": 600, "label": "set", "score": 0.9},    # 20.0 s, boundary
        ],
    }]
    kept = filter_events_to_spans(predictions, [(0.0, 2.0), (20.0, 25.0)], fps=30.0)
    assert [ev["frame"] for ev in kept[0]["events"]] == [30, 600]
    assert kept[0]["video"] == "v"
