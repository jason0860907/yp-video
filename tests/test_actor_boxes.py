from yp_video.extraction import feet


def test_actor_boxes_normalize_the_contact_box_and_skip_unnamed_actors(monkeypatch):
    records = {
        "a": {"id": "a", "box": [192, 108, 384, 540]},
        "b": {"id": "b", "box": [0, 0, 10, 10], "resolution": "occluded"},
        "c": {"id": "c", "box": [0, 0, 10, 10], "resolution": "unresolved"},
        "d": {"id": "d"},
    }
    monkeypatch.setattr(feet, "_placeable", lambda stem: (1920, 1080, 30.0, records))
    monkeypatch.setattr(feet, "event_id", lambda event: event["id"])
    events = [{"id": eid, "frame": 1} for eid in ("a", "b", "c", "d", "e")] + [{"id": "f", "frame": None}]

    assert feet.actor_boxes("stem", events) == {"a": [0.1, 0.1, 0.2, 0.5]}
