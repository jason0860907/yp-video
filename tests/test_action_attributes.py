from yp_video.action.attributes import attribute_defaults, derive_sides

FPS = 10.0


def _rally(index: int, winner: str | None) -> dict:
    return {"rally_id": index + 1, "start": index * 10.0, "end": index * 10.0 + 5, "winner": winner}


def _serve_and_receive(index: int, serve_y: float) -> list[dict]:
    start = int(index * 10 * FPS)
    return [
        {"frame": start + 5, "label": "serve", "xy": [0.5, serve_y]},
        {"frame": start + 20, "label": "receive", "xy": [0.5, 0.5]},
    ]


def _match(winners: list[str], serve_ys: list[float]) -> tuple[list[dict], list[dict]]:
    """Rally i is served by rally i-1's winner from ``serve_ys[i]``."""
    rallies = [_rally(i, w) for i, w in enumerate(winners)]
    events = [e for i, y in enumerate(serve_ys) for e in _serve_and_receive(i, y)]
    return rallies, events


def test_previous_winner_serves_and_the_receive_is_across() -> None:
    rallies, events = _match(["near", "far", "far"], [0.9, 0.2, 0.8])

    sides, report = derive_sides(rallies, events, FPS)

    assert report["status"] == "ok"
    # Rally 1 has no previous winner; rally 2 near serves, rally 3 far.
    assert sides == [None, None, "near", "far", "far", "near"]


def test_serve_position_direction_is_learned_per_video() -> None:
    """Near serves struck HIGHER in the frame (a toss close to the camera)
    still split cleanly — only consistency matters."""
    rallies, events = _match(["near", "far", "far"], [0.5, 0.2, 0.8])

    sides, report = derive_sides(rallies, events, FPS)

    assert report["status"] == "ok"
    assert sides[2:] == ["near", "far", "far", "near"]


def test_a_serve_on_the_wrong_half_drops_only_its_rally() -> None:
    winners = ["near", "far"] * 6
    # Rally i is served by winners[i-1]: near serves sit low (y≈0.9), far
    # ones high (y≈0.1) — except rally 5, a near serve struck at 0.1.
    serve_ys = [0.5] + [0.9 if winners[i - 1] == "near" else 0.1 for i in range(1, 12)]
    serve_ys[5] = 0.1
    rallies, events = _match(winners, serve_ys)

    sides, report = derive_sides(rallies, events, FPS)

    assert report["status"] == "ok"
    assert report["kept_rallies"] == 10
    assert sides[10:12] == [None, None]
    assert sides[12:14] == ["far", "near"]


def test_a_video_whose_serves_disagree_supervises_nothing() -> None:
    rallies, events = _match(["near", "far", "near", "far"], [0.5, 0.9, 0.9, 0.1])

    sides, report = derive_sides(rallies, events, FPS)

    assert report["status"] == "serve positions disagree with winners"
    assert sides == [None] * len(events)


def test_no_winner_labels_no_sides() -> None:
    rallies, events = _match([None, None], [0.5, 0.5])

    assert derive_sides(rallies, events, FPS) == ([None] * 4, {"status": "no winner labels"})


def test_jump_defaults_cover_spike_block_receive_only() -> None:
    events = [
        {"frame": i, "label": label, "xy": [0.5, 0.5]}
        for i, label in enumerate(["serve", "receive", "set", "spike", "block", "score"])
    ]

    defaults, _ = attribute_defaults([], events, FPS)

    assert [d["jump"] for d in defaults] == [None, False, None, True, True, None]
