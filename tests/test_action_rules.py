from yp_video.action.rules import attacks, rally_outcomes


def _events(*touches: tuple[str, float]) -> list[dict]:
    return [{"id": f"f{round(t * 30)}", "label": label, "time": t} for label, t in touches]


RALLY = [{"index": 1, "start": 0.0, "end": 10.0}]


def test_attack_counts_a_spike_even_when_receive_or_set_was_missed() -> None:
    events = _events(("serve", 1.0), ("set", 2.0), ("spike", 3.0), ("spike", 5.0))

    assert attacks(RALLY, events) == [
        {"rally_index": 1, "event_indices": [1, 2]},
        {"rally_index": 1, "event_indices": [3]},
    ]


def test_full_build_up_is_one_attack() -> None:
    events = _events(("serve", 1.0), ("receive", 2.0), ("set", 3.0), ("spike", 4.0))

    assert attacks(RALLY, events) == [{"rally_index": 1, "event_indices": [1, 2, 3]}]


def test_a_touch_never_feeds_two_attacks() -> None:
    """After a spike the next possession starts fresh: the second spike's
    build-up may not reach back past the first."""
    events = _events(("receive", 1.0), ("set", 2.0), ("spike", 3.0), ("set", 4.0), ("spike", 5.0))

    assert [a["event_indices"] for a in attacks(RALLY, events)] == [[0, 1, 2], [3, 4]]


def test_receive_and_set_without_spike_is_no_attack() -> None:
    assert attacks(RALLY, _events(("receive", 1.0), ("set", 2.0))) == []


def test_touches_sharing_a_frame_id_stay_distinct() -> None:
    """Ids are f<frame>; two labels on one frame share an id, so rules name
    touches by position instead."""
    events = _events(("receive", 36.92), ("set", 36.94), ("spike", 37.0))
    assert events[0]["id"] == events[1]["id"]

    assert attacks([{"index": 1, "start": 36.0, "end": 40.0}], events) == [
        {"rally_index": 1, "event_indices": [0, 1, 2]},
    ]


def test_positions_index_the_shipped_order() -> None:
    events = _events(("spike", 3.0), ("receive", 1.0))

    assert attacks(RALLY, events) == [{"rally_index": 1, "event_indices": [1, 0]}]


def test_serve_point_allows_a_receive_but_not_set_spike_or_block() -> None:
    rallies = [
        {"index": 1, "start": 0.0, "end": 4.0},
        {"index": 2, "start": 5.0, "end": 9.0},
        {"index": 3, "start": 10.0, "end": 14.0},
        {"index": 4, "start": 15.0, "end": 19.0},
        {"index": 5, "start": 20.0, "end": 24.0},
    ]
    events = _events(
        ("serve", 1.0), ("score", 2.0),                   # ace / service error
        ("serve", 6.0), ("receive", 7.0), ("score", 8.0),  # shanked receive
        ("serve", 11.0), ("set", 12.0), ("score", 13.0),
        ("serve", 16.0), ("block", 17.0),
    )

    assert [o["serve_point"] for o in rally_outcomes(rallies, events)] == [
        True, True, False, False, None,
    ]


def test_duplicate_whistles_collapse_to_the_last() -> None:
    events = _events(("serve", 1.0), ("score", 2.0), ("score", 2.4))

    assert rally_outcomes(RALLY, events)[0]["score_event_index"] == 2


def test_deciding_play_is_the_last_attack_before_the_whistle() -> None:
    events = _events(
        ("receive", 1.0), ("set", 2.0), ("spike", 3.0),
        ("receive", 4.0), ("spike", 5.0),
        ("score", 6.0),
        ("spike", 7.0),  # after the whistle: cannot have decided it
    )

    assert rally_outcomes(RALLY, events)[0]["deciding_event_indices"] == [3, 4]


def test_without_a_spike_the_last_touch_decides() -> None:
    events = _events(("serve", 1.0), ("receive", 2.0), ("score", 3.0))

    outcome = rally_outcomes(RALLY, events)[0]
    assert outcome["deciding_event_indices"] == [1]
    assert outcome["score_event_index"] == 2


def test_rally_membership_is_inclusive_without_tolerance() -> None:
    rallies = [{"index": 1, "start": 1.0, "end": 3.0}]
    events = _events(("spike", 0.99), ("spike", 1.0), ("spike", 3.0), ("spike", 3.01))

    assert [a["event_indices"] for a in attacks(rallies, events)] == [[1], [2]]
