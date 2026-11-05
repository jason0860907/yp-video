from yp_video.action.rules import attacks, rally_outcomes, serving_sides, touch_sides


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


def _touches(*labels: str) -> list[dict]:
    return [{"label": label} for label in labels]


def test_touch_sides_cross_after_serve_and_spike() -> None:
    sides = touch_sides(
        _touches("serve", "receive", "set", "spike", "receive", "spike", "score"), "near"
    )

    assert sides == ["near", "far", "far", "far", "near", "near", None]


def test_touch_sides_block_answers_the_spike_then_stops() -> None:
    """After a block the ball can land on either side — nothing after it is
    placed."""
    sides = touch_sides(_touches("serve", "receive", "set", "spike", "block", "receive"), "left")

    assert sides == ["left", "right", "right", "right", "left", None]


def test_touch_sides_stop_at_a_mid_possession_receive() -> None:
    """A second receive means a free ball came over or a second pass — the
    order cannot tell which."""
    sides = touch_sides(_touches("serve", "receive", "set", "receive", "spike"), "far")

    assert sides == ["far", "near", "near", None, None]


def test_touch_sides_need_the_serve_first() -> None:
    assert touch_sides(_touches("receive", "set", "spike"), "near") == [None, None, None]


def test_touch_sides_stop_at_a_fourth_touch() -> None:
    sides = touch_sides(_touches("serve", "receive", "set", "set", "set"), "near")

    assert sides == ["near", "far", "far", "far", None]


def test_an_antenna_hit_ends_the_rally() -> None:
    events = _events(("serve", 1.0), ("receive", 2.0), ("spike", 3.0), ("antenna", 3.6))

    assert touch_sides([{"label": e["label"]} for e in events], "near") == ["near", "far", "far", None]
    assert rally_outcomes(RALLY, events) == [
        {"rally_index": 1, "serve_point": False, "score_event_index": 3, "deciding_event_indices": [1, 2],
         "serving_side": None, "serving_side_from": None},
    ]


def test_a_net_contact_is_no_touch_and_ground_ends_the_rally() -> None:
    """A serve off the net cord still crosses to the receiver, and the floor
    landing is the rally's end like a score."""
    events = _events(("serve", 1.0), ("net", 1.3), ("receive", 2.0), ("spike", 3.0), ("ground", 3.6))

    assert touch_sides([{"label": e["label"]} for e in events], "near") == [
        "near", None, "far", "far", None,
    ]
    assert rally_outcomes(RALLY, events) == [
        {"rally_index": 1, "serve_point": False, "score_event_index": 4, "deciding_event_indices": [2, 3],
         "serving_side": None, "serving_side_from": None},
    ]


def _served_match(winners, serve_ys, sets=None):
    """Sideline rallies 10 s apart, each opening on a serve struck at the
    given frame height (None: no serve spotted), with the given winners."""
    rallies, events = [], []
    for i, (winner, y) in enumerate(zip(winners, serve_ys)):
        start = i * 10.0
        rallies.append({"index": i + 1, "set": (sets or [1] * len(winners))[i],
                        "start": start, "end": start + 8.0, "winner": winner})
        if y is not None:
            events.append({"id": f"s{i}", "label": "serve", "time": start + 1.0, "xy": [0.5, y]})
        events.append({"id": f"r{i}", "label": "receive", "time": start + 2.0, "xy": [0.5, 0.5]})
    return rallies, events


def _sides(rallies, events) -> dict:
    return {index: side for index, (side, _) in serving_sides(rallies, events).items()}


def test_each_rally_is_served_from_where_its_serve_was_struck() -> None:
    """The previous winners anchor which half each side serves from; a rally
    whose previous winner is wrong still reads its own serve."""
    near, far = 0.8, 0.3
    winners = ["near", "near", "far", "far", "near", "far", "near", "far"]
    ys = [near, near, near, far, far, near, far, near]  # rally 7's anchor (far) is wrong
    rallies, events = _served_match(winners, ys)

    expected = ["near", "near", "near", "far", "far", "near", "far", "near"]
    assert serving_sides(rallies, events) == {i + 1: (side, "serve") for i, side in enumerate(expected)}
    outcomes = rally_outcomes(rallies, events)
    assert [o["serving_side"] for o in outcomes] == expected
    assert {o["serving_side_from"] for o in outcomes} == {"serve"}


def test_a_rally_not_opening_on_a_serve_names_no_server() -> None:
    rallies, events = _served_match(["near", "far", "near", "far", "near"], [0.8, None, 0.3, 0.8, 0.3])

    assert serving_sides(rallies, events)[2] == (None, None)
    assert _sides(rallies, events) == {1: "near", 2: None, 3: "far", 4: "near", 5: "far"}


def test_without_winners_nothing_names_a_server() -> None:
    assert set(serving_sides(*_served_match([None] * 3, [0.8, 0.3, 0.8])).values()) == {(None, None)}


def test_serves_that_do_not_split_fall_back_to_the_previous_winner() -> None:
    """An end-line camera sees only the far server: every serve sits on one
    half, so the previous rally's winner serves — a rally opening on no
    serve too — and the set's first rally names none."""
    rallies, events = _served_match(
        ["near", "far", "far", "near", "far"], [0.3, 0.3, None, 0.3, 0.3], sets=[1, 1, 1, 2, 2]
    )

    assert serving_sides(rallies, events) == {
        1: (None, None),
        2: ("near", "previous_winner"),
        3: ("far", "previous_winner"),
        4: (None, None),
        5: ("near", "previous_winner"),
    }
    outcome = rally_outcomes(rallies, events)[1]
    assert (outcome["serving_side"], outcome["serving_side_from"]) == ("near", "previous_winner")


def test_a_new_set_does_not_anchor_on_the_last_set() -> None:
    """The first serve of a set is not the previous winner's: counted, rally
    3's anchor (near, on the far half) would sink the split under the gate."""
    rallies, events = _served_match(
        ["far", "near", "near", "far"], [0.8, 0.3, 0.3, 0.8], sets=[1, 1, 2, 2]
    )

    assert _sides(rallies, events) == {1: "near", 2: "far", 3: "far", 4: "near"}
