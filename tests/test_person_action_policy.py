import pytest

from yp_video.actor.candidates import boxes_on, normalized_paths
from yp_video.actor.person_action import policy_from_answers, policy_from_spot_picks
from yp_video.tracklets.geometry import TrackRef


def _spot_picks(tmp_path, picks):
    from yp_video.core.actor_picks import load_actor_picks, save_actor_picks
    events = [{"frame": f, "label": l, "score": .9, **({"actor": a} if a is not None else {})} for f, l, a in picks]
    checkpoint = tmp_path / "checkpoint_best.pt"
    checkpoint.write_bytes(b"")
    saved = save_actor_picks([{"video": "v", "events": events}], tmp_path / "picks.json", checkpoint)
    return saved, load_actor_picks(tmp_path / "picks.json")


def test_spot_pass_picks_become_boxes_in_source_pixels(tmp_path):
    saved, picks = _spot_picks(tmp_path, [
        (7, "set", {"box": [.1, .2, .3, .8], "candidates": 12}),
        (9, "spike", {"box": None, "candidates": 0}),
        (11, "score", None),
    ])
    assert saved == 2
    events = [{"id": "a", "frame": 7, "label": "set"}, {"id": "b", "frame": 9, "label": "spike"},
              {"id": "c", "frame": 7, "label": "receive"}, {"id": "d", "frame": 30, "label": "set"}]
    policy = policy_from_spot_picks(events, picks, width=1920, height=1080)
    assert not policy.needs_tracklets and policy.name == "fusion-spot-pass"
    pick = policy.decide("a")
    assert pick.box == pytest.approx((192, 216, 576, 864)) and pick.candidates == 12
    assert not policy.decide("b").decided and policy.decide("b").diagnostic["status"] == "no_candidate"
    # A label changed since, or a touch the pass never spotted: no pick.
    for event in ("c", "d"):
        assert not policy.decide(event).decided
        assert policy.decide(event).diagnostic["status"] == "not_spotted"


@pytest.mark.parametrize("actor", [
    {"box": [0, 0, float("nan"), 1], "candidates": 1}, {"box": [0, 0, 2, 1], "candidates": 1},
    {"box": [.3, 0, .2, 1], "candidates": 1}, {"box": [.1, .2, .3, .8], "candidates": 0},
    {"box": None, "candidates": 3}, {"box": [.1, .2, .3, .8], "candidates": -1},
])
def test_invalid_spot_picks_are_rejected(tmp_path, actor):
    with pytest.raises(ValueError):
        _spot_picks(tmp_path, [(7, "set", actor)])


def test_duplicate_spot_picks_are_rejected(tmp_path):
    actor = {"box": [.1, .2, .3, .8], "candidates": 1}
    with pytest.raises(ValueError):
        _spot_picks(tmp_path, [(7, "set", actor), (7, "set", actor)])


@pytest.mark.parametrize("change", [
    {"id": "wrong"}, {"frame": 8}, {"label": "spike"}, {"pick": True}, {"status": "no_candidate"},
])
def test_invalid_tracklet_output_is_rejected(change):
    row = {"id": "f7", "frame": 7, "label": "set"}
    answer = {**row, "box": [.1, .2, .3, .8], "pick": 0, "num_candidates": 2, "status": "selected", **change}
    with pytest.raises(ValueError):
        policy_from_answers([row], [answer], keys={"f7": ["3:1", "3:4"]})


def test_missing_event_does_not_silently_produce_empty_identity():
    with pytest.raises(ValueError):
        policy_from_answers([{"id": "f7", "frame": 7, "label": "set"}], [], keys={"f7": []})


def test_tracklet_pick_names_the_tracklet_not_a_box():
    rows = [{"id": "f7", "frame": 7, "label": "set", "candidates": [[.1, .2, .3, .8], [.5, .2, .6, .8]]},
            {"id": "f9", "frame": 9, "label": "set", "candidates": []}]
    answers = [
        {"id": "f7", "frame": 7, "label": "set", "box": [.5, .2, .6, .8], "pick": 1,
         "num_candidates": 2, "status": "selected"},
        {"id": "f9", "frame": 9, "label": "set", "box": None, "pick": None,
         "num_candidates": 0, "status": "no_candidate"},
    ]
    policy = policy_from_answers(rows, answers, keys={"f7": ["3:1", "3:4"], "f9": []})
    assert policy.needs_tracklets and policy.name == "fusion-person-action:tracklet"
    pick = policy.decide("f7")
    assert pick.track == TrackRef(3, 4) and pick.box is None
    assert not policy.decide("f9").decided
    with pytest.raises(KeyError):
        policy.decide("missing")


@pytest.mark.parametrize("change", [{"pick": 2}, {"pick": -1}, {"pick": 1.0}, {"num_candidates": 3, "pick": 2}])
def test_tracklet_pick_out_of_range_is_rejected(change):
    row = {"id": "f7", "frame": 7, "label": "set"}
    answer = {**row, "box": None, "pick": 0, "num_candidates": 2, "status": "selected", **change}
    with pytest.raises(ValueError):
        policy_from_answers([row], [answer], keys={"f7": ["3:1", "3:4"]})


def test_boxes_on_takes_each_tracklets_nearest_box_within_reach():
    paths = {"1:1": {10: [0, 0, 100, 200]}, "1:2": {13: [200, 0, 300, 200], 20: [0, 0, 1, 1]},
             "1:3": {14: [400, 0, 500, 200]},
             # Hanging off the frame keeps its visible part; wholly outside is dropped.
             "1:4": {10: [900, 0, 1100, 200]}, "1:5": {10: [1010, 0, 1100, 200]}}
    assert boxes_on(paths, 10, 1000, 1000) == [
        ("1:1", [0.0, 0.0, 0.1, 0.2]), ("1:2", [0.2, 0.0, 0.3, 0.2]), ("1:4", [0.9, 0.0, 1.0, 0.2])]


def test_normalized_paths_carry_the_named_tracklets_in_frame_order():
    paths = {"1:1": {12: [0, 0, 100, 200], 10: [100, 0, 200, 200], 14: [1010, 0, 1100, 200]},
             "1:2": {10: [0, 0, 1, 1]}}
    assert normalized_paths(paths, ["1:1"], 1000, 1000) == {
        "1:1": {"frames": [10, 12], "boxes": [[0.1, 0.0, 0.2, 0.2], [0.0, 0.0, 0.1, 0.2]]}}
