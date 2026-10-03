import pytest

from yp_video.actor.candidates import boxes_on
from yp_video.actor.person_action import policy_from_answers
from yp_video.actor.policy import EventContext
from yp_video.tracklets.geometry import TrackRef


def context(event_id):
    return EventContext(frame=7, event_id=event_id)


def test_box_uses_source_aspect_ratio_without_contact_or_tracks():
    row = {"id": "f7", "frame": 7, "label": "set"}
    answer = {**row, "box": [.1, .2, .3, .8], "pick": 3, "num_candidates": 12, "status": "selected"}
    policy = policy_from_answers([row], [answer], width=1920, height=1080)
    pick = policy.decide(context("f7"))
    assert pick.box == (192, 216, 576, 864)
    assert not policy.needs_tracklets
    with pytest.raises(KeyError):
        policy.decide(context("missing"))


def test_no_detection_abstains():
    row = {"id": "f7", "frame": 7, "label": "set"}
    answer = {**row, "box": None, "pick": None, "num_candidates": 0, "status": "no_candidate"}
    policy = policy_from_answers([row], [answer], width=1920, height=1080)
    assert not policy.decide(context("f7")).decided


@pytest.mark.parametrize("change", [
    {"id": "wrong"}, {"frame": 8}, {"label": "spike"},
    {"box": [0, 0, float("nan"), 1]}, {"box": [0, 0, 2, 1]},
    {"box": [.3, 0, .2, 1]}, {"pick": 1}, {"pick": True}, {"status": "no_candidate"},
])
def test_invalid_model_output_is_rejected(change):
    row = {"id": "f7", "frame": 7, "label": "set"}
    answer = {**row, "box": [.1, .2, .3, .8], "pick": 0, "num_candidates": 1, "status": "selected", **change}
    with pytest.raises(ValueError):
        policy_from_answers([row], [answer], width=1920, height=1080)


def test_missing_event_does_not_silently_produce_empty_identity():
    with pytest.raises(ValueError):
        policy_from_answers([{"id": "f7", "frame": 7, "label": "set"}], [], width=1, height=1)


def test_tracklet_pick_names_the_tracklet_not_a_box():
    rows = [{"id": "f7", "frame": 7, "label": "set", "candidates": [[.1, .2, .3, .8], [.5, .2, .6, .8]]},
            {"id": "f9", "frame": 9, "label": "set", "candidates": []}]
    answers = [
        {"id": "f7", "frame": 7, "label": "set", "box": [.5, .2, .6, .8], "pick": 1,
         "num_candidates": 2, "status": "selected"},
        {"id": "f9", "frame": 9, "label": "set", "box": None, "pick": None,
         "num_candidates": 0, "status": "no_candidate"},
    ]
    policy = policy_from_answers(rows, answers, width=1920, height=1080,
                                 keys={"f7": ["3:1", "3:4"], "f9": []})
    assert policy.needs_tracklets and policy.name == "fusion-person-action:tracklet"
    pick = policy.decide(context("f7"))
    assert pick.track == TrackRef(3, 4) and pick.box is None
    assert not policy.decide(context("f9")).decided


@pytest.mark.parametrize("change", [{"pick": 2}, {"pick": -1}, {"pick": 1.0}, {"num_candidates": 3, "pick": 2}])
def test_tracklet_pick_out_of_range_is_rejected(change):
    row = {"id": "f7", "frame": 7, "label": "set"}
    answer = {**row, "box": None, "pick": 0, "num_candidates": 2, "status": "selected", **change}
    with pytest.raises(ValueError):
        policy_from_answers([row], [answer], width=1, height=1, keys={"f7": ["3:1", "3:4"]})


def test_boxes_on_takes_each_tracklets_nearest_box_within_reach():
    paths = {"1:1": {10: [0, 0, 100, 200]}, "1:2": {13: [200, 0, 300, 200], 20: [0, 0, 1, 1]},
             "1:3": {14: [400, 0, 500, 200]},
             # Hanging off the frame keeps its visible part; wholly outside is dropped.
             "1:4": {10: [900, 0, 1100, 200]}, "1:5": {10: [1010, 0, 1100, 200]}}
    assert boxes_on(paths, 10, 1000, 1000) == [
        ("1:1", [0.0, 0.0, 0.1, 0.2]), ("1:2", [0.2, 0.0, 0.3, 0.2]), ("1:4", [0.9, 0.0, 1.0, 0.2])]
