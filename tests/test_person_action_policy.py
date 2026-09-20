import pytest

from yp_video.actor.person_action import policy_from_answers
from yp_video.actor.policy import EventContext


def context(event_id):
    return EventContext(frame=7, contact=None, visible=False, event_id=event_id)


def test_box_uses_source_aspect_ratio_without_contact_or_tracks():
    row = {"id": "f7", "frame": 7, "label": "set"}
    answer = {**row, "box": [.1, .2, .3, .8], "candidates": 12, "status": "selected"}
    policy = policy_from_answers([row], [answer], width=1920, height=1080)
    pick = policy.decide(context("f7"))
    assert pick.box == (192, 216, 576, 864)
    assert not policy.needs_tracklets
    with pytest.raises(KeyError):
        policy.decide(context("missing"))


def test_no_detection_abstains():
    row = {"id": "f7", "frame": 7, "label": "set"}
    answer = {**row, "box": None, "candidates": 0, "status": "no_person_detection"}
    policy = policy_from_answers([row], [answer], width=1920, height=1080)
    assert not policy.decide(context("f7")).decided


@pytest.mark.parametrize("change", [
    {"id": "wrong"}, {"frame": 8}, {"label": "spike"},
    {"box": [0, 0, float("nan"), 1]}, {"box": [0, 0, 2, 1]},
    {"box": [.3, 0, .2, 1]},
])
def test_invalid_model_output_is_rejected(change):
    row = {"id": "f7", "frame": 7, "label": "set"}
    answer = {**row, "box": [.1, .2, .3, .8], "candidates": 1, "status": "selected", **change}
    with pytest.raises(ValueError):
        policy_from_answers([row], [answer], width=1920, height=1080)


def test_missing_event_does_not_silently_produce_empty_identity():
    with pytest.raises(ValueError):
        policy_from_answers([{"id": "f7", "frame": 7, "label": "set"}], [], width=1, height=1)
