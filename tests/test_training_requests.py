"""The DEBUG App's training requests: yp-video takes them from the Worker's
admin API, runs the import, and reports back — except on a transient failure,
which leaves the request pending for the next round."""

from botocore.exceptions import EndpointConnectionError

from yp_video.web.app_review import sources, training_requests


def setup(monkeypatch, pending, add):
    posted = []
    monkeypatch.setattr(training_requests.sources, "admin_get", lambda path: pending)
    monkeypatch.setattr(
        training_requests.sources, "admin_post", lambda path, body: posted.append((path, body))
    )
    monkeypatch.setattr(training_requests.training, "add_to_training", add)
    return posted


def request(rid, match):
    return {"id": rid, "user_id": "u1", "match_id": match, "angle": "high", "created_at": 1}


def test_done_and_failed_are_reported(monkeypatch):
    def add(user, match, angle):
        assert (user, angle) == ("u1", "high")
        if match == "m2":
            raise ValueError("This match has no analysis result yet")
        return {"video": "m1.mp4", "kind": "user-high", "status": "added", "rallies": 1, "events": 2}

    posted = setup(monkeypatch, [request("r1", "m1"), request("r2", "m2")], add)
    assert training_requests.process_pending() == 2
    assert posted == [
        ("/training-requests/r1", {"status": "done", "error": None}),
        ("/training-requests/r2", {"status": "failed", "error": "This match has no analysis result yet"}),
    ]


def test_missing_match_fails_the_request(monkeypatch):
    def add(user, match, angle):
        raise sources.NotFound("Match not in this user's library")

    posted = setup(monkeypatch, [request("r1", "m1")], add)
    assert training_requests.process_pending() == 1
    assert posted[0][1]["status"] == "failed"


def test_transient_error_leaves_request_pending(monkeypatch):
    def add(user, match, angle):
        raise EndpointConnectionError(endpoint_url="https://r2")

    posted = setup(monkeypatch, [request("r1", "m1")], add)
    assert training_requests.process_pending() == 0
    assert posted == []

    def admin_down(user, match, angle):
        raise sources.AdminApiError("VolleyIQ admin API unreachable")

    posted = setup(monkeypatch, [request("r1", "m1")], admin_down)
    assert training_requests.process_pending() == 0
    assert posted == []


def test_missing_object_fails_the_request(monkeypatch):
    from botocore.exceptions import ClientError

    def add(user, match, angle):
        raise ClientError({"Error": {"Code": "NoSuchKey", "Message": "gone"}}, "CopyObject")

    posted = setup(monkeypatch, [request("r1", "m1")], add)
    assert training_requests.process_pending() == 1
    assert posted[0][1]["status"] == "failed"
