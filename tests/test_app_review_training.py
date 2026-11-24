"""Sending a user's App match into training: the copy, the pre-labels, and
what a second send must never touch. Every write lands under tmp_path."""

import json

import pytest

from yp_video.action import rules
from yp_video.web import action_annotations
from yp_video.web.app_review import training
from yp_video.web.app_review.models import Bundle


def make_bundle(corrections=None):
    events = [
        {"id": "f300", "label": "serve", "time": 10.0, "frame": 300, "visible": True,
         "xy": [0.1, 0.5], "score": 0.9},
        {"id": "f330", "label": "receive", "time": 11.0, "frame": 330, "visible": True,
         "xy": [0.6, 0.4], "score": 0.8},
        {"id": "f420", "label": "score", "time": 14.0, "frame": 420, "visible": False,
         "xy": [0.5, 0.7], "score": 0.7},
    ]
    rallies = [
        {"index": 1, "set": 1, "start": 9.0, "end": 15.0, "winner": "near"},
        {"index": 2, "set": 1, "start": 30.0, "end": 40.0, "winner": "far"},
    ]
    corr = {
        "schema_version": "7.0", "match_id": "m1", "updated_at": "now", "roster": [],
        "actions": [], "scores": [], "deleted_rally_indices": [], "rally_winner_overrides": {},
        **(corrections or {}),
    }
    return Bundle.model_validate({
        "result": {
            "job_id": "job1", "user_id": "u1", "match_id": "m1",
            "video_r2_key": "videos/u1/m1.mp4", "total_duration": 100,
            "action_events": events, "rallies": rallies,
            "attacks": rules.attacks(rallies, events),
            "rally_outcomes": rules.rally_outcomes(rallies, events),
        },
        "corrections": corr,
    })


def test_prelabels_apply_user_corrections():
    bundle = make_bundle({
        "deleted_rally_indices": [2],
        "rally_winner_overrides": {"1": "far"},
        "actions": [{"key": "11.000", "removed": True, "tag_ids": []}],
    })
    rallies, events = training.prelabels(bundle)
    assert rallies == [{"start": 9.0, "end": 15.0, "label": "rally", "winner": "far"}]
    assert [e["label"] for e in events] == ["serve", "score"]
    assert events[1] == {"frame": 420, "label": "score", "xy": [0.5, 0.7], "visible": False, "score": 0.7}


def test_removed_score_drops_only_the_score():
    bundle = make_bundle({"scores": [{"key": "14.000", "removed": True, "tag_ids": []}]})
    _, events = training.prelabels(bundle)
    assert [e["label"] for e in events] == ["serve", "receive"]


class FakeR2:
    """Objects as key -> user metadata, like R2's head_object Metadata."""

    bucket = "pipeline"
    configured = True

    def __init__(self):
        self.objects: dict[str, dict] = {}
        self.copies: list[tuple[str, str, str]] = []
        self.deleted: list[str] = []
        self.uploaded: list[str] = []

    def object_exists(self, key):
        return key in self.objects

    def copy_from(self, bucket, src, dst, metadata):
        self.copies.append((bucket, src, dst))
        self.objects[dst] = dict(metadata)

    def object_metadata(self, key):
        return self.objects[key]

    def delete_object(self, key):
        self.deleted.append(key)
        self.objects.pop(key, None)

    def upload_file(self, path, key):
        self.uploaded.append(key)

    def generate_presigned_url(self, key, expires=3600):
        return f"https://r2/{key}"

    def list_objects_cached(self, prefix="", ttl=30.0):
        return [{"key": k} for k in sorted(self.objects) if k.startswith(prefix)]


@pytest.fixture
def env(tmp_path, monkeypatch):
    dirs = {
        "rally_pre": tmp_path / "rally-pre",
        "rally": tmp_path / "rally",
        "action_pre": tmp_path / "action-pre",
        "action": tmp_path / "action",
    }
    monkeypatch.setattr(training, "RALLY_SPOT_PRE_ANNOTATIONS_DIR", dirs["rally_pre"])
    monkeypatch.setattr(training, "RALLY_ANNOTATIONS_DIR", dirs["rally"])
    monkeypatch.setattr(action_annotations, "ACTION_PRE_ANNOTATIONS_DIR", dirs["action_pre"])
    monkeypatch.setattr(action_annotations, "ACTION_ANNOTATIONS_DIR", dirs["action"])
    r2 = FakeR2()
    monkeypatch.setattr(training, "r2_client", r2)
    monkeypatch.setattr(training, "probe_video_metadata", lambda url: {"fps": 30.0, "num_frames": 3000})
    cuts: list = []
    monkeypatch.setattr(training, "all_cut_paths", lambda: cuts)
    monkeypatch.setattr(training, "_match_ids", {})
    row = {
        "id": "m1", "owner_id": "u1", "angle": "corner", "title": "霖度C/03",
        "r2_keys": {"result": "results/u1/m1.json", "corrections": None},
        "source_video": {"r2_key": "videos/u1/m1.mp4", "public_url": "https://x"},
    }
    state = {"row": row, "bundle": make_bundle()}
    monkeypatch.setattr(training.sources, "library_match", lambda user, match: ({}, state["row"]))
    monkeypatch.setattr(training.sources, "match_bundle", lambda lib, row: (state["bundle"], []))
    monkeypatch.setattr(training.sources.customer, "_config", {"R2_BUCKET_CUSTOMER": "customer"})
    monkeypatch.setattr(training.sources.customer, "_ensure_config", lambda: None)
    return {"r2": r2, "dirs": dirs, "state": state, "cuts": cuts}


NAME = "霖度C⧸03"


def test_add_copies_source_and_writes_prelabels(env):
    out = training.add_to_training("u1", "m1", None)
    assert out == {"video": f"{NAME}.mp4", "kind": "user-corner", "status": "added", "rallies": 2, "events": 3}
    assert env["r2"].copies == [("customer", "videos/u1/m1.mp4", f"cuts-user-corner/{NAME}.mp4")]
    assert env["r2"].objects[f"cuts-user-corner/{NAME}.mp4"] == {"match-id": "m1"}

    rally_file = env["dirs"]["rally_pre"] / f"{NAME}_annotations.jsonl"
    meta, *rows = [json.loads(line) for line in rally_file.read_text().splitlines()]
    assert meta["source"] == {"type": "app", "user_id": "u1", "match_id": "m1", "job_id": "job1"}
    assert meta["video"] == f"{NAME}.mp4" and meta["max_rally_id"] == 2
    assert [(r["start"], r["end"], r["winner"], r["rally_id"]) for r in rows] == [
        (9.0, 15.0, "near", 1), (30.0, 40.0, "far", 2)
    ]

    action_file = env["dirs"]["action_pre"] / f"{NAME}_actions.jsonl"
    meta, *events = [json.loads(line) for line in action_file.read_text().splitlines()]
    assert meta["video"] == NAME
    assert meta["source"]["type"] == "app" and meta["fps"] == 30.0 and meta["num_frames"] == 3000
    assert [e["score"] for e in events] == [0.9, 0.8, 0.7]
    assert not env["dirs"]["action"].exists() and not env["dirs"]["rally"].exists()
    assert sorted(env["r2"].uploaded) == [
        f"action/pre-annotations/{NAME}_actions.jsonl",
        f"rally-spot/pre-annotations/{NAME}_annotations.jsonl",
    ]


def test_add_again_is_noop_and_keeps_labels(env):
    training.add_to_training("u1", "m1", None)
    human = env["dirs"]["action"] / f"{NAME}_actions.jsonl"
    human.parent.mkdir(parents=True)
    human.write_text('{"_meta": true}\n')
    pre = env["dirs"]["action_pre"] / f"{NAME}_actions.jsonl"
    before = (human.read_bytes(), pre.read_bytes())
    env["r2"].copies.clear()

    out = training.add_to_training("u1", "m1", "corner")
    assert out["status"] == "exists"
    assert env["r2"].copies == []
    assert (human.read_bytes(), pre.read_bytes()) == before


def test_retitled_match_keeps_its_training_name(env):
    training.add_to_training("u1", "m1", None)
    env["state"]["row"]["title"] = "renamed"
    out = training.add_to_training("u1", "m1", None)
    assert out["status"] == "exists" and out["video"] == f"{NAME}.mp4"


def test_angle_change_moves_only_the_video(env):
    training.add_to_training("u1", "m1", "corner")
    labels = [env["dirs"]["rally_pre"] / f"{NAME}_annotations.jsonl", env["dirs"]["action_pre"] / f"{NAME}_actions.jsonl"]
    before = [p.read_bytes() for p in labels]
    env["r2"].copies.clear()
    out = training.add_to_training("u1", "m1", "high")
    assert out["status"] == "moved" and out["kind"] == "user-high" and out["video"] == f"{NAME}.mp4"
    assert env["r2"].copies == [("pipeline", f"cuts-user-corner/{NAME}.mp4", f"cuts-user-high/{NAME}.mp4")]
    assert env["r2"].objects[f"cuts-user-high/{NAME}.mp4"] == {"match-id": "m1"}
    assert env["r2"].deleted == [f"cuts-user-corner/{NAME}.mp4"]
    assert [p.read_bytes() for p in labels] == before


def test_taken_name_gets_the_match_id(env):
    from pathlib import Path

    env["cuts"].append(Path(f"/v/cuts-sideline/{NAME}.mp4"))
    assert training.add_to_training("u1", "m1", None)["video"] == f"{NAME}-m1.mp4"


def test_name_with_labels_but_no_cut_is_taken(env):
    other = env["dirs"]["action"] / f"{NAME}_actions.jsonl"
    other.parent.mkdir(parents=True)
    other.write_text('{"_meta": true}\n')
    assert training.add_to_training("u1", "m1", None)["video"] == f"{NAME}-m1.mp4"


def test_blank_title_falls_back_to_match_id(env):
    env["state"]["row"]["title"] = "  "
    assert training.add_to_training("u1", "m1", None)["video"] == "m1.mp4"


def test_refuses_operator_cut(env):
    env["state"]["row"]["source_video"] = {"r2_key": "cuts-sideline/x.mp4", "public_url": None}
    with pytest.raises(ValueError, match="already training data"):
        training.add_to_training("u1", "m1", None)
    assert env["r2"].copies == []


def test_no_result_fails_before_copy(env, monkeypatch):
    def no_result(lib, row):
        raise ValueError("This match has no analysis result yet")

    monkeypatch.setattr(training.sources, "match_bundle", no_result)
    with pytest.raises(ValueError, match="no analysis result"):
        training.add_to_training("u1", "m1", None)
    assert env["r2"].copies == []


def test_unknown_angle_is_rejected(env):
    env["state"]["row"]["angle"] = "phoneSideline"
    with pytest.raises(ValueError, match="angle"):
        training.add_to_training("u1", "m1", None)


def test_http_training_endpoint(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from yp_video.web.routers import app_review

    calls = []

    def fake_add(user, match, angle):
        calls.append((user, match, angle))
        if match == "bad":
            raise ValueError("This match has no analysis result yet")
        return {"video": f"{match}.mp4", "kind": "user-high", "status": "added", "rallies": 3, "events": 40}

    monkeypatch.setattr(app_review.training, "add_to_training", fake_add)
    app = FastAPI()
    app.include_router(app_review.router, prefix="/api/app-review")
    with TestClient(app) as client:
        ok = client.post("/api/app-review/users/u1/matches/m1/training", json={"angle": "high"})
        assert ok.status_code == 200 and ok.json()["kind"] == "user-high"
        assert client.post("/api/app-review/users/u1/matches/m2/training", json={"angle": None}).status_code == 200
        bad = client.post("/api/app-review/users/u1/matches/bad/training", json={"angle": None})
        assert bad.status_code == 409 and "no analysis result" in bad.json()["detail"]
        assert client.post(
            "/api/app-review/users/u1/matches/m1/training", json={"angle": "phoneSideline"}
        ).status_code == 422
    assert calls == [("u1", "m1", "high"), ("u1", "m2", None), ("u1", "bad", None)]


def test_failure_after_copy_is_finished_by_the_next_send(env, monkeypatch):
    def offline(url):
        raise OSError("ffprobe: connection reset")

    monkeypatch.setattr(training, "probe_video_metadata", offline)
    with pytest.raises(OSError):
        training.add_to_training("u1", "m1", None)
    assert f"cuts-user-corner/{NAME}.mp4" in env["r2"].objects
    assert not (env["dirs"]["action_pre"] / f"{NAME}_actions.jsonl").exists()

    monkeypatch.setattr(training, "probe_video_metadata", lambda url: {"fps": 30.0, "num_frames": 3000})
    out = training.add_to_training("u1", "m1", None)
    assert out["status"] == "exists" and out["events"] == 3
    assert len(env["r2"].copies) == 1
    assert (env["dirs"]["action_pre"] / f"{NAME}_actions.jsonl").exists()


def test_refuses_a_result_analysed_from_another_video(env):
    env["state"]["row"]["source_video"] = {"r2_key": "videos/u1/m1-reupload.mp4", "public_url": "https://x"}
    with pytest.raises(ValueError, match="another video"):
        training.add_to_training("u1", "m1", None)
    assert env["r2"].copies == []


def test_sent_lists_matches_already_in_training(env):
    env["r2"].objects.update({"cuts-user-high/x.mp4": {"match-id": "m2"}, "cuts-sideline/m9.mp4": {}})
    assert training.sent() == {"m2": "user-high"}
    training.add_to_training("u1", "m1", None)
    assert training.sent() == {"m1": "user-corner", "m2": "user-high"}
    training.add_to_training("u1", "m1", "baseline")
    assert training.sent() == {"m1": "user-baseline", "m2": "user-high"}


def test_http_sent_endpoint(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from yp_video.web.routers import app_review

    monkeypatch.setattr(app_review.training, "sent", lambda: {"m1": "user-corner"})
    app = FastAPI()
    app.include_router(app_review.router, prefix="/api/app-review")
    with TestClient(app) as client:
        assert client.get("/api/app-review/training").json() == {"m1": "user-corner"}
