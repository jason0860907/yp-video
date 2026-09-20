"""Projection parity with iOS TouchContextTests, and feedback write safety."""

import pytest
from pydantic import ValidationError

from yp_video.core.jsonl import read_jsonl, write_jsonl
from yp_video.web.app_review import feedback
from yp_video.web.app_review.models import Bundle
from yp_video.web.app_review.projection import project


def correction(key, **values):
    return {"key": key, "removed": False, "tag_ids": [], **values}


def bundle(events=(), rallies=((1, 9, 20),), **changes):
    corr = {
        "schema_version": "6.0",
        "match_id": "match",
        "updated_at": "now",
        "roster": [],
        "actions": [],
        "scores": [],
        "deleted_rally_indices": [],
        **changes.pop("corrections", {}),
    }
    return Bundle.model_validate(
        {
            "result": {
                "job_id": "job",
                "user_id": "user",
                "match_id": "match",
                "video_r2_key": "videos/user/source.mp4",
                "total_duration": 100,
                "action_events": [
                    {
                        "id": f"f{round(t * 30)}",
                        "label": label,
                        "time": t,
                        "frame": round(t * 30),
                    }
                    for label, t in events
                ],
                "rallies": [
                    {"index": i, "set": 1, "start": lo, "end": hi}
                    for i, lo, hi in rallies
                ],
            },
            "corrections": corr,
            **changes,
        }
    )


@pytest.mark.parametrize(
    "mode,bounds,kinds",
    [
        ("full_play", (10, 15), ["receive", "set", "spike"]),
        ("to_next", (12, 15), ["set", "spike"]),
        ("whole_rally", (9, 20), ["receive", "set", "spike"]),
    ],
)
def test_spike_build_up_three_windows(mode, bounds, kinds):
    b = bundle(
        [("serve", 10), ("receive", 11), ("set", 12), ("spike", 13), ("receive", 14)]
    )
    c = project(b, mode=mode)["actions"][3]
    assert (c["start"], c["end"]) == bounds
    assert [t["kind"] for t in c["touches"]] == kinds


def test_exact_containment_and_outside_window():
    b = bundle([("receive", 36.92), ("set", 36.94), ("spike", 37)], [(1, 36.93, 40)])
    r = project(b)
    assert r["actions"][0]["rally_index"] is None
    assert [t["kind"] for t in r["actions"][2]["touches"]] == ["set", "spike"]
    assert project(b, mode="whole_rally")["actions"][0]["start"] == pytest.approx(35.92)


def test_score_last_whistle_and_spike_not_defensive_receive():
    b = bundle(
        [
            ("receive", 11),
            ("set", 12),
            ("spike", 13),
            ("receive", 14),
            ("score", 15),
            ("score", 16),
        ]
    )
    for mode in ("full_play", "to_next", "whole_rally"):
        clips = project(b, mode=mode)["scores"]
        assert len(clips) == 1
        c = clips[0]
        assert c["key"] == "16.000"
        assert c["event_id"] == "f390"
        assert (c["start"], c["end"]) == (10, 17)


def test_empty_rally_and_standalone_score():
    r = project(bundle([("score", 25)]))
    assert [(c["start"], c["end"]) for c in r["scores"]] == [(9, 20), (24, 26)]
    assert r["scores"][0]["key"] == "unmapped-rally-1"


def test_non_spike_deciding_touch():
    c = project(bundle([("receive", 11), ("set", 12), ("score", 14)]))["scores"][0]
    assert [t["kind"] for t in c["touches"]] == ["set"]
    assert (c["start"], c["end"]) == (11, 15)


def test_trim_soft_delete_and_outcomes():
    b = bundle(
        [("set", 12), ("spike", 13), ("score", 15)],
        corrections={
            "actions": [
                correction("12.000", removed=True),
                correction("13.000", trim_start=12.5, trim_end=14),
            ],
            "scores": [correction("15.000", result="loss", loss_reason="net")],
            "deleted_rally_indices": [1],
        },
    )
    raw, changed = project(b, corrected=False), project(b)
    assert len(raw["actions"]) == 2
    assert len(changed["actions"]) == 1
    assert (changed["actions"][0]["start"], changed["actions"][0]["end"]) == (12.5, 14)
    assert changed["actions"][0]["rally_index"] is None
    assert changed["actions"][0]["result"] is None
    assert changed["scores"][0]["result"] == "loss"
    assert changed["loss_reasons"] == {"net": 1}


def test_player_run_scope_override_and_removed_units():
    b = bundle(
        [("set", 12), ("spike", 13), ("receive", 14)],
        corrections={
            "roster": [
                {"number": n, "name": str(n), "position": "", "hue": 0} for n in (7, 8)
            ],
            "player_identification": {
                "result_id": "identify",
                "unit_roster": {"a": 7, "b": 7},
                "removed_units": ["b"],
                "event_overrides": {"f390": 8},
            },
        },
        identification={
            "version": 4,
            "job_id": "identify",
            "user_id": "user",
            "match_id": "match",
            "units": [
                {"key": "a", "events": ["f360", "f390"]},
                {"key": "b", "events": ["f420"]},
            ],
        },
    )

    def numbers():
        return [
            a["player"]["number"] if a["player"] else None
            for a in project(b)["actions"]
        ]

    assert numbers() == [7, 8, None]
    b.identification.job_id = "new"
    assert numbers() == [None, 8, None]
    assert any("result_id" in w for w in project(b)["warnings"])


def test_library_uuid_trim_and_synthetic_score():
    uid = "11111111-1111-4111-8111-111111111111"
    b = bundle(
        [],
        corrections={"scores": [correction(f"rally-{uid}", result="loss")]},
        library_rallies=[
            {
                "id": uid,
                "match_id": "match",
                "index": 1,
                "set": 1,
                "start": 10,
                "end": 19,
            }
        ],
    )
    c = project(b)["scores"][0]
    assert c["key"] == f"rally-{uid}"
    assert (c["start"], c["end"]) == (10, 19)
    assert c["result"] == "loss"
    assert project(b, corrected=False)["scores"][0]["start"] == 9


def test_court_counts_all_rallies():
    b = bundle([], [(1, 0, 10), (2, 20, 30), (3, 40, 50)])
    b.result.rallies[0].winner = "left"
    b.result.rallies[1].winner = "near"
    r = project(b)
    assert r["court_totals"][1] == {
        "left": 1,
        "right": 0,
        "near": 1,
        "far": 0,
        "unknown": 1,
    }
    assert r["scores"][0]["court_score"]["unknown"] == 0


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": "5.0"},
        {"match_id": "wrong"},
        {"actions": [correction("1", trim_start=3, trim_end=2)]},
    ],
)
def test_bad_corrections(change):
    with pytest.raises(ValidationError):
        bundle(corrections=change)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback, "REVIEW_DIR", tmp_path / "reviews")
    monkeypatch.setattr(feedback, "RALLY_ANNOTATIONS_DIR", tmp_path / "rallies")
    monkeypatch.setattr(
        feedback,
        "annotation_path",
        lambda video: tmp_path / "actions" / f"{video}_actions.jsonl",
    )
    flags = []
    monkeypatch.setattr(
        feedback.label_done, "set_done", lambda *args: flags.append(args)
    )
    return flags


def action_file():
    path = feedback.annotation_path("game.mp4")
    write_jsonl(
        path,
        {"fps": 30, "num_frames": 3000},
        [
            {
                "id": "local-id",
                "frame": 300,
                "label": "spike",
                "xy": [0.5, 0.5],
                "visible": True,
            },
            {
                "id": "other",
                "frame": 360,
                "label": "set",
                "xy": [0.2, 0.5],
                "visible": True,
            },
        ],
    )
    return path


def review():
    return feedback.create(
        bundle(
            [("spike", 10)],
            corrections={
                "actions": [correction("10.000", removed=True)],
                "deleted_rally_indices": [1],
            },
        ),
        "reader",
    )


def apply(r, **overrides):
    options = {
        "video": "game.mp4",
        "same_source": True,
        "revision": feedback.target_revision("game.mp4", "remove_event"),
        **overrides,
    }
    return feedback.decide(
        r["id"],
        "actions:10.000",
        "remove_event",
        "No contact at frame 300",
        "reviewer",
        **options,
    )


def test_snapshot_dedup_provenance_no_implicit_labels(store):
    path = action_file()
    before = path.read_bytes()
    r = review()
    assert r == review()
    feedback.decide(
        r["id"], "actions:10.000", "accepted_feedback", "Curated hide", "reviewer"
    )
    assert path.read_bytes() == before
    assert feedback.load(r["id"])["decisions"]["actions:10.000"]["actor"] == "reviewer"
    assert feedback.list_reviews()[0]["reviewed"] == 1


def test_false_positive_import_preserves_others_clears_done(store):
    path = action_file()
    r = review()
    saved, changed = apply(r)
    assert changed == path
    assert [e["id"] for e in read_jsonl(path)[1]] == ["other"]
    assert store == [("game", "action", False)]
    decision = saved["decisions"]["actions:10.000"]
    assert decision["before_sha256"] != decision["after_sha256"]
    with pytest.raises(ValueError, match="already applied"):
        apply(r)


def test_missing_human_file_stale_source_and_wrong_duration(store):
    r = review()
    with pytest.raises(ValueError, match="完整審核"):
        apply(r)
    path = action_file()
    with pytest.raises(ValueError, match="Confirm identical"):
        apply(r, same_source=False)
    with pytest.raises(ValueError, match="已變更"):
        apply(r, revision="old")
    meta, rows = read_jsonl(path)
    write_jsonl(path, {**meta, "num_frames": 1000}, rows)
    with pytest.raises(ValueError, match="duration"):
        apply(r)
    assert len(read_jsonl(path)[1]) == 2


def test_ambiguous_event_no_join(store):
    b = bundle(
        [("spike", 10), ("spike", 10)],
        corrections={"actions": [correction("10.000", removed=True)]},
    )
    assert not feedback.candidates(b)[0]["can_remove_event"]


def test_rally_delete_preserves_id_high_water(store):
    path = feedback.target_path("game.mp4", "remove_rally")
    write_jsonl(
        path,
        {"duration": 100, "max_rally_id": 99},
        [{"rally_id": 42, "start": 9, "end": 20, "label": "rally", "winner": "left"}],
    )
    r = review()
    feedback.decide(
        r["id"],
        "rally:1",
        "remove_rally",
        "No rally here",
        "reviewer",
        video="game.mp4",
        revision=feedback.target_revision("game.mp4", "remove_rally"),
        same_source=True,
    )
    meta, rows = read_jsonl(path)
    assert rows == [] and meta["max_rally_id"] == 99
    assert store == [("game", "rally", False)]


def test_crash_after_label_write_recovers(store, monkeypatch):
    path = action_file()
    r = review()
    original = feedback.save

    def crash(p, value):
        if value["decisions"]:
            raise OSError("disk failed")
        original(p, value)

    monkeypatch.setattr(feedback, "save", crash)
    with pytest.raises(OSError):
        apply(r)
    assert len(read_jsonl(path)[1]) == 1
    assert feedback.load(r["id"])["application"] is not None
    monkeypatch.setattr(feedback, "save", original)
    result, _ = feedback.recover(r["id"])
    assert result["application"] is None
    assert len(read_jsonl(path)[1]) == 1


def test_recovery_rejects_external_edits(store, monkeypatch):
    path = action_file()
    r = review()

    def crash(*args):
        raise OSError("stop")

    monkeypatch.setattr(feedback.label_done, "set_done", crash)
    with pytest.raises(OSError):
        apply(r)
    write_jsonl(path, {"fps": 30}, [])
    with pytest.raises(ValueError, match="conflicts"):
        feedback.recover(r["id"])


def test_bad_paths_partial_snapshots(store):
    for value in ("../game.mp4", "/tmp/game.mp4", "game", "../../secret.json"):
        with pytest.raises(ValueError):
            feedback.target_path(value, "remove_event")
    with pytest.raises(ValueError):
        feedback.load("../x")
    b = bundle()
    b.result.partial = True
    with pytest.raises(ValueError, match="Partial"):
        feedback.create(b, "reader")


def test_reviewed_library_bounds_update_preserves_winner_and_identity(store):
    b = bundle(
        library_rallies=[
            {
                "id": "11111111-1111-4111-8111-111111111111",
                "match_id": "match",
                "index": 1,
                "set": 1,
                "start": 10,
                "end": 21,
            }
        ]
    )
    b.corrections = None
    path = feedback.target_path("game.mp4", "update_rally")
    write_jsonl(
        path,
        {"duration": 100, "max_rally_id": 50},
        [{"rally_id": 42, "start": 9, "end": 20, "label": "rally", "winner": "left"}],
    )
    review = feedback.create(b, "reader")
    feedback.decide(
        review["id"],
        "rally_bounds:1",
        "update_rally",
        "Verified serve and dead ball",
        "reviewer",
        video="game.mp4",
        revision=feedback.target_revision("game.mp4", "update_rally"),
        same_source=True,
    )
    assert read_jsonl(path)[1] == [
        {"rally_id": 42, "start": 10, "end": 21, "label": "rally", "winner": "left"}
    ]


def test_library_bounds_cannot_overlap_neighbor(store):
    b = bundle(
        library_rallies=[
            {
                "id": "11111111-1111-4111-8111-111111111111",
                "match_id": "match",
                "index": 1,
                "set": 1,
                "start": 10,
                "end": 31,
            }
        ]
    )
    path = feedback.target_path("game.mp4", "update_rally")
    rows = [
        {"rally_id": 1, "start": 9, "end": 20, "label": "rally"},
        {"rally_id": 2, "start": 30, "end": 40, "label": "rally"},
    ]
    write_jsonl(path, {"duration": 100}, rows)
    r = feedback.create(b, "reader")
    with pytest.raises(ValueError, match="overlap"):
        feedback.decide(
            r["id"],
            "rally_bounds:1",
            "update_rally",
            "Verified",
            "reviewer",
            video="game.mp4",
            revision=feedback.target_revision("game.mp4", "update_rally"),
            same_source=True,
        )
    assert read_jsonl(path)[1] == rows


def test_http_preview_review_apply_and_reload(store, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from yp_video.web.routers import app_review

    app = FastAPI()
    app.include_router(app_review.router, prefix="/api/app-review")
    monkeypatch.setattr(app_review, "current_actor", lambda: "reviewer@example.com")
    mirrored = []
    monkeypatch.setattr(app_review, "sync_to_r2", lambda *args: mirrored.append(args))
    path = action_file()
    b = bundle(
        [("spike", 10)], corrections={"actions": [correction("10.000", removed=True)]}
    )
    with TestClient(app) as client:
        raw = client.post(
            "/api/app-review/preview",
            json={"bundle": b.model_dump(), "corrected": False},
        )
        assert raw.status_code == 200 and len(raw.json()["actions"]) == 1
        created = client.post("/api/app-review/reviews", json=b.model_dump())
        assert created.status_code == 200
        data = created.json()
        assert data["preview"]["actions"] == []
        assert data["candidates"][0]["clip"]["event_id"] == "f300"
        target = client.get(
            "/api/app-review/target", params={"video": "game.mp4"}
        ).json()
        body = {
            "candidate_id": "actions:10.000",
            "decision": "remove_event",
            "note": "Verified no contact",
            "video": "game.mp4",
            "revision": target["remove_event"],
            "same_source": True,
        }
        applied = client.post(
            f"/api/app-review/reviews/{data['id']}/decisions", json=body
        )
        assert applied.status_code == 200, applied.text
        assert mirrored == [(path, "action/annotations")]
        assert (
            client.post(
                f"/api/app-review/reviews/{data['id']}/decisions", json=body
            ).status_code
            == 409
        )
        saved = client.get(f"/api/app-review/reviews/{data['id']}").json()
        assert saved["decisions"]["actions:10.000"]["actor"] == "reviewer@example.com"
        assert (
            client.post(
                "/api/app-review/reviews", json={**b.model_dump(), "typo": 1}
            ).status_code
            == 422
        )
        assert client.get("/api/app-review/reviews").json()[0]["reviewed"] == 1


def test_customer_source_joins_explicit_run_and_preserves_artifact_metadata(
    monkeypatch,
):
    from yp_video.web.app_review import sources

    b = bundle(
        [("spike", 10)], corrections={"actions": [correction("10.000", removed=True)]}
    )
    result = b.result.model_dump()
    result["actions_r2_key"] = "results/user/match/job.actions.jsonl"
    payloads = {
        "results/user/match/job.json": result,
        "corrections/user/match.json": b.corrections.model_dump(),
    }
    monkeypatch.setattr(sources.customer, "read_json", payloads.__getitem__)
    loaded = sources.cloud_bundle("user", "match", "job")
    assert loaded.model_dump()["result"]["actions_r2_key"] == result["actions_r2_key"]
    result["job_id"] = "other"
    with pytest.raises(ValueError, match="identity"):
        sources.cloud_bundle("user", "match", "job")
    with pytest.raises(ValueError, match="identifier"):
        sources.cloud_bundle("../user", "match", "job")
