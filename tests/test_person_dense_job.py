"""POST /api/person/dense: which videos may start a dense pass. The pass
itself is stubbed — nothing here touches the GPU or a real video."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from yp_video.web import person_dense_jobs
from yp_video.web.jobs import JobType, job_manager
from yp_video.web.routers import person


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(person, "resolve_cut", lambda name: None if name == "gone.mp4" else tmp_path / name)
    monkeypatch.setattr(person, "load_rallies", lambda stem: [] if stem == "norally" else [{"start": 0, "end": 1}])
    monkeypatch.setattr(person, "dense_path", lambda stem: SimpleNamespace(exists=lambda: stem == "has"))
    spawned = []
    monkeypatch.setattr(person, "spawn_batch_video_job", lambda job, paths, **kw: spawned.append((job, paths, kw)))
    monkeypatch.setattr(job_manager, "jobs", {})
    app = FastAPI()
    app.include_router(person.router, prefix="/api/person")
    with TestClient(app) as c:
        yield c, spawned


def test_starts_one_job_for_the_requested_videos(client):
    c, spawned = client
    r = c.post("/api/person/dense", json={"videos": ["a.mp4", "b.mp4"]})
    assert r.status_code == 200 and r.json()["type"] == JobType.PERSON_DENSE.value
    (job, paths, kw), = spawned
    assert [p.name for p in paths] == ["a.mp4", "b.mp4"]
    assert kw["work"] is person_dense_jobs.run
    assert person_dense_jobs.queued() == {"a.mp4", "b.mp4"}


@pytest.mark.parametrize("video, status", [("gone.mp4", 404), ("norally.mp4", 400), ("has.mp4", 400)])
def test_refuses_what_it_cannot_or_need_not_run(client, video, status):
    c, spawned = client
    assert c.post("/api/person/dense", json={"videos": [video]}).status_code == status
    assert spawned == []


def test_one_dense_job_at_a_time(client):
    c, _ = client
    assert c.post("/api/person/dense", json={"videos": ["a.mp4"]}).status_code == 200
    assert c.post("/api/person/dense", json={"videos": ["b.mp4"]}).status_code == 409


def test_the_model_is_released_even_when_a_video_fails(monkeypatch, tmp_path):
    released = []
    monkeypatch.setattr(person_dense_jobs, "release_detector", lambda: released.append(True))
    monkeypatch.setattr(person_dense_jobs, "materialized_cut", lambda path: (_ for _ in ()).throw(FileNotFoundError(path)))
    with pytest.raises(FileNotFoundError):
        person_dense_jobs.run(Path(tmp_path / "a.mp4"), None)
    assert released == [True]
