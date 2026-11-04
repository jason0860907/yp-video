"""The rally editor's video route serves cuts only: whatever path the client
sends, just its basename is resolved, so no other file on disk is reachable."""
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from yp_video.web.routers import annotate as routes


def _client():
    app = FastAPI()
    app.include_router(routes.router, prefix="/api/annotate")
    return TestClient(app)


def test_only_the_basename_names_a_cut(tmp_path):
    cut = tmp_path / "game.mp4"
    cut.write_bytes(b"mp4")
    seen = []

    def resolve(name):
        seen.append(name)
        return cut if name == "game.mp4" else None

    with patch.object(routes, "resolve_cut", side_effect=resolve):
        client = _client()
        assert client.get("/api/annotate/video/game.mp4").content == b"mp4"
        assert client.get("/api/annotate/video/%2Fetc%2Fpasswd").status_code == 404
        assert client.get("/api/annotate/video/..%2F..%2Fsecret.mp4").status_code == 404
    assert seen == ["game.mp4", "passwd", "secret.mp4"]
