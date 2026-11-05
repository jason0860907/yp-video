"""The rally editor names a cut by its filename: the video route resolves
exactly that name (a path never matches the route), and a save or clip
request carrying a path instead of a filename is refused."""
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from yp_video.web.routers import annotate as routes


def _client():
    app = FastAPI()
    app.include_router(routes.router, prefix="/api/annotate")
    return TestClient(app)


def test_the_route_resolves_a_cut_name_only(tmp_path):
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
    assert seen == ["game.mp4"]


@pytest.mark.parametrize("video", ["/videos/cuts-sideline/game.mp4", "cuts/game.mp4", "game"])
def test_a_save_names_the_cut_by_filename(video):
    with pytest.raises(ValidationError):
        routes.SaveAnnotationsRequest(video=video, duration=1.0, annotations=[])
    with pytest.raises(ValidationError):
        routes.ClipRequest(video=video, segment={"start": 0, "end": 1})
