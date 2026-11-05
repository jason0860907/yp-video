"""The server already percent-decodes path parameters once; a second decode
in the route would turn a literal ``%41`` in a filename into ``A``.

Driven through httpx's ASGITransport, which decodes the path once like
uvicorn — Starlette's TestClient decodes it twice and would hide the bug."""
import asyncio
from unittest.mock import patch

import httpx
from fastapi import FastAPI
from fastapi.responses import Response

from yp_video.web.routers import action_annotate as routes


def test_action_video_name_is_decoded_once(tmp_path):
    cut = tmp_path / "50%41off.mp4"
    seen = []

    def resolve(name):
        seen.append(name)
        return cut

    app = FastAPI()
    app.include_router(routes.router, prefix="/api/action-annotate")
    with (
        patch.object(routes, "resolve_cut", side_effect=resolve),
        patch.object(routes, "serve_video_or_r2_redirect", return_value=Response(b"mp4")),
    ):
        res = asyncio.run(_get(app, "/api/action-annotate/video/50%2541off.mp4"))
    assert res.content == b"mp4"
    assert seen == ["50%41off.mp4"]


async def _get(app, path):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get(path)
