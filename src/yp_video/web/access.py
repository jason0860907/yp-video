"""Cloudflare Access is the login. This module is the origin-side check.

Access sits in front of the named tunnel and stamps every proxied request with
a signed JWT in `Cf-Access-Jwt-Assertion`. We verify that signature against the
team's JWKS. The plaintext `Cf-Access-Authenticated-User-Email` header riding
alongside it is NOT trusted: it is a header like any other, and the whole point
of an audit trail is that the name on a row cannot be chosen by the person
making the request.

There is no unauthenticated path and no exemption list — not even the SPA
shell. uvicorn listens on loopback and cloudflared is its only client, so an
unauthenticated request reaching the origin is by definition an anomaly.
"Unconfigured means open" would be the one hole in the trail, and it would stay
invisible until somebody used it.
"""

import asyncio
import logging
from contextvars import ContextVar
from typing import Any

import jwt
from jwt import PyJWKClient
from starlette.responses import JSONResponse

from yp_video.config import ENV_PATH, load_env

log = logging.getLogger(__name__)

#: The verified identity serving the current request. Read by web.audit and by
#: JobManager.create_job, which stamps it onto jobs that outlive the request.
#: uvicorn runs each request in a fresh contextvars.Context, so there is no
#: leakage between requests and nothing to reset.
_actor: ContextVar[str] = ContextVar("access_actor", default="")


def current_actor() -> str:
    """Who is making the request being served on this task, or ""."""
    return _actor.get()


class AccessNotConfigured(RuntimeError):
    """The workspace .env is missing the team domain or the AUD tag."""


class AccessDenied(Exception):
    """The request carried no usable Access identity."""


class _Verifier:
    """Verifies Access assertions for one configured application."""

    def __init__(self) -> None:
        self._jwks: PyJWKClient | None = None
        self._issuer = ""
        self._aud = ""

    def configure(self) -> None:
        """Read the Access keys. Raises AccessNotConfigured if incomplete.

        Called from the app lifespan, never at import: the layering tests
        import web.app on machines with no .env at all.
        """
        env = load_env()
        team = env.get("CF_ACCESS_TEAM_DOMAIN", "").strip()
        aud = env.get("CF_ACCESS_AUD", "").strip()
        if not team or not aud:
            raise AccessNotConfigured(
                "CF_ACCESS_TEAM_DOMAIN and CF_ACCESS_AUD must both be set in "
                f"{ENV_PATH}. Copy .env.example beside it and fill both in from "
                "the Cloudflare Zero Trust dashboard "
                "(Access → Applications → your app → Application Audience Tag)."
            )
        self._issuer = f"https://{team}"
        self._aud = aud
        # Access rotates signing keys every six weeks and honours the old key
        # for a week, so a ten-minute cache is generous. The constructor makes
        # no network call.
        self._jwks = PyJWKClient(
            f"{self._issuer}/cdn-cgi/access/certs",
            cache_jwk_set=True,
            lifespan=600,
            timeout=5,
        )
        log.info("Cloudflare Access configured for %s", self._issuer)

    def email(self, token: str) -> str:
        """The verified identity in *token*.

        Blocking: on an unknown key id PyJWKClient refetches the JWKS over
        urllib. Callers run this in a thread — doing it inline would stall the
        event loop for every bogus request.
        """
        if self._jwks is None:
            raise AccessNotConfigured("access verifier is not configured")
        if not token:
            raise AccessDenied("no Access assertion")
        try:
            key = self._jwks.get_signing_key_from_jwt(token).key
            claims: dict[str, Any] = jwt.decode(
                token,
                key,
                algorithms=["RS256"],
                audience=self._aud,
                issuer=self._issuer,
            )
        except Exception as e:  # noqa: BLE001 — every verify failure is a denial
            raise AccessDenied(str(e)) from e
        # Service tokens carry no email; a human always does. Falling back to
        # the common name or subject keeps such a row attributable.
        identity = claims.get("email") or claims.get("common_name") or claims.get("sub")
        if not identity:
            raise AccessDenied("assertion carries no identity")
        return identity


verifier = _Verifier()

_HEADER = b"cf-access-jwt-assertion"


class AccessAuth:
    """Pure ASGI: no request is served without a verified identity.

    Not BaseHTTPMiddleware — the job SSE streams must pass through untouched,
    the same reason _ApiNoStore is written this way.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        token = next((v.decode() for k, v in scope["headers"] if k == _HEADER), "")
        try:
            email = await asyncio.to_thread(verifier.email, token)
        except AccessDenied as e:
            log.warning("Access assertion rejected for %s: %s", scope["path"], e)
            # 403, not 401: a 401 promises a WWW-Authenticate challenge the
            # browser cannot satisfy. Access is the real gate; this is the
            # second lock behind it.
            response = JSONResponse(
                {"detail": "Cloudflare Access identity required"}, status_code=403
            )
            await response(scope, receive, send)
            return

        _actor.set(email)
        scope.setdefault("state", {})["actor"] = email
        await self.app(scope, receive, send)
