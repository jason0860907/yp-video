"""Cloudflare Access is the only way in.

The middleware verifies a signed assertion rather than reading the plaintext
email header, because a header can be typed by whoever is making the request
and the audit trail is worthless if the name on a row can be chosen.

These tests sign their own tokens with a throwaway key and stub the JWKS
lookup, so nothing here touches the network or a real Cloudflare team.
"""

from __future__ import annotations

import contextlib
import unittest
from datetime import UTC, datetime, timedelta

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI
from fastapi.testclient import TestClient

from yp_video.web import access

TEAM = "example.cloudflareaccess.com"
ISSUER = f"https://{TEAM}"
AUD = "aud-tag-under-test"

_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _token(**overrides) -> str:
    now = datetime.now(UTC)
    claims = {
        "iss": ISSUER,
        "aud": AUD,
        "email": "labeler@example.com",
        "sub": "subject-1",
        "iat": now,
        "exp": now + timedelta(hours=1),
    }
    claims.update(overrides)
    return jwt.encode(claims, _KEY, algorithm="RS256")


class _StubJwks:
    """Stands in for PyJWKClient, handing back the one key we signed with."""

    class _Key:
        key = _KEY.public_key()

    def get_signing_key_from_jwt(self, token):  # noqa: ARG002 — one key only
        return self._Key()


@contextlib.contextmanager
def _configured():
    """The verifier wired to our stub, restored afterwards."""
    saved = (access.verifier._jwks, access.verifier._issuer, access.verifier._aud)
    access.verifier._jwks = _StubJwks()
    access.verifier._issuer = ISSUER
    access.verifier._aud = AUD
    try:
        yield
    finally:
        (
            access.verifier._jwks,
            access.verifier._issuer,
            access.verifier._aud,
        ) = saved


def _client() -> TestClient:
    app = FastAPI()

    @app.get("/api/who")
    def who() -> dict:
        return {"email": access.current_actor()}

    @app.get("/")
    def shell() -> dict:
        return {"ok": True}

    app.add_middleware(access.AccessAuth)
    return TestClient(app)


class AccessGateTests(unittest.TestCase):
    def test_valid_assertion_identifies_the_caller(self) -> None:
        with _configured(), _client() as client:
            res = client.get("/api/who", headers={"Cf-Access-Jwt-Assertion": _token()})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"email": "labeler@example.com"})

    def test_service_token_without_email_still_attributable(self) -> None:
        """A row with no name at all would be worse than one naming the client."""
        token = _token(email=None, common_name="ci-runner")
        with _configured(), _client() as client:
            res = client.get("/api/who", headers={"Cf-Access-Jwt-Assertion": token})
        self.assertEqual(res.json(), {"email": "ci-runner"})

    def test_missing_header_is_denied(self) -> None:
        with _configured(), _client() as client:
            self.assertEqual(client.get("/api/who").status_code, 403)

    def test_wrong_audience_is_denied(self) -> None:
        """An assertion minted for a different Access app must not be reusable."""
        with _configured(), _client() as client:
            res = client.get(
                "/api/who",
                headers={"Cf-Access-Jwt-Assertion": _token(aud="someone-elses-app")},
            )
        self.assertEqual(res.status_code, 403)

    def test_wrong_issuer_is_denied(self) -> None:
        with _configured(), _client() as client:
            res = client.get(
                "/api/who",
                headers={"Cf-Access-Jwt-Assertion": _token(iss="https://evil.example")},
            )
        self.assertEqual(res.status_code, 403)

    def test_expired_assertion_is_denied(self) -> None:
        past = datetime.now(UTC) - timedelta(hours=2)
        with _configured(), _client() as client:
            res = client.get(
                "/api/who",
                headers={"Cf-Access-Jwt-Assertion": _token(exp=past, iat=past)},
            )
        self.assertEqual(res.status_code, 403)

    def test_forged_signature_is_denied(self) -> None:
        other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        forged = jwt.encode(
            {
                "iss": ISSUER,
                "aud": AUD,
                "email": "attacker@example.com",
                "exp": datetime.now(UTC) + timedelta(hours=1),
            },
            other,
            algorithm="RS256",
        )
        with _configured(), _client() as client:
            res = client.get("/api/who", headers={"Cf-Access-Jwt-Assertion": forged})
        self.assertEqual(res.status_code, 403)

    def test_the_plaintext_email_header_is_not_trusted(self) -> None:
        """The header Access also sets is the one thing an attacker can forge."""
        with _configured(), _client() as client:
            res = client.get(
                "/api/who",
                headers={"Cf-Access-Authenticated-User-Email": "boss@example.com"},
            )
        self.assertEqual(res.status_code, 403)

    def test_no_path_is_exempt(self) -> None:
        """Including the SPA shell: an exemption list is a hole you forget."""
        with _configured(), _client() as client:
            self.assertEqual(client.get("/").status_code, 403)

    def test_unconfigured_verifier_refuses_rather_than_opens(self) -> None:
        with self.assertRaises(access.AccessNotConfigured):
            _Verifier = type(access.verifier)
            _Verifier().email(_token())


class ConfigTests(unittest.TestCase):
    def test_incomplete_env_fails_loudly(self) -> None:
        """The workspace .env carries many keys; missing either Access one is fatal."""
        from unittest.mock import patch

        partials = [
            {},
            {"CF_ACCESS_TEAM_DOMAIN": TEAM},
            {"CF_ACCESS_AUD": AUD},
            {"CF_ACCESS_TEAM_DOMAIN": "", "CF_ACCESS_AUD": AUD},
        ]
        for env in partials:
            with patch.object(access, "load_env", return_value=env):
                with self.assertRaises(access.AccessNotConfigured):
                    type(access.verifier)().configure()

    def test_a_complete_env_configures(self) -> None:
        from unittest.mock import patch

        env = {"CF_ACCESS_TEAM_DOMAIN": TEAM, "CF_ACCESS_AUD": AUD, "R2_ACCOUNT_ID": "x"}
        with patch.object(access, "load_env", return_value=env):
            v = type(access.verifier)()
            v.configure()
        self.assertEqual(v._issuer, ISSUER)


if __name__ == "__main__":
    unittest.main()
