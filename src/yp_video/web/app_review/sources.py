"""Read-only view of VolleyIQ App libraries: the Worker's admin API for D1 rows,
the customer bucket for the artifacts those rows point at."""

import json
from urllib.parse import quote

import httpx
from botocore.exceptions import ClientError

from yp_video.config import load_env
from yp_video.web.r2_client import R2Client, r2_client

from .models import CORRECTIONS_VERSION, Bundle


class CustomerArtifacts(R2Client):
    @property
    def bucket(self) -> str:
        self._ensure_config()
        return self._config.get("R2_BUCKET_CUSTOMER", "")

    @property
    def configured(self) -> bool:
        self._ensure_config()
        return bool(
            self.bucket
            and self._config.get("R2_ACCESS_KEY_ID")
            and self._config.get("R2_SECRET_ACCESS_KEY")
        )

    def read_json(self, key: str) -> dict:
        response = self._get_client().get_object(Bucket=self.bucket, Key=key)
        with response["Body"] as body:
            payload = body.read(20_000_001)
        if len(payload) > 20_000_000:
            raise ValueError("Artifact exceeds 20 MB")
        return json.loads(payload)


customer = CustomerArtifacts()


class AdminApiError(Exception):
    """The Worker's admin API is unreachable or refused the request."""


class NotFound(Exception):
    """No such user, match or source video in the App library."""


def admin_get(path: str):
    env = load_env()
    base, token = env.get("UPLOAD_SERVICE_URL"), env.get("AUTH_TOKEN")
    if not base or not token:
        raise AdminApiError("UPLOAD_SERVICE_URL and AUTH_TOKEN must be configured")
    try:
        response = httpx.get(
            f"{base.rstrip('/')}/admin{path}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
    except httpx.HTTPError as exc:
        raise AdminApiError(f"VolleyIQ admin API unreachable: {exc}") from exc
    if response.status_code == 404:
        raise NotFound("Not found in the VolleyIQ library")
    if response.is_error:
        raise AdminApiError(f"VolleyIQ admin API returned {response.status_code}")
    return response.json()


def users() -> list[dict]:
    return admin_get("/users")


def library(user: str) -> dict:
    return admin_get(f"/users/{quote(user, safe='')}/library")


def library_match(user: str, match: str) -> tuple[dict, dict]:
    lib = library(user)
    row = next((m for m in lib["matches"] if m["id"] == match), None)
    if row is None or row["deleted_at"] is not None:
        raise NotFound("Match not in this user's library")
    return lib, row


def match_bundle(lib: dict, row: dict) -> tuple[Bundle, list[str]]:
    """The App's inputs for one match: the latest analysis the App shows, the
    user's corrections, the identification they point at, and the rallies as
    synced — trims, UUIDs and deletions included. Also returns what the App
    would have silently dropped."""
    keys = row["r2_keys"]
    if not keys["result"]:
        raise ValueError("This match has no analysis result yet")
    notes = []
    corrections = (
        customer.read_json(keys["corrections"]) if keys["corrections"] else None
    )
    # The App discards a blob of any other schema version (it re-uploads its
    # own on the next edit), so the user sees this match uncorrected.
    if corrections and corrections.get("schema_version") != CORRECTIONS_VERSION:
        notes.append(
            f"修正檔是 {corrections.get('schema_version')} 版，App 只讀 "
            f"{CORRECTIONS_VERSION}：使用者目前看到的是未修正的結果。"
        )
        corrections = None
    bundle = Bundle.model_validate(
        {
            "result": customer.read_json(keys["result"]),
            "corrections": corrections,
            "library_rallies": [
                r for r in lib["rallies"] if r["match_id"] == row["id"]
            ],
        }
    )
    # The corrections name the identification run their unit mapping belongs
    # to, which need not be the match's latest one.
    pi = bundle.corrections.player_identification if bundle.corrections else None
    if pi and pi.result_id:
        run = safe_component(pi.result_id)
        key = f"reid/{row['owner_id']}/{row['id']}/{run}.json"
        try:
            identification = customer.read_json(key)
        except ClientError as exc:
            # A pruned run: the projection says the unit mapping is missing.
            if exc.response["Error"]["Code"] != "NoSuchKey":
                raise
        else:
            bundle = Bundle.model_validate(
                {**bundle.model_dump(), "identification": identification}
            )
    return bundle, notes


def safe_component(value: str) -> str:
    if not value or any(
        not (c.isascii() and (c.isalnum() or c in "_-")) for c in value
    ):
        raise ValueError("Invalid artifact identifier")
    return value


def video_url(row: dict) -> str:
    """Where the App would stream the source from: its public customer URL,
    or a signed pipeline-bucket URL for operator-published cuts."""
    source = row["source_video"]
    if not source or not source["r2_key"]:
        raise NotFound("This match has no source video")
    return source["public_url"] or r2_client.generate_presigned_url(source["r2_key"])
