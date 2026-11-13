"""Training requests sent from the DEBUG App (upload-service
``training_requests``): taken from the Worker's admin API, run through
``training.add_to_training``, reported back.

A request whose match cannot go in (``ValueError`` / ``NotFound``, or an R2
object it needs that is gone) is failed with the reason, which the App shows. Anything else — the admin API or R2
unreachable — leaves it pending for the next round.
"""

import logging
import time

from botocore.exceptions import ClientError

from yp_video.config import load_env

from . import sources, training

log = logging.getLogger(__name__)


def process_pending() -> int:
    """Run every pending request once; returns how many were finished."""
    finished = 0
    for req in sources.admin_get("/training-requests?status=pending"):
        try:
            training.add_to_training(req["user_id"], req["match_id"], req["angle"])
        except (ValueError, sources.NotFound) as exc:
            outcome = {"status": "failed", "error": str(exc)}
        except ClientError as exc:
            if not _missing_object(exc):
                log.warning("training request %s deferred", req["id"], exc_info=True)
                continue
            outcome = {"status": "failed", "error": f"A file of this match is gone from R2: {exc}"}
        except Exception:  # noqa: BLE001 — transient: retried next round
            log.warning("training request %s deferred", req["id"], exc_info=True)
            continue
        else:
            outcome = {"status": "done", "error": None}
        sources.admin_post(f"/training-requests/{req['id']}", outcome)
        finished += 1
    return finished


def _missing_object(exc: ClientError) -> bool:
    return exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404", "NotFound")


def poll_forever(interval_s: float = 120.0) -> None:
    """The yp-app daemon thread: one round every ``interval_s``."""
    while True:
        try:
            process_pending()
        except Exception:  # noqa: BLE001
            log.warning("training request poll failed", exc_info=True)
        time.sleep(interval_s)


def configured() -> bool:
    env = load_env()
    return bool(env.get("UPLOAD_SERVICE_URL") and env.get("AUTH_TOKEN"))
