"""System router - identity and presence."""

import time

from fastapi import APIRouter
from pydantic import Field

from yp_video.web.access import current_actor
from yp_video.web.schemas import StrictModel

router = APIRouter()


@router.get("/me")
def me() -> dict:
    """Who Cloudflare Access says is making this request.

    The top bar shows it, so a labeler can see which identity their actions
    are being recorded under.
    """
    return {"email": current_actor()}

# ── Presence (who has the page open right now) ────────────────────
# Each browser sends a heartbeat with a persistent random id every ~30 s;
# a client counts as online while its last beat is younger than the TTL, and
# as active while its latest beat says the user recently interacted (the
# idle threshold lives client-side). In-memory on purpose: a restart
# repopulates within one heartbeat.
_PRESENCE_TTL_S = 75.0
_presence: dict[str, tuple[float, bool]] = {}  # client_id -> (last_seen, is_active)


class PresenceBeat(StrictModel):
    client_id: str = Field(min_length=8, max_length=64)
    # False once the user has gone idle (no input past the client threshold).
    active: bool = True


@router.post("/presence")
def presence(beat: PresenceBeat) -> dict:
    """Record one heartbeat and return online/active client counts."""
    now = time.monotonic()
    _presence[beat.client_id] = (now, beat.active)
    for cid in [c for c, (seen, _a) in _presence.items() if now - seen > _PRESENCE_TTL_S]:
        del _presence[cid]
    return {
        "online": len(_presence),
        "active": sum(1 for _seen, is_active in _presence.values() if is_active),
    }

