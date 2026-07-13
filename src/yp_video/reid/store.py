"""Where ReID data lives on disk — paths and record-level policy.

The lowest layer of the reid package: pipeline (orchestration), identity
(matching), tracking (tracklets) and the web router all depend on this
module, never on each other, for where files live and which records count.

Layout under videos/player-reid/:
    embeddings/<stem>_reid.jsonl    per-event extraction records
    crops/<stem>/<event>.jpg        actor crops (display box)
    players/<stem>_players.json     assignments + actor fixes
    tracks/<stem>_tracks.jsonl      per-rally ByteTrack tracklets
"""

from __future__ import annotations

from pathlib import Path

from yp_video.config import (
    ACTION_ANNOTATIONS_DIR,
    ACTION_PRE_ANNOTATIONS_DIR,
    PLAYER_REID_DIR,
)

EMBEDDINGS_DIR = PLAYER_REID_DIR / "embeddings"
CROPS_DIR = PLAYER_REID_DIR / "crops"
PLAYERS_DIR = PLAYER_REID_DIR / "players"
TRACKS_DIR = PLAYER_REID_DIR / "tracks"

# Action labels with nobody to re-identify: "score" marks where the ball
# lands, not a person. Applied at extraction AND at read time, so old
# extractions that predate the rule stay filtered too.
SKIP_LABELS = frozenset({"score"})


def reid_path(stem: str) -> Path:
    return EMBEDDINGS_DIR / f"{stem}_reid.jsonl"


def crop_dir(stem: str) -> Path:
    return CROPS_DIR / stem


def players_path(stem: str) -> Path:
    return PLAYERS_DIR / f"{stem}_players.json"


def tracks_path(stem: str) -> Path:
    return TRACKS_DIR / f"{stem}_tracks.jsonl"


def action_annotation_path(stem: str) -> Path | None:
    """Manual action annotations win over pre-annotations."""
    for directory in (ACTION_ANNOTATIONS_DIR, ACTION_PRE_ANNOTATIONS_DIR):
        path = directory / f"{stem}_actions.jsonl"
        if path.exists():
            return path
    return None
