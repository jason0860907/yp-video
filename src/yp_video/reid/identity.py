"""Identity layer over extracted ReID embeddings.

Two consumers, one data source (the per-video reid jsonl):

- ``cluster``: unsupervised grouping of a video's embeddings — the zero-label
  view that shows whether the appearance features separate players at all.
- Player assignments + ``match``: the user names events (usually by naming a
  cluster), which defines per-player centroids; every unassigned event is then
  matched to its nearest centroid with a cosine similarity score. The UI
  decides how to render low-similarity matches.

Assignments persist in player-reid/players/<stem>_players.json as a flat
``{event_id: player_name}`` map — the smallest thing that can express both
cluster-level and single-event corrections.

The same file carries ``actor_fixes``: box-level corrections for events where
the automatic contact-point association picked the wrong person
(``{event_id: {"box": [x0,y0,x1,y1]} | {"none": true}}``). Fixes are the
durable human record — extraction replays them (matching boxes by IoU against
fresh detections), so re-extraction never loses correction work. They are
also the training set for a learned actor-association model later: each fix
is a labeled (frame, action point, candidate boxes) → correct-box example.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

from yp_video.config import PLAYER_REID_DIR
from yp_video.core.jsonl import atomic_write, read_jsonl_cached
from yp_video.reid.embedder import DEFAULT_EMBEDDER
from yp_video.reid.pipeline import SKIP_LABELS, reid_path

PLAYERS_DIR = PLAYER_REID_DIR / "players"

# Serializes read-modify-write of the players file: the UI auto-saves
# assignments while actor fixes land, and interleaving would drop one edit.
_players_lock = threading.Lock()

# Average-linkage cosine-distance cutoff on CLIP-ReID's scale — its ViT
# features sit in a tight cone (pairwise distances p5–p95 ≈ 0.12–0.32), so
# cutoffs are far smaller than typical CNN-feature values. The UI exposes it.
DEFAULT_CLUSTER_THRESHOLD = 0.15


def players_path(stem: str) -> Path:
    return PLAYERS_DIR / f"{stem}_players.json"


def load_embeddings(stem: str, model: str = DEFAULT_EMBEDDER) -> tuple[list[dict], np.ndarray]:
    """Records carrying the chosen model's embedding, plus their (N, 512) matrix."""
    path = reid_path(stem)
    if not path.exists():
        raise FileNotFoundError(f"No ReID results for {stem}")
    _meta, records = read_jsonl_cached(path)  # read-only from here on

    def vec(r: dict) -> list[float] | None:
        return (r.get("embeddings") or {}).get(model)

    # SKIP_LABELS guards extractions that predate the skip rule.
    with_emb = [r for r in records if vec(r) and r.get("label") not in SKIP_LABELS]
    matrix = np.array([vec(r) for r in with_emb], dtype=np.float32)
    if len(with_emb):
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    return with_emb, matrix


def cluster(matrix: np.ndarray, threshold: float = DEFAULT_CLUSTER_THRESHOLD) -> np.ndarray:
    """Average-linkage agglomerative clustering on cosine distance.

    Returns int labels aligned with the matrix rows; clusters are renumbered
    by descending size so cluster 0 is always the biggest.
    """
    from scipy.cluster.hierarchy import fcluster, linkage

    n = len(matrix)
    if n == 0:
        return np.empty(0, dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)
    links = linkage(matrix, method="average", metric="cosine")
    raw = fcluster(links, t=threshold, criterion="distance")
    order = sorted(set(raw), key=lambda c: -(raw == c).sum())
    remap = {c: i for i, c in enumerate(order)}
    return np.array([remap[c] for c in raw], dtype=int)


def _load_players_file(stem: str) -> dict:
    path = players_path(stem)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_players_file(stem: str, data: dict) -> None:
    # Atomic replace — auto-save and actor fixes hit this file concurrently.
    with atomic_write(players_path(stem)) as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


def seeded_groups(
    records: list[dict],
    matrix: np.ndarray,
    seeds: dict[str, list[str]],
    cutoff: float,
) -> tuple[dict[str, list[str]], list[str]]:
    """Distribute every non-seed event to its nearest seed centroid.

    ``seeds`` maps a caller-chosen key to the event ids anchoring that group.
    Each seed group's centroid is the mean of its members' embeddings; every
    other embedded event joins the closest centroid when its cosine distance
    is within ``cutoff``, otherwise it lands in the returned leftover list.
    Turns clustering into classification once the user has pinned one clean
    group per player.
    """
    index = {r["id"]: i for i, r in enumerate(records)}
    seed_members = {i for ids in seeds.values() for i in ids}
    keys: list[str] = []
    centroids = []
    for key, ids in seeds.items():
        rows = [index[i] for i in ids if i in index]
        if not rows:
            continue
        c = matrix[rows].mean(axis=0)
        centroids.append(c / (np.linalg.norm(c) + 1e-12))
        keys.append(key)
    out: dict[str, list[str]] = {k: [] for k in keys}
    leftover: list[str] = []
    if not keys:
        return out, [r["id"] for r in records if r["id"] not in seed_members]
    sims = matrix @ np.stack(centroids).T  # (N, S)
    for i, record in enumerate(records):
        if record["id"] in seed_members:
            continue
        best = int(np.argmax(sims[i]))
        if 1.0 - float(sims[i][best]) <= cutoff:
            out[keys[best]].append(record["id"])
        else:
            leftover.append(record["id"])
    return out, leftover


def load_assignments(stem: str) -> dict[str, str]:
    data = _load_players_file(stem)
    return {str(k): str(v) for k, v in data.get("assignments", {}).items()}


def save_assignments(stem: str, assignments: dict[str, str]) -> None:
    with _players_lock:
        data = _load_players_file(stem)
        data["assignments"] = {k: v.strip() for k, v in assignments.items() if v and v.strip()}
        _save_players_file(stem, data)


def load_actor_fixes(stem: str) -> dict[str, dict]:
    data = _load_players_file(stem)
    return {str(k): v for k, v in data.get("actor_fixes", {}).items() if isinstance(v, dict)}


def save_actor_fix(stem: str, event_id: str, box: list[float] | None) -> None:
    """Record a manual actor pick; ``box=None`` means "nobody is the actor"."""
    with _players_lock:
        data = _load_players_file(stem)
        fixes = data.setdefault("actor_fixes", {})
        fixes[event_id] = {"none": True} if box is None else {"box": [round(float(v), 1) for v in box]}
        _save_players_file(stem, data)


def remove_actor_fix(stem: str, event_id: str) -> None:
    """Revert an event to the automatic pick."""
    with _players_lock:
        data = _load_players_file(stem)
        if data.get("actor_fixes", {}).pop(event_id, None) is not None:
            _save_players_file(stem, data)


def match(
    records: list[dict], matrix: np.ndarray, assignments: dict[str, str]
) -> dict[str, dict]:
    """Nearest-centroid match for every embedded event.

    Returns ``{event_id: {player, sim, assigned}}``. Assigned events keep
    their label with sim 1.0; the rest get the closest player centroid and
    the cosine similarity to it.
    """
    if not assignments or not len(matrix):
        return {}
    index = {r["id"]: i for i, r in enumerate(records)}
    by_player: dict[str, list[int]] = {}
    for event_id, player in assignments.items():
        if event_id in index:
            by_player.setdefault(player, []).append(index[event_id])
    if not by_player:
        return {}

    players = sorted(by_player)
    centroids = np.stack([matrix[rows].mean(axis=0) for rows in (by_player[p] for p in players)])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12
    sims = matrix @ centroids.T  # (N, P)

    out: dict[str, dict] = {}
    for i, record in enumerate(records):
        event_id = record["id"]
        if event_id in assignments:
            out[event_id] = {"player": assignments[event_id], "sim": 1.0, "assigned": True}
        else:
            best = int(np.argmax(sims[i]))
            out[event_id] = {"player": players[best], "sim": round(float(sims[i][best]), 4), "assigned": False}
    return out
