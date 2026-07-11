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
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from yp_video.config import PLAYER_REID_DIR
from yp_video.core.jsonl import read_jsonl
from yp_video.reid.embedder import DEFAULT_EMBEDDER
from yp_video.reid.pipeline import reid_path

PLAYERS_DIR = PLAYER_REID_DIR / "players"

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
    _meta, records = read_jsonl(path)

    def vec(r: dict) -> list[float] | None:
        return (r.get("embeddings") or {}).get(model)

    with_emb = [r for r in records if vec(r)]
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


def load_assignments(stem: str) -> dict[str, str]:
    path = players_path(stem)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in data.get("assignments", {}).items()}


def save_assignments(stem: str, assignments: dict[str, str]) -> None:
    PLAYERS_DIR.mkdir(parents=True, exist_ok=True)
    clean = {k: v.strip() for k, v in assignments.items() if v and v.strip()}
    players_path(stem).write_text(
        json.dumps({"assignments": clean}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )


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
