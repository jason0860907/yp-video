"""Identity layer over extracted ReID embeddings.

Two consumers, one data source (the per-video reid jsonl):

- ``cluster``: unsupervised grouping of a video's embeddings — the zero-label
  view that shows whether the appearance features separate players at all.
- Player assignments + ``match``: the user names events (usually by naming a
  cluster), which defines per-player centroids; every unassigned event is then
  matched to its nearest centroid with a cosine similarity score. The UI
  decides how to render low-similarity matches.

Assignments persist in reid/annotations/<stem>_players.json as a flat
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

import numpy as np

from yp_video.core.cache import StatCache
from yp_video.core.jsonl import atomic_write, read_jsonl_cached
from yp_video.reid.embedder import DEFAULT_EMBEDDER
from yp_video.reid.store import SKIP_LABELS, load_embedding_matrix, players_path, reid_path, require_embedding_path

# Serializes read-modify-write of the players file: the UI auto-saves
# assignments while actor fixes land, and interleaving would drop one edit.
_players_lock = threading.Lock()

# Average-linkage cosine-distance cutoff on CLIP-ReID's scale — its ViT
# features sit in a tight cone (pairwise distances p5–p95 ≈ 0.12–0.32), so
# cutoffs are far smaller than typical CNN-feature values. The UI exposes it.
DEFAULT_CLUSTER_THRESHOLD = 0.15

# The threshold slider's hot path, keyed (stem, model) on the two source
# files. Values are shared — read-only, like everything read_jsonl_cached
# hands out. The linkage tree is threshold-independent, so a slider drag
# re-runs only fcluster (see cluster_video).
_emb_cache: StatCache = StatCache()
_linkage_cache: StatCache = StatCache()


def load_embeddings(stem: str, model: str = DEFAULT_EMBEDDER) -> tuple[list[dict], np.ndarray]:
    """Records with an embedding under ``model``, plus their (N, dim)
    L2-normalized matrix. Cached on the source files — SHARED, read-only.

    Records come from the reid jsonl, vectors from the model's npy sidecar
    (row i ↔ record i, NaN row = not embedded — see reid/store.py).
    """
    path = reid_path(stem)
    if not path.exists():
        raise FileNotFoundError(f"No ReID results for {stem}")
    return _emb_cache.get(
        (stem, model), [path, require_embedding_path(stem, model)], lambda: _load_embeddings(stem, model, path)
    )


def _load_embeddings(stem: str, model: str, path) -> tuple[list[dict], np.ndarray]:
    _meta, records = read_jsonl_cached(path)  # read-only from here on
    matrix = load_embedding_matrix(stem, model)
    if len(matrix) != len(records):
        raise ValueError(
            f"{model} embeddings for {stem} have {len(matrix)} rows for {len(records)} records — re-run embedding"
        )

    embedded = [bool(v) for v in np.asarray(np.isfinite(matrix).all(axis=1))]
    # SKIP_LABELS guards extractions that predate the skip rule.
    keep = [i for i, r in enumerate(records) if embedded[i] and r.get("label") not in SKIP_LABELS]
    matrix = matrix[keep]
    if len(keep):
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    return [records[i] for i in keep], matrix


def cluster(matrix: np.ndarray, threshold: float = DEFAULT_CLUSTER_THRESHOLD) -> np.ndarray:
    """Average-linkage agglomerative clustering on cosine distance.

    Returns int labels aligned with the matrix rows; clusters are renumbered
    by descending size so cluster 0 is always the biggest.
    """
    return _cut(_linkage(matrix), matrix, threshold)


def cluster_sweep(matrix: np.ndarray, thresholds) -> list[np.ndarray]:
    """Cluster labels at each threshold, sharing ONE linkage.

    The tree is threshold-independent, so a sweep costs a single O(n²) build
    plus an O(n) cut per threshold — calling cluster() in a loop would rebuild
    it every time. Used by the threshold calibration (see reid/evaluate.py).
    """
    links = _linkage(matrix)
    return [_cut(links, matrix, t) for t in thresholds]


def cluster_video(stem: str, model: str, threshold: float) -> tuple[list[dict], np.ndarray]:
    """One video's records + cluster labels, linkage cached on the source
    files — a threshold change (the UI slider) re-runs only the O(n) cut,
    not the O(n²) linkage."""
    records, matrix = load_embeddings(stem, model=model)
    links = _linkage_cache.get(
        (stem, model), [reid_path(stem), require_embedding_path(stem, model)], lambda: _linkage(matrix)
    )
    return records, _cut(links, matrix, threshold)


def _linkage(matrix: np.ndarray):
    from scipy.cluster.hierarchy import linkage

    return linkage(matrix, method="average", metric="cosine") if len(matrix) > 1 else None


def _cut(links, matrix: np.ndarray, threshold: float) -> np.ndarray:
    from scipy.cluster.hierarchy import fcluster

    n = len(matrix)
    if n == 0:
        return np.empty(0, dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)
    raw = fcluster(links, t=threshold, criterion="distance")
    order = sorted(set(raw), key=lambda c: -(raw == c).sum())
    remap = {c: i for i, c in enumerate(order)}
    return np.array([remap[c] for c in raw], dtype=int)


# Readers go through the cache; writers (under _players_lock) read fresh via
# _read_players_file — they mutate the loaded dict before saving.
_players_cache: StatCache = StatCache()


def _read_players_file(stem: str) -> dict:
    path = players_path(stem)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_players_file(stem: str) -> dict:
    path = players_path(stem)
    if not path.exists():
        return {}
    return _players_cache.get(stem, [path], lambda: _read_players_file(stem))


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
        data = _read_players_file(stem)
        data["assignments"] = {k: v.strip() for k, v in assignments.items() if v and v.strip()}
        _save_players_file(stem, data)


def load_actor_fixes(stem: str) -> dict[str, dict]:
    data = _load_players_file(stem)
    return {str(k): v for k, v in data.get("actor_fixes", {}).items() if isinstance(v, dict)}


def save_actor_fix(
    stem: str, event_id: str, box: list[float] | None, frame: int | None = None, snap: bool = True
) -> None:
    """Record a manual actor pick; ``box=None`` means "nobody is the actor".

    ``frame`` marks a cross-frame pick (the box lives on that frame, not the
    event's); ``snap=False`` forbids the IoU snap onto a stored detection.
    Both replay identically on re-extraction.
    """
    with _players_lock:
        data = _read_players_file(stem)
        fixes = data.setdefault("actor_fixes", {})
        if box is None:
            fixes[event_id] = {"none": True}
        else:
            fixes[event_id] = {"box": [round(float(v), 1) for v in box]}
            if frame is not None:
                fixes[event_id]["frame"] = int(frame)
            if not snap:
                fixes[event_id]["snap"] = False
        _save_players_file(stem, data)


def remove_assignment(stem: str, event_id: str) -> None:
    """Drop one event's player assignment — a re-picked actor is a different
    person, so whatever identity the event carried no longer applies."""
    with _players_lock:
        data = _read_players_file(stem)
        if data.get("assignments", {}).pop(event_id, None) is not None:
            _save_players_file(stem, data)


def remove_actor_fix(stem: str, event_id: str) -> None:
    """Revert an event to the automatic pick."""
    with _players_lock:
        data = _read_players_file(stem)
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
