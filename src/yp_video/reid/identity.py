"""Identity layer over extracted ReID embeddings.

Two consumers, one data source (the per-video extraction records):

- ``cluster``: unsupervised grouping of a video's embeddings — the zero-label
  view that shows whether the appearance features separate players at all.
- Player assignments + ``match``: the user names events (usually by naming a
  cluster), which defines per-player centroids; every unassigned event is then
  matched to its nearest centroid with a cosine similarity score. The UI
  decides how to render low-similarity matches.

Assignments persist in reid/annotations/<stem>_players.json as a flat
``{event_id: player_name}`` map — the smallest thing that can express both
cluster-level and single-event corrections. A ``done: true`` flag marks a
video's labeling as finished (the Label page's Done button) — a human verdict
the counts can't derive.

Who each crop DEPICTS is this module's question; which person performed the
action the crop was cut from is not, and its labels live in their own file
(see actor/labels.py). Keeping them apart is why naming a player and fixing
an actor no longer contend for the same lock.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import json
import threading

import numpy as np

from yp_video.core.cache import StatCache
from yp_video.core.jsonl import atomic_write, read_jsonl_cached
from yp_video.extraction.store import SKIP_LABELS, records_path
from yp_video.reid.embedder import DEFAULT_EMBEDDER
from yp_video.reid.store import load_embedding_matrix, players_path, require_embedding_path

PLAYERS_SCHEMA_VERSION = 2

# Serializes read-modify-write of the players file: the UI auto-saves
# assignments while a Done verdict lands, and interleaving would drop one edit.
_players_lock = threading.RLock()

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

    Records come from the extraction jsonl, vectors from the npy sidecar
    (row i ↔ record i, NaN row = not embedded — see reid/store.py).
    """
    path = records_path(stem)
    if not path.exists():
        raise FileNotFoundError(f"No extraction records for {stem}")
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


def cluster_video(
    stem: str, model: str, threshold: float, links: Mapping[str, str]
) -> tuple[list[Unit], np.ndarray]:
    """One video's units + cluster labels.

    Clustering runs over UNITS, not crops: two frames of one player are not
    independent evidence that they are the same person, and letting them vote
    twice made every cluster look tighter than it is.

    Linkage is cached on the source files plus a fingerprint of the links —
    re-running tracking regroups the crops, so the tree it produced no longer
    applies. A threshold change alone still re-runs only the O(n) cut.
    """
    records, matrix = load_embeddings(stem, model=model)
    units, unit_matrix = unit_embeddings(records, matrix, links)
    tree = _linkage_cache.get(
        (stem, model, _links_fingerprint(links)),
        [records_path(stem), require_embedding_path(stem, model)],
        lambda: _linkage(unit_matrix),
    )
    return units, _cut(tree, unit_matrix, threshold)


def _links_fingerprint(links: Mapping[str, str]) -> int:
    return hash(frozenset(links.items()))


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
    # Atomic replace — assignment auto-save and the Done button can race.
    with atomic_write(players_path(stem)) as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


@contextmanager
def players_write_transaction() -> Iterator[None]:
    """Hold the players file across a multi-file transaction."""
    with _players_lock:
        yield


# ── Units: what identity is actually about ───────────────────────
# A name belongs to a PERSON, and the longest-lived handle we have on a
# person is the tracklet that follows them through a rally. Naming a
# tracklet names every action it performed; an event with no tracklet is
# still its own unit, so downstream code sees one vocabulary, not two cases.

UNIT_TRACK_PREFIX = "t:"
UNIT_EVENT_PREFIX = "e:"

#: stem → {event_id: "rally:track"}. Injected, because deriving it needs both
#: tracklets and extraction records and reid may import neither together.
LinksFor = Callable[[str], Mapping[str, str]]


def unit_key(event_id: str, track_key: str | None) -> str:
    """The unit an event belongs to — its tracklet, or itself."""
    return f"{UNIT_TRACK_PREFIX}{track_key}" if track_key else f"{UNIT_EVENT_PREFIX}{event_id}"


def track_of_unit(key: str) -> str | None:
    """The "rally:track" a unit stands for, or None for a lone event."""
    return key[len(UNIT_TRACK_PREFIX):] if key.startswith(UNIT_TRACK_PREFIX) else None


@dataclass(frozen=True)
class Unit:
    """One person's crops within a video, as far as tracking can tell."""

    key: str
    event_ids: tuple[str, ...]
    #: Rows into the (n_records, dim) matrix — the crops that show this person.
    rows: tuple[int, ...]


def build_units(records: list[dict], links: Mapping[str, str]) -> list[Unit]:
    """Group records into units, in first-appearance order."""
    order: list[str] = []
    events: dict[str, list[str]] = {}
    rows: dict[str, list[int]] = {}
    for row, record in enumerate(records):
        event_id = record["id"]
        key = unit_key(event_id, links.get(event_id))
        if key not in events:
            order.append(key)
            events[key], rows[key] = [], []
        events[key].append(event_id)
        rows[key].append(row)
    return [Unit(key, tuple(events[key]), tuple(rows[key])) for key in order]


def unit_embeddings(
    records: list[dict], matrix: np.ndarray, links: Mapping[str, str]
) -> tuple[list[Unit], np.ndarray]:
    """Units and their (n_units, dim) embeddings — the mean of their crops.

    Averaging happens here, in memory, and never on disk: the npy stays one
    row per record, so Contract C and every one-row actor-fix patch are
    untouched. A centroid over several crops of one person is also a strictly
    better query vector than any single crop of them.
    """
    units = build_units(records, links)
    if not units:
        return [], np.empty((0, matrix.shape[1] if matrix.ndim == 2 else 0))
    stacked = np.stack([matrix[list(u.rows)].mean(axis=0) for u in units])
    stacked /= np.linalg.norm(stacked, axis=1, keepdims=True) + 1e-12
    return units, stacked


def seeded_groups(
    units: list[Unit],
    matrix: np.ndarray,
    seeds: dict[str, list[str]],
    cutoff: float,
) -> tuple[dict[str, list[str]], list[str]]:
    """Distribute every non-seed unit to its nearest seed centroid.

    ``seeds`` maps a caller-chosen key to the unit keys anchoring that group.
    Each seed group's centroid is the mean of its members' embeddings; every
    other unit joins the closest centroid when its cosine distance is within
    ``cutoff``, otherwise it lands in the returned leftover list. Turns
    clustering into classification once the user has pinned one clean group
    per player.
    """
    index = {u.key: i for i, u in enumerate(units)}
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
        return out, [u.key for u in units if u.key not in seed_members]
    sims = matrix @ np.stack(centroids).T  # (N, S)
    for i, unit in enumerate(units):
        if unit.key in seed_members:
            continue
        best = int(np.argmax(sims[i]))
        if 1.0 - float(sims[i][best]) <= cutoff:
            out[keys[best]].append(unit.key)
        else:
            leftover.append(unit.key)
    return out, leftover


@dataclass(frozen=True)
class PlayersFile:
    """``<stem>_players.json`` — who each unit depicts.

        {"version": 2,
         "tracks":      {"12:3": "王小明"},
         "assignments": {"<event_id>": "王小明"},
         "done": true}

    ``tracks`` is the unit: name a tracklet once and every action it performed
    carries the name. ``assignments`` exists for the two things a tracklet
    cannot say — an event no tracklet reaches, and an event that contradicts
    its tracklet, which is what a ByteTrack identity switch looks like from
    the outside. An event override therefore WINS; a tracklet that gets
    contradicted is evidence, not an error.
    """

    tracks: dict[str, str]
    assignments: dict[str, str]
    done: bool


def _clean(names: Mapping[str, str]) -> dict[str, str]:
    return {str(k): v.strip() for k, v in names.items() if v and v.strip()}


def load_players(stem: str) -> PlayersFile:
    data = _load_players_file(stem)
    return PlayersFile(
        tracks=_clean(data.get("tracks") or {}),
        assignments=_clean(data.get("assignments") or {}),
        done=bool(data.get("done")),
    )


def save_players(
    stem: str,
    *,
    tracks: Mapping[str, str] | None = None,
    assignments: Mapping[str, str] | None = None,
) -> None:
    """Replace the naming maps. Omitted maps are left as they are."""
    with _players_lock:
        data = _read_players_file(stem)
        data["version"] = PLAYERS_SCHEMA_VERSION
        if tracks is not None:
            data["tracks"] = _clean(tracks)
        if assignments is not None:
            data["assignments"] = _clean(assignments)
        for key in ("tracks", "assignments"):
            if not data.get(key):
                data.pop(key, None)
        _save_players_file(stem, data)


def unit_names(units: Iterable[Unit], players: PlayersFile) -> dict[str, str]:
    """unit key → player name, for the units that have one.

    A tracklet's own name wins. Failing that, the unit takes the name its
    events already agree on: naming events one by one is what labeling looked
    like before units existed, and a tracklet whose every named crop says
    "王小明" IS 王小明 — reading it any other way would make an entire video's
    existing work vanish from the board. Events that DISAGREE name nobody:
    that is an identity switch mid-track, and picking a winner would bury it.
    """
    out: dict[str, str] = {}
    for unit in units:
        track = track_of_unit(unit.key)
        if track is not None and track in players.tracks:
            out[unit.key] = players.tracks[track]
            continue
        named = {
            players.assignments[event_id]
            for event_id in unit.event_ids
            if event_id in players.assignments
        }
        if len(named) == 1:
            out[unit.key] = next(iter(named))
    return out


def resolve_names(
    event_ids: Iterable[str], links: Mapping[str, str], players: PlayersFile
) -> dict[str, str]:
    """event id → player name. The one place precedence is decided.

    An explicit assignment wins over the tracklet's name: it is the only way
    to say "this tracklet is right about everything except here".
    """
    out: dict[str, str] = {}
    for event_id in event_ids:
        if event_id in players.assignments:
            out[event_id] = players.assignments[event_id]
            continue
        track = links.get(event_id)
        if track and track in players.tracks:
            out[event_id] = players.tracks[track]
    return out


def load_assignments(stem: str, links: Mapping[str, str] | None = None) -> dict[str, str]:
    """Every named event, tracklet names expanded across their events.

    ``links`` omitted means "only what is named explicitly" — correct for
    callers that have no tracklets to expand, and NOT a silent default: with
    tracklet names present it would under-report, so callers that can pass it
    must.
    """
    players = load_players(stem)
    if links is None:
        return dict(players.assignments)
    ids = set(players.assignments) | {
        event_id for event_id, track in links.items() if track in players.tracks
    }
    return resolve_names(ids, links, players)


def load_done(stem: str) -> bool:
    """Whether the user marked this video's labeling as finished."""
    return bool(_load_players_file(stem).get("done"))


def save_done(stem: str, done: bool) -> None:
    """Persist the human "labeling finished" verdict.

    A judgment call, not derived state: assigned/actionable counts can't tell
    "done" from "gave up halfway", so the mark means what the user says it
    means. Absent rather than false when unset.
    """
    with _players_lock:
        data = _read_players_file(stem)
        if done:
            data["done"] = True
        else:
            data.pop("done", None)
        _save_players_file(stem, data)


def drop_assignment(stem: str, event_id: str) -> None:
    """Forget who one event depicts.

    Called when its actor changes: the crop now shows a different person, so
    the name attached to the old crop is not evidence about the new one. Only
    the event's own override is dropped — its old tracklet keeps its name,
    because that name was never a claim about this one event.
    """
    with _players_lock:
        data = _read_players_file(stem)
        if data.get("assignments", {}).pop(event_id, None) is not None:
            _save_players_file(stem, data)


def match(
    units: list[Unit], matrix: np.ndarray, names: dict[str, str]
) -> dict[str, dict]:
    """Nearest-centroid match for every unit.

    Returns ``{unit_key: {player, sim, assigned}}``. Named units keep their
    label with sim 1.0; the rest get the closest player centroid and the
    cosine similarity to it.
    """
    if not names or not len(matrix):
        return {}
    index = {u.key: i for i, u in enumerate(units)}
    by_player: dict[str, list[int]] = {}
    for key, player in names.items():
        if key in index:
            by_player.setdefault(player, []).append(index[key])
    if not by_player:
        return {}

    players = sorted(by_player)
    centroids = np.stack([matrix[rows].mean(axis=0) for rows in (by_player[p] for p in players)])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12
    sims = matrix @ centroids.T  # (N, P)

    out: dict[str, dict] = {}
    for i, unit in enumerate(units):
        if unit.key in names:
            out[unit.key] = {"player": names[unit.key], "sim": 1.0, "assigned": True}
        else:
            best = int(np.argmax(sims[i]))
            out[unit.key] = {"player": players[best], "sim": round(float(sims[i][best]), 4), "assigned": False}
    return out
