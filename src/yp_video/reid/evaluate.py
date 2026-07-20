"""How well do the extracted embedders separate players within a session?

Identities only need to be consistent within one recording session (see
reid/sessions.py), so that is exactly what this measures: for each session
group and each model, does the embedding rank a player's other crops above
everyone else's? A global evaluation would pool identities across sessions
and reward cross-day ReID — which nobody here needs, and which would make the
numbers look worse than the product cares about.

Retrieval protocol is leave-one-out over every labeled crop. CLIP-ReIdent's
metrics exclude gallery entries where (same pid AND same camid); feeding a
full NxN distance matrix with ``cams = arange(N)`` makes that filter drop
precisely the self-match, so every crop is a query and every other crop is
gallery. No split, no seed, nothing discarded. (If the self-match ever leaked
in, Rank-1 would be 1.0 for any input — the random-features check in the
verification script is what pins this down.)

Distances follow their squared-Euclidean convention (``2 - 2*cos``, lower is
more similar) because that is what their metrics expect. The threshold
calibration reports COSINE distance (``1 - cos``) instead, because that is the
scale EMBEDDER_THRESHOLDS and the Label page's slider live on.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from yp_video.reid.metrics import cmc, mean_ap
from yp_video.reid.identity import cluster_sweep, load_assignments, load_embeddings
from yp_video.reid.sessions import SessionGroup

# A fully-merged corpus would make the NxN distance matrix and the pairwise
# sweep the biggest allocations in the web process. Fail loudly instead of
# taking the box down; subsample here if this ever legitimately trips.
MAX_GROUP_CROPS = 20_000

#: Candidate cutoffs per sweep. The useful band always sits between typical
#: same-player and typical different-player distance, and those scales differ
#: by an order of magnitude across models — so the band is derived, not fixed.
SWEEP_STEPS = 48


@dataclass(frozen=True)
class Scores:
    m_ap: float
    rank1: float
    rank5: float
    n_query: int


@dataclass(frozen=True)
class ThresholdSuggestion:
    """A calibrated clustering cutoff, in cosine distance.

    Tuned against what the user actually experiences — the quality of the
    groups the Label page shows — rather than against pairwise separability.
    The two are not the same: average-linkage chains, so the cutoff that best
    classifies PAIRS sits above the one that best CLUSTERS, and past the peak
    the whole board collapses into a handful of mixed groups.
    """

    suggested: float
    #: Adjusted Rand index at ``suggested`` (1.0 = groups match the labels).
    ari: float
    #: Groups produced at ``suggested``, against the true player count. More
    #: is expected and fine: merging two groups of one player is a drag, while
    #: a group holding two players has to be noticed before it can be split.
    n_clusters: int
    n_ids: int
    #: The whole sweep — [{t, ari, n}] — so the page can show the plateau.
    curve: list
    #: Pairwise separability, independent of any cutoff. Chance = 0.5.
    auc: float
    same_p50: float
    same_p95: float
    diff_p05: float
    diff_p50: float
    n_pos: int
    n_neg: int
    #: paste-ready EMBEDDER_THRESHOLDS entry
    slider: dict


#: What every degenerate path returns — no pairs, no signal, AUC at chance.
EMPTY_THRESHOLD = ThresholdSuggestion(
    suggested=0.0, ari=0.0, n_clusters=0, n_ids=0, curve=[], auc=0.5,
    same_p50=0.0, same_p95=0.0, diff_p05=0.0, diff_p50=0.0,
    n_pos=0, n_neg=0, slider={},
)


@dataclass(frozen=True)
class VideoEval:
    """One video, scored on its own labels.

    The video is the unit because that is how labeling happens — you open one
    cut and cluster its crops. Merging a session's videos measures something
    harder (more identities in the gallery) and depends on their names having
    been kept consistent, which is a separate question answered by CrossEval.
    """

    stem: str
    model: str
    n_ids: int
    n_crops: int
    n_assigned: int
    coverage: float
    dropped_singletons: int
    dropped_unembedded: int
    scores: Scores
    threshold: ThresholdSuggestion


# ── retrieval ────────────────────────────────────────────────────────────


def _distmat(q: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Squared Euclidean between L2-normalized rows: 2 - 2*cos, lower = closer."""
    return (2.0 - 2.0 * (q @ g.T)).astype(np.float32)


def _scores(distmat, query_ids, gallery_ids, query_cams, gallery_cams, n_query: int) -> Scores:
    topk = max(1, min(100, len(gallery_ids)))
    curve = cmc(distmat, query_ids, gallery_ids, query_cams, gallery_cams, topk=topk)
    return Scores(
        m_ap=float(mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)),
        rank1=float(curve[0]),
        rank5=float(curve[min(4, topk - 1)]),
        n_query=n_query,
    )


def loo_scores(matrix: np.ndarray, pids: np.ndarray) -> Scores:
    """Leave-one-out: every crop queries every other crop of the group."""
    n = len(matrix)
    if n < 2:
        return Scores(0.0, 0.0, 0.0, 0)
    cams = np.arange(n)  # unique per row -> the validity filter drops only self
    return _scores(_distmat(matrix, matrix), pids, pids, cams, cams, n)


def split_scores(q: np.ndarray, qpid: np.ndarray, g: np.ndarray, gpid: np.ndarray) -> Scores | None:
    """Disjoint query/gallery (used for the cross-video check).

    cams 0 vs 1 excludes nothing, so the whole gallery is scored. Returns None
    when no query has a true match — the metrics raise in that case.
    """
    if not len(q) or not len(g) or not np.isin(qpid, gpid).any():
        return None
    return _scores(
        _distmat(q, g), qpid, gpid, np.zeros(len(q), int), np.ones(len(g), int), len(q)
    )


# ── threshold calibration ────────────────────────────────────────────────


def _pairs(matrix: np.ndarray, pids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Upper-triangle pairwise COSINE distances and their same-player mask."""
    iu = np.triu_indices(len(matrix), k=1)
    return (1.0 - matrix @ matrix.T)[iu], pids[iu[0]] == pids[iu[1]]


def adjusted_rand(true: np.ndarray, pred: np.ndarray) -> float:
    """Adjusted Rand index — cluster agreement, corrected for chance.

    Implemented here rather than imported: sklearn is a heavy import to pull
    into a web request for fifteen lines of contingency-table arithmetic.
    Cross-checked against sklearn.metrics.adjusted_rand_score in verification.
    """
    n = len(true)
    if n < 2:
        return 0.0
    _, t = np.unique(true, return_inverse=True)
    _, p = np.unique(pred, return_inverse=True)
    contingency = np.zeros((int(t.max()) + 1, int(p.max()) + 1), dtype=np.int64)
    np.add.at(contingency, (t, p), 1)

    def comb2(x: np.ndarray) -> float:
        return float((x * (x - 1) // 2).sum())

    index = comb2(contingency)
    a, b = comb2(contingency.sum(1)), comb2(contingency.sum(0))
    total = n * (n - 1) / 2
    expected = a * b / total
    maximum = (a + b) / 2
    return (index - expected) / (maximum - expected) if maximum != expected else 0.0


def sweep_candidates(dist: np.ndarray, same: np.ndarray) -> np.ndarray:
    """The band worth sweeping, derived from the labeled pair distances."""
    if not same.any() or not (~same).any():
        return np.empty(0)
    lo = float(np.quantile(dist[same], 0.05))
    hi = float(np.quantile(dist[~same], 0.5))
    return np.linspace(lo, hi, SWEEP_STEPS) if hi > lo else np.array([lo])


def threshold_curve(
    matrix: np.ndarray, pids: np.ndarray, thresholds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(ari, n_clusters) at each threshold, over ONE shared linkage."""
    labels = cluster_sweep(matrix, [float(t) for t in thresholds])
    return (
        np.array([adjusted_rand(pids, lab) for lab in labels]),
        np.array([len(set(lab.tolist())) for lab in labels]),
    )


def _auc(dist: np.ndarray, same: np.ndarray) -> float:
    """Mann-Whitney over the pair distances; lower distance = positive."""
    n_pos, n_neg = int(same.sum()), int((~same).sum())
    if not n_pos or not n_neg:
        return 0.5
    order = np.argsort(dist, kind="stable")
    ranks = np.empty(len(dist), dtype=np.float64)
    ranks[order] = np.arange(1, len(dist) + 1)
    return float((n_pos * n_neg + n_pos * (n_pos + 1) / 2 - ranks[same].sum()) / (n_pos * n_neg))


def build_suggestion(
    dist: np.ndarray,
    same: np.ndarray,
    thresholds: np.ndarray,
    ari: np.ndarray,
    sizes: np.ndarray,
    n_ids: int,
) -> ThresholdSuggestion:
    """Pick the cutoff, and hand back the curve that justified it."""
    n_pos, n_neg = int(same.sum()), int((~same).sum())
    if not n_pos or not n_neg or not len(thresholds):
        return EMPTY_THRESHOLD
    # argmax returns the FIRST maximum, so a plateau resolves to its low end —
    # the safe side, where a player splits across groups rather than two
    # players sharing one.
    k = int(np.argmax(ari))
    suggested = _round2(float(thresholds[k]))
    # Keep the slider inside the band that still clusters usefully, so its
    # whole travel is meaningful instead of mostly dead.
    useful = np.where(ari >= ari[k] * 0.25)[0]
    lo = _round2(float(thresholds[useful[0]])) if len(useful) else suggested
    hi = _round2(float(thresholds[useful[-1]])) if len(useful) else suggested
    return ThresholdSuggestion(
        suggested=suggested,
        ari=float(ari[k]),
        n_clusters=int(sizes[k]),
        n_ids=n_ids,
        curve=[
            {"t": _round2(float(t)), "ari": round(float(a), 3), "n": int(n)}
            for t, a, n in zip(thresholds, ari, sizes)
        ],
        auc=_auc(dist, same),
        same_p50=float(np.quantile(dist[same], 0.5)),
        same_p95=float(np.quantile(dist[same], 0.95)),
        diff_p05=float(np.quantile(dist[~same], 0.05)),
        diff_p50=float(np.quantile(dist[~same], 0.5)),
        n_pos=n_pos,
        n_neg=n_neg,
        slider={"min": lo, "max": hi, "default": suggested, "step": _nice_step(hi - lo)},
    )


def suggest_threshold(matrix: np.ndarray, pids: np.ndarray) -> ThresholdSuggestion:
    """Calibrate the clustering cutoff for one group's labeled crops."""
    if len(matrix) < 2:
        return EMPTY_THRESHOLD
    dist, same = _pairs(matrix, pids)
    thresholds = sweep_candidates(dist, same)
    if not len(thresholds):
        return EMPTY_THRESHOLD
    ari, sizes = threshold_curve(matrix, pids, thresholds)
    return build_suggestion(dist, same, thresholds, ari, sizes, len(set(pids.tolist())))


def _round2(v: float) -> float:
    """Two significant digits — reads like the hand-tuned entries already in
    embedder.EMBEDDER_THRESHOLDS (0.022, 0.15)."""
    if not v:
        return 0.0
    from math import floor, log10

    return round(v, -int(floor(log10(abs(v)))) + 1)


#: Slider stops to aim for. Enough to tune finely, few enough that every stop
#: is distinguishable at the 2-3 decimals the UI renders.
SLIDER_STOPS = 30


def _nice_step(span: float) -> float:
    """A 1-2-5 round step covering ``span`` in roughly SLIDER_STOPS stops.

    The raw span/N is a number like 0.00048, which the Label page's slider
    renders at 3 decimals — adjacent stops would print identically. Snapping
    to 1/2/5 x 10^k keeps every stop legible.
    """
    if span <= 0:
        return 0.001
    from math import floor, log10

    raw = span / SLIDER_STOPS
    mag = 10 ** floor(log10(raw))
    return float(next((m * mag for m in (1, 2, 5) if raw <= m * mag), 10 * mag))


# ── evaluation ───────────────────────────────────────────────────────────


def gather(stems: Sequence[str], model: str):
    """(matrix, pids, names, stem_idx, drops) for these videos under one model.

    load_embeddings already filters to embedded, non-SKIP records and
    L2-normalizes, so this only has to keep the rows that carry an assignment
    and drop players left too small to score.
    """
    rows: list[np.ndarray] = []
    names: list[str] = []
    stem_idx: list[int] = []
    n_assigned = 0
    embedded_ids: set[str] = set()

    for i, stem in enumerate(stems):
        assignments = load_assignments(stem)
        n_assigned += len(assignments)
        records, matrix = load_embeddings(stem, model=model)
        for row, record in enumerate(records):
            name = assignments.get(record["id"])
            if name is None:
                continue
            embedded_ids.add(record["id"])
            rows.append(matrix[row])
            names.append(name)
            stem_idx.append(i)

    drops = {"unembedded": n_assigned - len(embedded_ids), "singletons": 0, "n_assigned": n_assigned}
    if not rows:
        return np.empty((0, 0), np.float32), np.empty(0, int), [], np.empty(0, int), drops

    # Players with a single crop can never be retrieved; count them out loud.
    counts: dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    keep = [i for i, name in enumerate(names) if counts[name] >= 2]
    drops["singletons"] = sum(1 for _n, c in counts.items() if c < 2)

    matrix = np.asarray([rows[i] for i in keep], dtype=np.float32)
    kept_names = [names[i] for i in keep]
    order = {name: pid for pid, name in enumerate(sorted(set(kept_names)))}
    pids = np.array([order[n] for n in kept_names], dtype=int)
    return matrix, pids, kept_names, np.array([stem_idx[i] for i in keep], dtype=int), drops


def evaluate_video(stem: str, model: str) -> tuple[VideoEval, np.ndarray, np.ndarray] | None:
    """One video's evaluation, plus its (matrix, pids) so a caller pooling
    across videos doesn't have to gather them a second time."""
    matrix, pids, _names, _stem_idx, drops = gather([stem], model)
    if len(matrix) < 2:
        return None
    if len(matrix) > MAX_GROUP_CROPS:
        raise ValueError(
            f"{stem} has {len(matrix)} crops, over MAX_GROUP_CROPS={MAX_GROUP_CROPS} — "
            "subsample or raise the cap before evaluating"
        )
    n_assigned = drops["n_assigned"]
    ev = VideoEval(
        stem=stem,
        model=model,
        n_ids=int(len(set(pids.tolist()))),
        n_crops=len(matrix),
        n_assigned=n_assigned,
        coverage=(len(matrix) / n_assigned) if n_assigned else 0.0,
        dropped_singletons=drops["singletons"],
        dropped_unembedded=drops["unembedded"],
        scores=loo_scores(matrix, pids),
        threshold=suggest_threshold(matrix, pids),
    )
    return ev, matrix, pids


def cross_video_eval(group: SessionGroup, model: str) -> dict | None:
    """Does an identity survive into another recording of the same session?

    Query is the session's first video, gallery the rest. Only players named
    in BOTH sides can be scored — reid.metrics silently skips a query
    with no true match in the gallery, so the skipped count is reported
    rather than hidden: it is the players who simply were not on court for
    the other clip, which is normal and not a labeling gap.
    """
    if len(group.stems) < 2:
        return None
    matrix, pids, _names, stem_idx, _drops = gather(group.stems, model)
    if len(matrix) < 2:
        return None
    q = stem_idx == 0
    qpid, gpid = pids[q], pids[~q]
    scored = int(np.isin(qpid, gpid).sum())
    scores = split_scores(matrix[q], qpid, matrix[~q], gpid)
    if scores is None:
        return None
    return {
        "session_id": group.id,
        "query_stem": group.stems[0],
        "gallery_stems": list(group.stems[1:]),
        "n_ids_shared": int(len(set(qpid.tolist()) & set(gpid.tolist()))),
        "n_scored": scored,
        "n_skipped": int(len(qpid) - scored),
        "scores": asdict(Scores(scores.m_ap, scores.rank1, scores.rank5, scored)),
    }


def _pooled_threshold(collected: list[tuple[np.ndarray, np.ndarray]]) -> ThresholdSuggestion:
    """One cutoff for the model, from every video's curve at once.

    Pooling matters: a single video's ARI curve is jagged (agglomerative
    merges are discrete), so its argmax can land on a spike. Every video is
    swept over the same pooled band and the curves are averaged by crop count.
    """
    if not collected:
        return EMPTY_THRESHOLD
    pairs = [_pairs(m, p) for m, p in collected]
    dist = np.concatenate([d for d, _s in pairs])
    same = np.concatenate([s for _d, s in pairs])
    grid = sweep_candidates(dist, same)
    if not len(grid):
        return EMPTY_THRESHOLD

    weights = np.array([len(m) for m, _p in collected], dtype=float)
    curves = [threshold_curve(m, p, grid) for m, p in collected]
    ari = (np.stack([c[0] for c in curves]) * weights[:, None]).sum(0) / weights.sum()
    sizes = np.stack([c[1] for c in curves]).sum(0)
    n_ids = sum(len(set(p.tolist())) for _m, p in collected)
    return build_suggestion(dist, same, grid, ari, sizes, n_ids)


def evaluate_models(groups: Sequence[SessionGroup], models: Sequence[str]) -> dict:
    """The /performance payload: per-VIDEO scores plus a cross-video section.

    The video is the unit because that is the labeling unit. Sessions still
    matter for two things: a model is only scored on a video whose matrix
    exists (otherwise it could look better on an easier subset), and the
    cross-video check needs to know which videos share a name-space.
    """
    from yp_video.reid.embedder import threshold_calibration
    from yp_video.reid.store import embedded_models

    stems = [stem for g in groups for stem in g.stems]
    out = []
    for model in models:
        evals: list[VideoEval] = []
        skipped: list[str] = []
        collected: list[tuple[np.ndarray, np.ndarray]] = []
        for stem in stems:
            if model not in embedded_models(stem):
                skipped.append(stem)
                continue
            result = evaluate_video(stem, model)
            if result is None:
                skipped.append(stem)
                continue
            ev, matrix, pids = result
            evals.append(ev)
            collected.append((matrix, pids))

        cross = [c for g in groups if (c := cross_video_eval(g, model)) is not None] if evals else []

        if not evals:
            out.append({"model": model, "videos": [], "cross_video": [], "skipped": skipped,
                        "current_threshold": threshold_calibration(model)})
            continue

        total_q = sum(e.scores.n_query for e in evals) or 1
        weighted = Scores(
            m_ap=sum(e.scores.m_ap * e.scores.n_query for e in evals) / total_q,
            rank1=sum(e.scores.rank1 * e.scores.n_query for e in evals) / total_q,
            rank5=sum(e.scores.rank5 * e.scores.n_query for e in evals) / total_q,
            n_query=total_q,
        )
        macro = Scores(
            m_ap=float(np.mean([e.scores.m_ap for e in evals])),
            rank1=float(np.mean([e.scores.rank1 for e in evals])),
            rank5=float(np.mean([e.scores.rank5 for e in evals])),
            n_query=len(evals),
        )
        n_crops = sum(e.n_crops for e in evals)
        n_assigned = sum(e.n_assigned for e in evals)
        out.append({
            "model": model,
            "crop_weighted": asdict(weighted),
            "macro": asdict(macro),
            "totals": {
                "n_videos": len(evals),
                "n_ids": sum(e.n_ids for e in evals),
                "n_crops": n_crops,
                "coverage": (n_crops / n_assigned) if n_assigned else 0.0,
            },
            "threshold": asdict(_pooled_threshold(collected)),
            "current_threshold": threshold_calibration(model),
            "skipped": skipped,
            "videos": [asdict(e) for e in evals],
            "cross_video": cross,
        })

    out.sort(key=lambda m: m.get("crop_weighted", {}).get("m_ap", -1.0), reverse=True)
    return {"models": out}
