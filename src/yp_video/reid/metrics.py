"""Retrieval metrics: CMC (first-match-break) and mean average precision.

Pure-numpy port of ``clipreid.metrics`` (DeepSportRadar challenge code),
reduced to the call shapes evaluate.py actually uses — cmc is always called
with ``first_match_break=True``, never with the camera-set or single-shot
variants, so those branches don't exist here. The validity filter is
unchanged: a gallery item is excluded for a query iff it shares BOTH the id
and the cam (evaluate.py exploits this with per-row cams to drop only
self-matches in leave-one-out scoring).

``mean_ap`` reproduces sklearn's ``average_precision_score`` numerically
(same tie grouping, same step-wise interpolation) without the dependency;
verified against both references at port time. yp_reid/metrics.py mirrors
this file — keep them in lockstep.
"""

from __future__ import annotations

import numpy as np


def _average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AP over descending score with ties grouped, as sklearn computes it."""
    order = np.argsort(-y_score, kind="stable")
    y = y_true[order].astype(np.float64)
    scores = y_score[order]
    # Last index of each tie group — metrics are defined per distinct threshold.
    boundary = np.r_[np.nonzero(np.diff(scores))[0], len(scores) - 1]
    tps = np.cumsum(y)[boundary]
    precision = tps / (boundary + 1)
    recall = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def _valid_matches(distmat: np.ndarray, query_ids, gallery_ids, query_cams, gallery_cams):
    """Per query: gallery order by distance, validity mask, match mask."""
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    indices = np.argsort(distmat, axis=1)
    matches = gallery_ids[indices] == query_ids[:, np.newaxis]
    # Exclude only same-id AND same-cam entries (self-matches, typically).
    valid = (gallery_ids[indices] != query_ids[:, np.newaxis]) | (
        gallery_cams[indices] != query_cams[:, np.newaxis]
    )
    return indices, valid, matches


def cmc(distmat: np.ndarray, query_ids, gallery_ids, query_cams, gallery_cams, topk: int) -> np.ndarray:
    """Cumulative match curve, first-match-break: P(true match within rank k)."""
    _indices, valid, matches = _valid_matches(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    ret = np.zeros(topk)
    num_valid_queries = 0
    for i in range(len(distmat)):
        hits = np.nonzero(matches[i, valid[i]])[0]
        if not len(hits):
            continue
        if hits[0] < topk:
            ret[hits[0]] += 1
        num_valid_queries += 1
    if num_valid_queries == 0:
        raise RuntimeError("No valid query")
    return ret.cumsum() / num_valid_queries


def mean_ap(distmat: np.ndarray, query_ids, gallery_ids, query_cams, gallery_cams) -> float:
    """Mean AP over queries that have at least one valid true match."""
    distmat = np.asarray(distmat)
    indices, valid, matches = _valid_matches(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    aps = []
    for i in range(len(distmat)):
        y_true = matches[i, valid[i]]
        if not np.any(y_true):
            continue
        y_score = -distmat[i][indices[i]][valid[i]]
        aps.append(_average_precision(y_true, y_score))
    if not aps:
        raise RuntimeError("No valid query")
    return float(np.mean(aps))
