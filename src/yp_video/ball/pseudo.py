"""Ball pseudo-labels: the hand-marked contacts pick the detector's ball.

Action annotations mark where the ball is at every contact. Between two
consecutive contacts the ball flies free, so its image path is a smooth curve
from one marked point to the other. The detector finds most of the ball's
frames, but in a crowd it also scores heads, shirts and gym clutter — yet
those false boxes never line up on one curve. So:

1. ``detect_candidates`` (GPU) runs the ball detector on the frames of every
   contact-to-contact flight and caches every box above a low floor:
   ``<stem>_candidates.npz`` in BALL_CANDIDATES_DIR (``frames`` int32 cv2
   indices, ``counts`` int32, ``boxes`` float32 xyxy normalized to the frame,
   ``scores`` float32, ``meta`` JSON).
2. ``write_pseudo_labels`` (CPU) RANSACs, per flight, an x(t), y(t) quadratic
   through the two marked contacts and keeps the candidates on it
   (``flight`` labels). Each marked contact becomes a box too, sized like the
   nearest flight label (``contact`` labels) — the crowded frames a detector
   misses. ``<stem>_ball.jsonl`` in BALL_PSEUDO_LABELS_DIR.

The detector is the one fine-tuned (ball/train.py) on the previous round of
these labels; the first round used COCO RF-DETR's "sports ball" at the same
input size. Each round's labels feed the next round's training set.

Only manual annotations seed this; pre-annotations are model output.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from yp_video.ball import detector
from yp_video.config import (
    ACTION_ANNOTATIONS_DIR,
    BALL_CANDIDATES_DIR,
    BALL_PSEUDO_LABELS_DIR,
)
from yp_video.contracts.action import LABEL_FILE_SUFFIX
from yp_video.core.jsonl import atomic_binary, read_jsonl, write_jsonl

# A flight: consecutive visible contacts this far apart. Shorter is one touch
# marked twice; longer spans a dead ball or a missed contact.
MIN_FLIGHT_S = 0.2
MAX_FLIGHT_S = 4.0

CANDIDATES_SUFFIX = "_candidates.npz"
LABELS_SUFFIX = "_ball.jsonl"

# RANSAC, in units of frame width (y scaled by height/width).
CURVE_DEGREE = 2
INLIER_TOL = 0.006           # ~12 px at 1920
ENDPOINT_TOL = 0.02          # the curve must pass this near both marked contacts
RANSAC_ITERS = 1500
MIN_FLIGHT_LABELS = 4
# A kept box must be ball-shaped and ball-sized for its flight.
SIZE_RATIO = 1.6
MAX_ASPECT = 2.0
# Static clutter (a fan, a yellow pad on a post, a lamp) sits at one spot
# with one size through flight after flight; the ball passes a spot briefly.
# A candidate whose place-and-size neighbourhood holds boxes from this many
# distinct flights is clutter (see _drop_static for the neighbourhood).
STATIC_RADIUS = 0.003
STATIC_SIZE_RATIO = 1.25
STATIC_MIN_FLIGHTS = 40
# A contact borrows the size of a flight label at most this far away.
CONTACT_SIZE_S = 0.4


def candidates_path(stem: str) -> Path:
    return BALL_CANDIDATES_DIR / f"{stem}{CANDIDATES_SUFFIX}"


def labels_path(stem: str) -> Path:
    return BALL_PSEUDO_LABELS_DIR / f"{stem}{LABELS_SUFFIX}"


def contacts(stem: str) -> tuple[dict, list[dict]]:
    """The manual annotation's meta and its events, sorted by frame."""
    meta, rows = read_jsonl(ACTION_ANNOTATIONS_DIR / f"{stem}{LABEL_FILE_SUFFIX}")
    return meta, sorted((r for r in rows if r.get("frame") is not None), key=lambda r: r["frame"])


def flights(events: list[dict], fps: float) -> list[tuple[dict, dict]]:
    """Consecutive contact pairs that bound one free flight of the ball."""
    out = []
    for a, b in zip(events, events[1:]):
        gap = (b["frame"] - a["frame"]) / fps
        if a.get("visible") and b.get("visible") and a.get("xy") and b.get("xy") \
                and MIN_FLIGHT_S <= gap <= MAX_FLIGHT_S:
            out.append((a, b))
    return out


def spans(pairs: list[tuple[dict, dict]]) -> list[tuple[int, int]]:
    """Flights merged into contiguous [first, last] frame spans (a rally's
    flights chain end to start), so each span costs one seek."""
    out: list[list[int]] = []
    for a, b in pairs:
        if out and a["frame"] <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b["frame"])
        else:
            out.append([a["frame"], b["frame"]])
    return [(a, b) for a, b in out]


def detect_candidates(video_path: Path, *, should_pause: Callable[[], bool]) -> dict:
    """Cache the ball detector's boxes on every flight frame. Raises
    detector.Paused when ``should_pause`` stops it."""
    stem = video_path.stem
    meta, events = contacts(stem)
    pairs = flights(events, detector.video_fps(video_path))
    found = detector.detect(video_path, spans(pairs), should_pause=should_pause)
    out = candidates_path(stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    with atomic_binary(out) as f:
        np.savez_compressed(
            f,
            frames=np.asarray(found.frames, dtype=np.int32),
            counts=np.asarray([len(b) for b in found.boxes], dtype=np.int32),
            boxes=np.concatenate([b[:, :4] for b in found.boxes]) if found.boxes else np.zeros((0, 4), np.float32),
            scores=np.concatenate([b[:, 4] for b in found.boxes]) if found.boxes else np.zeros((0,), np.float32),
            meta=np.array(json.dumps({
                "video": stem,
                "model": detector.VARIANT,
                "weights": str(detector.WEIGHTS),
                "resolution": detector.RESOLUTION,
                "class_id": detector.CLASS_ID,
                "score_floor": detector.SCORE_FLOOR,
                "fps": found.fps,
                "stride": found.stride,
                "frame_size": list(found.frame_size),
                "annotation_events": meta.get("num_events"),
                "created_at": time.time(),
            })),
        )
    return {"frames": len(found.frames), "boxes": int(sum(len(b) for b in found.boxes))}


def _load_candidates(stem: str) -> tuple[dict, dict[int, np.ndarray]]:
    """Meta and {frame: (N, 5) xyxy+score}."""
    with np.load(candidates_path(stem)) as z:
        meta = json.loads(str(z["meta"]))
        ends = np.cumsum(z["counts"])
        rows = np.hstack([z["boxes"], z["scores"][:, None]])
        by_frame = {int(f): rows[e - c:e] for f, c, e in zip(z["frames"], z["counts"], ends)}
    return meta, by_frame


def _drop_static(by_frame: dict[int, np.ndarray], pairs: list[tuple[dict, dict]],
                 aspect: float) -> tuple[dict[int, np.ndarray], int]:
    """``by_frame`` without static clutter, and how many boxes went.

    Boxes are binned by position (STATIC_RADIUS cells) and log size
    (STATIC_SIZE_RATIO steps); a box is clutter when its bin and the 26
    around it hold boxes from STATIC_MIN_FLIGHTS distinct flights. Work and
    memory scale with occupied bins — a pairwise neighbour search goes
    quadratic in a cluttered gym (tens of thousands of boxes on one fan).
    """
    flight_of = {f: i for i, (a, b) in enumerate(pairs) for f in range(a["frame"] + 1, b["frame"])}
    frames = [f for f in by_frame if f in flight_of]
    if not frames:
        return by_frame, 0
    rows = np.concatenate([by_frame[f] for f in frames])
    owner = np.concatenate([[flight_of[f]] * len(by_frame[f]) for f in frames]).astype(np.int64)
    size = np.sqrt(np.maximum((rows[:, 2] - rows[:, 0]) * (rows[:, 3] - rows[:, 1]) * aspect, 1e-12))
    keys = np.c_[
        np.floor((rows[:, 0] + rows[:, 2]) / 2 / STATIC_RADIUS),
        np.floor((rows[:, 1] + rows[:, 3]) / 2 * aspect / STATIC_RADIUS),
        np.floor(np.log(size) / np.log(STATIC_SIZE_RATIO)),
    ].astype(np.int64)
    bins, inverse = np.unique(keys, axis=0, return_inverse=True)
    flights_in: dict[tuple, set] = {}
    for b, o in np.unique(np.c_[inverse, owner], axis=0):
        flights_in.setdefault(tuple(bins[b]), set()).add(int(o))
    offsets = [(dx, dy, ds) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for ds in (-1, 0, 1)]
    static_bin = np.array([
        len(set().union(*(flights_in.get((x + dx, y + dy, z + ds), ()) for dx, dy, ds in offsets)))
        >= STATIC_MIN_FLIGHTS
        for x, y, z in bins
    ])
    static = static_bin[inverse.ravel()]
    kept = dict(by_frame)
    start = 0
    for f in frames:
        n = len(by_frame[f])
        kept[f] = by_frame[f][~static[start:start + n]]
        start += n
    return kept, int(static.sum())


def _fit_flight(a: dict, b: dict, by_frame: dict[int, np.ndarray], aspect: float,
                rng: np.random.Generator) -> dict[int, np.ndarray]:
    """{frame: xyxy} of the candidates on the flight's curve, or {}.

    ``aspect`` is height/width: distances are in frame-width units.
    """
    fa, fb = a["frame"], b["frame"]
    pts = [(f - fa, box) for f in range(fa + 1, fb) for box in by_frame.get(f, ())]
    if len(pts) < MIN_FLIGHT_LABELS:
        return {}
    t = np.array([p[0] for p in pts], dtype=np.float64)
    box = np.array([p[1][:4] for p in pts], dtype=np.float64)
    x = (box[:, 0] + box[:, 2]) / 2
    y = (box[:, 1] + box[:, 3]) / 2 * aspect
    et = np.array([0.0, fb - fa])
    ex = np.array([a["xy"][0], b["xy"][0]])
    ey = np.array([a["xy"][1], b["xy"][1]]) * aspect

    def curve(ts, xs, ys):
        return np.polyfit(ts, xs, CURVE_DEGREE), np.polyfit(ts, ys, CURVE_DEGREE)

    def dist(c, ts, xs, ys):
        return np.hypot(np.polyval(c[0], ts) - xs, np.polyval(c[1], ts) - ys)

    best_n, best = 0, None
    for _ in range(RANSAC_ITERS):
        # CURVE_DEGREE candidates plus one marked contact define a curve.
        k = rng.choice(len(t), CURVE_DEGREE, replace=False)
        e = rng.integers(2)
        ts = np.r_[t[k], et[e]]
        if len(set(ts)) <= CURVE_DEGREE:
            continue
        c = curve(ts, np.r_[x[k], ex[e]], np.r_[y[k], ey[e]])
        if dist(c, et, ex, ey).max() > ENDPOINT_TOL:
            continue
        inl = dist(c, t, x, y) < INLIER_TOL
        n = len(set(t[inl]))
        if n > best_n:
            best_n, best = n, inl
    if best is None or best_n < MIN_FLIGHT_LABELS:
        return {}
    c = curve(np.r_[t[best], et], np.r_[x[best], ex], np.r_[y[best], ey])
    if dist(c, et, ex, ey).max() > ENDPOINT_TOL:
        return {}
    d = dist(c, t, x, y)
    w, h = box[:, 2] - box[:, 0], (box[:, 3] - box[:, 1]) * aspect
    size = np.sqrt(w * h)
    ok = (d < INLIER_TOL) & (np.maximum(w, h) <= MAX_ASPECT * np.minimum(w, h))
    if not ok.any():
        return {}
    median = np.median(size[ok])
    ok &= (size <= SIZE_RATIO * median) & (size >= median / SIZE_RATIO)
    # One box per frame: the closest to the curve.
    labels: dict[int, np.ndarray] = {}
    for i in np.flatnonzero(ok)[np.argsort(d[ok])]:
        labels.setdefault(int(t[i]) + fa, box[i])
    return labels if len(labels) >= MIN_FLIGHT_LABELS else {}


def write_pseudo_labels(stem: str) -> dict:
    """Pick the ball among the cached candidates; write ``<stem>_ball.jsonl``."""
    cmeta, by_frame = _load_candidates(stem)
    fps = cmeta["fps"]
    width, height = cmeta["frame_size"]
    aspect = height / width
    _meta, events = contacts(stem)
    rng = np.random.default_rng(0)
    pairs = flights(events, fps)
    clean, n_static = _drop_static(by_frame, pairs, aspect)

    flight_labels: dict[int, np.ndarray] = {}
    fitted = 0
    for a, b in pairs:
        labels = _fit_flight(a, b, clean, aspect, rng)
        fitted += bool(labels)
        flight_labels.update(labels)

    rows = [{"frame": f, "box": [round(float(v), 5) for v in box], "source": "flight"}
            for f, box in sorted(flight_labels.items())]
    # A visible contact takes the size of the nearest flight label.
    near = int(round(CONTACT_SIZE_S * fps))
    labelled = np.array(sorted(flight_labels), dtype=np.int64)
    n_contacts = 0
    for e in events:
        if not (e.get("visible") and e.get("xy")) or not len(labelled):
            continue
        j = labelled[np.abs(labelled - e["frame"]).argmin()]
        if abs(j - e["frame"]) > near or e["frame"] in flight_labels:
            continue
        x0, y0, x1, y1 = flight_labels[int(j)]
        hw, hh = (x1 - x0) / 2, (y1 - y0) / 2
        cx, cy = e["xy"]
        rows.append({
            "frame": e["frame"],
            "box": [round(float(v), 5) for v in (max(cx - hw, 0), max(cy - hh, 0), min(cx + hw, 1), min(cy + hh, 1))],
            "source": "contact",
            "label": e["label"],
        })
        n_contacts += 1
    rows.sort(key=lambda r: r["frame"])

    meta = {
        "video": stem,
        "fps": fps,
        "frame_size": [width, height],
        "candidates": {k: cmeta[k] for k in ("model", "weights", "resolution", "score_floor", "stride")},
        "flights": len(pairs),
        "flights_fitted": fitted,
        "flight_frames": len(by_frame),
        "static_boxes_dropped": n_static,
        "flight_labels": len(flight_labels),
        "contact_labels": n_contacts,
        "created_at": time.time(),
    }
    out = labels_path(stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, meta, rows)
    return {k: meta[k] for k in ("flights", "flights_fitted", "flight_frames", "static_boxes_dropped",
                                   "flight_labels", "contact_labels")}
