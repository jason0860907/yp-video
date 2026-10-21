"""A rally's 3D ball path from the ball's image positions and the camera.

Between touches the ball moves as p(t) = p0 + v0·t + ½·g·t² — six unknowns,
gravity known. One camera sees only each point's ray, but gravity fixes the
scale: the same image path at twice the depth would need twice the
acceleration. So the ball's image positions over the flight are enough.

Each image point (x, y) of p(t) satisfies x·(P3·p) = P1·p and y·(P3·p) = P2·p
(P the camera's projection rows) — linear in (p0, v0) — so three points at
distinct times solve a candidate exactly. The detector's boxes include heads
and lamps, so RANSAC picks the candidate most boxes agree with, and a robust
least-squares fit on reprojection error polishes it.

Gravity's scale is a weak hold on depth, though: a sideline camera behind an
end line looks along the court, so a flight over the net moves mostly in
depth, and the fit alone spreads ±15–20% there (two calibrated videos,
against contacts placed from the actor's feet; across the ray they agree to
~0.1 m). Where a touch was placed that way (court/camera.lift) it anchors
the polish: the image fixes each ray, the anchors where along it.

A rally is then fitted as one chain (`fit_chain`): the ball has one position
at each touch, and a flight is the gravity path between its two touches'
positions — so the unknowns are three per touch, consecutive arcs meet by
construction, and each touch is held by the boxes of both flights around it.
Fitted flight by flight, neighbouring arcs missed each other by 1.3 m
(median) at their shared touch.

Image distances are in frame heights (as in court/camera.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from yp_video.court.camera import GRAVITY, Camera
from yp_video.court.geometry import COURT_LENGTH, COURT_WIDTH

#: A box this close to the trajectory's image (frame heights) is the ball:
#: ~6 px at 1080p, about a ball's radius.
INLIER_TOL = 0.006
RANSAC_ITERS = 400
#: A trajectory must be seen on at least this many frames...
MIN_FRAMES = 6
#: ...spread over at least this share of the flight, so it is interpolated,
#: never extrapolated from one end — except an open flight's (`fit_open`),
#: whose share is of the time from its touch to the last box on it.
MIN_COVERAGE = 0.5
#: What a ball in play can do: anything else is a wrong fit.
MAX_SPEED_MS = 35.0
MAX_HEIGHT_M = 15.0
#: Below the floor by more than this is a wrong fit (the floor itself is
#: where a ball lands).
FLOOR_SLACK_M = 0.3
#: How far outside the court lines a flight may pass.
COURT_SLACK_M = 8.0
#: How far a touch's time may move from where it was spotted, seconds: a
#: touch is marked to a frame or two, and at 20 m/s a frame is 0.7 m.
TOUCH_SLACK_S = 0.1
#: How far an anchoring touch may sit from the trajectory, metres, before it
#: costs as much as a box INLIER_TOL off: the ball is not exactly above the
#: actor's feet, and the feet are a box's bottom edge.
ANCHOR_SIGMA_M = 0.5

_G = np.array([0.0, 0.0, -GRAVITY])


@dataclass(frozen=True)
class Trajectory:
    p0: np.ndarray
    v0: np.ndarray
    #: Distinct frames whose box lies on it.
    frames: int
    #: RMS reprojection error over those boxes, in frame heights.
    rmse: float
    #: Seconds from the start to the last of those boxes: past it the path
    #: is extrapolated.
    last: float

    def at(self, t: np.ndarray) -> np.ndarray:
        return _positions(np.r_[self.p0, self.v0][None], np.asarray(t, dtype=np.float64))[0]

    def landing(self) -> float:
        """Seconds until the ball comes down to the floor."""
        vz, z0 = self.v0[2], self.p0[2]
        return float((vz + np.sqrt(max(vz**2 + 2 * GRAVITY * z0, 0.0))) / GRAVITY)


def _positions(theta: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Court points (H, N, 3) of the trajectories theta (H, 6) at times t (N)."""
    return theta[:, None, :3] + theta[:, None, 3:] * t[:, None] + 0.5 * _G * (t**2)[:, None]


def _solve(p: np.ndarray, t: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """(p0, v0) of each row's three image points: t (H, 3), xy (H, 3, 2) → (H, 6)."""
    gravity = 0.5 * (t**2)[..., None] * _G
    a_rows, b_rows = [], []
    for axis in (0, 1):
        r = xy[..., axis : axis + 1] * p[2, :3] - p[axis, :3]
        a_rows.append(np.concatenate([r, r * t[..., None]], axis=-1))
        b_rows.append(-(r * gravity).sum(axis=-1) - (xy[..., axis] * p[2, 3] - p[axis, 3]))
    a, b = np.concatenate(a_rows, axis=1), np.concatenate(b_rows, axis=1)
    return (np.linalg.pinv(a) @ b[..., None])[..., 0]


def _solve_from(p: np.ndarray, p0: np.ndarray, t: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """(p0, v0) of each row's two image points from the known start p0:
    t (H, 2), xy (H, 2, 2) → (H, 6). With p0 known only v0 is left, linear
    as in `_solve`."""
    known = p0 + 0.5 * (t**2)[..., None] * _G
    a_rows, b_rows = [], []
    for axis in (0, 1):
        r = xy[..., axis : axis + 1] * p[2, :3] - p[axis, :3]
        a_rows.append(r * t[..., None])
        b_rows.append(-(r * known).sum(axis=-1) - (xy[..., axis] * p[2, 3] - p[axis, 3]))
    a, b = np.concatenate(a_rows, axis=1), np.concatenate(b_rows, axis=1)
    v0 = (np.linalg.pinv(a) @ b[..., None])[..., 0]
    return np.concatenate([np.broadcast_to(p0, v0.shape), v0], axis=1)


def _offsets(p: np.ndarray, theta: np.ndarray, t: np.ndarray, xy: np.ndarray, aspect: float) -> np.ndarray:
    """Reprojection offset (H, N, 2) of each trajectory from each point, frame
    heights; a whole frame (finite, so least squares can still step) behind
    the camera."""
    h = _positions(theta, t) @ p[:, :3].T + p[:, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (h[..., :2] / h[..., 2:3] - xy) * [aspect, 1.0]
    return np.where(h[..., 2:3] > 0, d, 1.0)


def _errors(p: np.ndarray, theta: np.ndarray, t: np.ndarray, xy: np.ndarray, aspect: float) -> np.ndarray:
    """Reprojection error (H, N), frame heights."""
    return np.linalg.norm(_offsets(p, theta, t, xy, aspect), axis=-1)


def _plausible(theta: np.ndarray, duration: float) -> np.ndarray:
    """(H,) whether each trajectory is one a ball in play can fly."""
    x, y, z = np.moveaxis(_positions(theta, np.linspace(0, duration, 9)), -1, 0)
    return (
        (np.linalg.norm(theta[:, 3:], axis=1) <= MAX_SPEED_MS)
        & (z.min(axis=1) >= -FLOOR_SLACK_M) & (z.max(axis=1) <= MAX_HEIGHT_M)
        & (x.min(axis=1) >= -COURT_SLACK_M) & (x.max(axis=1) <= COURT_LENGTH + COURT_SLACK_M)
        & (y.min(axis=1) >= -COURT_SLACK_M) & (y.max(axis=1) <= COURT_WIDTH + COURT_SLACK_M)
    )


def _anchor_offsets(theta: np.ndarray, anchors) -> np.ndarray:
    """Each anchor's offset (A, 3) from the trajectory theta (6,) at its
    time, in image-error units (ANCHOR_SIGMA_M ↔ INLIER_TOL)."""
    if not len(anchors):
        return np.zeros((0, 3))
    at = _positions(theta[None], np.array([a[0] for a in anchors], dtype=np.float64))[0]
    return (at - np.array([a[1] for a in anchors])) * (INLIER_TOL / ANCHOR_SIGMA_M)


def _closest_per_time(err: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Indices of the inliers, one per distinct time: the closest."""
    inl = np.flatnonzero(err < INLIER_TOL)
    order = inl[np.lexsort((err[inl], t[inl]))]
    return np.sort(order[np.r_[True, np.diff(t[order]) != 0]]) if len(order) else order


def fit(
    camera: Camera,
    aspect: float,
    t: np.ndarray,
    xy: np.ndarray,
    duration: float,
    rng: np.random.Generator,
    anchors: list[tuple[float, np.ndarray]] = (),
) -> Trajectory | None:
    """The flight's trajectory, or None when the boxes name none.

    ``t`` seconds from the flight's start (several boxes may share one),
    ``xy`` their normalized image points, ``duration`` the flight's length,
    ``aspect`` the frame's width / height, ``anchors`` (seconds, court
    point) where touches of the flight were placed.
    """
    order = np.argsort(t, kind="stable")
    t = np.asarray(t, dtype=np.float64)[order]
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)[order]
    times, first, count = np.unique(t, return_index=True, return_counts=True)
    if len(times) < MIN_FRAMES:
        return None
    p = np.array(camera.projection)

    # Hypotheses from three boxes at distinct times, all at once.
    k = rng.random((RANSAC_ITERS, len(times))).argsort(axis=1)[:, :3]
    pick = first[k] + (rng.random(k.shape) * count[k]).astype(np.int64)
    theta = _solve(p, t[pick], xy[pick])
    found = _consensus(p, theta[_plausible(theta, duration)], t, xy, first, aspect, anchors)
    if found is None:
        return None
    theta, keep, err = found
    if np.ptp(t[keep]) < MIN_COVERAGE * duration or not _plausible(theta[None], duration)[0]:
        return None
    return Trajectory(
        p0=theta[:3], v0=theta[3:], frames=len(keep), rmse=float(np.sqrt(np.mean(err[keep] ** 2))),
        last=float(t[keep].max()),
    )


def fit_open(
    camera: Camera,
    aspect: float,
    t: np.ndarray,
    xy: np.ndarray,
    start: np.ndarray,
    rng: np.random.Generator,
) -> Trajectory | None:
    """The trajectory of a flight with no far end — the ball went out, or
    left the shot — from ``start``, its touch's placed court point, or None.

    With nothing holding its end, it is judged only as far as it was seen:
    in the air up to its last box, which half the time since the touch must
    cover. Each hypothesis leaves from ``start`` (two boxes then fix its
    velocity), so a still lamp seen over the whole window cannot outvote a
    ball seen only until it left the shot — moved onto the ray of the box
    at the touch itself when there is one: the image is exact there, the
    placement only says how far along. Where it came down is the
    trajectory's to say (`Trajectory.landing`).
    """
    order = np.argsort(t, kind="stable")
    t = np.asarray(t, dtype=np.float64)[order]
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)[order]
    times, first, count = np.unique(t, return_index=True, return_counts=True)
    if len(times) < MIN_FRAMES:
        return None
    p = np.array(camera.projection)
    start = np.asarray(start, dtype=np.float64)
    if times[0] == 0:
        c, d = np.array(camera.center), camera.ray(tuple(xy[0]))
        start = c + d * ((start - c) @ d)

    # Hypotheses from two boxes after the touch, all at once.
    later = np.flatnonzero(times > 0)
    k = later[rng.random((RANSAC_ITERS, len(later))).argsort(axis=1)[:, :2]]
    pick = first[k] + (rng.random(k.shape) * count[k]).astype(np.int64)
    theta = _solve_from(p, start, t[pick], xy[pick])
    found = _consensus(p, theta[_plausible(theta, times[-1])], t, xy, first, aspect, [(0.0, start)])
    if found is None:
        return None
    theta, keep, err = found
    seen = float(t[keep].max())
    if np.ptp(t[keep]) < MIN_COVERAGE * seen or not _plausible(theta[None], seen)[0]:
        return None
    return Trajectory(
        p0=theta[:3], v0=theta[3:], frames=len(keep), rmse=float(np.sqrt(np.mean(err[keep] ** 2))), last=seen,
    )


def _consensus(
    p: np.ndarray, theta: np.ndarray, t: np.ndarray, xy: np.ndarray, first: np.ndarray, aspect: float, anchors,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """The hypothesis (of ``theta``, H×6) most boxes agree with, polished on
    its boxes and the anchors: (theta, the boxes kept, every box's error),
    or None when fewer than MIN_FRAMES agree."""
    if not len(theta):
        return None
    inliers = _errors(p, theta, t, xy, aspect) < INLIER_TOL
    seen = np.logical_or.reduceat(inliers, first, axis=1).sum(axis=1)
    if seen.max() < MIN_FRAMES:
        return None

    theta = theta[seen.argmax()]
    for _ in range(2):
        keep = _closest_per_time(_errors(p, theta[None], t, xy, aspect)[0], t)
        if len(keep) < MIN_FRAMES:
            return None
        # Residuals in INLIER_TOL units, so the solver's tolerances mean something.
        theta = least_squares(
            lambda th, k=keep: np.concatenate([
                _offsets(p, th[None], t[k], xy[k], aspect)[0].ravel(), _anchor_offsets(th, anchors).ravel(),
            ]) / INLIER_TOL,
            theta, loss="soft_l1", x_scale="jac",
        ).x
    err = _errors(p, theta[None], t, xy, aspect)[0]
    keep = _closest_per_time(err, t)
    return (theta, keep, err) if len(keep) >= MIN_FRAMES else None


@dataclass(frozen=True)
class Flight:
    """The boxes between two consecutive touches: ``t`` seconds from the
    first touch (several boxes may share one), ``xy`` normalized image points."""

    t: np.ndarray
    xy: np.ndarray
    duration: float


@dataclass(frozen=True)
class Chain:
    #: The ball at each touch, court metres; None where nothing placed it.
    touches: list[np.ndarray | None]
    #: How far each touch's time moved, seconds (0 where unplaced).
    shifts: list[float]
    #: Per flight: distinct frames whose box lies on its arc, and their RMS
    #: reprojection error (frame heights; nan when none).
    seen: list[tuple[int, float]]


def _theta(pa: np.ndarray, pb: np.ndarray, duration: float) -> np.ndarray:
    """(p0, v0) of the gravity path from pa to pb in ``duration`` seconds."""
    return np.r_[pa, (pb - pa) / duration - 0.5 * _G * duration]


def plausible(pa: np.ndarray, pb: np.ndarray, duration: float) -> bool:
    """Whether a ball in play can fly the gravity path from pa to pb."""
    return bool(_plausible(_theta(pa, pb, duration)[None], duration)[0])


def fit_chain(
    camera: Camera,
    aspect: float,
    flights: list[Flight],
    anchors: dict[int, np.ndarray],
    rng: np.random.Generator,
) -> Chain:
    """The ball at each touch of a run of consecutive flights (touch k starts
    flight k), fitted to every flight's boxes at once, with ``anchors``
    (touch index → court point placed otherwise) held at ANCHOR_SIGMA_M.

    Each flight first gets its own RANSAC trajectory (`fit`), which seeds
    its touches and picks its boxes; a touch neither flight nor anchor
    places splits the run, so a chain never guesses through it.
    """
    p = np.array(camera.projection)
    n = len(flights) + 1
    own = [
        fit(camera, aspect, f.t, f.xy, f.duration, rng,
            [(at, anchors[j]) for at, j in ((0.0, k), (f.duration, k + 1)) if j in anchors])
        if len(f.t) else None
        for k, f in enumerate(flights)
    ]
    # The boxes each flight's own trajectory kept: the chain starts from
    # these, as seed touches averaged from two disagreeing fits sit off both.
    own_boxes = [
        _closest_per_time(_errors(p, np.r_[o.p0, o.v0][None], f.t, f.xy, aspect)[0], f.t) if o else None
        for f, o in zip(flights, own)
    ]
    seeds: list[np.ndarray | None] = []
    for k in range(n):
        votes = [own[k - 1].at([flights[k - 1].duration])[0]] if k > 0 and own[k - 1] else []
        votes += [own[k].at([0.0])[0]] if k < n - 1 and own[k] else []
        votes += [anchors[k]] if k in anchors else []
        seeds.append(np.mean(votes, axis=0) if votes else None)

    touches: list[np.ndarray | None] = list(seeds)
    shifts = [0.0] * n
    start = 0
    while start < n:
        if seeds[start] is None:
            start += 1
            continue
        end = start
        while end + 1 < n and seeds[end + 1] is not None:
            end += 1
        if end > start:
            touches[start:end + 1], shifts[start:end + 1] = _polish(
                p, aspect, flights[start:end], seeds[start:end + 1],
                {k - start: a for k, a in anchors.items() if start <= k <= end},
                [own_boxes[k] for k in range(start, end)],
            )
        start = end + 1

    seen = []
    for k, f in enumerate(flights):
        if touches[k] is None or touches[k + 1] is None or not len(f.t):
            seen.append((0, float("nan")))
            continue
        err = _flight_errors(p, aspect, f, touches[k], touches[k + 1], shifts[k], shifts[k + 1])
        keep = _closest_per_time(err, f.t)
        seen.append((len(keep), float(np.sqrt(np.mean(err[keep] ** 2))) if len(keep) else float("nan")))
    return Chain(touches=touches, shifts=shifts, seen=seen)


def _flight_offsets(p, aspect, f: Flight, pa, pb, sa: float, sb: float, keep=slice(None)) -> np.ndarray:
    """Reprojection offsets of a flight's boxes from the arc between its
    touches, the touches' times moved by sa and sb seconds."""
    duration = max(f.duration + sb - sa, 1e-3)
    return _offsets(p, _theta(pa, pb, duration)[None], f.t[keep] - sa, f.xy[keep], aspect)[0]


def _flight_errors(p, aspect, f: Flight, pa, pb, sa: float, sb: float) -> np.ndarray:
    return np.linalg.norm(_flight_offsets(p, aspect, f, pa, pb, sa, sb), axis=-1)


def _polish(p, aspect, flights: list[Flight], seeds: list[np.ndarray], anchors: dict[int, np.ndarray],
            own_boxes: list[np.ndarray | None]):
    """Least squares over one run's touches — position and a time shift of
    at most TOUCH_SLACK_S each: every flight's boxes on its arc, plus the
    anchors. The first round takes the boxes each flight's own fit kept
    (twice the tolerance off the seed arcs where it had none); later rounds
    re-pick them off the chain. Returns (positions, shifts)."""
    n = len(seeds)
    x = np.concatenate([np.r_[seed, 0.0] for seed in seeds])
    lo = np.tile([-np.inf, -np.inf, -np.inf, -TOUCH_SLACK_S], n)
    hi = -lo

    def touch(x, k):
        return x[4 * k:4 * k + 3], x[4 * k + 3]

    for round_ in range(3):
        picked = []
        for k, f in enumerate(flights):
            if round_ == 0 and own_boxes[k] is not None:
                picked.append(own_boxes[k])
                continue
            (pa, sa), (pb, sb) = touch(x, k), touch(x, k + 1)
            err = _flight_errors(p, aspect, f, pa, pb, sa, sb) if len(f.t) else np.zeros(0)
            picked.append(_closest_per_time(err / (2 if round_ == 0 else 1), f.t))

        def residuals(x):
            out = []
            for k, (f, keep) in enumerate(zip(flights, picked)):
                (pa, sa), (pb, sb) = touch(x, k), touch(x, k + 1)
                out.append(_flight_offsets(p, aspect, f, pa, pb, sa, sb, keep).ravel())
            out += [(touch(x, k)[0] - a) * (INLIER_TOL / ANCHOR_SIGMA_M) for k, a in anchors.items()]
            return np.concatenate(out) / INLIER_TOL

        if len(residuals(x)) < 3 * n:
            break  # too little left to hold every touch: keep what it has
        # A flight's boxes move only its two touches; an anchor only its own.
        rows = [np.repeat([[k, k + 1]], 2 * len(keep), axis=0) for k, keep in enumerate(picked)]
        rows += [np.repeat([[k, k]], 3, axis=0) for k in anchors]
        owners = np.concatenate(rows)
        sparsity = lil_matrix((len(owners), 4 * n), dtype=np.int8)
        for i, (a, b) in enumerate(owners):
            sparsity[i, 4 * a:4 * a + 4] = 1
            sparsity[i, 4 * b:4 * b + 4] = 1
        # A few chains crawl along a flat valley for a thousand steps to gain
        # nothing visible; the cap stops them where they already fit.
        x = least_squares(residuals, x, bounds=(lo, hi), loss="soft_l1", x_scale="jac",
                          jac_sparsity=sparsity, ftol=1e-6, xtol=1e-6, max_nfev=100).x
    return [touch(x, k)[0] for k in range(n)], [float(touch(x, k)[1]) for k in range(n)]
