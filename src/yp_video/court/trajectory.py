"""A flight's 3D trajectory from the ball's image positions and the camera.

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

Image distances are in frame heights (as in court/camera.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from yp_video.court.camera import GRAVITY, Camera
from yp_video.court.geometry import COURT_LENGTH, COURT_WIDTH

#: A box this close to the trajectory's image (frame heights) is the ball:
#: ~6 px at 1080p, about a ball's radius.
INLIER_TOL = 0.006
RANSAC_ITERS = 400
#: A trajectory must be seen on at least this many frames...
MIN_FRAMES = 6
#: ...spread over at least this share of the flight, so it is interpolated,
#: never extrapolated from one end.
MIN_COVERAGE = 0.5
#: What a ball in play can do: anything else is a wrong fit.
MAX_SPEED_MS = 35.0
MAX_HEIGHT_M = 15.0
#: Below the floor by more than this is a wrong fit (the floor itself is
#: where a ball lands).
FLOOR_SLACK_M = 0.3
#: How far outside the court lines a flight may pass.
COURT_SLACK_M = 8.0
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

    def at(self, t: np.ndarray) -> np.ndarray:
        return _positions(np.r_[self.p0, self.v0][None], np.asarray(t, dtype=np.float64))[0]


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
    theta = theta[_plausible(theta, duration)]
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
    if len(keep) < MIN_FRAMES or np.ptp(t[keep]) < MIN_COVERAGE * duration or not _plausible(theta[None], duration)[0]:
        return None
    return Trajectory(
        p0=theta[:3], v0=theta[3:], frames=len(keep), rmse=float(np.sqrt(np.mean(err[keep] ** 2))),
    )
