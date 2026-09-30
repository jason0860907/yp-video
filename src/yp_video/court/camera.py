"""Full pinhole camera from a court calibration: where the camera stands,
where it looks, and its focal length — what a height needs.

The floor homography (geometry.fit) cannot tell a far point from a high one.
A camera can: every image point becomes a ray in court space. The model is
the plain pinhole — principal point at the frame centre, square pixels, no
lens distortion — leaving focal length plus pose, seven unknowns. The floor
marks alone pin those down (a plane's homography fixes them); net-top marks,
standing net_height_m above the floor, are the off-plane check that makes
the solve trustworthy.

Image units here are frame heights: u = x·aspect, v = y for normalized
(x, y). Focal length and reprojection error are in the same units, so a
calibration is independent of the video's resolution.
"""

from __future__ import annotations

import cv2
import numpy as np
from pydantic import BaseModel
from scipy.optimize import minimize_scalar

from yp_video.court import geometry
from yp_video.court.annotations import Calibration

GRAVITY = 9.81

Matrix = list[list[float]]


class Camera(BaseModel):
    #: 3×4 projection: homogeneous court point (x, y, z, 1) → normalized
    #: image point (x, y, 1), up to scale.
    projection: Matrix
    #: Camera centre in court metres.
    center: tuple[float, float, float]
    focal: float
    #: RMS reprojection error over every mark, in frame heights.
    rmse: float
    #: How many of the marks stand off the floor (net tops).
    off_floor: int

    def ray(self, xy: tuple[float, float]) -> np.ndarray:
        """Unit direction, in court space, of the ray through image point xy.

        The projection's third row is camera depth, so solving for image
        (x, y, 1) yields a direction at depth +1 — already pointing forward.
        """
        d = np.linalg.solve(np.array(self.projection)[:, :3], np.array([xy[0], xy[1], 1.0]))
        return d / np.linalg.norm(d)


class CameraError(ValueError):
    pass


def _world_points(calibration: Calibration) -> tuple[np.ndarray, np.ndarray, int]:
    world, image, off_floor = [], [], 0
    for name, xy in calibration.points.items():
        if name in geometry.LANDMARKS:
            x, y = geometry.LANDMARKS[name]
            world.append((x, y, 0.0))
        else:
            x, y = geometry.NET_LANDMARKS[name]
            world.append((x, y, calibration.net_height_m))
            off_floor += 1
        image.append(xy)
    return np.array(world, dtype=np.float64), np.array(image, dtype=np.float64), off_floor


def _pose(world: np.ndarray, image: np.ndarray, focal: float, aspect: float):
    k = np.array([[focal, 0, aspect / 2], [0, focal, 0.5], [0, 0, 1]])
    ok, rvec, tvec = cv2.solvePnP(world, image, k, None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(world, image, k, None, rvec, tvec)
    projected, _ = cv2.projectPoints(world, rvec, tvec, k, None)
    rmse = float(np.sqrt(((projected.reshape(-1, 2) - image) ** 2).sum(axis=1).mean()))
    return rmse, k, rvec, tvec


def solve(calibration: Calibration) -> Camera:
    """Fit focal length and pose to every mark (floor and net tops)."""
    floor = sum(1 for n in calibration.points if n in geometry.LANDMARKS)
    if floor < geometry.MIN_POINTS:
        raise CameraError(f"Mark at least {geometry.MIN_POINTS} floor points ({floor} marked)")
    width, height = calibration.frame_size
    aspect = width / height
    world, norm, off_floor = _world_points(calibration)
    image = norm * np.array([aspect, 1.0])

    # Focal length is the one non-linear unknown; pose is solved exactly per
    # focal. The cost is not unimodal over the whole range, so a coarse log
    # sweep brackets the best basin before the bounded 1-D minimiser polishes.
    def cost(log_f: float) -> float:
        r = _pose(world, image, float(np.exp(log_f)), aspect)
        return r[0] if r else np.inf

    grid = np.linspace(np.log(0.2), np.log(8.0), 80)
    best = int(np.argmin([cost(g) for g in grid]))
    bounds = (grid[max(best - 1, 0)], grid[min(best + 1, len(grid) - 1)])
    focal = float(np.exp(minimize_scalar(cost, bounds=bounds, method="bounded").x))
    solved = _pose(world, image, focal, aspect)
    if solved is None:
        raise CameraError("Camera solve failed for these marks")
    rmse, k, rvec, tvec = solved
    rot, _ = cv2.Rodrigues(rvec)
    # K works in frame-height units; scale u back to a normalized x.
    to_norm = np.diag([1 / aspect, 1.0, 1.0])
    projection = to_norm @ k @ np.hstack([rot, tvec])
    center = (-rot.T @ tvec).ravel()
    return Camera(
        projection=projection.tolist(),
        center=tuple(float(v) for v in center),
        focal=focal,
        rmse=rmse,
        off_floor=off_floor,
    )


def project(camera: Camera, points: np.ndarray) -> np.ndarray:
    """Court points (N, 3) → normalized image points (N, 2)."""
    p = np.array(camera.projection)
    h = np.hstack([points, np.ones((len(points), 1))]) @ p.T
    return h[:, :2] / h[:, 2:3]


def lift(camera: Camera, ball_xy: tuple[float, float], foot: tuple[float, float]) -> np.ndarray | None:
    """The ball's court position at a contact, or None when the feet name no
    point in front of the camera.

    The ball lies somewhere on its image ray; which depth is what one camera
    cannot say. The contact supplies it: the ball is (near) above the actor's
    feet. So take the point on the ray closest to the vertical line standing
    on the feet — the ray keeps the ball exactly where the image shows it,
    the line only chooses how far along the ray.
    """
    c = np.array(camera.center)
    d = camera.ray(ball_xy)
    f = np.array([foot[0], foot[1], 0.0])
    up = np.array([0.0, 0.0, 1.0])
    # Minimise |c + s·d − (f + z·up)| over s, z.
    a = np.array([[d @ d, -(d @ up)], [d @ up, -(up @ up)]])
    b = np.array([(f - c) @ d, (f - c) @ up])
    s, _z = np.linalg.solve(a, b)
    return c + s * d if s > 0 else None


def arc(p0: np.ndarray, p1: np.ndarray, duration: float, samples: int = 16) -> np.ndarray:
    """The ballistic path from p0 to p1 taking `duration` seconds (no drag).

    Two points and a flight time fix a parabola under gravity: horizontally
    the ball moves at constant velocity, vertically it decelerates at g.
    """
    g = np.array([0.0, 0.0, -GRAVITY])
    v0 = (p1 - p0) / duration - 0.5 * g * duration
    t = np.linspace(0.0, duration, samples)[:, None]
    return p0 + v0 * t + 0.5 * g * t**2
