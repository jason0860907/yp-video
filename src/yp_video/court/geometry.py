"""Volleyball court geometry and the image↔court homography.

Court coordinates are metres on the floor plane, laid out for sideline
footage: x runs along the court from the end line on the image's left (0) to
the one on its right (18), the net at x = 9; y runs across it from the
sideline nearest the camera (0) to the far one (9).

Image coordinates are normalized to the frame, [0, 1] on both axes — the
convention every other label in this project uses, so a calibration survives
a re-encode at another resolution.

A homography maps one plane onto another, so it only locates points ON the
floor: a player's feet, a ball landing. A ball in the air projects to where
its line of sight meets the floor, not to the spot beneath it. The net-top
landmarks are marked for exactly that reason — they are the known heights a
full camera solve needs — and never enter the floor fit.
"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel

COURT_LENGTH = 18.0
COURT_WIDTH = 9.0
ATTACK_LINE_OFFSET = 3.0

FloorLandmark = Literal[
    "left_near", "left_far",
    "left_attack_near", "left_attack_far",
    "center_near", "center_far",
    "right_attack_near", "right_attack_far",
    "right_near", "right_far",
]
#: Where the net's top band meets each sideline (the antennas stand there).
NetLandmark = Literal["net_top_near", "net_top_far"]
Landmark = FloorLandmark | NetLandmark

#: Every point a user can mark: the line intersections on the floor.
LANDMARKS: dict[FloorLandmark, tuple[float, float]] = {
    "left_near": (0.0, 0.0),
    "left_far": (0.0, COURT_WIDTH),
    "left_attack_near": (COURT_LENGTH / 2 - ATTACK_LINE_OFFSET, 0.0),
    "left_attack_far": (COURT_LENGTH / 2 - ATTACK_LINE_OFFSET, COURT_WIDTH),
    "center_near": (COURT_LENGTH / 2, 0.0),
    "center_far": (COURT_LENGTH / 2, COURT_WIDTH),
    "right_attack_near": (COURT_LENGTH / 2 + ATTACK_LINE_OFFSET, 0.0),
    "right_attack_far": (COURT_LENGTH / 2 + ATTACK_LINE_OFFSET, COURT_WIDTH),
    "right_near": (COURT_LENGTH, 0.0),
    "right_far": (COURT_LENGTH, COURT_WIDTH),
}

#: Each net-top landmark's floor position; its height is the calibration's
#: net height (2.43 m men, 2.24 m women).
NET_LANDMARKS: dict[NetLandmark, tuple[float, float]] = {
    "net_top_near": (COURT_LENGTH / 2, 0.0),
    "net_top_far": (COURT_LENGTH / 2, COURT_WIDTH),
}

#: The painted lines, as court-space segments.
LINES: list[tuple[float, float, float, float]] = [
    (0.0, 0.0, COURT_LENGTH, 0.0),
    (0.0, COURT_WIDTH, COURT_LENGTH, COURT_WIDTH),
    *((x, 0.0, x, COURT_WIDTH) for x in sorted({x for x, _ in LANDMARKS.values()})),
]

MIN_POINTS = 4

Matrix = list[list[float]]


class Fit(BaseModel):
    """A solved calibration. Both directions are served so no client ever
    has to invert a matrix."""

    image_to_court: Matrix
    court_to_image: Matrix
    #: RMS distance, in metres, between each marked point projected onto the
    #: court and the landmark it names. Zero with exactly four points — four
    #: pin a homography down exactly, so only a fifth can disagree.
    rmse_m: float


class FitError(ValueError):
    pass


def fit(points: dict[Landmark, tuple[float, float]]) -> Fit:
    """Solve the floor homography from the marked floor landmarks (least
    squares past four); net-top marks are not on the floor and are skipped."""
    points = {n: xy for n, xy in points.items() if n in LANDMARKS}
    if len(points) < MIN_POINTS:
        raise FitError(f"Mark at least {MIN_POINTS} floor points ({len(points)} marked)")
    names = list(points)
    image = np.array([points[n] for n in names], dtype=np.float64)
    court = np.array([LANDMARKS[n] for n in names], dtype=np.float64)
    h, _ = cv2.findHomography(image, court, 0)
    if h is None or abs(np.linalg.det(h)) < 1e-12:
        raise FitError("These points do not span the floor — three of them may lie on one line")
    residual = project(h, image) - court
    return Fit(
        image_to_court=h.tolist(),
        court_to_image=np.linalg.inv(h).tolist(),
        rmse_m=float(np.sqrt((residual**2).sum(axis=1).mean())),
    )


def project(h: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Apply a homography to an (N, 2) array of points."""
    return cv2.perspectiveTransform(xy.reshape(-1, 1, 2), h).reshape(-1, 2)
