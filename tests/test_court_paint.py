"""Pre-drawn guides: paint found in a rendered court lands on its lines;
grooves between floor mats alone give nothing."""

import cv2
import numpy as np
import pytest

from yp_video.court import paint
from yp_video.court.geometry import LINES

W, H = 1920, 1080


def _camera(focal=1.1, center=(9.0, -11.0, 2.5), target=(9.0, 4.5, 0.0)):
    """A pinhole behind the near sideline: court metres → pixels."""
    c, t = np.array(center), np.array(target)
    z = (t - c) / np.linalg.norm(t - c)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    rot = np.vstack([x, np.cross(z, x), z])
    k = np.array([[focal * H, 0, W / 2], [0, focal * H, H / 2], [0, 0, 1]])

    def see(x_m, y_m):
        u = k @ (rot @ np.array([x_m, y_m, 0.0]) - rot @ c)
        return u[:2] / u[2]

    return see


def _floor(see=None, grooves=False):
    image = np.full((H, W, 3), (70, 120, 170), np.uint8)
    rng = np.random.default_rng(0)
    image = np.clip(image + rng.normal(0, 3, image.shape), 0, 255).astype(np.uint8)
    if grooves:
        for y in range(400, H, 60):
            cv2.line(image, (0, y), (W, y + 120), (40, 80, 120), 2)
    if see is not None:
        for x0, y0, x1, y1 in LINES:
            a, b = see(x0, y0), see(x1, y1)
            cv2.line(image, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), (235, 235, 235), 6, cv2.LINE_AA)
    return image


def _distance_to(line_ends, point):
    (ax, ay), (bx, by) = line_ends
    return abs((bx - ax) * (ay - point[1]) - (ax - point[0]) * (by - ay)) / np.hypot(bx - ax, by - ay)


@pytest.mark.parametrize(
    ("center", "grooves"),
    [
        ((-2.0, -6.0, 3.0), True),  # behind a corner, mats seamed parallel to the paint
        ((9.0, -11.0, 2.5), False),  # square on to the sideline: its lines run level
    ],
)
def test_guides_lie_on_the_painted_court(center, grooves):
    see = _camera(center=center)
    guides = paint.guides([_floor(see, grooves=grooves)] * 3, floor_top=0.25)
    painted = [(see(x0, y0), see(x1, y1)) for x0, y0, x1, y1 in LINES]
    assert len(guides) == len(LINES)
    for a, b in guides:
        ends = [np.array(a) * (W, H), np.array(b) * (W, H)]
        # Each guide runs along one 6 px painted line, end to end.
        assert min(max(_distance_to(line, e) for e in ends) for line in painted) < 5


def test_grooves_alone_give_no_guides():
    assert paint.guides([_floor(grooves=True)] * 3, floor_top=0.25) == []


def test_court_guides_reads_the_floor_off_the_tracks(monkeypatch):
    from types import SimpleNamespace

    from yp_video.extraction import court_guides as module

    see = _camera()
    image = _floor(see)
    # Feet in pixels, the highest just above the far sideline (y ≈ 0.33 H).
    records = [{"frames": list(range(rally * 100, rally * 100 + 50)),
                "boxes": [[900, 200, 960, 0.34 * H]] * 50, "scores": [0.9] * 50} for rally in range(3)]
    tracks = SimpleNamespace(meta={"frame_size": [W, H]}, records=records)
    picked = []
    monkeypatch.setattr(module, "tracklet_data", lambda stem: tracks)
    monkeypatch.setattr(module, "read_frames", lambda video, picks: picked.extend(picks) or [(i, image) for i in picks])
    guides = module.court_guides("game", "game.mp4")
    assert len(guides) == len(LINES)
    # Spread over the tracked rallies, not the gaps between them.
    assert len(picked) == module.FRAMES and picked[0] == 0 and picked[-1] == 249
