"""Guide lines drawn ahead of the marking: find the court's paint in a video.

The marker draws guides along the paint and snaps marks to their crossings
(annotations.Calibration.guides). This pre-draws them. Its answer is only
ever guides — straight runs of real paint — never marks: a wrong guide costs
one delete, a wrong mark a bad calibration.

The court model is what tells paint from clutter. Line segments come from
the median of frames spread over the video (the players walk out of it),
below the highest feet seen (the walls are gone), minus the dark grooves
between floor mats. Every pairing of two lines along the court with two
across it, each assigned court positions, is a candidate floor homography,
scored by how much of the court's paint it lays on segments; the best few
are re-solved against every segment they match. The best that a real
camera could have seen wins — clutter that lines up rarely makes one — and
the answer is one guide per court line it found paint on, drawn across the
frame.

Pixel units throughout; the result is normalized to the frame.
"""

from __future__ import annotations

import itertools

import cv2
import numpy as np

from yp_video.court import camera
from yp_video.court.geometry import COURT_LENGTH, COURT_WIDTH, LANDMARKS, LINES

Segment = tuple[tuple[float, float], tuple[float, float]]

#: Detections this confident place feet; the highest of them, less the
#: margin, is where the floor ends.
FEET_SCORE = 0.5
FEET_PERCENTILE = 1.0
FLOOR_MARGIN = 0.03
#: Segments shorter than this (px) are texture, not paint.
MIN_SEGMENT = 25.0
#: A segment along a line this much darker (Lab L) than the same floor on
#: both sides, with no paint beside it, is a groove between floor mats, which
#: run parallel to the paint.
GROOVE_DEPTH = 4.0
#: Segments within this distance (px) and angle of one line merge into it.
MERGE_DIST = 5.0
MERGE_ANGLE = np.deg2rad(1.5)
#: How many merged lines, longest first, the vanishing-point split weighs.
VP_LINES = 40
#: A line belongs to a vanishing point when the direction to it is within
#: this angle of the line's own.
VP_ANGLE = np.deg2rad(1.5)
#: Candidates per family: lines along the court, lines across it.
ALONG, ACROSS = 8, 10
#: Model paint within this distance (px) of a segment counts as found.
HIT = 4.0
#: A court whose corners enclose less of the frame than this is folded flat
#: onto a line or two — or too far off to mark anyway.
MIN_COURT_AREA = 0.02
#: Scoring grid (px): paint landing in a cell already counted adds nothing.
SCORE_CELL = 4
#: Hypotheses re-solved against every segment they match.
REFINE_TOP = 30
#: Hypotheses whose landmarks all lie within this (px) of each other are one.
SAME_FIT = 10.0
#: Matching radii (px) of the successive re-solves, coarse to fine; the last
#: also decides which court lines found paint, so get a guide.
REFINE_RADII = (12.0, 8.0, 6.0, 6.0, 6.0, 6.0)
REFINE_ANGLE = np.deg2rad(2.0)
#: What a fit must look like to be a court rather than clutter that lined up
#: (doors, another sport's lines, floor boards): a pinhole camera (centred,
#: square pixels) reproduces it within MAX_CAMERA_RMSE frame heights — real
#: courts land under 0.04 even through a wide-angle lens's bend — from no
#: farther than MAX_CAMERA_DISTANCE metres from the court's centre (a court
#: folded onto one line needs a camera hundreds of metres off), and at least
#: MIN_PAINT_FOUND of its paint in view lies on segments.
MAX_CAMERA_RMSE = 0.04
MAX_CAMERA_DISTANCE = 40.0
MIN_PAINT_FOUND = 0.6

#: Court positions of the lines along the court (y) and across it (x).
ALONG_Y = sorted({y0 for _, y0, _, y1 in LINES if y0 == y1})
ACROSS_X = sorted({x0 for x0, _, x1, _ in LINES if x0 == x1})
#: Each of LINES in homogeneous form: y = k → (0, 1, -k); x = k → (1, 0, -k).
COURT_LINES = [
    np.array([0.0, 1.0, -y0]) if y0 == y1 else np.array([1.0, 0.0, -x0]) for x0, y0, x1, y1 in LINES
]
#: y → COURT_WIDTH - y: the court onto itself, mirrored.
_MIRROR = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, COURT_WIDTH], [0.0, 0.0, 1.0]])
#: x → COURT_LENGTH - x: the court onto itself, mirrored the other way.
_FLIP = np.array([[-1.0, 0.0, COURT_LENGTH], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
#: Every way the court's paint maps onto itself.
_SYMMETRIES = (np.eye(3), _MIRROR, _FLIP, _MIRROR @ _FLIP)
#: The court's corners.
_CORNERS = np.array([[0.0, 0.0], [COURT_LENGTH, 0.0], [COURT_LENGTH, COURT_WIDTH], [0.0, COURT_WIDTH]])
#: The paint, sampled every 10 cm, for scoring.
_PAINT = np.concatenate(
    [
        np.linspace((x0, y0), (x1, y1), int(max(abs(x1 - x0), abs(y1 - y0)) * 10) + 1)
        for x0, y0, x1, y1 in LINES
    ]
)


def guides(frames: list[np.ndarray], floor_top: float) -> list[Segment]:
    """Guide lines along the court paint in these frames (BGR, one video),
    normalized; empty when no court fits.

    ``floor_top`` is the normalized height above which nothing is floor —
    the highest feet seen."""
    background = np.median(np.stack(frames), axis=0).astype(np.uint8)
    h, w = background.shape[:2]
    segs = _segments(background, floor_top * h)
    families = _vanishing_families(_merge(segs))
    if len(families) < 2:
        return []
    homography = _fit(segs, families, floor_top * h, (w, h))
    if homography is None:
        return []
    return [
        ((round(a[0] / w, 4), round(a[1] / h, 4)), (round(b[0] / w, 4), round(b[1] / h, 4)))
        for a, b in _paint_in_view(homography, segs, (w, h))
    ]


def floor_top(boxes: np.ndarray, scores: np.ndarray) -> float:
    """Normalized height above which nothing is floor, from person boxes
    (xyxy normalized) seen over the video: the highest feet."""
    feet = boxes[scores > FEET_SCORE, 3]
    return float(np.percentile(feet, FEET_PERCENTILE)) - FLOOR_MARGIN


def _segments(image: np.ndarray, top: float) -> np.ndarray:
    """(N, 4) x0 y0 x1 y1 line segments below ``top``, grooves dropped."""
    lum = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0]
    found = cv2.createLineSegmentDetector().detect(lum)[0]
    if found is None:
        return np.empty((0, 4))
    segs = found.reshape(-1, 4).astype(np.float64)
    length = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    segs = segs[(length > MIN_SEGMENT) & (segs[:, 1] > top) & (segs[:, 3] > top)]
    darkest, brightest, left, right = _profile(lum, segs)
    floor = np.minimum(left, right)
    groove = (
        (darkest < floor - GROOVE_DEPTH)  # a dark line runs along it
        & (np.abs(left - right) < GROOVE_DEPTH)  # through one floor, not along a boundary
        & (brightest < np.maximum(left, right) + GROOVE_DEPTH)  # with no paint beside it
    )
    return segs[~groove]


def _profile(lum: np.ndarray, segs: np.ndarray, offset: float = 6.0, samples: int = 12):
    """Mean luminance along each segment — the darkest and the brightest of
    the rows within 2 px of it — and ``offset`` px to either side."""
    d = segs[:, 2:] - segs[:, :2]
    normal = np.c_[-d[:, 1], d[:, 0]] / np.hypot(d[:, 0], d[:, 1])[:, None]
    along = segs[:, None, :2] + np.linspace(0.1, 0.9, samples)[None, :, None] * d[:, None, :]
    limit = np.array([lum.shape[1] - 1, lum.shape[0] - 1])

    def sample(o: float) -> np.ndarray:
        q = np.clip(along + o * normal[:, None, :], 0, limit).astype(int)
        return lum[q[..., 1], q[..., 0]].astype(np.float64).mean(axis=1)

    rows = [sample(o) for o in (-2, -1, 0, 1, 2)]
    return np.min(rows, axis=0), np.max(rows, axis=0), sample(-offset), sample(offset)


def _line(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Homogeneous line through two points, normalized so l·(x, y, 1) is a
    signed distance."""
    line = np.cross([*p, 1.0], [*q, 1.0])
    return line / np.hypot(line[0], line[1])


def _merge(segs: np.ndarray) -> list[tuple[np.ndarray, float, np.ndarray]]:
    """Collinear segments as lines: (line, total segment length, weighted
    centre), longest first."""
    length = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    theta = np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0]) % np.pi
    used = np.zeros(len(segs), bool)
    lines = []
    for seed in np.argsort(-length):
        if used[seed]:
            continue
        line = _line(segs[seed, :2], segs[seed, 2:])
        for _ in range(2):
            near = (np.abs(segs[:, :2] @ line[:2] + line[2]) < MERGE_DIST) & (
                np.abs(segs[:, 2:] @ line[:2] + line[2]) < MERGE_DIST
            )
            turn = np.abs((theta - np.arctan2(-line[0], line[1]) % np.pi + np.pi / 2) % np.pi - np.pi / 2)
            members = ~used & near & (turn < MERGE_ANGLE)
            members[seed] = True
            points = np.vstack([segs[members, :2], segs[members, 2:]])
            weight = np.tile(length[members], 2)
            centre = weight @ points / weight.sum()
            direction = np.linalg.svd((points - centre) * np.sqrt(weight[:, None]))[2][0]
            line = np.array([-direction[1], direction[0], 0.0])
            line[2] = -line[:2] @ centre
        used |= members
        lines.append((line, float(length[members].sum()), centre))
    return sorted(lines, key=lambda x: -x[1])


def _vanishing_families(lines) -> list[list[np.ndarray]]:
    """The two dominant vanishing points' lines, each longest first: the
    paint along the court and the paint across it (which is which is left
    to the fit)."""
    top = lines[:VP_LINES]
    eqs = np.array([line for line, _, _ in top])
    weight = np.array([s for _, s, _ in top])
    centres = np.array([c for _, _, c in top])
    free = np.ones(len(top), bool)
    families = []
    for _ in range(2):
        best, members = 0.0, None
        for i, j in itertools.combinations(np.flatnonzero(free), 2):
            agree = _through(eqs, centres, np.cross(eqs[i], eqs[j])) & free
            if weight[agree].sum() > best:
                best, members = weight[agree].sum(), agree
        if members is None:
            break
        families.append([eqs[k] for k in np.flatnonzero(members)])
        free &= ~members
    return families


def _through(eqs: np.ndarray, centres: np.ndarray, vp: np.ndarray) -> np.ndarray:
    """Which lines head for this vanishing point, judged from each line's
    own centre."""
    direction = np.c_[-eqs[:, 1], eqs[:, 0]]
    if abs(vp[2]) < 1e-9 * np.linalg.norm(vp):
        toward = np.tile(vp[:2], (len(eqs), 1))
    else:
        toward = vp[:2] / vp[2] - centres
    cos = np.abs((direction * toward).sum(axis=1)) / np.maximum(np.linalg.norm(toward, axis=1), 1e-9)
    return cos > np.cos(VP_ANGLE)


def _fit(segs: np.ndarray, families, top: float, size: tuple[int, int]) -> np.ndarray | None:
    """The court → image homography whose paint best lies on the segments."""
    w, h = size
    mask = np.full((h, w), 255, np.uint8)
    for x0, y0, x1, y1 in segs:
        cv2.line(mask, (int(x0), int(y0)), (int(x1), int(y1)), 0, 1)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)

    hypotheses = []
    for along, across in (families, families[::-1]):
        for s0, s1 in itertools.permutations(along[:ALONG], 2):
            for c0, c1 in itertools.permutations(across[:ACROSS], 2):
                corners = np.array([np.cross(s, c) for s in (s0, s1) for c in (c0, c1)])
                if (np.abs(corners[:, 2]) < 1e-9).any():
                    continue
                image = (corners[:, :2] / corners[:, 2:]).astype(np.float32)
                for xa, xb in itertools.combinations(ACROSS_X, 2):
                    court = np.float32([[xa, 0], [xb, 0], [xa, COURT_WIDTH], [xb, COURT_WIDTH]])
                    hom = cv2.getPerspectiveTransform(court, image).astype(np.float64)
                    hypotheses.append((_score(hom, distance, top), hom))
    hypotheses.sort(key=lambda x: -x[0])

    refined = [r for r in (_refine(hom, segs, size) for hom in _distinct(hypotheses, REFINE_TOP)) if r is not None]
    refined.sort(key=lambda hom: -_score(hom, distance, top))
    return next((hom for hom in refined if _plausible(hom, distance, top, size)), None)


def _distinct(hypotheses: list[tuple[float, np.ndarray]], count: int) -> list[np.ndarray]:
    """The best ``count`` hypotheses that differ: one whose landmarks all sit
    within SAME_FIT px of a better one's, the court's symmetries allowed, is
    that one again — typically its lines taken from the other edge of the
    same painted stripes."""
    court = np.array(list(LANDMARKS.values()), np.float64).reshape(-1, 1, 2)
    kept: list[np.ndarray] = []
    seen: list[np.ndarray] = []  # each kept hypothesis's landmarks, every symmetry
    for score, hom in hypotheses:
        if not np.isfinite(score) or len(kept) == count:
            break
        at = cv2.perspectiveTransform(court, hom).reshape(-1, 2)
        if all(np.abs(at - other).max() > SAME_FIT for other in seen):
            kept.append(hom)
            seen += [cv2.perspectiveTransform(court, hom @ f).reshape(-1, 2) for f in _SYMMETRIES]
    return kept


def _plausible(hom: np.ndarray, distance: np.ndarray, top: float, size: tuple[int, int]) -> bool:
    found, missed = _coverage(hom, distance, top)
    if found < MIN_PAINT_FOUND * (found + missed):
        return False
    # The court is symmetric, so the lines fit it mirrored as well — as a
    # camera under the floor would see it. Either way round may be the one
    # a camera above it saw.
    return any(_camera_fits(h, size) for h in (hom, hom @ _MIRROR))


def _camera_fits(hom: np.ndarray, size: tuple[int, int]) -> bool:
    court = np.array(list(LANDMARKS.values()), np.float64)
    image = cv2.perspectiveTransform(court.reshape(-1, 1, 2), hom).reshape(-1, 2) / size
    try:
        cam = camera.fit(np.c_[court, np.zeros(len(court))], image, size)
    except (camera.CameraError, cv2.error):  # the court collapses to a sliver
        return False
    centre = (COURT_LENGTH / 2, COURT_WIDTH / 2, 0.0)
    return cam.rmse <= MAX_CAMERA_RMSE and np.linalg.norm(np.subtract(cam.center, centre)) <= MAX_CAMERA_DISTANCE


def _score(hom: np.ndarray, distance: np.ndarray, top: float) -> float:
    found, missed = _coverage(hom, distance, top)
    return found - 0.5 * missed


def _coverage(hom: np.ndarray, distance: np.ndarray, top: float) -> tuple[float, float]:
    """Image length of the court's paint in view (the floor, inside the
    frame) that lies on a segment, and that lies on none. Each SCORE_CELL-px
    cell counts once, so a court folded onto one strong line counts as that
    line; a court straddling the horizon, folded flat or barely in view
    counts as nothing found."""
    p = np.c_[_PAINT, np.ones(len(_PAINT))] @ hom.T
    if (p[:, 2] <= 0).any():
        return 0.0, np.inf
    xy = p[:, :2] / p[:, 2:]
    h, w = distance.shape
    corners = cv2.perspectiveTransform(_CORNERS.reshape(-1, 1, 2), hom).reshape(-1, 2)
    if abs(cv2.contourArea(corners.astype(np.float32))) < MIN_COURT_AREA * w * h:
        return 0.0, np.inf
    seen = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= top) & (xy[:, 1] < h)
    if seen.sum() < 50:
        return 0.0, np.inf
    step = np.minimum(np.r_[np.hypot(*np.diff(xy, axis=0).T), 0.0], 20.0)[seen]
    q = xy[seen].astype(int)
    cell = (q[:, 1] // SCORE_CELL) * w + q[:, 0] // SCORE_CELL
    _, first = np.unique(cell, return_index=True)
    hit = distance[q[first, 1], q[first, 0]] < HIT
    return float(step[first][hit].sum()), float(step[first][~hit].sum())


def _matches(hom: np.ndarray, segs: np.ndarray, radius: float, angle: float):
    """Per court line: its image line and which segments lie on it."""
    p0 = np.c_[segs[:, :2], np.ones(len(segs))]
    p1 = np.c_[segs[:, 2:], np.ones(len(segs))]
    d = segs[:, 2:] - segs[:, :2]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    inverse = np.linalg.inv(hom)
    for court_line in COURT_LINES:
        line = inverse.T @ court_line
        line /= np.hypot(line[0], line[1])
        on = (np.abs(p0 @ line) < radius) & (np.abs(p1 @ line) < radius) & (np.abs(d @ line[:2]) < np.sin(angle))
        yield court_line, line, on


def _refine(hom: np.ndarray, segs: np.ndarray, size: tuple[int, int]) -> np.ndarray | None:
    """Re-solve the homography from every segment on a projected court line,
    minimising pixel distance from segment ends to their lines.

    With G = H⁻¹ the image line of court line L is GᵀL, so an end p on it
    gives Lᵀ G p = 0 — linear in G. Dividing each line's rows by |(GᵀL)₁,₂|
    from the previous round turns that into a distance (iteratively
    reweighted least squares). Points are scaled to about ±1 first."""
    w, h = size
    scale = np.array([[2 / w, 0, -1], [0, 2 / w, -h / w], [0, 0, 1.0]])
    unscale = np.linalg.inv(scale)
    p0 = np.c_[segs[:, :2], np.ones(len(segs))] @ scale.T
    p1 = np.c_[segs[:, 2:], np.ones(len(segs))] @ scale.T
    weight = np.sqrt(np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]) / 50)
    for radius in REFINE_RADII:
        g = np.linalg.inv(hom) @ unscale  # court ← scaled image
        rows, hit = [], {0.0: 0, 1.0: 0}
        for court_line, _, on in _matches(hom, segs, radius, REFINE_ANGLE):
            if not on.any():
                continue
            hit[court_line[0]] += 1
            line = g.T @ court_line
            w_line = weight[on] / np.hypot(line[0], line[1])
            for p in (p0[on], p1[on]):
                rows.append(np.einsum("i,nj->nij", court_line, p).reshape(-1, 9) * w_line[:, None])
        if min(hit.values()) < 2:  # one family alone leaves the fit free
            return None
        g = np.linalg.svd(np.concatenate(rows))[2][-1].reshape(3, 3)
        try:
            hom = np.linalg.inv(g @ scale)
        except np.linalg.LinAlgError:
            return None
        hom /= hom[2, 2]
    return hom


def _paint_in_view(hom: np.ndarray, segs: np.ndarray, size: tuple[int, int]) -> list[tuple[np.ndarray, np.ndarray]]:
    """One guide per court line with paint found on it: the line as the fit
    lays it, across the frame — a short visible stretch still gives a guide
    long enough to see and to cross the others."""
    w, h = size
    out = []
    for (x0, y0, x1, y1), (_, _, on) in zip(LINES, _matches(hom, segs, REFINE_RADII[-1], REFINE_ANGLE), strict=True):
        if not on.any():
            continue
        a, b = cv2.perspectiveTransform(np.array([[[x0, y0]], [[x1, y1]]], np.float64), hom).reshape(2, 2)
        inside, a, b = cv2.clipLine((0, 0, w, h), tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)))
        if inside:
            out.append((np.array(a, float), np.array(b, float)))
    return out
