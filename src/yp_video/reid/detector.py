"""RF-DETR keypoint detection + contact-point → player association.

The keypoint model (GroupPose-style DETR head) predicts boxes and 17 COCO
keypoints for every person in one pass, at constant cost regardless of player
count (~29 ms/frame on a 4090). The keypoints buy a far more physical
association than box geometry: a volleyball contact happens at a hand, so the
annotated xy should sit next to somebody's wrist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

# Detection floor. Deliberately low: every box above it is stored on the
# record and offered in the UI's actor picker (which has its own score
# slider), so even a barely-detected player can still be clicked. The
# automatic pick applies its own stricter floor below.
PERSON_SCORE_THRESHOLD = 0.1
# Automatic contact-point association only trusts confident detections — the
# 0.1–0.5 band exists solely to give the human picker more boxes.
AUTO_PICK_MIN_SCORE = 0.5

# Detection (boxes + scores) is ALWAYS RF-DETR; what's selectable is who
# estimates the 17 COCO keypoints on those boxes. name → weights identifier,
# recorded in extraction headers.
DETECTOR_NAME = "rf-detr-keypoint-preview"
KEYPOINT_SOURCES = {
    "rf-detr": "rf-detr-keypoint-preview head",
    "sam-3d-body": "sam-3d-body-dinov3 (MHR projection)",
}
DEFAULT_KEYPOINT_SOURCE = "rf-detr"

# COCO keypoint indices for left/right wrist.
WRIST_IDXS = (9, 10)
MIN_KEYPOINT_CONF = 0.3
# A wrist match counts when the contact point is within this fraction of the
# person's box height from the wrist — roughly ball-diameter reach at contact.
WRIST_REACH_FRAC = 0.6

# Box-geometry fallback for people whose wrists weren't found: the contact
# point may sit up to 35% of box height above the top (ball above the raised
# hand) and 20% of box width outside the horizontal span. Validated on
# annotated sideline footage.
X_PAD_FRAC = 0.20
Y_ABOVE_FRAC = 0.35
# Fallback candidates always rank below any wrist match.
FALLBACK_PENALTY = 10.0


@dataclass(frozen=True)
class PersonBox:
    xyxy: tuple[float, float, float, float]
    score: float
    keypoints: np.ndarray | None = None  # (17, 2) pixel coords
    keypoint_conf: np.ndarray | None = None  # (17,)


class KeypointSource(Protocol):
    """One entry in the keypoint-source registry (see build_keypoint_sources).

    Boxes and scores always come from RF-DETR; implementations differ only in
    who estimates the 17 COCO keypoints. ``focus`` (the event's contact point)
    is a hint — whole-frame detectors ignore it, top-down ones use it to skip
    implausible actors.
    """

    def detect(self, frame_bgr: np.ndarray, focus: tuple[float, float] | None = None) -> list[PersonBox]: ...


def iou(a: list[float], b: list[float]) -> float:
    """Plain box IoU, xyxy lists."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter or 1.0)


class PersonDetector:
    """RF-DETR keypoint wrapper returning person boxes with their skeletons.

    Loads lazily on first detect() — the model download / CUDA init must not
    happen at import time inside the web server.
    """

    def __init__(self, score_threshold: float = PERSON_SCORE_THRESHOLD):
        self.score_threshold = score_threshold
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        from rfdetr import RFDETRKeypointPreview

        self._model = RFDETRKeypointPreview()

    def detect(self, frame_bgr: np.ndarray, focus: tuple[float, float] | None = None) -> list[PersonBox]:
        # ``focus`` (event contact point) is part of the shared detector
        # interface; RF-DETR is single-pass whole-frame, nothing to narrow.
        del focus
        import cv2
        from PIL import Image

        self._ensure_model()
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        kp = self._model.predict(img, threshold=self.score_threshold)
        if kp.xy is None or not len(kp.xy):
            return []
        det = kp.as_detections()
        confs = det.confidence if det.confidence is not None else np.ones(len(det.xyxy))
        kp_conf = np.asarray(kp.keypoint_confidence)
        return [
            PersonBox(
                tuple(float(v) for v in xyxy),
                float(score),
                keypoints=np.asarray(points, dtype=np.float32),
                keypoint_conf=kp_conf[i].astype(np.float32),
            )
            for i, (xyxy, score, points) in enumerate(zip(det.xyxy, confs, kp.xy))
        ]


def associate(boxes: list[PersonBox], x: float, y: float) -> list[PersonBox]:
    """Rank persons by how plausibly they own the contact point (x, y).

    Wrist distance first — the contact IS at a hand — with the old box-top
    geometry as a fallback for players whose wrists weren't confidently
    detected. Scores are normalized by box height so near and far players
    compare fairly. Returns candidates best-first; empty when nobody is
    geometrically compatible. Pixel coordinates.

    Low-confidence detections (< AUTO_PICK_MIN_SCORE) never compete here —
    they exist only as manual-picker choices.
    """
    scored: list[tuple[float, PersonBox]] = []
    for box in boxes:
        if box.score < AUTO_PICK_MIN_SCORE:
            continue
        x0, y0, x1, y1 = box.xyxy
        w, h = max(x1 - x0, 1.0), max(y1 - y0, 1.0)

        wrist_d = None
        if box.keypoints is not None and box.keypoint_conf is not None:
            dists = [
                float(np.hypot(box.keypoints[i][0] - x, box.keypoints[i][1] - y))
                for i in WRIST_IDXS
                if box.keypoint_conf[i] >= MIN_KEYPOINT_CONF
            ]
            if dists:
                wrist_d = min(dists)

        if wrist_d is not None and wrist_d <= WRIST_REACH_FRAC * h:
            scored.append((wrist_d / h, box))
            continue

        in_x = x0 - X_PAD_FRAC * w <= x <= x1 + X_PAD_FRAC * w
        in_y = y0 - Y_ABOVE_FRAC * h <= y <= y1
        if in_x and in_y:
            d = float(np.hypot(x - (x0 + x1) / 2, y - y0))
            scored.append((d / h + FALLBACK_PENALTY, box))

    return [box for _, box in sorted(scored, key=lambda t: t[0])]


# Constructed sources — instances persist so loaded models stay resident;
# availability is re-checked on every build_keypoint_sources call.
_rf: PersonDetector | None = None
_sam3d: KeypointSource | None = None


def build_keypoint_sources() -> dict[str, KeypointSource]:
    """Every available keypoint source; SAM 3D Body joins when its weights
    exist (re-checked per call, so a download while the server runs appears
    without a restart). Both entries share ONE RF-DETR instance —
    "sam-3d-body" wraps it for boxes and replaces only the keypoints
    (see reid/sam3d.py).
    """
    global _rf, _sam3d

    from yp_video.reid.sam3d import Sam3dBodyDetector, sam3d_available

    if _rf is None:
        _rf = PersonDetector()
    out: dict[str, KeypointSource] = {"rf-detr": _rf}
    if sam3d_available():
        if _sam3d is None:
            _sam3d = Sam3dBodyDetector(_rf)
        out["sam-3d-body"] = _sam3d
    return out
