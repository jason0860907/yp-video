"""RF-DETR keypoint detection: every person on a frame, with their skeleton.

The keypoint model (GroupPose-style DETR head) predicts boxes and 17 COCO
keypoints for every person in one pass, at constant cost regardless of player
count (~29 ms/frame on a 4090). The keypoints exist because a volleyball
contact happens at a hand — but what to DO with that fact is actor
association's business (see yp_video/actor/ranking.py), not the detector's.
This module knows only what is on the frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

# Detection floor. Deliberately low: every box above it is stored on the
# record and offered in the UI's actor picker (which has its own score
# slider), so even a barely-detected player can still be clicked. Consumers
# that want a stricter floor own that threshold themselves.
PERSON_SCORE_THRESHOLD = 0.1

# Detection (boxes + scores) is ALWAYS RF-DETR; what's selectable is who
# estimates the 17 COCO keypoints on those boxes. name → weights identifier,
# recorded in extraction headers.
DETECTOR_NAME = "rf-detr-keypoint-preview"
KEYPOINT_SOURCES = {
    "rf-detr": "rf-detr-keypoint-preview head",
    "sam-3d-body": "sam-3d-body-dinov3 (MHR projection)",
}
DEFAULT_KEYPOINT_SOURCE = "rf-detr"

# COCO keypoint indices for left/right wrist — a property of the skeleton
# format, so it belongs to whoever produces skeletons.
WRIST_IDXS = (9, 10)


@dataclass(frozen=True)
class PersonBox:
    xyxy: tuple[float, float, float, float]
    score: float
    keypoints: np.ndarray | None = None  # (17, 2) pixel coords
    keypoint_conf: np.ndarray | None = None  # (17,)


def person_from_detection(detection: dict) -> PersonBox:
    """Rebuild a PersonBox from the form extraction stores it in.

    Lives with the type rather than with any one reader: the dataset builder,
    the association policies and the fix path all need the same three lines,
    and had each grown their own copy.
    """
    keypoints = detection.get("keypoints")
    return PersonBox(
        xyxy=(
            float(detection["box"][0]),
            float(detection["box"][1]),
            float(detection["box"][2]),
            float(detection["box"][3]),
        ),
        score=float(detection.get("score") or 0.0),
        keypoints=(
            np.asarray([[p[0], p[1]] for p in keypoints], dtype=np.float32)
            if keypoints
            else None
        ),
        keypoint_conf=(
            np.asarray([p[2] for p in keypoints], dtype=np.float32)
            if keypoints
            else None
        ),
    )


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
        # Keypoint-hull boxes computed here instead of kp.as_detections():
        # that helper drags kp.data through fancy indexing, and its
        # source_image entry carries a full frame PER detection — ~300 ms of
        # pure memcpy per call. Same boxes, same [0,0]=missing convention.
        xy = np.asarray(kp.xy)  # (N, 17, 2)
        valid = ~np.all(xy == 0, axis=2)
        x_min = np.where(valid, xy[..., 0], np.inf).min(axis=1)
        y_min = np.where(valid, xy[..., 1], np.inf).min(axis=1)
        x_max = np.where(valid, xy[..., 0], -np.inf).max(axis=1)
        y_max = np.where(valid, xy[..., 1], -np.inf).max(axis=1)
        keep = valid.any(axis=1) & (x_max > x_min) & (y_max > y_min)
        confs = (
            kp.detection_confidence
            if kp.detection_confidence is not None
            else np.ones(len(xy))
        )
        kp_conf = np.asarray(kp.keypoint_confidence)
        return [
            PersonBox(
                (float(x_min[i]), float(y_min[i]), float(x_max[i]), float(y_max[i])),
                float(confs[i]),
                keypoints=xy[i].astype(np.float32),
                keypoint_conf=kp_conf[i].astype(np.float32),
            )
            for i in np.flatnonzero(keep)
        ]


# Constructed sources — instances persist so loaded models stay resident;
# availability is re-checked on every build_keypoint_sources call.
_rf: PersonDetector | None = None
_sam3d: KeypointSource | None = None


def build_keypoint_sources() -> dict[str, KeypointSource]:
    """Every available keypoint source; SAM 3D Body joins when its weights
    exist (re-checked per call, so a download while the server runs appears
    without a restart). Both entries share ONE RF-DETR instance —
    "sam-3d-body" wraps it for boxes and replaces only the keypoints
    (see person/sam3d.py).
    """
    global _rf, _sam3d

    from yp_video.person.sam3d import Sam3dBodyDetector, sam3d_available

    if _rf is None:
        _rf = PersonDetector()
    out: dict[str, KeypointSource] = {"rf-detr": _rf}
    if sam3d_available():
        if _sam3d is None:
            _sam3d = Sam3dBodyDetector(_rf)
        out["sam-3d-body"] = _sam3d
    return out
