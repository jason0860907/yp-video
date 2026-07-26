"""RF-DETR Seg person detection for annotated action frames.

The segmentation model returns both masks and boxes. Sparse player detection
stores the person boxes; dense tracking owns the masks it needs over time.
Actor association then decides which detected person performed the action.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Keep weak detections for the human actor picker. Association policies own
# any stricter confidence threshold.
PERSON_SCORE_THRESHOLD = 0.1
PERSON_CLASS_ID = 1
DETECTOR_NAME = "rf-detr-seg-medium"


@dataclass(frozen=True)
class PersonBox:
    xyxy: tuple[float, float, float, float]
    score: float


def person_from_detection(detection: dict) -> PersonBox:
    """Rebuild a PersonBox from the persisted extraction representation."""
    x0, y0, x1, y1 = detection["box"]
    return PersonBox(
        xyxy=(float(x0), float(y0), float(x1), float(y1)),
        score=float(detection.get("score") or 0.0),
    )


def iou(a: list[float], b: list[float]) -> float:
    """Plain box IoU, xyxy lists."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter or 1.0)


class PersonDetector:
    """Lazy RF-DETR Seg wrapper returning every detected person box."""

    def __init__(self, score_threshold: float = PERSON_SCORE_THRESHOLD):
        self.score_threshold = score_threshold
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        import torch
        from rfdetr import RFDETRSegMedium

        model = RFDETRSegMedium()
        model.optimize_for_inference(dtype=torch.float16, batch_size=1)
        self._model = model

    def detect(
        self,
        frame_bgr: np.ndarray,
        focus: tuple[float, float] | None = None,
    ) -> list[PersonBox]:
        # ``focus`` remains accepted so callers can pass the event point; the
        # whole-frame segmenter does not need it.
        del focus
        import cv2
        from PIL import Image

        self._ensure_model()
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        detections = self._model.predict(
            image,
            threshold=self.score_threshold,
            include_source_image=False,
        )
        if detections.xyxy is None or not len(detections.xyxy):
            return []
        class_ids = np.asarray(detections.class_id)
        scores = np.asarray(detections.confidence)
        people = []
        for index in np.flatnonzero(class_ids == PERSON_CLASS_ID):
            x0, y0, x1, y1 = detections.xyxy[index]
            people.append(
                PersonBox(
                    (float(x0), float(y0), float(x1), float(y1)),
                    float(scores[index]),
                )
            )
        return people


_detector: PersonDetector | None = None


def person_detector() -> PersonDetector:
    """Return the process-wide detector so loaded weights stay resident."""
    global _detector
    if _detector is None:
        _detector = PersonDetector()
    return _detector
