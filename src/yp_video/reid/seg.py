"""RF-DETR Seg person masks — background suppression for embedder crops.

Pile-up crops (blocks, digs) routinely contain two or three players, and the
appearance embedders latch onto whoever fills the most pixels. Masking runs
on the SAVED crop, not the video frame, so it lives in the embed stage: old
videos backfill without re-extraction, and extraction pays nothing.

The masked result is a normal BGR image (non-target pixels flattened to
neutral grey), so every embedder consumes it unchanged — see
embedder.MaskedEmbedder for how a variant opts in.
"""

from __future__ import annotations

import numpy as np

from yp_video.reid.detector import iou

# COCO class id for person in the RF-DETR Seg heads.
PERSON_CLASS_ID = 1
SEG_SCORE_THRESHOLD = 0.4
# The seg detection must overlap the actor's box this much to count as the
# same person; below it the crop passes through unmasked (a silent no-op
# beats grey-ing out the actual player on a seg miss).
MIN_TARGET_IOU = 0.3
# Neutral grey; keeps masked crops in the distribution the embedders saw in
# training better than hard black.
MASK_FILL = 128

# Medium over Nano: +7 ms/crop on a 4090 for visibly cleaner boundaries —
# benchmarked on real match frames (box↔mask IoU 0.62–0.86).
SEG_WEIGHTS = "rf-detr-seg-medium"


class CropMasker:
    """Lazy RF-DETR Seg Medium; one instance per process keeps the weights
    resident (mirrors detector.PersonDetector)."""

    def __init__(self):
        self._model = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _ensure(self):
        if self._model is not None:
            return
        import torch
        from rfdetr import RFDETRSegMedium

        model = RFDETRSegMedium()
        # Crops go through predict() one at a time; the fp16-compiled graph
        # (batch 1) halves per-crop latency and keeps masks in the output.
        model.optimize_for_inference(dtype=torch.float16, batch_size=1)
        self._model = model

    def mask_crop(self, crop_bgr: np.ndarray, target_xyxy: list[float]) -> np.ndarray:
        """The crop with everything outside the target person's mask greyed.

        ``target_xyxy`` is the actor's box in CROP pixels. Returns the crop
        unchanged when no person mask overlaps it by ``MIN_TARGET_IOU``.
        """
        import cv2
        from PIL import Image

        self._ensure()
        det = self._model.predict(
            Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)), threshold=SEG_SCORE_THRESHOLD
        )
        if det.mask is None or not len(det):
            return crop_bgr
        best, best_iou = None, MIN_TARGET_IOU
        for i in range(len(det)):
            if int(det.class_id[i]) != PERSON_CLASS_ID:
                continue
            overlap = iou([float(v) for v in det.xyxy[i]], target_xyxy)
            if overlap > best_iou:
                best, best_iou = i, overlap
        if best is None:
            return crop_bgr
        m = det.mask[best].astype(np.uint8)[..., None]
        return (crop_bgr * m + MASK_FILL * (1 - m)).astype(np.uint8)


# One instance per process — the weights stay resident across jobs.
_masker: CropMasker | None = None


def crop_masker() -> CropMasker:
    global _masker
    if _masker is None:
        _masker = CropMasker()
    return _masker
