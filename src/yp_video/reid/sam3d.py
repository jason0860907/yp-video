"""SAM 3D Body (Meta) as a keypoint upgrade over RF-DETR detections.

3DB is a promptable single-image human mesh recovery model, not a detector —
it wants person boxes as prompts. So this "detector" composes the two:
RF-DETR finds the people (boxes + scores, fast), 3DB re-estimates each
person's pose through the full MHR rig, and the 70 projected 2D keypoints
are mapped down to the COCO-17 schema the rest of the pipeline speaks
(association, display box, skeleton overlay).

Lives in its own checkout (see config.SAM3D_DIR), imported lazily via
sys.path; the weights are gated on Hugging Face, so everything degrades
gracefully until they are downloaded:

    hf download facebook/sam-3d-body-dinov3 \
        --local-dir third_party/sam-3d-body/checkpoints/sam-3d-body-dinov3
"""

from __future__ import annotations

import numpy as np

from yp_video.config import SAM3D_DIR
from yp_video.reid.detector import PersonBox, PersonDetector

CHECKPOINT = SAM3D_DIR / "checkpoints" / "sam-3d-body-dinov3" / "model.ckpt"
MHR_MODEL = SAM3D_DIR / "checkpoints" / "sam-3d-body-dinov3" / "assets" / "mhr_model.pt"


def sam3d_available() -> bool:
    return CHECKPOINT.exists() and MHR_MODEL.exists()


# Only people whose box (expanded by this fraction of its height) contains the
# contact point get 3DB keypoints — matches associate()'s reach: a wrist match
# needs the contact within WRIST_REACH_FRAC (0.6) of a wrist, and wrists live
# inside the box. Everyone else keeps RF-DETR keypoints; they only ever appear
# in the manual picker.
FOCUS_PAD_FRAC = 0.7


class Sam3dBodyDetector:
    """Same detect() contract as PersonDetector.

    Boxes and scores come from the wrapped RF-DETR (so the picker/associate
    thresholds keep their meaning); only the keypoints are replaced by 3DB's.
    3DB emits no per-keypoint confidence, so keypoints carry confidence 1.0.

    3DB is top-down (one forward per person), so cost scales with people —
    given ``focus`` (the event's contact point), only plausible actors are
    re-estimated: body decoder only (no hand refinement), and only confident
    boxes near the contact point.
    """

    def __init__(self, box_detector: PersonDetector):
        self._boxes = box_detector
        self._estimator = None
        self._coco_idx: list[int] | None = None

    def _ensure(self):
        if self._estimator is not None:
            return
        import sys

        import torch

        if str(SAM3D_DIR) not in sys.path:
            sys.path.insert(0, str(SAM3D_DIR))
        from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
        from sam_3d_body.metadata import MHR70_TO_OPENPOSE, OPENPOSE_TO_COCO

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, cfg = load_sam_3d_body(str(CHECKPOINT), device=device, mhr_path=str(MHR_MODEL))
        # No detector / segmentor / FOV estimator: boxes come from RF-DETR and
        # the default FOV is fine for fixed match footage.
        self._estimator = SAM3DBodyEstimator(model, cfg)
        # MHR-70 → OpenPose → COCO-17 index chain.
        self._coco_idx = [MHR70_TO_OPENPOSE[op] for op in OPENPOSE_TO_COCO]

    def detect(self, frame_bgr: np.ndarray, focus: tuple[float, float] | None = None) -> list[PersonBox]:
        import cv2

        from yp_video.reid.detector import AUTO_PICK_MIN_SCORE

        people = self._boxes.detect(frame_bgr)
        if not people:
            return []

        def near_focus(p: PersonBox) -> bool:
            if focus is None:
                return True
            x0, y0, x1, y1 = p.xyxy
            pad = FOCUS_PAD_FRAC * max(y1 - y0, 1.0)
            return x0 - pad <= focus[0] <= x1 + pad and y0 - pad <= focus[1] <= y1 + pad

        # 3DB only for plausible actors; the rest keep RF-DETR keypoints.
        targets = [i for i, p in enumerate(people) if p.score >= AUTO_PICK_MIN_SCORE and near_focus(p)]
        if not targets:
            return people
        self._ensure()
        boxes = np.array([people[i].xyxy for i in targets], dtype=np.float32)
        outs = self._estimator.process_one_image(
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), bboxes=boxes, inference_type="body"
        )
        result = list(people)
        for i, out in zip(targets, outs):
            kps = np.asarray(out["pred_keypoints_2d"], dtype=np.float32)[self._coco_idx, :2]
            result[i] = PersonBox(
                people[i].xyxy,
                people[i].score,
                keypoints=kps,
                keypoint_conf=np.ones(len(kps), dtype=np.float32),
            )
        return result
