"""Appearance embedder for person crops.

CLIP-ReID ViT-B/16 (Market-1501 fine-tune) through its ONNX export
(occurra/person_vit_clip_reid, MIT). It replaced the OSNet baseline after an
A/B on identical crops: cluster sizes came out far more balanced (OSNet glued
several players into one 86-event blob at every threshold that kept recall).

Runs on CPU via onnxruntime — a few hundred crops per video take seconds, and
it sidesteps onnxruntime-gpu / CUDA wheel matching entirely.
"""

from __future__ import annotations

import numpy as np

CLIP_REID_HF_REPO = "occurra/person_vit_clip_reid"
CLIP_REID_ONNX = "person_vit_clip_reid.onnx"
# ReID-standard input aspect (h, w) the checkpoint was trained at.
INPUT_H, INPUT_W = 256, 128
EMBEDDING_DIM = 512


class ClipReidEmbedder:
    """Batch person-crop → L2-normalized 512-d embedding. Lazy session load.

    The ONNX output is NOT L2-normalized despite the model card's claim
    (measured ‖v‖ ≈ 8) — we normalize here so cosine math holds.
    """

    def __init__(self):
        self._session = None

    def _ensure_session(self):
        if self._session is not None:
            return
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(CLIP_REID_HF_REPO, CLIP_REID_ONNX)
        self._session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def embed(self, crops_bgr: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
        """Embed BGR person crops → (N, 512) float32, L2-normalized."""
        import cv2

        self._ensure_session()
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        feats: list[np.ndarray] = []
        for start in range(0, len(crops_bgr), batch_size):
            batch = []
            for crop in crops_bgr[start : start + batch_size]:
                img = cv2.cvtColor(cv2.resize(crop, (INPUT_W, INPUT_H)), cv2.COLOR_BGR2RGB)
                t = img.transpose(2, 0, 1).astype(np.float32) / 255.0
                batch.append((t - mean) / std)
            feats.append(self._session.run(None, {"input": np.stack(batch)})[0])
        if not feats:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        matrix = np.concatenate(feats).astype(np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
        return matrix


# name → weights identifier, recorded in each extraction's header.
EMBEDDER_WEIGHTS = {"clip-reid": CLIP_REID_ONNX}
DEFAULT_EMBEDDER = "clip-reid"


def build_embedders() -> dict[str, ClipReidEmbedder]:
    return {"clip-reid": ClipReidEmbedder()}
