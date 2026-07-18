"""Appearance embedder for person crops.

CLIP-ReID ViT-B/16 (Market-1501 fine-tune) through its ONNX export
(occurra/person_vit_clip_reid, MIT). It replaced the OSNet baseline after an
A/B on identical crops: cluster sizes came out far more balanced (OSNet glued
several players into one 86-event blob at every threshold that kept recall).

Runs on CPU via onnxruntime — a few hundred crops per video take seconds, and
it sidesteps onnxruntime-gpu / CUDA wheel matching entirely.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np

CLIP_REID_HF_REPO = "occurra/person_vit_clip_reid"
CLIP_REID_ONNX = "person_vit_clip_reid.onnx"
# ReID-standard input aspect (h, w) the checkpoint was trained at.
INPUT_H, INPUT_W = 256, 128
EMBEDDING_DIM = 512


class Embedder(Protocol):
    """One entry in the embedder registry (see build_embedders).

    Embedding dimension may differ per model — downstream cosine math never
    assumes one.
    """

    def embed(self, crops_bgr: list[np.ndarray], batch_size: int = 32) -> np.ndarray: ...

    @property
    def loaded(self) -> bool:
        """Whether the weights are resident — first embed() otherwise stalls
        on model load, and progress reporting wants to say so."""
        ...


class ClipReidEmbedder:
    """Batch person-crop → L2-normalized 512-d embedding. Lazy session load.

    The ONNX output is NOT L2-normalized despite the model card's claim
    (measured ‖v‖ ≈ 8) — we normalize here so cosine math holds.
    """

    def __init__(self):
        self._session = None

    @property
    def loaded(self) -> bool:
        return self._session is not None

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


class MaskedEmbedder:
    """A registered embedder re-run on background-suppressed crops.

    Same weights as ``base`` (delegation keeps ONE loaded copy for both
    variants); the difference is the input — embed_video sees
    ``masked_input`` and feeds crops with non-actor pixels greyed out
    (reid/seg.py), so appearance features can't latch onto teammates in
    pile-up crops. A/B against the base variant on identical extractions.
    """

    masked_input = True

    def __init__(self, base: Embedder):
        self._base = base

    @property
    def loaded(self) -> bool:
        return self._base.loaded

    def embed(self, crops_bgr: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
        return self._base.embed(crops_bgr, batch_size)


# name → weights identifier, recorded in each extraction's header.
EMBEDDER_WEIGHTS = {
    "clip-reid": CLIP_REID_ONNX,
    "clip-reid-masked": f"{CLIP_REID_ONNX} + rf-detr-seg-medium",
    "clip-reident": "ViT-L-14_openai/all_data_seed_1/weights_e4.pth",
    "clip-reident-masked": "ViT-L-14_openai/all_data_seed_1/weights_e4.pth + rf-detr-seg-medium",
}
# The optional models may be unregistered (weights not downloaded);
# /reid/options falls back to the first registered embedder then. The masked
# variants are always registered — their base (clip-reid) ships with the repo.
DEFAULT_EMBEDDER = "clip-reid-masked"

# Cluster-threshold slider calibration per embedder, served to the UI via
# /reid/options. Cosine-distance scales differ wildly per model: CLIP-ReID's
# ViT features sit in a tight cone, CLIP-ReIdent's fine-tuned ViT-L is extremely tight (labeled-pair optimum
# ~0.02). Calibrated on real match footage (~12 people on court); models
# without an entry get the wide neutral fallback.
EMBEDDER_THRESHOLDS = {
    "clip-reid": {"min": 0.08, "max": 0.3, "default": 0.15, "step": 0.01},
    "clip-reident": {"min": 0.008, "max": 0.06, "default": 0.022, "step": 0.002},
    # Masked variants start from their base model's scale; not yet tuned on
    # masked crops (they measure larger distances — expect the optimum higher).
    "clip-reid-masked": {"min": 0.08, "max": 0.3, "default": 0.15, "step": 0.01},
    "clip-reident-masked": {"min": 0.008, "max": 0.06, "default": 0.022, "step": 0.002},
}
FALLBACK_THRESHOLD = {"min": 0.05, "max": 0.95, "default": 0.3, "step": 0.01}


def threshold_calibration(name: str) -> dict:
    return EMBEDDER_THRESHOLDS.get(name, FALLBACK_THRESHOLD)


# Constructed embedders, by name. Construction is cheap (models load lazily on
# first embed) but instances must persist across calls so loaded weights stay
# resident for the whole process.
_instances: dict[str, Embedder] = {}


def build_embedders() -> dict[str, Embedder]:
    """Every available embedder; optional ones join when their checkout + weights exist.

    Availability is re-checked on every call (a stat per optional model), so
    weights downloaded while the server runs appear without a restart.
    """
    from yp_video.reid.clip_reident import ClipReidentEmbedder, clip_reident_available

    available: dict[str, Callable[[], Embedder]] = {"clip-reid": ClipReidEmbedder}
    if clip_reident_available():
        available["clip-reident"] = ClipReidentEmbedder
    # Masked variants wrap the base instances — dict order guarantees a base
    # is constructed by the time its factory runs.
    available["clip-reid-masked"] = lambda: MaskedEmbedder(_instances["clip-reid"])
    if "clip-reident" in available:
        available["clip-reident-masked"] = lambda: MaskedEmbedder(_instances["clip-reident"])
    for name, make in available.items():
        _instances.setdefault(name, make())
    return {name: _instances[name] for name in available}
