"""Appearance embedder for person crops.

CLIP-ReID ViT-B/16 (Market-1501 fine-tune) through its ONNX export
(occurra/person_vit_clip_reid, MIT). It replaced the OSNet baseline after an
A/B on identical crops: cluster sizes came out far more balanced (OSNet glued
several players into one 86-event blob at every threshold that kept recall).

Runs on CPU via onnxruntime — a few hundred crops per video take seconds, and
it sidesteps onnxruntime-gpu / CUDA wheel matching entirely.
"""

from __future__ import annotations

import tempfile
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
    assumes one. ``prompts[i]`` optionally carries keypoint prompts (see
    ``build_crop_prompt``); non-promptable embedders ignore them.
    """

    def embed(self, crops_bgr: list[np.ndarray], prompts: list[dict] | None = None, batch_size: int = 32) -> np.ndarray: ...

    @property
    def loaded(self) -> bool:
        """Whether the weights are resident — first embed() otherwise stalls
        on model load, and progress reporting wants to say so."""
        ...


def build_crop_prompt(record: dict, person_box: list[float]) -> dict:
    """Keypoint prompts for promptable embedders (KPR) from one extraction
    record: the chosen person's keypoints form the positive prompt and every
    other detected person's keypoints that fall inside the saved crop become
    negatives — all in crop-pixel coordinates of the display crop.

    ``person_box`` is the chosen person's raw detector box (frame pixels),
    used only to exclude them from the negatives.
    """
    from yp_video.reid.detector import iou

    dx0, dy0, dx1, dy1 = record["box"]
    cw, ch = max(dx1 - dx0, 1), max(dy1 - dy0, 1)

    # KPR's transform stack validates that every keypoint lies inside the
    # crop, so clamp — points pushed out by frame-edge clipping land on the
    # border, which is where the person is cut off anyway.
    def clamped(kx: float, ky: float, c: float) -> list[float]:
        return [min(max(kx, 0.0), cw - 1.0), min(max(ky, 0.0), ch - 1.0), c]

    pos = None
    if record.get("keypoints"):
        # Stored crop-relative (0-1) → crop pixels.
        pos = np.array([clamped(k[0] * cw, k[1] * ch, k[2]) for k in record["keypoints"]], dtype=np.float32)
    negs = []
    for d in record.get("detections") or []:
        kps = d.get("keypoints")
        if not kps or iou(d["box"], person_box) > 0.8:
            continue  # no keypoints, or this IS the chosen person
        pts, any_inside = [], False
        for px, py, c in kps:
            kx, ky = px - dx0, py - dy0
            inside = 0 <= kx <= cw and 0 <= ky <= ch and c > 0
            any_inside = any_inside or inside
            pts.append(clamped(kx, ky, c if inside else 0.0))
        if any_inside:
            negs.append(pts)
    return {
        "keypoints_xyc": pos,
        "negative_kps": np.array(negs, dtype=np.float32) if negs else None,
    }


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

    def embed(self, crops_bgr: list[np.ndarray], prompts: list[dict] | None = None, batch_size: int = 32) -> np.ndarray:
        """Embed BGR person crops → (N, 512) float32, L2-normalized.

        ``prompts`` (keypoint prompts) is part of the shared embedder
        interface; CLIP-ReID is not promptable and ignores it.
        """
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


# SOLIDER (human-centric foundation) backbone: slightly below the ImageNet
# variant on Occluded-PoseTrack itself, but clearly stronger on the OTHER
# occlusion benchmarks (Occluded-ReID 82.6 vs 79.1, Partial-ReID 90.7 vs
# 86.0) — better cross-domain behaviour is what volleyball footage needs.
KPR_CONFIG = "configs/kpr/solider/kpr_occ_posetrack_test.yaml"
KPR_WEIGHTS = "kpr_occ_pt_SOLIDER_81.24_90.59_42326409.pth.tar"


class KprEmbedder:
    """KPR — Keypoint Promptable ReID (ECCV'24, Swin backbone), GPU.

    The crop's own keypoints act as a positive prompt and other people's
    keypoints as negatives, so the embedding locks onto the intended player
    in multi-person crops — exactly our spike/block pile-up failure mode.

    KPR is part-based; we keep only the batch-normed foreground embedding
    (test_embeddings[0], prompt-guided whole-person vector) so it drops into
    the same flat-cosine centroid math as the other embedders. Part-level
    matching with visibility scores is a possible future upgrade.
    """

    def __init__(self):
        self._extractor = None

    @property
    def loaded(self) -> bool:
        return self._extractor is not None

    def _ensure(self):
        if self._extractor is not None:
            return
        import sys

        import torch

        from yp_video.config import KPR_DIR

        if str(KPR_DIR) not in sys.path:
            sys.path.insert(0, str(KPR_DIR))
        from torchreid.scripts.builder import build_config
        from torchreid.tools.feature_extractor import KPRFeatureExtractor
        from yacs.config import CfgNode

        override = CfgNode({
            "model": CfgNode({"load_weights": str(KPR_DIR / "pretrained_models" / KPR_WEIGHTS)}),
            # build_config mints a run dir under save_dir on every init —
            # keep that noise out of the repo.
            "data": CfgNode({"save_dir": tempfile.mkdtemp(prefix="kpr-")}),
        })
        cfg = build_config(config_path=str(KPR_DIR / KPR_CONFIG), config=override)
        cfg.use_gpu = torch.cuda.is_available()
        # The extractor does NOT read input size / normalization from cfg —
        # they are constructor args. SOLIDER runs 384x128 with 0.5-norm,
        # ImageNet-Swin 256x128 with ImageNet stats; feed whatever the
        # loaded config says so the preprocessing always matches training.
        self._extractor = KPRFeatureExtractor(
            cfg,
            image_size=(cfg.data.height, cfg.data.width),
            pixel_mean=list(cfg.data.norm_mean),
            pixel_std=list(cfg.data.norm_std),
            verbose=False,
        )

    def embed(self, crops_bgr: list[np.ndarray], prompts: list[dict] | None = None, batch_size: int = 32) -> np.ndarray:
        """Embed BGR person crops → (N, 512) float32, L2-normalized.

        ``prompts[i]`` may carry ``keypoints_xyc`` (17, 3) and ``negative_kps``
        (M, 17, 3) in crop-pixel coordinates. A batch must be prompt-uniform
        (the extractor stacks prompt masks), so missing prompts are replaced
        with an all-zero-confidence dummy, which KPR treats as "no prompt".
        """
        if not len(crops_bgr):
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        self._ensure()
        import torch

        no_prompt = np.zeros((17, 3), dtype=np.float32)
        no_negs = np.empty((0, 17, 3), dtype=np.float32)
        feats: list[np.ndarray] = []
        for start in range(0, len(crops_bgr), batch_size):
            samples = []
            for i in range(start, min(start + batch_size, len(crops_bgr))):
                p = (prompts[i] if prompts else None) or {}
                pos = p.get("keypoints_xyc")
                negs = p.get("negative_kps")
                samples.append({
                    "image": crops_bgr[i],
                    "keypoints_xyc": np.asarray(pos, dtype=np.float32) if pos is not None else no_prompt,
                    "negative_kps": np.asarray(negs, dtype=np.float32) if negs is not None and len(negs) else no_negs,
                })
            with torch.inference_mode():
                _, emb, _vis, _masks = self._extractor(samples)
            feats.append(emb[:, 0].cpu().numpy())  # bn_foreg: prompt-guided whole-person vector
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

    def embed(self, crops_bgr: list[np.ndarray], prompts: list[dict] | None = None, batch_size: int = 32) -> np.ndarray:
        return self._base.embed(crops_bgr, prompts, batch_size)


# name → weights identifier, recorded in each extraction's header.
EMBEDDER_WEIGHTS = {
    "clip-reid": CLIP_REID_ONNX,
    "clip-reid-masked": f"{CLIP_REID_ONNX} + rf-detr-seg-medium",
    "kpr": KPR_WEIGHTS,
    "clip-reident": "ViT-L-14_openai/all_data_seed_1/weights_e4.pth",
    "clip-reident-masked": "ViT-L-14_openai/all_data_seed_1/weights_e4.pth + rf-detr-seg-medium",
}
# The optional models may be unregistered (weights not downloaded);
# /reid/options falls back to the first registered embedder then.
DEFAULT_EMBEDDER = "clip-reident"

# Cluster-threshold slider calibration per embedder, served to the UI via
# /reid/options. Cosine-distance scales differ wildly per model: CLIP-ReID's
# ViT features sit in a tight cone, KPR's foreground embeddings spread wide,
# CLIP-ReIdent's fine-tuned ViT-L is extremely tight (labeled-pair optimum
# ~0.02). Calibrated on real match footage (~12 people on court); models
# without an entry get the wide neutral fallback.
EMBEDDER_THRESHOLDS = {
    "clip-reid": {"min": 0.08, "max": 0.3, "default": 0.15, "step": 0.01},
    "kpr": {"min": 0.3, "max": 0.8, "default": 0.55, "step": 0.01},
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
    from yp_video.config import KPR_DIR
    from yp_video.reid.clip_reident import ClipReidentEmbedder, clip_reident_available

    available: dict[str, Callable[[], Embedder]] = {"clip-reid": ClipReidEmbedder}
    if (KPR_DIR / "pretrained_models" / KPR_WEIGHTS).exists():
        available["kpr"] = KprEmbedder
    if clip_reident_available():
        available["clip-reident"] = ClipReidentEmbedder
    # Masked variants wrap the base instances — dict order guarantees a base
    # is constructed by the time its factory runs. No kpr-masked: KPR's
    # keypoint prompts already lock onto the actor; masking on top is
    # redundant and confuses its negative prompts.
    available["clip-reid-masked"] = lambda: MaskedEmbedder(_instances["clip-reid"])
    if "clip-reident" in available:
        available["clip-reident-masked"] = lambda: MaskedEmbedder(_instances["clip-reident"])
    for name, make in available.items():
        _instances.setdefault(name, make())
    return {name: _instances[name] for name in available}
