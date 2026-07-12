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


# name → weights identifier, recorded in each extraction's header.
EMBEDDER_WEIGHTS = {"clip-reid": CLIP_REID_ONNX, "kpr": KPR_WEIGHTS}
DEFAULT_EMBEDDER = "clip-reid"


def build_embedders() -> dict:
    """Every available embedder; KPR joins when its checkout + weights exist."""
    from yp_video.config import KPR_DIR

    out: dict = {"clip-reid": ClipReidEmbedder()}
    if (KPR_DIR / "pretrained_models" / KPR_WEIGHTS).exists():
        out["kpr"] = KprEmbedder()
    return out
