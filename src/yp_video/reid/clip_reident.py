"""CLIP-ReIdent (MMSports'22) as a third appearance embedder.

Habel et al. reformulate CLIP's language-image contrastive training as
image-to-image InfoNCE and fine-tune on basketball broadcast crops (the
MMSports 2022 player-ReID challenge, which it won) — the closest training
domain to volleyball among our embedders: indoor arenas, jerseys, motion
blur, look-alike teammates. Compare against the Market-1501-trained
clip-reid to see how much that domain gap matters.

The model is OpenCLIP ViT-L/14 with the CLIP projection removed, so
features are the raw 1024-d visual width (embedding dims differ per model;
downstream cosine math never assumes one).

Lives in its own checkout (see config.CLIP_REIDENT_DIR), imported lazily
via sys.path (deps live in our venv). Weights are the repo's Google Drive
release, unzipped under <checkout>/model/:

    gdown 1Gm5J19okhLdnZTQLUsjfYoI0rwrLQ09i -O model/checkpoints.zip
    unzip model/checkpoints.zip -d model/
"""

from __future__ import annotations

import numpy as np

from yp_video.config import CLIP_REIDENT_DIR

# The all-data checkpoint (their fold checkpoints exist only for ensembling,
# which we skip — one forward per crop is expensive enough).
CHECKPOINT = CLIP_REIDENT_DIR / "model" / "ViT-L-14_openai" / "all_data_seed_1" / "weights_e4.pth"
EMBEDDING_DIM = 1024


def clip_reident_available() -> bool:
    return CHECKPOINT.exists()


class ClipReidentEmbedder:
    """Same embed() contract as the other embedders; GPU when available."""

    def __init__(self):
        self._model = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _ensure(self):
        if self._model is not None:
            return
        import sys

        import torch

        if str(CLIP_REIDENT_DIR) not in sys.path:
            sys.path.insert(0, str(CLIP_REIDENT_DIR))
        from clipreid.model import OpenClipModel
        from clipreid.transforms import get_transforms

        # The checkpoint was trained via create_model("ViT-L-14", "openai"),
        # which in the open_clip of that era silently meant QuickGELU
        # activations. Newer open_clip only guarantees that through the
        # explicit -quickgelu config; pretrained=None because the checkpoint
        # overwrites every weight anyway (strict=True proves it).
        model = OpenClipModel("ViT-L-14-quickgelu", None, remove_proj=True)
        model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"), strict=True)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = model.eval().to(self._device)
        size = model.get_image_size()
        # Zero-pad aspect-preserving resize + CLIP normalization — the exact
        # val transform the checkpoint was evaluated with.
        self._tf, _ = get_transforms(
            (size, size) if isinstance(size, int) else tuple(size),
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        )

    def embed(self, crops_bgr: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
        """Embed BGR person crops → (N, 1024) float32, L2-normalized."""
        if not len(crops_bgr):
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        self._ensure()
        import cv2
        import torch

        feats: list[np.ndarray] = []
        for start in range(0, len(crops_bgr), batch_size):
            batch = [
                self._tf(image=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))["image"]
                for crop in crops_bgr[start : start + batch_size]
            ]
            with torch.inference_mode():
                out = self._model(torch.stack(batch).to(self._device))
            feats.append(out.float().cpu().numpy())
        matrix = np.concatenate(feats).astype(np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
        return matrix
