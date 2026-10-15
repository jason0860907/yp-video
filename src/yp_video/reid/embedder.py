"""Appearance embedders for person crops, behind one path-level interface.

Embedders consume crop *files*, not in-memory images — extraction persists
every crop jpg anyway (the embedder input IS the reviewable artifact), and
the file boundary is what lets heavyweight models run out of process. The
engine is ``SubprocessEmbedder`` — the yp-reid model package (CLIP-ReIdent
lineage, basketball-domain ViT-L/14) reached across a subprocess boundary
(``python -m yp_reid.embed``, Contract C in contracts/reid.py). Weights come
from the best checkpoint package under reid/checkpoints/ — trained runs and
the imported paper release compete on their recorded metrics.

It returns ``(N, dim) float32`` L2-normalized matrices, row i ↔ path i, NaN
rows for unreadable files. The one registered embedder,
``clip-reident-masked``, runs it on background-suppressed crops; the
tracklet-window variant (extraction/windows.py) runs it on its own crops.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from yp_video.config import REID_EMBED_MODULE, REID_PKG_DIR, REID_PYTHON
from yp_video.contracts.reid import (
    REID_CONTRACT_VERSION,
    REID_CONTRACT_VERSION_ENV,
    REID_PROGRESS_PREFIX,
)
from yp_video.core.progress import ProgressFn
from yp_video.reid.checkpoints import (
    default_checkpoint,
    read_manifest,
    reid_engine_available,
)

# ReID-standard input aspect (h, w) the checkpoint was trained at.

class ReidInferenceError(RuntimeError):
    """yp-reid subprocess embedding failed; callers wanting graceful
    degradation should catch this and continue without the model."""


class Embedder(Protocol):
    """One entry in the embedder registry (see build_embedders)."""

    def embed_paths(
        self,
        paths: Sequence[Path],
        *,
        batch_size: int = 32,
        on_progress: ProgressFn | None = None,
        checkpoint: Path | None = None,
    ) -> np.ndarray:
        """Crop files → (N, dim) float32, L2-normalized, NaN row per unreadable file.

        ``checkpoint`` overrides which weight package to load; ``None`` means
        the default package.
        """
        ...

    @property
    def loaded(self) -> bool:
        """Whether the weights are resident — first embed otherwise stalls
        on model load, and progress reporting wants to say so."""
        ...


class SubprocessEmbedder:
    """The yp-reid engine across a subprocess boundary (Contract C).

    The checkpoint package is resolved per call (the best one may change
    while the server runs), and each call pays the weight load — so callers
    batch a whole video into one call, never crop-by-crop.
    """

    @property
    def loaded(self) -> bool:
        return False  # every call cold-loads in the subprocess; be honest

    def embed_paths(
        self,
        paths: Sequence[Path],
        *,
        batch_size: int = 32,
        on_progress: ProgressFn | None = None,
        checkpoint: Path | None = None,
    ) -> np.ndarray:
        package = checkpoint or default_checkpoint()
        if package is None:
            raise ReidInferenceError("No ReID checkpoint package available")
        dim = read_manifest(package)["model"]["embedding_dim"]
        if not len(paths):
            return np.full((0, dim), np.nan, dtype=np.float32)

        with tempfile.TemporaryDirectory(prefix="yp-reid-embed-") as tmp:
            crops_list = Path(tmp) / "crops.txt"
            crops_list.write_text("\n".join(str(p) for p in paths) + "\n", encoding="utf-8")
            out = Path(tmp) / "embeddings.npy"
            cmd = [
                str(REID_PYTHON), "-m", REID_EMBED_MODULE,
                "--checkpoint", str(package),
                "--crops-list", str(crops_list),
                "--out", str(out),
                "--batch-size", str(batch_size),
            ]
            env = {
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                REID_CONTRACT_VERSION_ENV: REID_CONTRACT_VERSION,
            }
            proc = subprocess.Popen(
                cmd, cwd=REID_PKG_DIR, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            )
            tail: deque[str] = deque(maxlen=20)
            assert proc.stdout is not None
            for raw in proc.stdout:
                # tqdm-style \r redraws can glue fragments to a line; split so
                # the prefix is always at a segment start.
                for line in raw.rstrip("\n").split("\r"):
                    progress = _parse_progress(line)
                    if progress is not None:
                        done, total = progress
                        if on_progress:
                            on_progress(done, total, f"crop {done}/{total}")
                    elif line.strip():
                        tail.append(line.strip())
            rc = proc.wait()
            if rc != 0:
                raise ReidInferenceError(f"yp-reid embed failed (rc={rc}): " + " | ".join(list(tail)[-5:]))
            if not out.exists():
                raise ReidInferenceError("yp-reid embed produced no output")
            matrix = np.load(out)
        if matrix.shape != (len(paths), dim):
            raise ReidInferenceError(f"yp-reid embed shape {matrix.shape} != ({len(paths)}, {dim})")
        return matrix


def _parse_progress(line: str) -> tuple[int, int] | None:
    """(done, total) from a REID_PROGRESS embed line; None otherwise.
    Only the prefix is a hard contract — the body is parsed defensively."""
    if not line.startswith(REID_PROGRESS_PREFIX):
        return None
    try:
        data = json.loads(line[len(REID_PROGRESS_PREFIX):])
        if data.get("phase") != "embed":
            return None
        return int(data["done"]), int(data["total"])
    except (ValueError, KeyError, TypeError):
        return None


class MaskedEmbedder:
    """The engine re-run on background-suppressed crops.

    embed_video sees ``masked_input`` and feeds crop paths under
    crops-masked/ (non-actor pixels greyed out, reid/seg.py), so appearance
    features can't latch onto teammates in pile-up crops.
    """

    masked_input = True

    def __init__(self, base: Embedder):
        self._base = base

    @property
    def loaded(self) -> bool:
        return self._base.loaded

    def embed_paths(
        self,
        paths: Sequence[Path],
        *,
        batch_size: int = 32,
        on_progress: ProgressFn | None = None,
        checkpoint: Path | None = None,
    ) -> np.ndarray:
        return self._base.embed_paths(
            paths, batch_size=batch_size, on_progress=on_progress, checkpoint=checkpoint
        )


#: The registered embedder, and the only one the API accepts.
#
#: clip-reident-masked measured best on the labeled crops: mAP 0.715 / Rank-1
#: 0.931 against CLIP-ReID's 0.636 / 0.874; unmasked, CLIP-ReIdent was the
#: worst of the four on mAP (0.559), so the masking is what makes it work. The
#: other three were removed 2026-10-16.
DEFAULT_EMBEDDER = "clip-reident-masked"
EMBEDDER_NAMES = (DEFAULT_EMBEDDER,)


def base_embedder_name(name: str) -> str:
    """Weight-sharing family for masked/unmasked input variants."""
    return name.removesuffix("-masked")


def weights_id(name: str) -> str:
    """Identifier of the weights a name currently binds to.

    The names follow the default checkpoint package, so this is the only
    honest answer to "which weights produced this matrix" — record it
    wherever that question will be asked later.
    """
    package = default_checkpoint()
    base = package.name if package else "<no checkpoint package>"
    return f"{base} + rf-detr-seg-medium" if name.endswith("-masked") else base


# Cluster-threshold slider calibration per embedder, served to the UI via
# /reid/options. Cosine-distance scales differ per variant (the tracklet
# window's mean sits well below a single crop's).
#
# Calibrated by the ReID Train page (reid/evaluate.py) against 547 labeled
# crops over 24 identities: it sweeps the cutoff, clusters at each stop, and
# takes the peak adjusted Rand index against the human labels. `default` is
# that peak, `min`/`max` bound the band that still clusters usefully. Note the
# peak sits on the OVER-SPLIT side by design — merging two groups of one
# player is a drag, a group holding two players has to be spotted first.
#
# Re-run the page after labeling more videos; these are a two-session fit —
# and recalibrate after every new fine-tune, distance scales move.
EMBEDDER_THRESHOLDS = {
    "clip-reident-masked": {"min": 0.019, "max": 0.043, "default": 0.034, "step": 0.001},
    # Tracklet-window mean (extraction/windows.py) over McByte++ tracks:
    # averaging tightens same-player distances, so its scale sits well below
    # the single masked crop's. Fit 2026-10-02 on 913 events / 64 identities
    # (6 sideline videos); ARI peaks at 0.017 then drops steeply, so the
    # default takes the over-split side of the peak.
    "clip-reident-masked-win30": {"min": 0.005, "max": 0.027, "default": 0.016, "step": 0.001},
}


def threshold_calibration(name: str) -> dict:
    return EMBEDDER_THRESHOLDS[name]


_engine = SubprocessEmbedder()


def reident_engine() -> Embedder:
    """The CLIP-ReIdent engine on whatever crops the caller hands it."""
    if not reid_engine_available():
        raise ReidInferenceError("yp-reid is not installed or has no checkpoint package")
    return _engine


def build_embedders() -> dict[str, Embedder]:
    """The registered embedders available right now — empty until yp-reid and
    a checkpoint package exist. Re-checked per call (a manifest stat), so a
    package landing while the server runs appears without a restart."""
    if not reid_engine_available():
        return {}
    return {DEFAULT_EMBEDDER: MaskedEmbedder(_engine)}
