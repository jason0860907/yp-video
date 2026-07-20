"""Appearance embedders for person crops, behind one path-level interface.

Embedders consume crop *files*, not in-memory images — extraction persists
every crop jpg anyway (the embedder input IS the reviewable artifact), and
the file boundary is what lets heavyweight models run out of process. Two
implementations exist behind the same protocol:

- ``ClipReidEmbedder`` — CLIP-ReID ViT-B/16 (Market-1501 fine-tune) through
  its ONNX export (occurra/person_vit_clip_reid, MIT). CPU, in-process; a few
  hundred crops per video take seconds.
- ``SubprocessEmbedder`` — the yp-reid model package (CLIP-ReIdent lineage,
  basketball-domain ViT-L/14) reached across a subprocess boundary
  (``python -m yp_reid.embed``, Contract C in contracts/reid.py). Weights
  come from the best checkpoint package under reid/checkpoints/ — trained
  runs and the imported paper release compete on their recorded metrics.

Both return ``(N, dim) float32`` L2-normalized matrices, row i ↔ path i,
NaN rows for unreadable files. Embedding dimension differs per model —
downstream cosine math never assumes one.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from yp_video.config import REID_EMBED_MODULE, REID_PKG_DIR, REID_PYTHON
from yp_video.contracts.reid import REID_CONTRACT_VERSION, REID_CONTRACT_VERSION_ENV, REID_PROGRESS_PREFIX
from yp_video.reid.checkpoints import default_checkpoint, read_manifest, reid_engine_available

CLIP_REID_HF_REPO = "occurra/person_vit_clip_reid"
CLIP_REID_ONNX = "person_vit_clip_reid.onnx"
# ReID-standard input aspect (h, w) the checkpoint was trained at.
INPUT_H, INPUT_W = 256, 128
CLIP_REID_DIM = 512

ProgressFn = Callable[[int, int, str], None]


class ReidInferenceError(RuntimeError):
    """yp-reid subprocess embedding failed; callers wanting graceful
    degradation should catch this and continue without the model."""


class Embedder(Protocol):
    """One entry in the embedder registry (see build_embedders)."""

    def embed_paths(
        self, paths: Sequence[Path], *, batch_size: int = 32, on_progress: ProgressFn | None = None
    ) -> np.ndarray:
        """Crop files → (N, dim) float32, L2-normalized, NaN row per unreadable file."""
        ...

    @property
    def loaded(self) -> bool:
        """Whether the weights are resident — first embed otherwise stalls
        on model load, and progress reporting wants to say so."""
        ...


class ClipReidEmbedder:
    """Batch person-crop → 512-d embedding via ONNX. Lazy session load.

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

    def embed_paths(
        self, paths: Sequence[Path], *, batch_size: int = 32, on_progress: ProgressFn | None = None
    ) -> np.ndarray:
        import cv2

        matrix = np.full((len(paths), CLIP_REID_DIM), np.nan, dtype=np.float32)
        if not len(paths):
            return matrix
        self._ensure_session()
        assert self._session is not None
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        for start in range(0, len(paths), batch_size):
            batch, rows = [], []
            for i, path in enumerate(paths[start : start + batch_size]):
                crop = cv2.imread(str(path))
                if crop is None:
                    continue
                img = cv2.cvtColor(cv2.resize(crop, (INPUT_W, INPUT_H)), cv2.COLOR_BGR2RGB)
                t = img.transpose(2, 0, 1).astype(np.float32) / 255.0
                batch.append((t - mean) / std)
                rows.append(start + i)
            if batch:
                feats = np.asarray(self._session.run(None, {"input": np.stack(batch)})[0], dtype=np.float32)
                feats /= np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12
                matrix[rows] = feats
            if on_progress:
                done = min(start + batch_size, len(paths))
                on_progress(done, len(paths), f"crop {done}/{len(paths)}")
        return matrix


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
        self, paths: Sequence[Path], *, batch_size: int = 32, on_progress: ProgressFn | None = None
    ) -> np.ndarray:
        package = default_checkpoint()
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
    """A registered embedder re-run on background-suppressed crops.

    Same weights as ``base`` (delegation keeps ONE loaded copy for both
    variants); the difference is the input — embed_video sees
    ``masked_input`` and feeds crop paths under crops-masked/ (non-actor
    pixels greyed out, reid/seg.py), so appearance features can't latch onto
    teammates in pile-up crops. A/B against the base variant on identical
    extractions.
    """

    masked_input = True

    def __init__(self, base: Embedder):
        self._base = base

    @property
    def loaded(self) -> bool:
        return self._base.loaded

    def embed_paths(
        self, paths: Sequence[Path], *, batch_size: int = 32, on_progress: ProgressFn | None = None
    ) -> np.ndarray:
        return self._base.embed_paths(paths, batch_size=batch_size, on_progress=on_progress)


# Registered embedder names — the API-facing whitelist. What weights a name
# stands for is answered by weights_id() at runtime.
EMBEDDER_NAMES = ("clip-reid", "clip-reid-masked", "clip-reident", "clip-reident-masked")


def weights_id(name: str) -> str:
    """Identifier of the weights a name currently binds to.

    The clip-reident names follow the best checkpoint package, so this is the
    only honest answer to "which weights produced this matrix" — record it
    wherever that question will be asked later.
    """
    if name.startswith("clip-reident"):
        package = default_checkpoint()
        base = package.name if package else "<no checkpoint package>"
    else:
        base = CLIP_REID_ONNX
    return f"{base} + rf-detr-seg-medium" if name.endswith("-masked") else base


# The optional models may be unregistered (yp-reid venv or checkpoint package
# missing); /reid/options falls back to the first registered embedder then.
# The masked variants are always registered — their base (clip-reid) ships
# with the repo.
#
# clip-reident-masked measured best on the labeled crops: mAP 0.715 / Rank-1
# 0.931 against clip-reid-masked's 0.636 / 0.874 (see the ReID Train page).
# Background suppression is what unlocks it — unmasked, CLIP-ReIdent is the
# WORST of the four on mAP (0.559), so the basketball-domain advantage only
# materializes once teammates are masked out of the crop.
DEFAULT_EMBEDDER = "clip-reident-masked"

# Cluster-threshold slider calibration per embedder, served to the UI via
# /reid/options. Cosine-distance scales differ wildly per model: CLIP-ReID's
# ViT features sit in a tight cone, CLIP-ReIdent's fine-tuned ViT-L is tighter
# still. Models without an entry get the wide neutral fallback.
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
    "clip-reid": {"min": 0.081, "max": 0.23, "default": 0.15, "step": 0.005},
    "clip-reident": {"min": 0.016, "max": 0.039, "default": 0.026, "step": 0.001},
    "clip-reid-masked": {"min": 0.083, "max": 0.23, "default": 0.16, "step": 0.005},
    "clip-reident-masked": {"min": 0.019, "max": 0.043, "default": 0.034, "step": 0.001},
}
FALLBACK_THRESHOLD = {"min": 0.05, "max": 0.95, "default": 0.3, "step": 0.01}


def threshold_calibration(name: str) -> dict:
    return EMBEDDER_THRESHOLDS.get(name, FALLBACK_THRESHOLD)


# Constructed embedders, by name. Construction is cheap (models load lazily on
# first embed) but instances must persist across calls so loaded weights stay
# resident for the whole process.
_instances: dict[str, Embedder] = {}


def build_embedders() -> dict[str, Embedder]:
    """Every available embedder; optional ones join when their engine + weights exist.

    Availability is re-checked on every call (a manifest stat per optional
    model), so a checkpoint package landing while the server runs appears
    without a restart.
    """
    available: dict[str, Callable[[], Embedder]] = {"clip-reid": ClipReidEmbedder}
    if reid_engine_available():
        available["clip-reident"] = SubprocessEmbedder
    # Masked variants wrap the base instances — dict order guarantees a base
    # is constructed by the time its factory runs.
    available["clip-reid-masked"] = lambda: MaskedEmbedder(_instances["clip-reid"])
    if "clip-reident" in available:
        available["clip-reident-masked"] = lambda: MaskedEmbedder(_instances["clip-reident"])
    for name, make in available.items():
        _instances.setdefault(name, make())
    return {name: _instances[name] for name in available}
