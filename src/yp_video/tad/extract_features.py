"""Extract V-JEPA 2.1 features from videos for TAD.

Uses V-JEPA 2.1 pretrained models to extract BF16 RGB features.
Supports ViT-B (768), ViT-L (1024), ViT-g (1408), ViT-G (1664).

Each clip samples 64 frames evenly across ``clip_seconds`` of wall-clock time,
and consecutive clip starts are spaced ``stride_seconds`` apart. This keeps the
temporal scale of features consistent across videos with different fps.
"""

import argparse
import os
import threading
import time
import traceback
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

warnings.filterwarnings("ignore", message=".*sdp_kernel.*", category=FutureWarning)

import numpy as np

from yp_video.config import FEATURES_DIR
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import tqdm

# Inference-time perf flags — bit-exact for our bf16 model (TF32 only kicks in
# on any residual fp32 ops; cudnn.benchmark picks the fastest algo but output
# is identical). No effect on training correctness since this module is
# inference-only.
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Per-batch timing breakdown when TAD_PROFILE=1. Prints decode / preprocess /
# gather / inference split after each video.
_PROFILE = os.environ.get("TAD_PROFILE") == "1"

try:
    from decord import VideoReader, cpu
    import decord.logging as _decord_logging
    _decord_logging.set_level(_decord_logging.QUIET)  # suppress FFmpeg mmco errors on stderr
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False
    import cv2

try:
    import PyNvVideoCodec as nvc
    HAS_NVDEC = True
except ImportError:
    HAS_NVDEC = False

# V-JEPA 2.1 constants (fixed by model design)
VJEPA_CLIP_FRAMES = 64   # frames per clip
CROP_SIZE = 384

# Time-based sampling defaults — match the per-feature time scale that the
# 60-fps "enterprise" source produced under the legacy frame-stride config
# (window=128 frames @ 60fps = 2.13s; clip stride=48 frames @ 60fps = 0.80s).
DEFAULT_CLIP_SECONDS = 2.13
DEFAULT_STRIDE_SECONDS = 0.80
# Upper bound for NvdecReader cache trim when the per-call window is unknown.
_DEFAULT_CACHE_FRAMES = 256


class ModelConfig(NamedTuple):
    hub_name: str
    feat_dim: int
    dir_suffix: str  # subdirectory under ~/videos/tad-features/


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "base":     ModelConfig("vjepa2_1_vit_base_384",      768,  "vjepa-b"),
    "large":    ModelConfig("vjepa2_1_vit_large_384",     1024, "vjepa-l"),
    "giant":    ModelConfig("vjepa2_1_vit_giant_384",     1408, "vjepa-g"),
    "gigantic": ModelConfig("vjepa2_1_vit_gigantic_384",  1664, "vjepa-gg"),
}

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# ── Model cache (avoids repeated torch.compile) ──────────────────────
_model_cache: dict[tuple[str, str], torch.nn.Module] = {}
_model_cache_lock = threading.Lock()


# ── Video readers ──────────────────────────────────────────────────────


class DecordReader:
    """CPU video reader using decord (random access)."""

    def __init__(self, video_path: Path):
        self._vr = VideoReader(str(video_path), ctx=cpu(0))

    def __len__(self) -> int:
        return len(self._vr)

    def get_batch(self, indices: list[int]) -> np.ndarray:
        return self._vr.get_batch(indices).asnumpy()


class Cv2Reader:
    """CPU video reader using OpenCV (sequential scan)."""

    def __init__(self, video_path: Path):
        self._path = video_path
        cap = cv2.VideoCapture(str(video_path))
        self._num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    def __len__(self) -> int:
        return self._num_frames

    def get_batch(self, indices: list[int]) -> np.ndarray:
        cap = cv2.VideoCapture(str(self._path))
        frames = []
        max_idx = max(indices)
        indices_set = set(indices)
        idx = 0
        while cap.isOpened() and idx <= max_idx:
            ret, frame = cap.read()
            if not ret:
                break
            if idx in indices_set:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            idx += 1
        cap.release()
        return np.array(frames)


class NvdecReader:
    """GPU video reader using NVDEC (sequential decode, cached overlap)."""

    def __init__(self, video_path: Path, num_frames: int, cache_window: int = _DEFAULT_CACHE_FRAMES):
        self._dmx = nvc.CreateDemuxer(str(video_path))
        self._dec = nvc.CreateDecoder(
            gpuid=0,
            codec=self._dmx.GetNvCodecId(),
            outputColorType=nvc.OutputColorType.RGB,
        )
        self._pkt_iter = iter(self._dmx)
        self._pos = 0
        self._cache: dict[int, np.ndarray] = {}
        self._exhausted = False
        self._num_frames = num_frames
        self._cache_window = cache_window

    def __len__(self) -> int:
        return self._num_frames

    def get_batch(self, indices: list[int]) -> np.ndarray:
        indices_set = set(indices)
        min_idx, max_idx = indices[0], indices[-1]

        self._cache = {k: v for k, v in self._cache.items() if k >= min_idx}
        results: dict[int, np.ndarray] = {
            i: self._cache[i] for i in indices if i in self._cache
        }

        while len(results) < len(indices) and not self._exhausted:
            try:
                pkt = next(self._pkt_iter)
            except StopIteration:
                self._exhausted = True
                break
            for f in self._dec.Decode(pkt):
                if self._pos in indices_set and self._pos not in results:
                    frame = torch.utils.dlpack.from_dlpack(f).numpy().copy()
                    results[self._pos] = frame
                    self._cache[self._pos] = frame
                self._pos += 1

        trim_below = max_idx - self._cache_window
        self._cache = {k: v for k, v in self._cache.items() if k >= trim_below}

        # Filter to only frames that were actually decoded (video may be shorter than metadata)
        available = [i for i in indices if i in results]
        if not available:
            raise RuntimeError("No frames decoded")
        return np.stack([results[i] for i in available])


def open_video(
    video_path: Path,
    use_gpu: bool = False,
    cache_window: int = _DEFAULT_CACHE_FRAMES,
) -> DecordReader | Cv2Reader | NvdecReader:
    """Open a video with the best available decoder."""
    if HAS_DECORD:
        reader = DecordReader(video_path)
    else:
        reader = Cv2Reader(video_path)

    if use_gpu and HAS_NVDEC:
        try:
            return NvdecReader(video_path, num_frames=len(reader), cache_window=cache_window)
        except Exception:
            pass
    return reader


def _get_video_fps(video_path: Path) -> float:
    """Read average fps from the container metadata."""
    if HAS_DECORD:
        try:
            return float(VideoReader(str(video_path), ctx=cpu(0)).get_avg_fps())
        except Exception:
            pass
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(fps) if fps and fps > 0 else 30.0


# ── Model ──────────────────────────────────────────────────────────────

_inductor_cache_configured = False


def _enable_inductor_cache() -> None:
    """Enable PyTorch inductor persistent caches (fx_graph + autotune).

    Caches stored under ~/.cache/torch/inductor/ survive process restarts,
    reducing compile time from ~30-60s to ~2-5s on warm start.
    """
    global _inductor_cache_configured
    if _inductor_cache_configured:
        return
    try:
        import torch._inductor.config as inductor_cfg
        inductor_cfg.fx_graph_cache = True
        inductor_cfg.autotune_cache = True
        _inductor_cache_configured = True
    except (ImportError, AttributeError):
        pass


def load_model(device: torch.device, model_name: str = "base") -> torch.nn.Module:
    """Load pretrained V-JEPA 2.1 model with BF16 and torch.compile.

    Cached in-process: repeated calls with the same (model_name, device)
    return the already-compiled module instantly.
    """
    cache_key = (model_name, str(device))

    cached = _model_cache.get(cache_key)
    if cached is not None:
        print(f"Using cached V-JEPA 2.1 {model_name} model")
        return cached

    with _model_cache_lock:
        # Re-check after acquiring lock
        cached = _model_cache.get(cache_key)
        if cached is not None:
            return cached

        cfg = MODEL_CONFIGS[model_name]
        print(f"Loading V-JEPA 2.1 {model_name} ({cfg.hub_name}, dim={cfg.feat_dim})...")

        _enable_inductor_cache()

        encoder, _ = torch.hub.load(
            "facebookresearch/vjepa2",
            cfg.hub_name,
            trust_repo=True,
            verbose=False,
        )
        encoder = encoder.to(device=device, dtype=torch.bfloat16,
                             memory_format=torch.channels_last_3d)
        encoder.eval()
        print("Compiling model (first time ~30-60s, cached thereafter)...")
        encoder = torch.compile(encoder, mode="max-autotune")
        _model_cache[cache_key] = encoder
        return encoder


def clear_model_cache() -> list[str]:
    """Drop all cached models, freeing GPU memory."""
    with _model_cache_lock:
        keys = list(_model_cache.keys())
        _model_cache.clear()
        return [f"{name}@{dev}" for name, dev in keys]


# ── Preprocessing ──────────────────────────────────────────────────────


# Chunk size for preprocess — caps the peak GPU/CPU memory spike.
# Chunking is numerically identical because bilinear resize and normalization
# are per-pixel with no cross-frame dependency.
_PREPROCESS_CHUNK = 64


def preprocess_frames_batch(frames_np: np.ndarray, device: torch.device) -> torch.Tensor:
    """Resize, crop, and normalize all unique frames on *device*.

    Args:
        frames_np: (N, H, W, C) uint8 numpy array
        device: Target device (CUDA recommended — GPU resize is ~10x faster).

    Returns:
        (N, C, CROP_SIZE, CROP_SIZE) float32 tensor on *device*
    """
    N, H, W, _ = frames_np.shape

    short_side = min(H, W)
    if short_side != CROP_SIZE:
        scale = CROP_SIZE / short_side
        new_h = int(H * scale + 0.5)
        new_w = int(W * scale + 0.5)
        needs_resize = True
    else:
        new_h, new_w = H, W
        needs_resize = False

    top = (new_h - CROP_SIZE) // 2
    left = (new_w - CROP_SIZE) // 2

    mean = _MEAN.to(device)
    std = _STD.to(device)
    out = torch.empty((N, 3, CROP_SIZE, CROP_SIZE), dtype=torch.float32, device=device)

    for i in range(0, N, _PREPROCESS_CHUNK):
        # Transfer uint8 first (4x less PCIe bandwidth than float32)
        chunk = torch.from_numpy(frames_np[i:i + _PREPROCESS_CHUNK]).to(device)
        chunk = chunk.float().div_(255.0).permute(0, 3, 1, 2)  # (n, C, H, W)
        if needs_resize:
            chunk = F.interpolate(chunk, size=(new_h, new_w), mode="bilinear", align_corners=False)
        chunk = chunk[:, :, top:top + CROP_SIZE, left:left + CROP_SIZE]
        out[i:i + chunk.shape[0]] = chunk.sub_(mean).div_(std)

    return out


# ── Feature extraction ─────────────────────────────────────────────────


def extract_features_from_video(
    video_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    clip_seconds: float = DEFAULT_CLIP_SECONDS,
    stride_seconds: float = DEFAULT_STRIDE_SECONDS,
    batch_size: int = 32,
    feat_dim: int = 768,
) -> np.ndarray:
    """Extract V-JEPA 2.1 features from a video file.

    Each clip's 64 sampled frames span ``clip_seconds`` of wall-clock time, and
    consecutive clip starts are ``stride_seconds`` apart. The resulting feature
    rate is therefore ``1 / stride_seconds`` features per second regardless of
    source fps.
    """
    fps = _get_video_fps(video_path)
    window_frames = max(VJEPA_CLIP_FRAMES, int(round(clip_seconds * fps)))
    clip_stride = max(1, int(round(stride_seconds * fps)))
    # 64 evenly-spaced frame offsets within each clip's window. When
    # window_frames < 64 (very low fps) some offsets repeat.
    frame_offsets = (
        np.linspace(0, window_frames - 1, VJEPA_CLIP_FRAMES).round().astype(int).tolist()
    )

    reader = open_video(video_path, use_gpu=device.type == "cuda", cache_window=window_frames)
    num_frames = len(reader)

    if num_frames < window_frames:
        print(f"  Warning: Video too short ({num_frames} frames < {window_frames})")
        return np.zeros((1, feat_dim), dtype=np.float32)

    clip_starts = list(range(0, num_frames - window_frames + 1, clip_stride))
    if not clip_starts:
        clip_starts = [0]

    # Pre-compute per-batch clip starts and frame indices
    batch_infos = []
    for batch_start in range(0, len(clip_starts), batch_size):
        batch_clip_starts = clip_starts[batch_start : batch_start + batch_size]
        all_indices = sorted({
            start + off
            for start in batch_clip_starts
            for off in frame_offsets
        })
        batch_infos.append((batch_clip_starts, all_indices))

    features = []
    timings = {"decode_wait": 0.0, "preprocess": 0.0, "gather": 0.0, "inference": 0.0}
    n_timed = 0

    def _now() -> float:
        if _PROFILE:
            torch.cuda.synchronize()
        return time.perf_counter()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(reader.get_batch, batch_infos[0][1])

        for bi in tqdm(range(len(batch_infos)), desc=f"  {video_path.name}", leave=False):
            t0 = _now()
            frames = future.result()
            t1 = _now()

            if bi + 1 < len(batch_infos):
                future = pool.submit(reader.get_batch, batch_infos[bi + 1][1])

            batch_clip_starts, all_indices = batch_infos[bi]

            # Truncate indices to match actual decoded frames (video may be shorter than metadata)
            actual_indices = all_indices[:len(frames)]

            # Preprocess unique frames on GPU, then assemble clips via indexing
            processed = preprocess_frames_batch(frames, device=device)
            # processed: (N_unique, C, H, W) on device
            t2 = _now()

            # Build index map: frame index → position in processed tensor
            idx_to_pos = {idx: pos for pos, idx in enumerate(actual_indices)}

            # Build gather indices for all valid clips at once
            gather_rows = []  # list of [64 positions in processed] per valid clip
            for start_idx in batch_clip_starts:
                clip_indices = [start_idx + off for off in frame_offsets]
                if not all(i in idx_to_pos for i in clip_indices):
                    continue
                gather_rows.append([idx_to_pos[i] for i in clip_indices])

            if not gather_rows:
                continue  # all clips in this batch were incomplete

            # Single advanced-index gather: (num_clips, 64, C, H, W) → permute to (num_clips, C, 64, H, W)
            gather_idx = torch.tensor(gather_rows, device=processed.device)  # (num_clips, 64)
            clips = processed[gather_idx.flatten()].view(
                len(gather_rows), VJEPA_CLIP_FRAMES, 3, CROP_SIZE, CROP_SIZE,
            ).permute(0, 2, 1, 3, 4).to(dtype=torch.bfloat16)

            actual_count = clips.shape[0]
            del processed  # free preprocessing memory before inference

            # Pad to fixed batch_size so CUDAGraph only records one graph
            if actual_count < batch_size:
                pad = torch.zeros(
                    (batch_size - actual_count, *clips.shape[1:]),
                    device=device, dtype=clips.dtype,
                )
                clips = torch.cat([clips, pad], dim=0)
            clips = clips.contiguous(memory_format=torch.channels_last_3d)
            t3 = _now()

            # autocast is load-bearing: V-JEPA's attention has fp32 Q/K (RoPE)
            # with bf16 V — autocast coerces them all to bf16 before SDPA.
            # Forcing FlashAttention prevents fallback to math backend (saves
            # ~20% peak VRAM with no throughput change).
            with torch.inference_mode(), \
                 torch.autocast("cuda", dtype=torch.bfloat16), \
                 sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                patch_feats = model(clips)
                feats = patch_feats.mean(dim=1)[:actual_count]  # trim padding
                features.append(feats.float().cpu().numpy())
            t4 = _now()

            if _PROFILE:
                timings["decode_wait"] += t1 - t0
                timings["preprocess"]  += t2 - t1
                timings["gather"]      += t3 - t2
                timings["inference"]   += t4 - t3
                n_timed += 1

    if _PROFILE and n_timed > 0:
        total = sum(timings.values())
        print(f"  [profile] {video_path.name} ({n_timed} batches, {total:.2f}s total)")
        for k, v in timings.items():
            pct = v / total * 100 if total else 0
            print(f"    {k:12s} {v:6.2f}s  avg {v / n_timed * 1000:6.1f}ms  ({pct:5.1f}%)")

    return np.concatenate(features, axis=0)


# ── Directory processing & CLI ─────────────────────────────────────────


def process_directory(
    input_dir,
    output_dir: Path,
    device: torch.device,
    clip_seconds: float = DEFAULT_CLIP_SECONDS,
    stride_seconds: float = DEFAULT_STRIDE_SECONDS,
    videos: list[str] | None = None,
    batch_size: int = 32,
    model_name: str = "base",
    on_progress: Callable[[int, int], None] | None = None,
):
    """Process all videos under one or more directories.

    Args:
        input_dir: A single Path or a sequence of Paths. Cuts are split across
            broadcast/sideline dirs, so this accepts both forms.
        on_progress: Optional callback ``(done, total) -> None`` called after
            each video is processed (or skipped).
    """
    cfg = MODEL_CONFIGS[model_name]
    model = load_model(device, model_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(input_dir, Path):
        input_dirs = [input_dir]
    else:
        input_dirs = list(input_dir)

    video_extensions = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
    video_files: list[Path] = []
    seen: set[str] = set()
    for d in input_dirs:
        if not d.exists():
            continue
        for f in d.iterdir():
            if f.suffix.lower() in video_extensions and f.name not in seen:
                seen.add(f.name)
                video_files.append(f)

    if videos:
        video_set = set(videos)
        video_files = [f for f in video_files if f.stem in video_set or f.name in video_set]

    video_files.sort()
    total = len(video_files)
    print(f"Found {total} videos in {[str(d) for d in input_dirs]}")
    decoder = "NVDEC (GPU)" if HAS_NVDEC else ("decord" if HAS_DECORD else "OpenCV")
    print(f"Using {decoder} for video loading")

    # Warmup: trigger AUTOTUNE + CUDAGraph with real batch_size so the
    # cost doesn't land inside the first video's tqdm.
    print("Warming up (AUTOTUNE + CUDAGraph)...")
    dummy = torch.zeros(batch_size, 3, VJEPA_CLIP_FRAMES, CROP_SIZE, CROP_SIZE,
                        device=device, dtype=torch.bfloat16).contiguous(
        memory_format=torch.channels_last_3d)
    with torch.no_grad(), \
         torch.autocast("cuda", dtype=torch.bfloat16), \
         sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        model(dummy)
    del dummy
    torch.cuda.empty_cache()
    print("Warmup done.")

    for i, video_path in enumerate(tqdm(video_files, desc="Processing videos")):
        output_path = output_dir / f"{video_path.stem}.npy"
        if output_path.exists():
            print(f"Skipping {video_path.name} (already exists)")
            if on_progress:
                on_progress(i + 1, total)
            continue

        try:
            features = extract_features_from_video(
                video_path, model, device,
                clip_seconds=clip_seconds, stride_seconds=stride_seconds,
                batch_size=batch_size, feat_dim=cfg.feat_dim,
            )
            np.save(output_path, features)
            print(f"Saved {output_path.name}: {features.shape}")
        except Exception as e:
            print(f"Error processing {video_path.name}: {e}")
            traceback.print_exc()

        if on_progress:
            on_progress(i + 1, total)


def main():
    parser = argparse.ArgumentParser(description="Extract V-JEPA 2.1 features from videos")
    parser.add_argument("--model", type=str, default="base",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="V-JEPA 2.1 model size (default: base)")
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=None,
        help="One or more cut dirs (default: cuts-broadcast + cuts-sideline)",
    )
    parser.add_argument("--output", type=Path, default=None,
                        help="Output dir (default: ~/videos/features/vjepa-{size}/)")
    parser.add_argument("--clip-seconds", type=float, default=DEFAULT_CLIP_SECONDS,
                        help=f"Wall-clock span of each clip's 64 sampled frames "
                             f"(default: {DEFAULT_CLIP_SECONDS}s)")
    parser.add_argument("--stride-seconds", type=float, default=DEFAULT_STRIDE_SECONDS,
                        help=f"Time gap between consecutive clip starts "
                             f"(default: {DEFAULT_STRIDE_SECONDS}s, "
                             f"= {1 / DEFAULT_STRIDE_SECONDS:.2f} features/sec)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--videos", type=str, nargs="*", default=None)
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    output_dir = args.output or (FEATURES_DIR / cfg.dir_suffix)

    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"V-JEPA 2.1 {args.model} ({cfg.hub_name}): dim={cfg.feat_dim}, "
          f"{VJEPA_CLIP_FRAMES} frames/clip spanning {args.clip_seconds}s, "
          f"clip stride {args.stride_seconds}s "
          f"(= {1 / args.stride_seconds:.2f} features/sec)")
    print(f"Output: {output_dir}")

    from yp_video.config import CUTS_DIRS as _CUTS_DIRS
    inputs = args.input if args.input else list(_CUTS_DIRS)
    process_directory(inputs, output_dir, device,
                      clip_seconds=args.clip_seconds, stride_seconds=args.stride_seconds,
                      videos=args.videos, batch_size=args.batch_size,
                      model_name=args.model)
    print("Feature extraction complete!")


if __name__ == "__main__":
    main()
