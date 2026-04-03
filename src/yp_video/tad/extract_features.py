"""Extract V-JEPA 2.1 features from videos for TAD.

Uses V-JEPA 2.1 pretrained models to extract BF16 RGB features.
Supports ViT-B (768), ViT-L (1024), ViT-g (1408), ViT-G (1664).

Each clip samples 64 frames at stride 2 (128 actual video frames per window).
"""

import argparse
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

warnings.filterwarnings("ignore", message=".*sdp_kernel.*", category=FutureWarning)

import numpy as np

from yp_video.config import FEATURES_DIR
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    from decord import VideoReader, cpu
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
VJEPA_FRAME_STRIDE = 2   # stride between sampled frames within a clip
VJEPA_WINDOW = VJEPA_CLIP_FRAMES * VJEPA_FRAME_STRIDE  # = 128 actual video frames
CROP_SIZE = 384


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

# Default for backward compatibility
FEAT_DIM = 768

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

    def __init__(self, video_path: Path, num_frames: int):
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

        trim_below = max_idx - VJEPA_WINDOW
        self._cache = {k: v for k, v in self._cache.items() if k >= trim_below}

        # Filter to only frames that were actually decoded (video may be shorter than metadata)
        available = [i for i in indices if i in results]
        if not available:
            raise RuntimeError("No frames decoded")
        return np.stack([results[i] for i in available])


def open_video(video_path: Path, use_gpu: bool = False) -> DecordReader | Cv2Reader | NvdecReader:
    """Open a video with the best available decoder."""
    if HAS_DECORD:
        reader = DecordReader(video_path)
    else:
        reader = Cv2Reader(video_path)

    if use_gpu and HAS_NVDEC:
        try:
            return NvdecReader(video_path, num_frames=len(reader))
        except Exception:
            pass
    return reader


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
        encoder = encoder.to(device=device, dtype=torch.bfloat16)
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


def preprocess_frames_batch(frames_np: np.ndarray) -> torch.Tensor:
    """Resize, crop, and normalize all unique frames at once.

    Args:
        frames_np: (N, H, W, C) uint8 numpy array

    Returns:
        (N, C, CROP_SIZE, CROP_SIZE) float32 tensor (CPU)
    """
    N, H, W, _ = frames_np.shape
    frames = torch.from_numpy(frames_np).float().div_(255.0)
    frames = frames.permute(0, 3, 1, 2)  # (N, C, H, W)

    short_side = min(H, W)
    if short_side != CROP_SIZE:
        scale = CROP_SIZE / short_side
        new_h = int(H * scale + 0.5)
        new_w = int(W * scale + 0.5)
        frames = F.interpolate(frames, size=(new_h, new_w), mode="bilinear", align_corners=False)
    else:
        new_h, new_w = H, W

    top = (new_h - CROP_SIZE) // 2
    left = (new_w - CROP_SIZE) // 2
    frames = frames[:, :, top:top + CROP_SIZE, left:left + CROP_SIZE]

    frames.sub_(_MEAN).div_(_STD)
    return frames


# ── Feature extraction ─────────────────────────────────────────────────


def extract_features_from_video(
    video_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    stride: int = 64,
    batch_size: int = 32,
    feat_dim: int = FEAT_DIM,
) -> np.ndarray:
    """Extract V-JEPA 2.1 features from a video file."""
    reader = open_video(video_path, use_gpu=device.type == "cuda")
    num_frames = len(reader)

    if num_frames < VJEPA_WINDOW:
        print(f"  Warning: Video too short ({num_frames} frames < {VJEPA_WINDOW})")
        return np.zeros((1, feat_dim), dtype=np.float32)

    clip_starts = list(range(0, num_frames - VJEPA_WINDOW + 1, stride))
    if not clip_starts:
        clip_starts = [0]

    # Pre-compute per-batch clip starts and frame indices
    batch_infos = []
    for batch_start in range(0, len(clip_starts), batch_size):
        batch_clip_starts = clip_starts[batch_start : batch_start + batch_size]
        all_indices = sorted({
            start + i * VJEPA_FRAME_STRIDE
            for start in batch_clip_starts
            for i in range(VJEPA_CLIP_FRAMES)
        })
        batch_infos.append((batch_clip_starts, all_indices))

    features = []

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(reader.get_batch, batch_infos[0][1])

        for bi in tqdm(range(len(batch_infos)), desc=f"  {video_path.name}", leave=False):
            frames = future.result()

            if bi + 1 < len(batch_infos):
                future = pool.submit(reader.get_batch, batch_infos[bi + 1][1])

            batch_clip_starts, all_indices = batch_infos[bi]

            # Truncate indices to match actual decoded frames (video may be shorter than metadata)
            actual_indices = all_indices[:len(frames)]

            # Preprocess unique frames, then assemble clips
            processed = preprocess_frames_batch(frames)
            proc_dict = {idx: processed[i] for i, idx in enumerate(actual_indices)}

            clips = []
            for start_idx in batch_clip_starts:
                clip_indices = [start_idx + i * VJEPA_FRAME_STRIDE for i in range(VJEPA_CLIP_FRAMES)]
                if not all(i in proc_dict for i in clip_indices):
                    continue  # skip incomplete clips at video end
                clips.append(torch.stack([proc_dict[i] for i in clip_indices], dim=1))

            if not clips:
                continue  # all clips in this batch were incomplete

            clips = torch.stack(clips, dim=0).to(device=device, dtype=torch.bfloat16)

            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                patch_feats = model(clips)
                feats = patch_feats.mean(dim=1)
                features.append(feats.float().cpu().numpy())

    return np.concatenate(features, axis=0)


# ── Directory processing & CLI ─────────────────────────────────────────


def process_directory(
    input_dir: Path,
    output_dir: Path,
    device: torch.device,
    stride: int = 64,
    videos: list[str] | None = None,
    batch_size: int = 32,
    model_name: str = "base",
    on_progress: "Callable[[int, int], None] | None" = None,
):
    """Process all videos in a directory.

    Args:
        on_progress: Optional callback ``(done, total) -> None`` called after
            each video is processed (or skipped).
    """
    cfg = MODEL_CONFIGS[model_name]
    model = load_model(device, model_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_extensions = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
    video_files = [f for f in input_dir.iterdir() if f.suffix.lower() in video_extensions]

    if videos:
        video_set = set(videos)
        video_files = [f for f in video_files if f.stem in video_set or f.name in video_set]

    video_files.sort()
    total = len(video_files)
    print(f"Found {total} videos in {input_dir}")
    decoder = "NVDEC (GPU)" if HAS_NVDEC else ("decord" if HAS_DECORD else "OpenCV")
    print(f"Using {decoder} for video loading")

    for i, video_path in enumerate(tqdm(video_files, desc="Processing videos")):
        output_path = output_dir / f"{video_path.stem}.npy"
        if output_path.exists():
            print(f"Skipping {video_path.name} (already exists)")
            if on_progress:
                on_progress(i + 1, total)
            continue

        try:
            features = extract_features_from_video(
                video_path, model, device, stride, batch_size,
                feat_dim=cfg.feat_dim,
            )
            np.save(output_path, features)
            print(f"Saved {output_path.name}: {features.shape}")
        except Exception as e:
            print(f"Error processing {video_path.name}: {e}")
            import traceback
            traceback.print_exc()

        if on_progress:
            on_progress(i + 1, total)


def main():
    parser = argparse.ArgumentParser(description="Extract V-JEPA 2.1 features from videos")
    parser.add_argument("--model", type=str, default="base",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="V-JEPA 2.1 model size (default: base)")
    parser.add_argument("--input", type=Path, default=Path.home() / "videos" / "cuts")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output dir (default: ~/videos/features/vjepa-{size}/)")
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--videos", type=str, nargs="*", default=None)
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    output_dir = args.output or (FEATURES_DIR / cfg.dir_suffix)

    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"V-JEPA 2.1 {args.model} ({cfg.hub_name}): dim={cfg.feat_dim}, "
          f"{VJEPA_CLIP_FRAMES} frames/clip, "
          f"frame stride {VJEPA_FRAME_STRIDE}, clip stride {args.stride}")
    print(f"Output: {output_dir}")

    process_directory(args.input, output_dir, device, args.stride, args.videos,
                      args.batch_size, model_name=args.model)
    print("Feature extraction complete!")


if __name__ == "__main__":
    main()
