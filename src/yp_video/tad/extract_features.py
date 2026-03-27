"""Extract V-JEPA 2.1 features from videos for TAD.

Uses V-JEPA 2.1 ViT-L pretrained model to extract RGB features.
Output: (T, 768) feature array saved as .npy file.

Each clip samples 64 frames at stride 2 (128 actual video frames per window).
Uses decord for efficient video decoding.
"""

import argparse
from pathlib import Path

import numpy as np

from yp_video.config import TAD_FEATURES_DIR
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    from decord import VideoReader, cpu
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False
    import cv2

# V-JEPA 2.1 constants (fixed by model design)
VJEPA_CLIP_FRAMES = 64   # frames per clip
VJEPA_FRAME_STRIDE = 2   # stride between sampled frames within a clip
VJEPA_WINDOW = VJEPA_CLIP_FRAMES * VJEPA_FRAME_STRIDE  # = 128 actual video frames
FEAT_DIM = 768            # ViT-B embed dim
CROP_SIZE = 384


def load_model(device: torch.device) -> torch.nn.Module:
    """Load pretrained V-JEPA 2.1 ViT-L model."""
    encoder, _ = torch.hub.load(
        "facebookresearch/vjepa2",
        "vjepa2_1_vit_base_384",
        trust_repo=True,
        verbose=False,
    )
    encoder = encoder.to(device)
    encoder.eval()
    return encoder


def preprocess_clip(frames_np: np.ndarray) -> torch.Tensor:
    """Preprocess video clip for V-JEPA 2.1.

    Args:
        frames_np: (T, H, W, C) uint8 numpy array

    Returns:
        (C, T, CROP_SIZE, CROP_SIZE) float32 tensor
    """
    T, H, W, _ = frames_np.shape
    clip = torch.from_numpy(frames_np).float() / 255.0
    clip = clip.permute(0, 3, 1, 2)  # (T, C, H, W)

    # Resize short side to CROP_SIZE
    short_side = min(H, W)
    if short_side != CROP_SIZE:
        scale = CROP_SIZE / short_side
        new_h = int(H * scale + 0.5)
        new_w = int(W * scale + 0.5)
        clip = F.interpolate(clip, size=(new_h, new_w), mode="bilinear", align_corners=False)
    else:
        new_h, new_w = H, W

    # Center crop
    top = (new_h - CROP_SIZE) // 2
    left = (new_w - CROP_SIZE) // 2
    clip = clip[:, :, top:top + CROP_SIZE, left:left + CROP_SIZE]

    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    clip = (clip - mean) / std

    return clip.permute(1, 0, 2, 3)  # (C, T, CROP_SIZE, CROP_SIZE)


def load_video_decord(video_path: Path, indices: list) -> np.ndarray:
    """Load specific frames using decord."""
    vr = VideoReader(str(video_path), ctx=cpu(0))
    frames = vr.get_batch(indices).asnumpy()  # (N, H, W, C)
    return frames


def load_video_cv2(video_path: Path, indices: list) -> np.ndarray:
    """Load specific frames using OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    max_idx = max(indices)

    idx = 0
    indices_set = set(indices)
    while cap.isOpened() and idx <= max_idx:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in indices_set:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        idx += 1

    cap.release()
    return np.array(frames)


def get_video_info(video_path: Path) -> tuple[int, float]:
    """Get video frame count and FPS."""
    if HAS_DECORD:
        vr = VideoReader(str(video_path), ctx=cpu(0))
        return len(vr), vr.get_avg_fps()
    else:
        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return frame_count, fps


def extract_features_from_video(
    video_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    stride: int = 64,
    batch_size: int = 16,
) -> np.ndarray:
    """Extract V-JEPA 2.1 features from a video file.

    Args:
        video_path: Path to video file
        model: Pretrained V-JEPA 2.1 encoder
        device: torch device
        stride: Stride between clips in actual video frames
        batch_size: Number of clips to process at once

    Returns:
        Feature array of shape (T, 1024)
    """
    num_frames, fps = get_video_info(video_path)

    if num_frames < VJEPA_WINDOW:
        print(f"  Warning: Video too short ({num_frames} frames < {VJEPA_WINDOW})")
        return np.zeros((1, FEAT_DIM), dtype=np.float32)

    # Clip start positions in actual video frames
    clip_starts = list(range(0, num_frames - VJEPA_WINDOW + 1, stride))
    if not clip_starts:
        clip_starts = [0]

    features = []
    load_fn = load_video_decord if HAS_DECORD else load_video_cv2

    for batch_start in tqdm(
        range(0, len(clip_starts), batch_size),
        desc=f"  {video_path.name}",
        leave=False,
    ):
        batch_clip_starts = clip_starts[batch_start : batch_start + batch_size]

        # Collect unique frame indices for this batch (sampled at VJEPA_FRAME_STRIDE)
        all_indices = sorted({
            start + i * VJEPA_FRAME_STRIDE
            for start in batch_clip_starts
            for i in range(VJEPA_CLIP_FRAMES)
        })

        frames = load_fn(video_path, all_indices)
        frames_dict = {idx: frames[i] for i, idx in enumerate(all_indices)}

        clips = []
        for start_idx in batch_clip_starts:
            clip_indices = [start_idx + i * VJEPA_FRAME_STRIDE for i in range(VJEPA_CLIP_FRAMES)]
            clip_frames = np.stack([frames_dict[i] for i in clip_indices], axis=0)  # (T, H, W, C)
            clips.append(preprocess_clip(clip_frames))  # (C, T, H', W')

        clips = torch.stack(clips, dim=0).to(device)  # (B, C, T, H', W')

        with torch.no_grad():
            patch_feats = model(clips)  # (B, num_patches, 1024)
            feats = patch_feats.mean(dim=1)  # (B, 1024)
            features.append(feats.cpu().numpy())

    return np.concatenate(features, axis=0)  # (T, 1024)


def process_directory(
    input_dir: Path,
    output_dir: Path,
    device: torch.device,
    stride: int = 64,
    videos: list[str] | None = None,
    batch_size: int = 16,
):
    """Process all videos in a directory."""
    model = load_model(device)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_extensions = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
    video_files = [f for f in input_dir.iterdir() if f.suffix.lower() in video_extensions]

    if videos:
        video_set = set(videos)
        video_files = [f for f in video_files if f.stem in video_set]

    video_files.sort()
    print(f"Found {len(video_files)} videos in {input_dir}")
    print(f"Using {'decord' if HAS_DECORD else 'OpenCV'} for video loading")

    for video_path in tqdm(video_files, desc="Processing videos"):
        output_path = output_dir / f"{video_path.stem}.npy"
        if output_path.exists():
            print(f"Skipping {video_path.name} (already exists)")
            continue

        try:
            features = extract_features_from_video(
                video_path, model, device, stride, batch_size
            )
            np.save(output_path, features)
            print(f"Saved {output_path.name}: {features.shape}")
        except Exception as e:
            print(f"Error processing {video_path.name}: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Extract V-JEPA 2.1 features from videos")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path.home() / "videos" / "cuts",
        help="Input directory containing videos",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=TAD_FEATURES_DIR,
        help="Output directory for features",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=64,
        help="Stride between clips in actual video frames (default: 64)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of clips to process at once (default: 16)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--videos",
        type=str,
        nargs="*",
        default=None,
        help="Specific video stems to process (optional)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"V-JEPA 2.1 ViT-L: {VJEPA_CLIP_FRAMES} frames/clip, "
          f"frame stride {VJEPA_FRAME_STRIDE}, clip stride {args.stride}")

    process_directory(
        args.input, args.output, device, args.stride, args.videos, args.batch_size
    )
    print("Feature extraction complete!")


if __name__ == "__main__":
    main()
