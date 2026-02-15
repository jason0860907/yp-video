"""Extract I3D features from videos for TAD.

Uses R3D-18 model pretrained on Kinetics-400 to extract RGB features.
Output: (T, 512) feature array saved as .npy file.

Uses decord for efficient video decoding.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.video import r3d_18, R3D_18_Weights
from tqdm import tqdm

try:
    from decord import VideoReader, cpu, gpu
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False
    import cv2


def load_model(device: torch.device) -> tuple:
    """Load pretrained R3D-18 model."""
    weights = R3D_18_Weights.KINETICS400_V1
    model = r3d_18(weights=weights)
    # Remove the final classification layer to get features
    model.fc = torch.nn.Identity()
    model = model.to(device)
    model.eval()
    return model


def preprocess_clip(clip: torch.Tensor) -> torch.Tensor:
    """Preprocess clip for R3D model.

    Args:
        clip: (B, C, T, H, W) tensor with values in [0, 1]

    Returns:
        Preprocessed clip
    """
    # Resize to 112x112
    B, C, T, H, W = clip.shape
    clip = clip.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # (B*T, C, H, W)
    clip = F.interpolate(clip, size=(112, 112), mode="bilinear", align_corners=False)
    clip = clip.reshape(B, T, C, 112, 112).permute(0, 2, 1, 3, 4)  # (B, C, T, 112, 112)

    # Normalize with Kinetics-400 stats
    mean = torch.tensor([0.43216, 0.394666, 0.37645], device=clip.device).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.22803, 0.22145, 0.216989], device=clip.device).view(1, 3, 1, 1, 1)
    clip = (clip - mean) / std

    return clip


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
    clip_len: int = 16,
    stride: int = 16,
    batch_size: int = 16,
) -> np.ndarray:
    """Extract features from a video file.

    Args:
        video_path: Path to video file
        model: Pretrained video model
        device: torch device
        clip_len: Number of frames per clip
        stride: Stride between clips
        batch_size: Number of clips to process at once

    Returns:
        Feature array of shape (T, 512)
    """
    num_frames, fps = get_video_info(video_path)

    if num_frames < clip_len:
        print(f"  Warning: Video too short ({num_frames} frames < {clip_len})")
        return np.zeros((1, 512), dtype=np.float32)

    # Calculate clip start indices
    clip_starts = list(range(0, num_frames - clip_len + 1, stride))
    if not clip_starts:
        clip_starts = [0]

    features = []
    load_fn = load_video_decord if HAS_DECORD else load_video_cv2

    # Process in batches
    for batch_start in tqdm(
        range(0, len(clip_starts), batch_size),
        desc=f"  {video_path.name}",
        leave=False,
    ):
        batch_clip_starts = clip_starts[batch_start : batch_start + batch_size]

        # Collect all frame indices needed for this batch
        all_indices = []
        for start_idx in batch_clip_starts:
            all_indices.extend(range(start_idx, start_idx + clip_len))
        all_indices = sorted(set(all_indices))

        # Load frames
        frames = load_fn(video_path, all_indices)  # (N, H, W, C)
        frames_dict = {idx: frames[i] for i, idx in enumerate(all_indices)}

        # Build clips
        clips = []
        for start_idx in batch_clip_starts:
            clip_frames = [frames_dict[i] for i in range(start_idx, start_idx + clip_len)]
            clip = np.stack(clip_frames, axis=0)  # (T, H, W, C)
            clips.append(clip)

        clips = np.stack(clips, axis=0)  # (B, T, H, W, C)
        clips = torch.from_numpy(clips).float().permute(0, 4, 1, 2, 3) / 255.0  # (B, C, T, H, W)

        # Preprocess and move to device
        clips = preprocess_clip(clips)
        clips = clips.to(device)

        # Extract features
        with torch.no_grad():
            feats = model(clips)  # (B, 512)
            features.append(feats.cpu().numpy())

    return np.concatenate(features, axis=0)  # (T, 512)


def process_directory(
    input_dir: Path,
    output_dir: Path,
    device: torch.device,
    clip_len: int = 16,
    stride: int = 16,
    videos: list[str] | None = None,
):
    """Process all videos in a directory.

    Args:
        input_dir: Directory containing videos
        output_dir: Directory for output features
        device: Torch device
        clip_len: Frames per clip
        stride: Stride between clips
        videos: Optional list of video stems to process (for partial processing)
    """
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
                video_path, model, device, clip_len, stride
            )
            np.save(output_path, features)
            print(f"Saved {output_path.name}: {features.shape}")
        except Exception as e:
            print(f"Error processing {video_path.name}: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Extract R3D features from videos")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path.home() / "videos" / "cuts",
        help="Input directory containing videos",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "data" / "features",
        help="Output directory for features",
    )
    parser.add_argument(
        "--clip-len",
        type=int,
        default=16,
        help="Number of frames per clip",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=16,
        help="Stride between clips (in frames)",
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

    process_directory(
        args.input, args.output, device, args.clip_len, args.stride, args.videos
    )
    print("Feature extraction complete!")


if __name__ == "__main__":
    main()
