"""
InternVideo-Next video feature extraction script.

Usage:
    uv run python run_internvideo_next.py --video path/to/video.mp4
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path

# Add InternVideo-Next to path
sys.path.insert(0, str(Path(__file__).parent / "InternVideo" / "InternVideo-Next"))

from decord import VideoReader, cpu
from torchvision import transforms
from einops import rearrange


def load_video(video_path: str, num_frames: int = 16, img_size: int = 224):
    """Load and preprocess video frames."""
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)

    # Sample frames uniformly
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = vr.get_batch(frame_indices).asnumpy()  # (T, H, W, C)

    # Preprocess
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    processed_frames = []
    for frame in frames:
        processed_frames.append(transform(frame))

    # Stack to (C, T, H, W) format
    video_tensor = torch.stack(processed_frames, dim=1)
    return video_tensor.unsqueeze(0)  # (1, C, T, H, W)


def load_model(model_name: str = "large", device: str = "cuda"):
    """Load InternVideo-Next model from HuggingFace."""
    from huggingface_hub import hf_hub_download
    from models.InternVideo_next import internvideo_next_large_patch14_224, internvideo_next_base_patch14_224

    if model_name == "large":
        # Download weights
        print("Downloading InternVideo-Next Large weights...")
        ckpt_path = hf_hub_download(
            repo_id="revliter/internvideo_next_large_p14_res224_f16",
            filename="internvideo_next_large.pth"
        )
        model = internvideo_next_large_patch14_224(
            num_frames=16,
            use_flash_attn=True,
            use_fused_rmsnorm=True,
            use_fused_mlp=True,
        )
    else:
        print("Downloading InternVideo-Next Base weights...")
        ckpt_path = hf_hub_download(
            repo_id="revliter/internvideo_next_base_p14_res224_f16",
            filename="internvideo_next_base.pth"
        )
        model = internvideo_next_base_patch14_224(
            num_frames=16,
            use_flash_attn=True,
            use_fused_rmsnorm=True,
            use_fused_mlp=True,
        )

    # Load weights
    print(f"Loading weights from {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict, strict=False)

    model = model.to(device).eval()
    print(f"Model loaded on {device}")
    return model


def main():
    parser = argparse.ArgumentParser(description="InternVideo-Next feature extraction")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--model", type=str, default="large", choices=["base", "large"], help="Model size")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames to sample")
    parser.add_argument("--output", type=str, default=None, help="Output path for features (.pt file)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load video
    print(f"Loading video: {args.video}")
    video_tensor = load_video(args.video, num_frames=args.num_frames)
    video_tensor = video_tensor.to(device)
    print(f"Video tensor shape: {video_tensor.shape}")  # (1, C, T, H, W)

    # Load model
    model = load_model(args.model, device)

    # Extract features
    print("Extracting features...")
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            # Get video features
            features = model(video_tensor)  # (B, T*H*W, C)

            # Also get projected features (for retrieval tasks)
            projected_features = model(video_tensor, projected=True)  # (B, C)

    print(f"Feature shape: {features.shape}")
    print(f"Projected feature shape: {projected_features.shape}")

    # Save features if output path specified
    if args.output:
        output_path = Path(args.output)
        torch.save({
            "features": features.cpu(),
            "projected_features": projected_features.cpu(),
            "video_path": args.video,
        }, output_path)
        print(f"Features saved to {output_path}")

    # Print feature statistics
    print(f"\nFeature statistics:")
    print(f"  Mean: {features.mean().item():.4f}")
    print(f"  Std:  {features.std().item():.4f}")
    print(f"  Min:  {features.min().item():.4f}")
    print(f"  Max:  {features.max().item():.4f}")

    return features, projected_features


if __name__ == "__main__":
    main()
