"""Run TAD inference on new videos using OpenTAD.

Pipeline:
1. Extract R3D features from video
2. Run OpenTAD inference
3. Convert output to annotator JSONL format
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from .extract_features import extract_features_from_video, load_model, get_video_info
from .output_converter import convert_tad_output_to_jsonl


def get_opentad_path() -> Path:
    """Get path to OpenTAD repository."""
    return Path(__file__).parent.parent / "OpenTAD"


def run_inference(
    video_path: Path,
    checkpoint_path: Path,
    config_path: Path,
    output_path: Path,
    device: str = "cuda",
    confidence_threshold: float = 0.3,
    cut_dir: Path | None = None,
):
    """Run full inference pipeline on a video.

    Args:
        video_path: Path to input video
        checkpoint_path: Path to trained checkpoint
        config_path: Path to config file
        output_path: Path for output JSONL
        device: Device to use
        confidence_threshold: Minimum confidence for detections
        cut_dir: If set, export rally clips to this directory
    """
    print(f"Processing: {video_path.name}")

    # Get video info
    num_frames, fps = get_video_info(video_path)
    duration = num_frames / fps if fps > 0 else 0

    # Step 1: Extract features (or load from cache)
    feature_cache = Path(__file__).parent / "data" / "features" / f"{video_path.stem}.npy"
    if feature_cache.exists():
        print(f"Step 1: Loading cached features from {feature_cache}")
        features = np.load(feature_cache)
    else:
        print("Step 1: Extracting R3D features...")
        torch_device = torch.device(device)
        model = load_model(torch_device)
        features = extract_features_from_video(
            video_path, model, torch_device, clip_len=16, stride=16
        )
        feature_cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(feature_cache, features)
    print(f"  Features shape: {features.shape}")

    feature_fps = features.shape[0] / duration if duration > 0 else 1.0

    # Save features to temp file for OpenTAD
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_feature_path = Path(tmpdir) / f"{video_path.stem}.npy"
        np.save(temp_feature_path, features)

        # Step 2: Run OpenTAD inference
        print("Step 2: Running TAD inference...")
        opentad_path = get_opentad_path()

        if not opentad_path.exists() or not checkpoint_path.exists():
            print("Warning: OpenTAD or checkpoint not found, using simple detection")
            detections = simple_inference(features, fps, confidence_threshold)
        else:
            detections = opentad_inference(
                temp_feature_path,
                checkpoint_path,
                config_path,
                opentad_path,
                device,
                confidence_threshold,
                duration,
                feature_fps,
                fps,
            )

        print(f"  Found {len(detections)} detections")

    # Step 3: Convert to JSONL
    print("Step 3: Converting to JSONL...")
    convert_tad_output_to_jsonl(
        detections=detections,
        video_path=video_path,
        duration=duration,
        feature_fps=feature_fps,
        output_path=output_path,
    )

    print(f"  Saved to: {output_path}")

    # Step 4 (optional): Cut rally clips
    if cut_dir is not None:
        from utils.ffmpeg import export_segment

        print("Step 4: Exporting rally clips...")
        cut_dir.mkdir(parents=True, exist_ok=True)

        sorted_dets = sorted(detections, key=lambda d: d["segment"][0])
        for i, det in enumerate(sorted_dets, 1):
            start, end = det["segment"]
            clip_path = cut_dir / f"rally_{i:03d}.mp4"
            export_segment(video_path, start, end, clip_path)

        # Print summary table
        print(f"\n{'Clip':<16} {'Start':>8} {'End':>8} {'Duration':>8} {'Conf':>6}")
        print("-" * 50)
        for i, det in enumerate(sorted_dets, 1):
            start, end = det["segment"]
            dur = end - start
            score = det["score"]
            print(f"rally_{i:03d}.mp4   {start:>7.1f}s {end:>7.1f}s {dur:>7.1f}s {score:>5.2f}")
        print(f"\n  {len(sorted_dets)} clips saved to: {cut_dir}")

    print("Inference complete!")


def simple_inference(
    features: np.ndarray,
    fps: float,
    confidence_threshold: float = 0.3,
) -> list[dict]:
    """Simple inference without OpenTAD (for testing).

    Uses feature magnitude changes to detect potential action boundaries.
    """
    from scipy.ndimage import gaussian_filter1d

    # Compute feature magnitudes
    magnitudes = np.linalg.norm(features, axis=1)

    # Smooth magnitudes
    smoothed = gaussian_filter1d(magnitudes, sigma=2)

    # Find peaks (potential rally starts)
    detections = []

    # Simple threshold-based detection
    threshold = np.mean(smoothed) + 0.5 * np.std(smoothed)
    above_threshold = smoothed > threshold

    # Find continuous regions
    in_region = False
    start_idx = 0

    for i, above in enumerate(above_threshold):
        if above and not in_region:
            in_region = True
            start_idx = i
        elif not above and in_region:
            in_region = False
            if i - start_idx >= 3:  # Minimum duration
                confidence = float(np.mean(smoothed[start_idx:i]) / np.max(smoothed))
                if confidence >= confidence_threshold:
                    # Convert to time (assuming features are at 1 per 16 frames at 30fps)
                    start_time = start_idx * 16 / fps
                    end_time = i * 16 / fps
                    detections.append(
                        {
                            "segment": [start_time, end_time],
                            "label": "rally",
                            "score": confidence,
                        }
                    )

    return detections


def opentad_inference(
    feature_path: Path,
    checkpoint_path: Path,
    config_path: Path,
    opentad_path: Path,
    device: str,
    confidence_threshold: float,
    duration: float,
    feature_fps: float,
    video_fps: float = 60.0,
) -> list[dict]:
    """Run OpenTAD inference by loading model directly."""
    import sys
    sys.path.insert(0, str(opentad_path))

    from mmengine.config import Config
    from opentad.models import build_detector

    # Load config
    cfg = Config.fromfile(str(config_path))

    # Build model
    model = build_detector(cfg.model)
    model = model.to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "state_dict_ema" in checkpoint:
        state_dict = checkpoint["state_dict_ema"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    # Strip "module." prefix from DataParallel/DDP training
    state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    # Load features
    features = np.load(feature_path)
    features_tensor = torch.from_numpy(features).float().to(device)
    features_tensor = features_tensor.unsqueeze(0).permute(0, 2, 1)  # [1, C, T]

    # Create masks
    seq_len = features_tensor.shape[2]
    masks = torch.ones(1, 1, seq_len, device=device)

    # Create metas for post-processing
    # Use padding mode (like ThumosPaddingDataset used in training)
    feature_stride = cfg.dataset["train"]["feature_stride"]
    sample_stride = cfg.dataset["train"].get("sample_stride", 1)
    offset_frames = cfg.dataset["train"].get("offset_frames", 0)
    snippet_stride = feature_stride * sample_stride

    metas = [{
        "video_name": feature_path.stem,
        "duration": duration,
        "fps": video_fps,
        "snippet_stride": snippet_stride,
        "offset_frames": offset_frames,
        "feat_stride": feature_stride,
        "feat_num_frames": 16,
    }]

    # Create post_cfg from config
    class PostCfg:
        def __init__(self, cfg_dict):
            self.sliding_window = False
            self.nms = None
            if "nms" in cfg_dict:
                self.nms = dict(cfg_dict["nms"])

    post_cfg = PostCfg(cfg.post_processing)

    # Create infer_cfg
    class InferCfg:
        load_from_raw_predictions = False
        save_raw_prediction = False

    infer_cfg = InferCfg()

    # Run inference using forward_test
    with torch.no_grad():
        predictions = model.forward_test(features_tensor, masks, metas, infer_cfg)

    # Post-processing using model's method
    ext_cls = ["rally"]  # Single class
    results = model.post_processing(predictions, metas, post_cfg, ext_cls)

    # Convert to detection format
    detections = []
    for video_id, video_results in results.items():
        for result in video_results:
            score = result["score"]
            if score >= confidence_threshold:
                segment = result["segment"]
                # Clamp to video duration
                start_time = max(0, min(segment[0], duration))
                end_time = max(0, min(segment[1], duration))

                if end_time > start_time:
                    detections.append({
                        "segment": [start_time, end_time],
                        "label": "rally",
                        "score": score,
                    })

    return detections


def main():
    parser = argparse.ArgumentParser(description="Run TAD inference on video")
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Input video path",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).parent / "checkpoints" / "actionformer" / "best.pth",
        help="Model checkpoint path",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs" / "volleyball_actionformer.py",
        help="Config file path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: ~/videos/tad-predictions/{video_stem}_annotations.jsonl)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--cut",
        action="store_true",
        help="Export detected rallies as individual video clips",
    )
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Error: Video not found: {args.video}")
        sys.exit(1)

    if args.output is None:
        args.output = (
            Path.home() / "videos" / "tad-predictions" / f"{args.video.stem}_annotations.jsonl"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    cut_dir = None
    if args.cut:
        cut_dir = Path.home() / "videos" / "rally_clips" / args.video.stem

    run_inference(
        args.video,
        args.checkpoint,
        args.config,
        args.output,
        args.device,
        args.threshold,
        cut_dir,
    )


if __name__ == "__main__":
    main()
