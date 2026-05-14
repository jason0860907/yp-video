"""Run TAD inference on new videos using ActionFormer.

Pipeline:
1. Extract V-JEPA 2.1 features from video
2. Run ActionFormer inference
3. Convert output to annotator JSONL format
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from yp_video.config import (
    ACTIONFORMER_DIR,
    FEATURES_DIR,
    TAD_CHECKPOINTS_DIR,
    TAD_CONFIGS_DIR,
    TAD_FEATURES_DIR,
)

from yp_video.core.sampling import frame_to_time, get_fps

from .extract_features import MODEL_CONFIGS, extract_features_from_video, load_model, open_video
from .output_converter import convert_tad_output_to_jsonl


def run_inference(
    video_path: Path,
    checkpoint_path: Path,
    config_path: Path,
    output_path: Path,
    device: str = "cuda",
    confidence_threshold: float = 0.3,
    cut_dir: Path | None = None,
    model_name: str = "base",
    on_message: "Callable[[str], None] | None" = None,
    on_progress: "Callable[[float], None] | None" = None,
) -> list[dict]:
    """Run full inference pipeline on a video.

    Args:
        video_path: Path to input video
        checkpoint_path: Path to trained checkpoint
        config_path: Path to config file
        output_path: Path for output JSONL
        device: Device to use
        confidence_threshold: Minimum confidence for detections
        cut_dir: If set, export rally clips to this directory
        model_name: V-JEPA model size (base/large/giant/gigantic)
        on_message: Optional callback for step-level status updates
        on_progress: Optional callback ``(fraction) -> None`` for progress bar

    Returns:
        List of detection dicts (also written to ``output_path`` as JSONL).
        Each dict has keys ``segment`` ([start, end] in seconds), ``label``,
        and ``score``.
    """
    def _msg(text: str):
        print(text)
        if on_message:
            on_message(text)

    def _prog(frac: float):
        if on_progress:
            on_progress(frac)

    _msg(f"Processing: {video_path.name}")
    _prog(0.0)

    mcfg = MODEL_CONFIGS[model_name]
    feat_dir = FEATURES_DIR / mcfg.dir_suffix

    # Get video info
    reader = open_video(video_path)
    num_frames = len(reader)
    fps = get_fps(video_path)
    duration = frame_to_time(num_frames, fps)

    # Step 1: Extract features (or load from cache)
    feature_cache = feat_dir / f"{video_path.stem}.npy"
    if feature_cache.exists():
        _msg("Step 1/3: Loading cached features")
        features = np.load(feature_cache)
    else:
        _msg(f"Step 1/3: Extracting V-JEPA 2.1 {model_name} features...")
        torch_device = torch.device(device)
        model = load_model(torch_device, model_name)
        features = extract_features_from_video(
            video_path, model, torch_device,
            feat_dim=mcfg.feat_dim,
        )
        feature_cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(feature_cache, features)
    print(f"  Features shape: {features.shape}")
    _prog(0.7)

    # Step 2: Run ActionFormer inference
    _msg("Step 2/3: Running TAD inference...")
    if not checkpoint_path.exists():
        print("Warning: Checkpoint not found, using simple detection")
        detections = simple_inference(features, fps, confidence_threshold)
    else:
        detections = actionformer_inference(
            features,
            checkpoint_path,
            config_path,
            device,
            confidence_threshold,
            duration,
            fps,
            model_name,
        )

    print(f"  Found {len(detections)} detections")
    _prog(0.9)

    # Step 3: Convert to JSONL
    _msg(f"Step 3/3: Saving {len(detections)} detections...")
    convert_tad_output_to_jsonl(
        detections=detections,
        video_path=video_path,
        duration=duration,
        feature_fps=1.0,
        output_path=output_path,
        checkpoint=str(checkpoint_path),
        model=model_name,
    )

    print(f"  Saved to: {output_path}")

    # Step 4 (optional): Cut rally clips
    if cut_dir is not None:
        import asyncio

        from yp_video.core.ffmpeg import export_segment

        print("Step 4: Exporting rally clips...")
        cut_dir.mkdir(parents=True, exist_ok=True)

        sorted_dets = sorted(detections, key=lambda d: d["segment"][0])

        async def _export_all():
            for i, det in enumerate(sorted_dets, 1):
                start, end = det["segment"]
                clip_path = cut_dir / f"rally_{i:03d}.mp4"
                await export_segment(video_path, start, end, clip_path)

        asyncio.run(_export_all())

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
    return detections


def simple_inference(
    features: np.ndarray,
    fps: float,
    confidence_threshold: float = 0.3,
) -> list[dict]:
    """Simple inference without ActionFormer (for testing).

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
                    # Features are at 1 per 16 frames at 30fps
                    start_time = frame_to_time(start_idx * 16, fps)
                    end_time = frame_to_time(i * 16, fps)
                    detections.append(
                        {
                            "segment": [start_time, end_time],
                            "label": "rally",
                            "score": confidence,
                        }
                    )

    return detections


def _setup_actionformer():
    """Add ActionFormer to sys.path for imports."""
    af_dir = str(ACTIONFORMER_DIR)
    af_utils = str(ACTIONFORMER_DIR / "libs" / "utils")
    if af_dir not in sys.path:
        sys.path.insert(0, af_dir)
    if af_utils not in sys.path:
        sys.path.insert(0, af_utils)


def actionformer_inference(
    features: np.ndarray,
    checkpoint_path: Path,
    config_path: Path,
    device: str,
    confidence_threshold: float,
    duration: float,
    video_fps: float = 60.0,
    model_name: str = "base",
) -> list[dict]:
    """Run ActionFormer inference on extracted features."""
    _setup_actionformer()

    from libs.core import load_config
    from libs.modeling import make_meta_arch

    # Load config
    cfg = load_config(str(config_path))

    # Override input_dim to match the feature model
    mcfg = MODEL_CONFIGS[model_name]
    cfg["dataset"]["input_dim"] = mcfg.feat_dim
    cfg["model"]["input_dim"] = mcfg.feat_dim

    # Use the same fps that the training data loader did so postprocessing's
    # `seconds = (feat_idx * stride + nframes/2) / fps` matches what the
    # loss converged on. ActionFormer's THUMOS dataset takes default_fps
    # from the YAML when it's set (it is — 60.0) and IGNORES the per-video
    # `fps` field in the annotation JSON. Passing the real video fps here
    # would scale every predicted boundary by default_fps / video_fps
    # (e.g. 60/30 = 2× too long), then duration-clamp would truncate to
    # the video length — exactly the "94% val mAP, garbage at predict
    # time" symptom.
    default_fps = cfg["dataset"].get("default_fps")
    fps_for_postprocess = float(default_fps) if default_fps is not None else float(video_fps)

    # Build model and wrap in DataParallel (ActionFormer checkpoints expect this)
    model = make_meta_arch(cfg["model_name"], **cfg["model"])
    model = torch.nn.DataParallel(model, device_ids=[0])

    # Load checkpoint (prefer EMA weights)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "state_dict_ema" in checkpoint:
        model.load_state_dict(checkpoint["state_dict_ema"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Prepare input: features are (T, C), ActionFormer expects (C, T)
    feats = torch.from_numpy(features).float().permute(1, 0)  # (C, T)

    feat_stride = cfg["dataset"]["feat_stride"]
    num_frames = cfg["dataset"]["num_frames"]

    video_item = {
        "video_id": "inference",
        "feats": feats,
        "segments": None,
        "labels": None,
        "fps": fps_for_postprocess,
        "duration": duration,
        "feat_stride": feat_stride,
        "feat_num_frames": num_frames,
    }

    # Run inference
    with torch.no_grad():
        results = model([video_item])

    # Convert results to detection format
    # ActionFormer postprocessing already converts to seconds
    detections = []
    for result in results:
        segs = result["segments"]
        scores = result["scores"]
        labels = result["labels"]

        for i in range(len(segs)):
            score = scores[i].item()
            if score >= confidence_threshold:
                start_time = max(0, min(segs[i][0].item(), duration))
                end_time = max(0, min(segs[i][1].item(), duration))

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
        default=TAD_CHECKPOINTS_DIR / "actionformer" / "best.pth",
        help="Model checkpoint path",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=TAD_CONFIGS_DIR / "volleyball_actionformer.yaml",
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
        "--model",
        type=str,
        default="base",
        choices=list(MODEL_CONFIGS.keys()),
        help="V-JEPA model size (must match the features used for training)",
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
        args.model,
    )


if __name__ == "__main__":
    main()
