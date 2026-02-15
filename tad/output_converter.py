"""Convert TAD output to annotator-compatible JSONL format.

TAD output format (from MambaTAD):
[
    {"segment": [start_frame, end_frame], "label": "rally", "score": 0.95},
    ...
]

Annotator JSONL format:
    Line 1: {"_meta": true, "video": "path/to/video.mp4", "duration": 1800.5}
    Line 2+: {"start": 10.5, "end": 25.3, "label": "rally"}
"""

import argparse
import json
from pathlib import Path


def convert_tad_output_to_jsonl(
    detections: list[dict],
    video_path: Path,
    duration: float,
    feature_fps: float,
    output_path: Path,
    min_duration: float = 1.0,
):
    """Convert TAD detections to annotator JSONL format.

    Args:
        detections: List of detection dicts with segment, label, score
        video_path: Path to the original video
        duration: Video duration in seconds
        feature_fps: Features per second (for frame->time conversion)
        output_path: Output JSONL path
        min_duration: Minimum detection duration in seconds
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        # Write metadata
        meta = {"_meta": True, "video": str(video_path), "duration": duration}
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

        # Write detections
        for det in detections:
            segment = det.get("segment", [0, 0])

            # Convert frame indices to time
            if feature_fps > 0:
                start_time = segment[0] / feature_fps
                end_time = segment[1] / feature_fps
            else:
                start_time = segment[0]
                end_time = segment[1]

            # Clamp to video duration
            start_time = max(0, min(start_time, duration))
            end_time = max(0, min(end_time, duration))

            # Skip too short detections
            if end_time - start_time < min_duration:
                continue

            annotation = {
                "start": round(start_time, 2),
                "end": round(end_time, 2),
                "label": det.get("label", "rally"),
            }

            # Optionally include score for reference
            if "score" in det:
                annotation["confidence"] = round(det["score"], 3)

            f.write(json.dumps(annotation, ensure_ascii=False) + "\n")

    return output_path


def convert_mambatad_results(
    results_path: Path,
    video_dir: Path,
    output_dir: Path,
    feature_fps: float = 1.0,
):
    """Convert MambaTAD results JSON to individual JSONL files.

    Args:
        results_path: Path to MambaTAD results JSON
        video_dir: Directory containing original videos
        output_dir: Directory for output JSONL files
    """
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)

    for video_name, video_results in results.get("results", {}).items():
        # Find video file
        video_path = None
        for ext in [".mp4", ".avi", ".mkv", ".mov", ".webm"]:
            candidate = video_dir / f"{video_name}{ext}"
            if candidate.exists():
                video_path = candidate
                break

        if video_path is None:
            print(f"Warning: Video not found for {video_name}")
            video_path = video_dir / f"{video_name}.mp4"

        # Get video duration
        duration = get_video_duration(video_path) if video_path.exists() else 0

        output_path = output_dir / f"{video_name}.jsonl"
        convert_tad_output_to_jsonl(
            detections=video_results,
            video_path=video_path,
            duration=duration,
            feature_fps=feature_fps,
            output_path=output_path,
        )
        print(f"Converted: {output_path.name}")


def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds."""
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return frame_count / fps if fps > 0 else 0
    except Exception:
        return 0


def main():
    parser = argparse.ArgumentParser(description="Convert TAD output to JSONL")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="MambaTAD results JSON path",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path.home() / "videos" / "cuts",
        help="Directory containing original videos",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "videos" / "tad-predictions",
        help="Output directory for JSONL files",
    )
    parser.add_argument(
        "--feature-fps",
        type=float,
        default=1.0,
        help="Features per second for time conversion",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return

    convert_mambatad_results(
        args.input, args.video_dir, args.output_dir, args.feature_fps
    )


if __name__ == "__main__":
    main()
