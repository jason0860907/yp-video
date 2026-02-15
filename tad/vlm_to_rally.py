"""Convert VLM clip detection results to rally annotations.

VLM detection format (JSONL):
    Line 1: {"_meta": true, "video": "...", "clip_duration": 6.0, ...}
    Line 2+: {"start_time": 0.0, "end_time": 6.0, "has_volleyball": true, "shot_type": "full_court", ...}

Rally annotation format (JSONL):
    Line 1: {"_meta": true, "video": "...", "duration": 1800.5}
    Line 2+: {"start": 10.5, "end": 25.3, "label": "rally"}

Rally detection logic:
- A rally = consecutive clips with has_volleyball=true AND shot_type="full_court"
- Merge overlapping/adjacent clips into single rally segments
- Filter out rallies shorter than min_duration
"""

import argparse
import json
from pathlib import Path


def read_vlm_jsonl(path: Path) -> tuple[dict, list[dict]]:
    """Read VLM detection JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        return {}, []

    meta = json.loads(lines[0])
    meta.pop("_meta", None)

    clips = []
    for line in lines[1:]:
        line = line.strip()
        if line:
            clips.append(json.loads(line))

    return meta, clips


def detect_rallies(
    clips: list[dict],
    min_duration: float = 3.0,
    max_gap: float = 6.0,
    require_full_court: bool = True,
) -> list[dict]:
    """Detect rallies from VLM clip detections.

    Args:
        clips: List of clip detection dicts
        min_duration: Minimum rally duration in seconds
        max_gap: Maximum gap between clips to merge (seconds)
        require_full_court: Only count full_court shots as rally

    Returns:
        List of rally annotations with start, end, label
    """
    # Filter clips that are part of a rally
    rally_clips = []
    for clip in clips:
        is_rally = clip.get("has_volleyball", False)
        if require_full_court:
            is_rally = is_rally and clip.get("shot_type") == "full_court"
        if is_rally:
            rally_clips.append(clip)

    if not rally_clips:
        return []

    # Sort by start time
    rally_clips.sort(key=lambda x: x["start_time"])

    # Merge overlapping/adjacent clips into rally segments
    rallies = []
    current_start = rally_clips[0]["start_time"]
    current_end = rally_clips[0]["end_time"]

    for clip in rally_clips[1:]:
        clip_start = clip["start_time"]
        clip_end = clip["end_time"]

        # Check if this clip can be merged with current rally
        if clip_start <= current_end + max_gap:
            # Extend current rally
            current_end = max(current_end, clip_end)
        else:
            # Save current rally and start new one
            if current_end - current_start >= min_duration:
                rallies.append(
                    {
                        "start": round(current_start, 2),
                        "end": round(current_end, 2),
                        "label": "rally",
                    }
                )
            current_start = clip_start
            current_end = clip_end

    # Don't forget the last rally
    if current_end - current_start >= min_duration:
        rallies.append(
            {
                "start": round(current_start, 2),
                "end": round(current_end, 2),
                "label": "rally",
            }
        )

    return rallies


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


def convert_vlm_to_rally(
    input_path: Path,
    output_path: Path,
    min_duration: float = 3.0,
    max_gap: float = 6.0,
) -> int:
    """Convert a single VLM JSONL to rally annotation JSONL.

    Returns:
        Number of rallies detected
    """
    meta, clips = read_vlm_jsonl(input_path)

    if not clips:
        print(f"No clips found in {input_path.name}")
        return 0

    # Get video path and duration
    video_path = Path(meta.get("video", ""))
    if video_path.exists():
        duration = get_video_duration(video_path)
    else:
        # Estimate from clips
        duration = max(c.get("end_time", 0) for c in clips)

    # Detect rallies
    rallies = detect_rallies(clips, min_duration, max_gap)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        # Write metadata
        rally_meta = {"_meta": True, "video": str(video_path), "duration": duration}
        f.write(json.dumps(rally_meta, ensure_ascii=False) + "\n")

        # Write rallies
        for rally in rallies:
            f.write(json.dumps(rally, ensure_ascii=False) + "\n")

    return len(rallies)


def convert_directory(
    input_dir: Path,
    output_dir: Path,
    min_duration: float = 3.0,
    max_gap: float = 6.0,
):
    """Convert all VLM JSONL files in a directory."""
    jsonl_files = sorted(input_dir.glob("*.jsonl"))

    # Filter out files that are already rally annotations
    vlm_files = []
    for f in jsonl_files:
        if "_annotations" in f.stem:
            continue  # Skip already converted files
        vlm_files.append(f)

    if not vlm_files:
        print(f"No VLM detection files found in {input_dir}")
        return

    print(f"Found {len(vlm_files)} VLM detection files")

    total_rallies = 0
    for vlm_path in vlm_files:
        output_name = f"{vlm_path.stem}_annotations.jsonl"
        output_path = output_dir / output_name

        n_rallies = convert_vlm_to_rally(vlm_path, output_path, min_duration, max_gap)
        total_rallies += n_rallies
        print(f"  {vlm_path.name}: {n_rallies} rallies")

    print(f"\nTotal: {total_rallies} rallies from {len(vlm_files)} videos")


def main():
    parser = argparse.ArgumentParser(
        description="Convert VLM clip detection to rally annotations"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path.home() / "videos" / "seg-annotations",
        help="Input directory with VLM JSONL files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.home() / "videos" / "rally-pre-annotations",
        help="Output directory for rally annotations",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=3.0,
        help="Minimum rally duration in seconds",
    )
    parser.add_argument(
        "--max-gap",
        type=float,
        default=6.0,
        help="Maximum gap between clips to merge",
    )
    args = parser.parse_args()

    convert_directory(args.input, args.output, args.min_duration, args.max_gap)


if __name__ == "__main__":
    main()
