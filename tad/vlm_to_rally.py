"""Convert VLM clip detection results to rally annotations.

VLM detection format (JSONL):
    Line 1: {"_meta": true, "video": "...", "clip_duration": 6.0, ...}
    Line 2+: {"start_time": 0.0, "end_time": 6.0, "in_rally": true, "shot_type": "full_court", ...}

Rally annotation format (JSONL):
    Line 1: {"_meta": true, "video": "...", "duration": 1800.5}
    Line 2+: {"start": 10.5, "end": 25.3, "label": "rally"}

Rally detection logic:
- Build per-slot rally score from overlapping clips (voting)
- Smooth scores with 3-slot moving average to tolerate isolated errors
- Threshold smoothed scores to get rally regions
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
    clip_duration: float = 6.0,
    slide_interval: float = 3.0,
    min_duration: float = 3.0,
    min_score: float = 0.5,
    require_full_court: bool = True,
) -> list[dict]:
    """Detect rallies using smoothed per-slot voting from overlapping clips.

    Instead of merging individual rally clips by time gap, this builds a
    per-slot rally score from all overlapping clips, smooths it with a
    moving average, and thresholds to find rally regions.

    A single misclassified clip can only contribute 0.5 to its slot score
    (since each slot is covered by ~2 overlapping clips). After smoothing
    over 3 slots, an isolated false positive stays below the threshold,
    preventing it from bridging two separate rallies.

    Args:
        clips: List of clip detection dicts
        clip_duration: Duration of each clip in seconds
        slide_interval: Sliding window interval in seconds
        min_duration: Minimum rally duration in seconds
        min_score: Minimum smoothed score to count as rally (0-1)
        require_full_court: Only count full_court shots as rally

    Returns:
        List of rally annotations with start, end, label
    """
    if not clips:
        return []

    clips_sorted = sorted(clips, key=lambda x: x["start_time"])

    # Build time slots at slide_interval resolution
    t_min = clips_sorted[0]["start_time"]
    t_max = max(c["end_time"] for c in clips_sorted)

    slot_times: list[float] = []
    t = t_min
    while t < t_max:
        slot_times.append(t)
        t += slide_interval

    if not slot_times:
        return []

    # Compute raw rally score per slot (ratio of rally clips among covering clips)
    raw_scores: list[float] = []
    for t in slot_times:
        rally_votes = 0
        total_votes = 0
        for c in clips_sorted:
            if c["start_time"] <= t < c["end_time"]:
                total_votes += 1
                is_rally = c.get("in_rally", False)
                if require_full_court:
                    is_rally = is_rally and c.get("shot_type") == "full_court"
                if is_rally:
                    rally_votes += 1
        raw_scores.append(rally_votes / total_votes if total_votes > 0 else 0.0)

    # Smooth with 3-slot moving average
    n = len(raw_scores)
    smoothed = []
    for i in range(n):
        window = raw_scores[max(0, i - 1) : i + 2]
        smoothed.append(sum(window) / len(window))

    # Threshold to get binary rally signal, then merge consecutive rally slots
    rallies: list[dict] = []
    current_start: float | None = None
    current_end: float = 0.0

    for t, score in zip(slot_times, smoothed):
        if score >= min_score:
            if current_start is None:
                current_start = t
            current_end = t + slide_interval
        else:
            if current_start is not None:
                if current_end - current_start >= min_duration:
                    rallies.append({
                        "start": round(current_start, 2),
                        "end": round(current_end, 2),
                        "label": "rally",
                    })
                current_start = None

    # Last segment
    if current_start is not None:
        if current_end - current_start >= min_duration:
            rallies.append({
                "start": round(current_start, 2),
                "end": round(current_end, 2),
                "label": "rally",
            })

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
    min_score: float = 0.5,
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

    # Detect rallies using sliding window params from VLM metadata
    clip_duration = meta.get("clip_duration", 6.0)
    slide_interval = meta.get("slide_interval", 3.0)
    rallies = detect_rallies(clips, clip_duration, slide_interval, min_duration, min_score)

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
    min_score: float = 0.5,
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

        n_rallies = convert_vlm_to_rally(vlm_path, output_path, min_duration, min_score)
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
        "--min-score",
        type=float,
        default=0.5,
        help="Minimum smoothed rally score to count as rally (0-1)",
    )
    args = parser.parse_args()

    convert_directory(args.input, args.output, args.min_duration, args.min_score)


if __name__ == "__main__":
    main()
