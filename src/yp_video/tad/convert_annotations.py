"""Convert annotator JSONL annotations to ActionFormer format.

Annotator JSONL format:
    Line 1: {"_meta": true, "video": "path/to/video.mp4", "duration": 1800.5}
    Line 2+: {"start": 10.5, "end": 25.3, "label": "rally"}

ActionFormer format (JSON):
{
    "database": {
        "video_name": {
            "subset": "training",  # or "validation"
            "duration": 1800.5,
            "fps": 30,
            "feature_fps": 1,
            "annotations": [
                {"segment": [10.5, 25.3], "label": "rally", "label_id": 0}
            ]
        }
    }
}
"""

import argparse
import json
from pathlib import Path

import numpy as np

from yp_video.config import TAD_ANNOTATIONS_FILE, TAD_FEATURES_DIR
from yp_video.core.jsonl import read_jsonl


def get_video_fps(video_path: Path) -> float:
    """Get video FPS using OpenCV."""
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps if fps > 0 else 30.0
    except Exception:
        return 30.0


def convert_annotations(
    annotations_dirs: list[Path] | Path,
    features_dir: Path,
    output_path: Path,
    train_ratio: float = 0.8,
    videos: list[str] | None = None,
) -> dict:
    """Convert JSONL annotations to ActionFormer format.

    Args:
        annotations_dirs: Directory or list of directories containing
            *_annotations.jsonl files.  When multiple directories are given,
            earlier directories take priority (per video stem).
        features_dir: Directory containing extracted features (.npy)
        output_path: Output path for ActionFormer JSON
        train_ratio: Ratio of videos to use for training
        videos: Optional list of video filenames to include (None = all)

    Returns:
        ActionFormer format dictionary
    """
    if isinstance(annotations_dirs, Path):
        annotations_dirs = [annotations_dirs]

    # Collect annotation files, earlier dirs take priority per video stem
    seen_stems: set[str] = set()
    jsonl_files: list[Path] = []
    video_set = {Path(v).stem for v in videos} if videos else None

    for d in annotations_dirs:
        if not d.exists():
            continue
        for f in sorted(d.glob("*_annotations.jsonl")):
            stem = f.stem.removesuffix("_annotations")
            if stem in seen_stems:
                continue
            if video_set and stem not in video_set:
                continue
            seen_stems.add(stem)
            jsonl_files.append(f)

    jsonl_files.sort(key=lambda f: f.name)
    print(f"Found {len(jsonl_files)} annotation files")

    database = {}
    label_set = set()

    for jsonl_path in jsonl_files:
        meta, annotations = read_jsonl(jsonl_path)
        if not meta or not annotations:
            print(f"Skipping empty file: {jsonl_path.name}")
            continue

        # Extract video name from the original video path
        video_path = Path(meta.get("video", ""))
        video_name = video_path.stem

        # Check if features exist
        feature_path = features_dir / f"{video_name}.npy"
        if not feature_path.exists():
            print(f"Warning: Features not found for {video_name}, skipping")
            continue

        # Load features to get temporal dimension
        features = np.load(feature_path)
        num_features = features.shape[0]

        # Calculate feature FPS (features per second)
        duration = meta.get("duration", 0)
        if duration > 0:
            feature_fps = num_features / duration
        else:
            feature_fps = 1.0

        # Get video FPS
        if video_path.exists():
            video_fps = get_video_fps(video_path)
        else:
            video_fps = 30.0

        # Convert annotations
        anno_list = []
        for ann in annotations:
            label = ann.get("label", "rally")
            label_set.add(label)
            anno_list.append(
                {"segment": [ann["start"], ann["end"]], "label": label, "label_id": 0}
            )

        # Calculate frame count (required by ActionFormer)
        frame_count = int(duration * video_fps)

        database[video_name] = {
            "subset": "training",  # Will be updated later
            "duration": duration,
            "fps": video_fps,
            "frame": frame_count,
            "feature_fps": feature_fps,
            "feature_frame": num_features,
            "annotations": anno_list,
        }

    # Split into train/validation (shuffled so each league appears in both)
    import random
    video_names = list(database.keys())
    random.Random(42).shuffle(video_names)
    n_train = max(1, int(len(video_names) * train_ratio))
    train_videos = set(video_names[:n_train])

    for name in database:
        database[name]["subset"] = "training" if name in train_videos else "validation"

    # Create final structure
    anno_data = {"database": database}

    # Save annotation JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(anno_data, f, indent=2, ensure_ascii=False)

    # Save category index file (one class per line, order determines class ID)
    category_path = output_path.parent / "category_idx.txt"
    with open(category_path, "w") as f:
        for label in sorted(label_set):
            f.write(f"{label}\n")

    print(f"Converted {len(database)} videos")
    print(f"Labels found: {sorted(label_set)}")
    print(f"Train: {n_train}, Val: {len(video_names) - n_train}")
    print(f"Saved to: {output_path}")
    print(f"Category map: {category_path}")

    return anno_data


def main():
    parser = argparse.ArgumentParser(
        description="Convert annotator JSONL to ActionFormer format"
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path.home() / "videos" / "rally-pre-annotations",
        help="Directory containing *_annotations.jsonl files",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=TAD_FEATURES_DIR,
        help="Directory containing extracted features (.npy)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=TAD_ANNOTATIONS_FILE,
        help="Output ActionFormer JSON path",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of videos to use for training",
    )
    args = parser.parse_args()

    convert_annotations(args.annotations, args.features, args.output, args.train_ratio)


if __name__ == "__main__":
    main()
