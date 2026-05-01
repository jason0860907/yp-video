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
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from yp_video.core.sampling import get_fps as get_video_fps

from yp_video.config import TAD_ANNOTATIONS_FILE, TAD_FEATURES_DIR
from yp_video.core.jsonl import read_jsonl


# ── Stratified split helpers ───────────────────────────────────────────


def _match_key(video_name: str) -> str:
    """Strip _setN suffix so sets of the same match stay together (no leakage)."""
    return re.sub(r"_set\d+$", "", video_name)


def _source_key(name: str) -> str:
    """Bucket videos by broadcast source so each style appears in train and val."""
    n_lower = name.lower()
    if "vnl" in n_lower:
        return "vnl"
    if "world champs" in n_lower or "u19" in n_lower:
        return "u19"
    if "u22" in n_lower or "cev" in n_lower:
        return "cev_u22"
    if any(k in n_lower for k in (
        "sv league", "svl", "sv.league", "suntory", "bluteon", "stings",
        "wolfdogs", "sunbirds", "jtekt", "toray", "phitsanulok", "sakai",
        "champions crowned", "final - stings",
    )):
        return "svl_japan"
    if "企業" in name or "甲級" in name:
        return "enterprise"
    if "tpvl" in n_lower or name.startswith("2025-") or any(
        team in name for team in ("臺北伊斯特", "臺中連莊", "桃園雲豹飛將", "台鋼天鷹", "台中連莊")
    ):
        return "tpvl"
    return "other"


def _stratified_split(video_names: list[str], train_ratio: float, seed: int = 42) -> set[str]:
    """Group videos by match, then split matches stratified by source.

    Guarantees:
    1. All sets of one match end up in the same subset (no info leakage)
    2. Each broadcast source has both training and validation representation

    Returns the set of training video names.
    """
    matches: dict[str, list[str]] = defaultdict(list)
    for v in video_names:
        matches[_match_key(v)].append(v)

    by_source: dict[str, list[str]] = defaultdict(list)
    for mkey, vids in matches.items():
        by_source[_source_key(vids[0])].append(mkey)

    rng = random.Random(seed)
    train: set[str] = set()
    summary: list[tuple[str, int, int, int, int]] = []
    for src, mkeys in sorted(by_source.items()):
        mkeys = sorted(mkeys)
        rng.shuffle(mkeys)
        n_train = max(1, int(len(mkeys) * train_ratio)) if len(mkeys) > 1 else len(mkeys)
        train_matches = mkeys[:n_train]
        val_matches = mkeys[n_train:]
        train_vids = sum((matches[m] for m in train_matches), [])
        val_vids = sum((matches[m] for m in val_matches), [])
        train.update(train_vids)
        summary.append((src, len(train_matches), len(val_matches), len(train_vids), len(val_vids)))

    print("Stratified split per source (matches | videos):")
    print(f"  {'source':<14s} {'train M':>8s} {'val M':>6s}  {'train V':>8s} {'val V':>6s}")
    for src, tm, vm, tv, vv in summary:
        print(f"  {src:<14s} {tm:>8d} {vm:>6d}  {tv:>8d} {vv:>6d}")

    return train


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

    # Stratified split: group by match (all sets together), stratify by source
    train_videos = _stratified_split(list(database.keys()), train_ratio, seed=42)
    n_train = len(train_videos)

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
    print(f"Train: {n_train}, Val: {len(database) - n_train}")
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
