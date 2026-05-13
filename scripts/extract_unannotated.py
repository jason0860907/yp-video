"""Extract V-JEPA features only for cuts videos that have no rally-annotation yet."""
from pathlib import Path

import torch

from yp_video.config import (
    ANNOTATIONS_DIR,
    CUTS_DIRS,
    FEATURES_DIR,
)
from yp_video.tad.extract_features import MODEL_CONFIGS, process_directory


def main() -> None:
    model_name = "base"
    cfg = MODEL_CONFIGS[model_name]
    output_dir = FEATURES_DIR / cfg.dir_suffix

    ann_stems = {
        p.name.removesuffix("_annotations.jsonl")
        for p in ANNOTATIONS_DIR.iterdir()
        if p.name.endswith("_annotations.jsonl")
    }

    video_exts = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
    todo: list[str] = []
    for d in CUTS_DIRS:
        if not d.exists():
            continue
        for f in d.iterdir():
            if f.suffix.lower() in video_exts and f.stem not in ann_stems:
                todo.append(f.name)

    if not todo:
        print("Nothing to do — every cut already has a rally-annotation.")
        return

    print(f"Will extract features for {len(todo)} unannotated videos -> {output_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    process_directory(
        list(CUTS_DIRS),
        output_dir,
        device,
        videos=todo,
        model_name=model_name,
    )
    print("Done.")


if __name__ == "__main__":
    main()
