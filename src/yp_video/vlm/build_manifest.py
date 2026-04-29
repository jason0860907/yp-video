"""Build a (clip_window → in-rally label) manifest for VLM fine-tuning.

For each cut video in `cuts/`, slide a fixed-length window (default 6s, stride 2s)
and label each window by whether it overlaps a ground-truth rally segment from
`rally-annotations/`.  The output is a JSONL where each line is one window:

    {"video": "cuts/X_set1.mp4", "start": 12.0, "end": 18.0,
     "label": "rally" | "non_rally",
     "source": "tpvl", "subset": "training" | "validation"}

The split mirrors `tad/convert_annotations.py` — same `_source_key` buckets,
same stratification — so an apples-to-apples comparison with the TAD pipeline
is possible.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from yp_video.config import (
    ANNOTATIONS_DIR,
    CUTS_DIR,
    VLM_MANIFEST_FILE,
    PRE_ANNOTATIONS_DIR,
)
from yp_video.core.jsonl import read_jsonl
from yp_video.tad.convert_annotations import _match_key, _source_key


def _load_rally_segments(stem: str) -> list[tuple[float, float]] | None:
    """Return list of (start, end) rally seconds for a video stem, or None if
    no annotation exists. Prefer manual annotation over pre-annotation."""
    for d in (ANNOTATIONS_DIR, PRE_ANNOTATIONS_DIR):
        path = d / f"{stem}_annotations.jsonl"
        if path.exists():
            _, records = read_jsonl(path)
            return [
                (float(r.get("start", r.get("start_time", 0))),
                 float(r.get("end", r.get("end_time", 0))))
                for r in records
                if (r.get("label", "rally") == "rally") or r.get("in_rally")
            ]
    return None


def _windows(duration: float, win: float, stride: float):
    t = 0.0
    while t + win <= duration + 1e-6:
        yield t, t + win
        t += stride


def _iou(a_start: float, a_end: float, segs: list[tuple[float, float]]) -> float:
    """Max IoU between [a_start, a_end] and any segment in segs."""
    best = 0.0
    a_len = a_end - a_start
    for s, e in segs:
        inter = max(0.0, min(a_end, e) - max(a_start, s))
        if inter <= 0:
            continue
        union = a_len + (e - s) - inter
        if union > 0:
            best = max(best, inter / union)
    return best


def _stratified_split(stems: list[str], train_ratio: float, seed: int) -> set[str]:
    """Group videos by match (no leakage between sets of the same match) and
    split matches stratified by source. Returns the *train* stem set."""
    matches: dict[str, list[str]] = defaultdict(list)
    for s in stems:
        matches[_match_key(s)].append(s)

    by_src: dict[str, list[str]] = defaultdict(list)
    for mkey, vids in matches.items():
        by_src[_source_key(vids[0])].append(mkey)

    rng = random.Random(seed)
    train: set[str] = set()
    print(f"  {'source':<14s} {'train M':>8s} {'val M':>6s}  {'train V':>8s} {'val V':>6s}")
    for src, mkeys in sorted(by_src.items()):
        mkeys = sorted(mkeys)
        rng.shuffle(mkeys)
        n_train = max(1, round(len(mkeys) * train_ratio))
        train_m, val_m = mkeys[:n_train], mkeys[n_train:]
        train_vids = [v for m in train_m for v in matches[m]]
        val_vids = [v for m in val_m for v in matches[m]]
        train.update(train_vids)
        print(f"  {src:<14s} {len(train_m):>8d} {len(val_m):>6d}  "
              f"{len(train_vids):>8d} {len(val_vids):>6d}")
    return train


def build_manifest(
    output_path: Path = VLM_MANIFEST_FILE,
    window: float = 6.0,
    stride: float = 2.0,
    iou_threshold: float = 0.5,
    train_ratio: float = 0.8,
    seed: int = 42,
    only_annotated: bool = True,
) -> dict:
    """Scan cuts/ + rally-annotations/, write window-level JSONL manifest.

    only_annotated: if True (default), skip videos without manual or
        pre-annotation. Set False to include cuts as 'non_rally' implicitly
        (but that's risky — better to annotate or skip).
    """
    if not CUTS_DIR.exists():
        raise FileNotFoundError(f"Cuts directory missing: {CUTS_DIR}")

    stems_with_anno: list[str] = []
    segments: dict[str, list[tuple[float, float]]] = {}
    durations: dict[str, float] = {}

    for video_path in sorted(CUTS_DIR.glob("*.mp4")):
        stem = video_path.stem
        segs = _load_rally_segments(stem)
        if segs is None and only_annotated:
            continue
        from yp_video.core.ffmpeg import get_video_duration
        try:
            dur = get_video_duration(video_path)
        except Exception:
            continue
        stems_with_anno.append(stem)
        segments[stem] = segs or []
        durations[stem] = dur

    print(f"\nFound {len(stems_with_anno)} annotated cuts.")
    print("\nStratified split (matches | videos):")
    train_stems = _stratified_split(stems_with_anno, train_ratio, seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_pos = n_neg = 0
    by_src_split: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"pos": 0, "neg": 0}
    )
    with open(output_path, "w") as f:
        for stem in stems_with_anno:
            subset = "training" if stem in train_stems else "validation"
            src = _source_key(stem)
            for w_start, w_end in _windows(durations[stem], window, stride):
                iou = _iou(w_start, w_end, segments[stem])
                label = "rally" if iou >= iou_threshold else "non_rally"
                rec = {
                    "video": str(CUTS_DIR / f"{stem}.mp4"),
                    "stem": stem,
                    "start": round(w_start, 3),
                    "end": round(w_end, 3),
                    "label": label,
                    "source": src,
                    "subset": subset,
                    "iou": round(iou, 4),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if label == "rally":
                    n_pos += 1
                    by_src_split[(src, subset)]["pos"] += 1
                else:
                    n_neg += 1
                    by_src_split[(src, subset)]["neg"] += 1

    print(f"\nWrote {n_pos + n_neg} windows ({n_pos} rally, {n_neg} non_rally) "
          f"to {output_path}")
    print(f"\nPer-source per-split window counts:")
    print(f"  {'source':<14s} {'split':<6s} {'rally':>7s} {'non':>7s} {'%pos':>6s}")
    for (src, split), c in sorted(by_src_split.items()):
        pct = 100 * c["pos"] / (c["pos"] + c["neg"]) if (c["pos"] + c["neg"]) else 0
        print(f"  {src:<14s} {split:<6s} {c['pos']:>7d} {c['neg']:>7d} {pct:>5.1f}%")

    return {
        "n_train_videos": len(train_stems),
        "n_val_videos": len(stems_with_anno) - len(train_stems),
        "n_windows": n_pos + n_neg,
        "n_rally": n_pos,
        "n_non_rally": n_neg,
        "output_path": str(output_path),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", type=Path, default=VLM_MANIFEST_FILE)
    p.add_argument("--window", type=float, default=6.0)
    p.add_argument("--stride", type=float, default=2.0)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    build_manifest(
        output_path=args.output,
        window=args.window,
        stride=args.stride,
        iou_threshold=args.iou_threshold,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
