"""Export the ball pseudo-labels as an RF-DETR (Roboflow COCO) training set.

``<BALL_DATASETS_DIR>/<name>/{train,valid}/`` each hold JPEG frames and an
``_annotations.coco.json`` with one category, "ball"; ``test`` links to
``valid``. Whole videos are held out (a fixed 1 in VALID_EVERY by name hash),
so the detector is scored on videos it never trained on. The action val set
cannot serve: it is broadcast footage, and these labels are sideline only.

Per video a fixed budget of frames is drawn: a share of hand-marked contacts
(the crowded frames a detector misses) and the rest flight labels.
Neighbouring flight frames are near-duplicates, so the budget, not the label
count, is what sizes the set.
"""

from __future__ import annotations

import json
import os
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from yp_video.ball.pseudo import LABELS_SUFFIX, labels_path
from yp_video.config import BALL_DATASETS_DIR, BALL_PSEUDO_LABELS_DIR
from yp_video.core.jsonl import read_jsonl

FRAMES_PER_VIDEO = 300
VALID_EVERY = 10
CONTACT_SHARE = 1 / 3
# Wider than the detector's 1024 input: storing more pixels buys nothing.
IMAGE_WIDTH = 1280
JPEG_QUALITY = 92
CATEGORY = {"id": 1, "name": "ball", "supercategory": "ball"}
# Below this gap, reading forward beats a seek (a seek decodes from the keyframe).
_SEEK_GAP = 30


def _sample(stem: str, rows: list[dict]) -> list[dict]:
    rng = np.random.default_rng(zlib.crc32(stem.encode()))
    contacts = [r for r in rows if r["source"] == "contact"]
    flights = [r for r in rows if r["source"] == "flight"]
    n_contact = min(len(contacts), round(FRAMES_PER_VIDEO * CONTACT_SHARE))
    n_flight = min(len(flights), FRAMES_PER_VIDEO - n_contact)
    picked = [contacts[i] for i in rng.choice(len(contacts), n_contact, replace=False)]
    picked += [flights[i] for i in rng.choice(len(flights), n_flight, replace=False)]
    return sorted(picked, key=lambda r: r["frame"])


def _export_video(stem: str, image_dir: Path) -> list[dict]:
    """Write one video's sampled frames; return their image+box records."""
    import cv2

    from yp_video.web.r2_client import materialized_cut, resolve_cut

    _meta, rows = read_jsonl(labels_path(stem))
    picked = _sample(stem, rows)
    out = []
    with materialized_cut(resolve_cut(f"{stem}.mp4")) as path:
        cap = cv2.VideoCapture(str(path))
        pos = -1
        for r in picked:
            if not 0 <= r["frame"] - pos <= _SEEK_GAP:
                cap.set(cv2.CAP_PROP_POS_FRAMES, r["frame"])
                pos = r["frame"]
            else:
                while pos < r["frame"]:
                    cap.grab()
                    pos += 1
            ok, img = cap.read()
            pos += 1
            if not ok:
                continue
            h, w = img.shape[:2]
            height = round(h * IMAGE_WIDTH / w)
            img = cv2.resize(img, (IMAGE_WIDTH, height), interpolation=cv2.INTER_AREA)
            name = f"{zlib.crc32(stem.encode()):08x}_{r['frame']:06d}.jpg"
            cv2.imwrite(str(image_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            x0, y0, x1, y1 = r["box"]
            out.append({
                "file_name": name, "width": IMAGE_WIDTH, "height": height,
                "video": stem, "frame": r["frame"], "source": r["source"],
                "bbox": [x0 * IMAGE_WIDTH, y0 * height, (x1 - x0) * IMAGE_WIDTH, (y1 - y0) * height],
            })
        cap.release()
    return out


def _write_split(split_dir: Path, records: list[dict]) -> None:
    images, annotations = [], []
    for i, r in enumerate(records, start=1):
        images.append({"id": i, "file_name": r["file_name"], "width": r["width"], "height": r["height"],
                       "video": r["video"], "frame": r["frame"], "source": r["source"]})
        w, h = r["bbox"][2], r["bbox"][3]
        annotations.append({"id": i, "image_id": i, "category_id": CATEGORY["id"],
                            "bbox": [round(v, 2) for v in r["bbox"]], "area": round(w * h, 2), "iscrowd": 0})
    payload = {"images": images, "annotations": annotations, "categories": [CATEGORY]}
    (split_dir / "_annotations.coco.json").write_text(json.dumps(payload))


def export(name: str, workers: int = 6) -> dict:
    """Export every pseudo-labelled video into ``BALL_DATASETS_DIR/name``."""
    root = BALL_DATASETS_DIR / name
    if root.exists():
        raise FileExistsError(root)
    stems = sorted(p.name.removesuffix(LABELS_SUFFIX) for p in BALL_PSEUDO_LABELS_DIR.glob(f"*{LABELS_SUFFIX}"))
    split_of = {s: "valid" if zlib.crc32(s.encode()) % VALID_EVERY == 0 else "train" for s in stems}
    for split in ("train", "valid"):
        (root / split).mkdir(parents=True)
    records: dict[str, list[dict]] = {"train": [], "valid": []}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {s: pool.submit(_export_video, s, root / split_of[s]) for s in stems}
        for i, (stem, fut) in enumerate(futures.items(), start=1):
            records[split_of[stem]] += fut.result()
            print(f"[{i}/{len(stems)}] {stem}", flush=True)
    for split, recs in records.items():
        _write_split(root / split, recs)
    os.symlink("valid", root / "test")
    summary = {
        split: {"videos": len({r["video"] for r in recs}), "images": len(recs),
                "contact": sum(r["source"] == "contact" for r in recs)}
        for split, recs in records.items()
    }
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    return summary


if __name__ == "__main__":
    import sys

    print(export(sys.argv[1]))
