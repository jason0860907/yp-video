"""TAD prediction review router.

Same editing workflow as annotate, but reads from tad-predictions/
instead of rally-pre-annotations/.  Saves to rally-annotations/
so corrected labels feed back into training.
"""

import json
from pathlib import Path
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from yp_video.config import (
    ANNOTATIONS_DIR,
    CUT_R2_CATEGORIES,
    CUTS_BROADCAST_DIR,
    PREDICTIONS_DIR,
    TAD_ANNOTATIONS_FILE,
    VIDEOS_DIR,
    cut_kind_of,
    find_cut,
)
from yp_video.core.jsonl import read_jsonl
from yp_video.web.r2_client import serve_video_or_r2_redirect

router = APIRouter()


# ── Per-video subset + model performance ─────────────────────────────────


_TIOU_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)


def _segment_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def _ap_at_tiou(
    preds: list[tuple[float, float, float]],  # (start, end, conf) sorted by conf desc
    gts: list[tuple[float, float]],
    tiou: float,
) -> float:
    """VOC-style all-points AP at a single tIoU threshold."""
    if not gts:
        return 0.0
    matched = [False] * len(gts)
    tp = [0] * len(preds)
    fp = [0] * len(preds)
    for i, (ps, pe, _) in enumerate(preds):
        best_j, best_iou = -1, tiou
        for j, g in enumerate(gts):
            if matched[j]:
                continue
            iou = _segment_iou((ps, pe), g)
            if iou >= best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0:
            tp[i] = 1
            matched[best_j] = True
        else:
            fp[i] = 1

    tp_cum = 0
    fp_cum = 0
    recalls, precisions = [], []
    for i in range(len(preds)):
        tp_cum += tp[i]
        fp_cum += fp[i]
        recalls.append(tp_cum / len(gts))
        precisions.append(tp_cum / (tp_cum + fp_cum))

    # All-points AP: for each rank, take max precision over the tail
    ap = 0.0
    for i in range(len(recalls)):
        p = max(precisions[i:])
        dr = recalls[i] - (recalls[i - 1] if i > 0 else 0.0)
        ap += p * dr
    return ap


def _load_eval_context() -> tuple[dict[str, str], dict[str, float]]:
    """Return ({stem: subset}, {stem: mAP averaged over tIoU [.3,.4,.5,.6,.7]}).

    Subset comes from volleyball_anno.json (the train/val split). mAP is
    computed directly from each tad-predictions/*.jsonl file vs GT — no
    intermediate pkl needed. The number therefore exactly describes the same
    predictions a user loads when clicking the 🤖 entry.

    Unlike recall, mAP penalises false positives, so it matches what you see
    in the training eval log. Computed for both training and validation
    videos — training mAP is biased high (memorization) but useful for
    spotting outlier failures even within the training set.
    """
    subset: dict[str, str] = {}
    m_ap: dict[str, float] = {}

    if not TAD_ANNOTATIONS_FILE.exists():
        return subset, m_ap
    try:
        gt_data = json.loads(TAD_ANNOTATIONS_FILE.read_text())["database"]
    except Exception:
        return subset, m_ap

    for stem, meta in gt_data.items():
        subset[stem] = meta.get("subset", "")

    if not PREDICTIONS_DIR.exists():
        return subset, m_ap

    for pred_file in PREDICTIONS_DIR.glob("*_annotations.jsonl"):
        stem = pred_file.stem.removesuffix("_annotations")
        meta = gt_data.get(stem)
        if not meta:
            continue
        gts = [(a["segment"][0], a["segment"][1]) for a in meta.get("annotations", [])]
        if not gts:
            continue

        try:
            preds: list[tuple[float, float, float]] = []
            for line in pred_file.read_text().splitlines():
                rec = json.loads(line)
                if "start" in rec and "end" in rec:
                    preds.append((rec["start"], rec["end"], rec.get("confidence", 0.0)))
        except Exception:
            continue

        preds.sort(key=lambda p: p[2], reverse=True)
        aps = [_ap_at_tiou(preds, gts, t) for t in _TIOU_THRESHOLDS]
        m_ap[stem] = sum(aps) / len(aps)

    return subset, m_ap


class Annotation(BaseModel):
    start: float
    end: float
    label: str


class SaveAnnotationsRequest(BaseModel):
    video: str
    duration: float
    annotations: list[Annotation]


def _read_jsonl_as_dict(path: Path) -> dict:
    """Read JSONL and return as {**meta, results: [...]}."""
    meta, records = read_jsonl(path)
    meta["results"] = records
    return meta


@router.get("/results")
def list_results() -> list[dict]:
    """List TAD predictions and reviewed annotations as separate entries.

    When a file exists in both sources (common: predicted, then reviewed), it
    appears twice so the user can load either version for side-by-side review.
    """
    by_source: dict[str, set[str]] = {"annotation": set(), "tad-prediction": set()}

    if PREDICTIONS_DIR.exists():
        for f in PREDICTIONS_DIR.glob("*.jsonl"):
            by_source["tad-prediction"].add(f.name)
    if ANNOTATIONS_DIR.exists():
        for f in ANNOTATIONS_DIR.glob("*.jsonl"):
            by_source["annotation"].add(f.name)

    subset, m_ap = _load_eval_context()

    def _stem(filename: str) -> str:
        return filename.removesuffix(".jsonl").removesuffix("_annotations")

    entries: list[dict] = []
    for s, names in by_source.items():
        for n in names:
            stem = _stem(n)
            e: dict = {"name": n, "source": s}
            if stem in subset and subset[stem]:
                e["subset"] = subset[stem]
            if stem in m_ap:
                e["map"] = m_ap[stem]
            # Tag each entry with the cut kind (broadcast/sideline) so the UI
            # can filter. If the cut isn't on disk anymore default to broadcast
            # to match cut_kind_of's fallback.
            cut_path = find_cut(f"{stem}.mp4")
            e["kind"] = cut_kind_of(cut_path) if cut_path else "broadcast"
            entries.append(e)

    # Sort by filename, then source so the two variants of a file appear together
    entries.sort(key=lambda e: (e["name"], e["source"]))
    return entries


@router.get("/results/{name}")
def get_result(name: str, source: str = "") -> dict:
    """Get result contents.

    When *source* is "tad-prediction" or "annotation", load that specific version.
    Otherwise fall back to legacy preference: annotation first, then prediction.
    """
    # Resolve the (local_dir, r2_category) pair based on requested source
    if source == "tad-prediction":
        search_order = [(PREDICTIONS_DIR, "tad-predictions")]
    elif source == "annotation":
        search_order = [(ANNOTATIONS_DIR, "rally-annotations")]
    else:
        search_order = [(ANNOTATIONS_DIR, "rally-annotations"), (PREDICTIONS_DIR, "tad-predictions")]

    for local_dir, category in search_order:
        path = local_dir / name
        if path.exists() and path.is_file():
            try:
                data = _read_jsonl_as_dict(path)
                data["source"] = category
                return data
            except json.JSONDecodeError:
                raise HTTPException(400, "Invalid JSONL file")

    raise HTTPException(404, "Results file not found")


@router.get("/video/{path:path}")
def stream_video(path: str):
    """Serve a cut video file for playback."""
    decoded_path = unquote(path)
    if decoded_path.startswith("/"):
        video_path = Path(decoded_path)
    else:
        # Search both cut dirs by basename; fall back to the broadcast path
        # so the R2 redirect can still resolve files that exist remotely.
        resolved = find_cut(Path(decoded_path).name)
        video_path = resolved if resolved is not None else CUTS_BROADCAST_DIR / decoded_path
    response = serve_video_or_r2_redirect(video_path, CUT_R2_CATEGORIES)
    if response:
        return response
    raise HTTPException(404, f"Video not found: {video_path}")


@router.post("/annotations")
def save_annotations(req: SaveAnnotationsRequest) -> dict:
    """Save reviewed annotations to rally-annotations/."""
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    video_path = Path(req.video)
    output_name = f"{video_path.stem}_annotations.jsonl"
    output_path = ANNOTATIONS_DIR / output_name

    with open(output_path, "w", encoding="utf-8") as f:
        meta = {"_meta": True, "video": req.video, "duration": req.duration}
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

        for a in req.annotations:
            annotation = {"start": a.start, "end": a.end, "label": a.label}
            f.write(json.dumps(annotation, ensure_ascii=False) + "\n")

    return {"saved": str(output_path), "count": len(req.annotations)}
