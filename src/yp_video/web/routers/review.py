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
    PREDICTIONS_DIR,
    CUTS_DIR,
    TAD_ANNOTATIONS_FILE,
    VIDEOS_DIR,
)
from yp_video.core.jsonl import read_jsonl
from yp_video.web.r2_client import serve_video_or_r2_redirect

router = APIRouter()


# ── Per-video subset + model performance ─────────────────────────────────


def _load_eval_context() -> tuple[dict[str, str], dict[str, float]]:
    """Return ({stem: subset}, {stem: recall@0.5}).

    Subset comes from volleyball_anno.json (the train/val split). Recall is
    computed directly from each tad-predictions/*.jsonl file vs GT — no
    intermediate pkl needed. The recall value therefore exactly describes
    the same predictions a user loads when clicking the 🤖 entry.

    Restricted to validation videos so the number reflects generalization,
    not memorization.
    """
    subset: dict[str, str] = {}
    recall: dict[str, float] = {}

    if not TAD_ANNOTATIONS_FILE.exists():
        return subset, recall
    try:
        gt_data = json.loads(TAD_ANNOTATIONS_FILE.read_text())["database"]
    except Exception:
        return subset, recall

    for stem, meta in gt_data.items():
        subset[stem] = meta.get("subset", "")

    if not PREDICTIONS_DIR.exists():
        return subset, recall

    for pred_file in PREDICTIONS_DIR.glob("*_annotations.jsonl"):
        stem = pred_file.stem.removesuffix("_annotations")
        meta = gt_data.get(stem)
        if not meta or meta.get("subset") != "validation":
            continue
        gt_segs = [(a["segment"][0], a["segment"][1]) for a in meta.get("annotations", [])]
        if not gt_segs:
            continue

        try:
            lines = pred_file.read_text().splitlines()
            preds = []
            for line in lines[1:]:  # skip _meta line
                rec = json.loads(line)
                if "start" in rec and "end" in rec:
                    preds.append((rec["start"], rec["end"]))
        except Exception:
            continue

        matched = 0
        for gs, ge in gt_segs:
            best = 0.0
            for ps, pe in preds:
                inter = max(0.0, min(ge, pe) - max(gs, ps))
                union = (ge - gs) + (pe - ps) - inter
                if union > 0 and inter / union > best:
                    best = inter / union
            if best >= 0.5:
                matched += 1
        recall[stem] = matched / len(gt_segs)

    return subset, recall


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

    subset, recall = _load_eval_context()

    def _stem(filename: str) -> str:
        return filename.removesuffix(".jsonl").removesuffix("_annotations")

    entries: list[dict] = []
    for s, names in by_source.items():
        for n in names:
            stem = _stem(n)
            e: dict = {"name": n, "source": s}
            if stem in subset and subset[stem]:
                e["subset"] = subset[stem]
            if stem in recall:
                e["recall"] = recall[stem]
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
        video_path = CUTS_DIR / decoded_path
    response = serve_video_or_r2_redirect(video_path, ("cuts",))
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
