"""Per-video ReID extraction: action events → person crops → embeddings.

Writes, per video:
  player-reid/embeddings/<stem>_reid.jsonl   header + one record per event
  player-reid/crops/<stem>/<event_id>.jpg    the associated person crop

Records keep the association outcome (ok / multi / miss) so downstream
matching and the UI can treat ambiguous events differently.

Every record also stores ``detections`` — ALL person boxes the detector found
on that frame, unfiltered by the association heuristic. The labeling UI needs
them so the user can re-point an event at the right person when the heuristic
picked the wrong one; those manual picks (identity.actor_fixes) are replayed
here on re-extraction and stashed alongside the auto pick (``auto_box``), so
the auto/manual disagreement set is preserved as future association training
data.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

from yp_video.config import (
    ACTION_ANNOTATIONS_DIR,
    ACTION_PRE_ANNOTATIONS_DIR,
    PLAYER_REID_DIR,
)
from yp_video.core.jsonl import read_jsonl, write_jsonl
from yp_video.reid.detector import (
    DEFAULT_KEYPOINT_SOURCE,
    DETECTOR_NAME,
    KEYPOINT_SOURCES,
    PersonBox,
    associate,
    build_keypoint_sources,
)
from yp_video.reid.embedder import EMBEDDER_WEIGHTS, build_embedders

EMBEDDINGS_DIR = PLAYER_REID_DIR / "embeddings"
CROPS_DIR = PLAYER_REID_DIR / "crops"

# Actions that are not performed BY a player — "score" marks where the ball
# lands, so there is nobody to re-identify at that point.
SKIP_LABELS = frozenset({"score"})

# One instance per process: the models stay loaded across jobs.
_keypoint_sources = build_keypoint_sources()
_embedders = build_embedders()

ProgressFn = Callable[[int, int, str], None]


def reid_path(stem: str) -> Path:
    return EMBEDDINGS_DIR / f"{stem}_reid.jsonl"


def crop_dir(stem: str) -> Path:
    return CROPS_DIR / stem


def action_annotation_path(stem: str) -> Path | None:
    """Manual action annotations win over pre-annotations."""
    for directory in (ACTION_ANNOTATIONS_DIR, ACTION_PRE_ANNOTATIONS_DIR):
        path = directory / f"{stem}_actions.jsonl"
        if path.exists():
            return path
    return None


def load_events(stem: str) -> list[dict]:
    """Visible action events with a location, sorted by frame."""
    path = action_annotation_path(stem)
    if path is None:
        return []
    _meta, rows = read_jsonl(path)
    events = [
        r for r in rows
        if r.get("visible", True)
        and r.get("xy")
        and r.get("frame") is not None
        and r.get("label") not in SKIP_LABELS
    ]
    events.sort(key=lambda e: e["frame"])
    return events


def _clamp_box(box: tuple[float, float, float, float], w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(w, int(x1)), min(h, int(y1))
    return x0, y0, x1, y1


def _display_box(person: PersonBox, x: float, y: float, w: int, h: int) -> tuple[int, int, int, int]:
    """The union of the person box, ALL predicted keypoints (regardless of
    confidence), and the contact point (the ball), plus 4% margin.

    Keypoints join the union so a fully extended limb (spike, jump serve) is
    never cropped off; the detector box stays in it because keypoints mark
    joints, not extremities — eyes/ankles alone would cut the scalp and feet.
    The saved crop and the video overlay both use this box.
    """
    x0, y0, x1, y1 = person.xyxy
    ux0, uy0, ux1, uy1 = min(x0, x), min(y0, y), max(x1, x), max(y1, y)
    if person.keypoints is not None:
        for px, py in person.keypoints:
            ux0, uy0 = min(ux0, float(px)), min(uy0, float(py))
            ux1, uy1 = max(ux1, float(px)), max(uy1, float(py))
    mx, my = 0.04 * (ux1 - ux0) + 4, 0.04 * (uy1 - uy0) + 4
    return _clamp_box((ux0 - mx, uy0 - my, ux1 + mx, uy1 + my), w, h)


def _serialize_detections(boxes: list[PersonBox], w: int, h: int) -> list[dict]:
    """All person detections of a frame as jsonl-friendly dicts, best first."""
    out = []
    for b in sorted(boxes, key=lambda b: -b.score):
        x0, y0, x1, y1 = _clamp_box(b.xyxy, w, h)
        d: dict = {"box": [x0, y0, x1, y1], "score": round(float(b.score), 3)}
        if b.keypoints is not None and b.keypoint_conf is not None:
            d["keypoints"] = [
                [round(float(px), 1), round(float(py), 1), round(float(c), 2)]
                for (px, py), c in zip(b.keypoints, b.keypoint_conf)
            ]
        out.append(d)
    return out


def _person_from_detection(d: dict) -> PersonBox:
    import numpy as np

    kps = d.get("keypoints")
    return PersonBox(
        xyxy=tuple(d["box"]),
        score=float(d.get("score") or 0.0),
        keypoints=np.array([[k[0], k[1]] for k in kps], dtype=np.float32) if kps else None,
        keypoint_conf=np.array([k[2] for k in kps], dtype=np.float32) if kps else None,
    )


def _crop_prompt(record: dict, person: PersonBox) -> dict:
    """Keypoint prompts for promptable embedders (KPR).

    The chosen person's keypoints form the positive prompt and every other
    detected person's keypoints that fall inside the saved crop become
    negatives — all in crop-pixel coordinates of the display crop.
    """
    import numpy as np

    dx0, dy0, dx1, dy1 = record["box"]
    cw, ch = max(dx1 - dx0, 1), max(dy1 - dy0, 1)

    # KPR's transform stack validates that every keypoint lies inside the
    # crop, so clamp — points pushed out by frame-edge clipping land on the
    # border, which is where the person is cut off anyway.
    def clamped(kx: float, ky: float, c: float) -> list[float]:
        return [min(max(kx, 0.0), cw - 1.0), min(max(ky, 0.0), ch - 1.0), c]

    pos = None
    if record.get("keypoints"):
        # Stored crop-relative (0-1) → crop pixels.
        pos = np.array([clamped(k[0] * cw, k[1] * ch, k[2]) for k in record["keypoints"]], dtype=np.float32)
    negs = []
    for d in record.get("detections") or []:
        kps = d.get("keypoints")
        if not kps or _iou(d["box"], list(person.xyxy)) > 0.8:
            continue  # no keypoints, or this IS the chosen person
        pts, any_inside = [], False
        for px, py, c in kps:
            kx, ky = px - dx0, py - dy0
            inside = 0 <= kx <= cw and 0 <= ky <= ch and c > 0
            any_inside = any_inside or inside
            pts.append(clamped(kx, ky, c if inside else 0.0))
        if any_inside:
            negs.append(pts)
    return {
        "keypoints_xyc": pos,
        "negative_kps": np.array(negs, dtype=np.float32) if negs else None,
    }


def _iou(a: list[float], b: list[float]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter or 1.0)


# A fix box must overlap a fresh detection this much to snap onto it (and
# inherit its keypoints); below that the fix box is embedded as drawn.
FIX_SNAP_IOU = 0.5


def _snap_to_detection(detections: list[dict], box: list[float]) -> PersonBox | None:
    """The stored detection a fix box refers to, matched by IoU."""
    best, best_iou = None, FIX_SNAP_IOU
    for d in detections:
        iou = _iou(d["box"], box)
        if iou >= best_iou:
            best, best_iou = d, iou
    return _person_from_detection(best) if best else None


def _attach_person(
    record: dict, frame, person: PersonBox, x: float, y: float, w: int, h: int, out_crops: Path, *, suffix: str = ""
):
    """Point ``record`` at ``person``: write the display crop and fill
    box/score/crop/keypoints. Returns the crop image (None if degenerate)."""
    import cv2

    x0, y0, x1, y1 = _clamp_box(person.xyxy, w, h)
    if x1 <= x0 or y1 <= y0:
        return None
    dx0, dy0, dx1, dy1 = _display_box(person, x, y, w, h)
    crop = frame[dy0:dy1, dx0:dx1]
    crop_file = out_crops / f"{record['id']}{suffix}.jpg"
    out_crops.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(crop_file), crop)
    # Keypoints ship as crop-relative data; the UI draws the skeleton as a
    # toggleable overlay, so the jpg stays raw — identical to what the
    # embedder sees.
    keypoints = None
    if person.keypoints is not None and person.keypoint_conf is not None:
        cw, ch = max(dx1 - dx0, 1), max(dy1 - dy0, 1)
        keypoints = [
            [round(float(px - dx0) / cw, 4), round(float(py - dy0) / ch, 4), round(float(c), 2)]
            for (px, py), c in zip(person.keypoints, person.keypoint_conf)
        ]
    record.update(
        box=[dx0, dy0, dx1, dy1],
        score=person.score,
        crop=crop_file.name,
        keypoints=keypoints,
    )
    return crop


def extract_video(
    video_path: Path, *, keypoints: str = DEFAULT_KEYPOINT_SOURCE, on_progress: ProgressFn | None = None
) -> dict:
    """Run the full detect → associate → crop → embed pass for one video.

    Detection is always RF-DETR; ``keypoints`` picks who estimates the
    skeletons on those boxes (see detector.build_keypoint_sources). Returns
    the summary counts also written to the jsonl header. Synchronous and
    GPU-bound — callers run it in an executor.
    """
    import cv2

    stem = video_path.stem
    events = load_events(stem)
    if not events:
        raise ValueError(f"No action events for {video_path.name}")

    # Deferred: identity imports this module at load time.
    from yp_video.reid.identity import load_actor_fixes

    fixes = load_actor_fixes(stem)

    out_crops = crop_dir(stem)
    out_crops.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0

    records: list[dict] = []
    crops: list = []
    crop_owners: list[int] = []  # records index each crop belongs to
    crop_prompts: list[dict] = []  # keypoint prompts, aligned with crops
    total = len(events)
    try:
        for i, event in enumerate(events):
            cap.set(cv2.CAP_PROP_POS_FRAMES, event["frame"])
            ok, frame = cap.read()
            record = {
                "id": event.get("id") or f"f{event['frame']}",
                "frame": event["frame"],
                "time": event.get("time"),
                "label": event.get("label"),
                "xy": event["xy"],
                "status": "miss",
                "box": None,
                "score": None,
                "candidates": 0,
                "crop": None,
            }
            if ok:
                x, y = event["xy"][0] * frame_w, event["xy"][1] * frame_h
                detections = _keypoint_sources[keypoints].detect(frame, focus=(x, y))
                # ALL person boxes, unfiltered — the UI's actor picker and the
                # future association training set both need the ones the
                # heuristic rejected.
                record["detections"] = _serialize_detections(detections, frame_w, frame_h)
                candidates = associate(detections, x, y)
                record["candidates"] = len(candidates)
                auto = candidates[0] if candidates else None
                fix = fixes.get(record["id"])
                if fix is None:
                    if auto is not None:
                        crop = _attach_person(record, frame, auto, x, y, frame_w, frame_h, out_crops)
                        if crop is not None:
                            record["status"] = "ok" if len(candidates) == 1 else "multi"
                            crops.append(crop)
                            crop_owners.append(len(records))
                            crop_prompts.append(_crop_prompt(record, auto))
                else:
                    # Replay the user's actor fix; keep the auto pick alongside
                    # so the disagreement survives re-extraction.
                    record["box_source"] = "manual"
                    if auto is not None:
                        record["auto_box"] = list(_display_box(auto, x, y, frame_w, frame_h))
                    if fix.get("box"):
                        person = _snap_to_detection(record["detections"], fix["box"]) or PersonBox(
                            xyxy=tuple(fix["box"]), score=0.0
                        )
                        crop = _attach_person(record, frame, person, x, y, frame_w, frame_h, out_crops)
                        if crop is not None:
                            record["status"] = "ok"
                            crops.append(crop)
                            crop_owners.append(len(records))
                            crop_prompts.append(_crop_prompt(record, person))
            records.append(record)
            if on_progress:
                on_progress(i + 1, total, record["status"])
    finally:
        cap.release()

    # Every registered embedder runs on the same crops so models can be
    # A/B-compared on identical inputs without re-extracting.
    for name, embedder in _embedders.items():
        matrix = embedder.embed(crops, prompts=crop_prompts)
        for owner, emb in zip(crop_owners, matrix):
            records[owner].setdefault("embeddings", {})[name] = [round(float(v), 5) for v in emb]

    counts = {
        "events": total,
        "ok": sum(r["status"] == "ok" for r in records),
        "multi": sum(r["status"] == "multi" for r in records),
        "miss": sum(r["status"] == "miss" for r in records),
    }
    header = {
        "video": stem,
        "source": {"detector": DETECTOR_NAME, "keypoints": KEYPOINT_SOURCES[keypoints], "embedders": {k: EMBEDDER_WEIGHTS[k] for k in _embedders}},
        "frame_size": [frame_w, frame_h],
        "fps": fps,
        "created_at": time.time(),
        **counts,
    }
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(reid_path(stem), header, records)
    return counts


def apply_actor_fix(video_path: Path, event_id: str, box: list[float] | None, *, none: bool = False) -> dict:
    """Re-point one extracted event at a user-chosen person, in place.

    Three modes: ``box`` given = manual pick (snapped by IoU onto a stored
    detection when possible, so keypoints carry over); ``none=True`` = nobody
    is the actor (crop/embedding cleared, so the event drops out of
    clustering and matching); neither = revert to the automatic pick, re-run
    from the stored detections. Persisting the fix into the players file is
    the caller's job — this only patches the derived jsonl.

    Returns the updated record without embeddings (the UI payload).
    """
    import cv2

    stem = video_path.stem
    path = reid_path(stem)
    meta, records = read_jsonl(path)
    record = next((r for r in records if r["id"] == event_id), None)
    if record is None:
        raise KeyError(f"No ReID record for event {event_id}")

    frame_w, frame_h = meta.get("frame_size") or [0, 0]
    x, y = record["xy"][0] * frame_w, record["xy"][1] * frame_h
    detections = record.get("detections") or []

    revert = box is None and not none
    if revert:
        record.pop("box_source", None)
        record.pop("auto_box", None)
    else:
        if record.get("box_source") != "manual":  # first fix stashes the auto pick
            record["auto_box"] = record.get("box")
        record["box_source"] = "manual"

    # Clear the previous pick; each branch below re-fills what applies.
    record.update(status="miss", box=None, score=None, crop=None, keypoints=None)
    record.pop("embeddings", None)

    person = None
    n_candidates = record.get("candidates", 0)
    if revert:
        candidates = associate([_person_from_detection(d) for d in detections], x, y)
        n_candidates = len(candidates)
        record["candidates"] = n_candidates
        person = candidates[0] if candidates else None
    elif box is not None:
        person = _snap_to_detection(detections, box) or PersonBox(xyxy=tuple(box), score=0.0)

    if person is not None:
        cap = cv2.VideoCapture(str(video_path))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, record["frame"])
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok:
            raise ValueError(f"Could not decode frame {record['frame']} of {video_path.name}")
        bx0, by0 = int(person.xyxy[0]), int(person.xyxy[1])
        suffix = "" if revert else f"_fix_{bx0}_{by0}"  # per-box name busts browser cache
        crop = _attach_person(record, frame, person, x, y, frame_w, frame_h, crop_dir(stem), suffix=suffix)
        if crop is None:
            raise ValueError("Degenerate person box")
        if revert:
            record["status"] = "ok" if n_candidates == 1 else "multi"
        else:
            record["status"] = "ok"
        prompt = _crop_prompt(record, person)
        for name, embedder in _embedders.items():
            emb = embedder.embed([crop], prompts=[prompt])[0]
            record.setdefault("embeddings", {})[name] = [round(float(v), 5) for v in emb]

    write_jsonl(path, meta, records)
    out = dict(record)
    out.pop("embeddings", None)
    return out
