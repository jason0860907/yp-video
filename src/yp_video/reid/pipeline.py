"""Per-video ReID extraction: action events → person crops → embeddings.

Two stages, coupled only through what extraction leaves on disk:

- ``extract_video``: decode + detect + associate + crop. Writes the record
  jsonl and the crop jpgs — everything embedding needs, nothing more.
- ``embed_video``: crops → one npy matrix per embedder (see reid/store.py).
  Reads the saved jpgs (the embedder input IS the reviewable artifact), so a
  new embedder backfills old videos without touching the video file, and
  extraction cost no longer scales with the number of registered models.

extract_video chains embed_video at the end so "Run ReID" stays one job.

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
import queue
import threading
import time
from collections.abc import Callable
from pathlib import Path

from yp_video.core.jsonl import read_jsonl, read_jsonl_cached, write_jsonl
from yp_video.reid.detector import (
    DEFAULT_KEYPOINT_SOURCE,
    DETECTOR_NAME,
    KEYPOINT_SOURCES,
    PersonBox,
    associate,
    build_keypoint_sources,
    iou,
)
from yp_video.reid.embedder import build_crop_prompt, build_embedders
from yp_video.reid.store import (
    EMBEDDINGS_DIR,
    SKIP_LABELS,
    action_annotation_path,
    crop_dir,
    embedded_models,
    embedding_path,
    load_embedding_matrix,
    reid_path,
    save_embedding_matrix,
)

ProgressFn = Callable[[int, int, str], None]


def load_events(stem: str) -> list[dict]:
    """Action events with a frame, sorted by frame.

    Invisible events (and ones without a contact point) are INCLUDED: they
    can't auto-associate, but they become miss records the user assigns by
    hand — usually with a cross-frame pick on a frame where the actor shows.
    Only SKIP_LABELS (nobody to identify) stay out.

    Cached parse (list_videos calls this for EVERY cut on every page load);
    events are read-only downstream — extract_video builds fresh records.
    """
    path = action_annotation_path(stem)
    if path is None:
        return []
    _meta, rows = read_jsonl_cached(path)
    events = [
        r for r in rows
        if r.get("frame") is not None
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


# A fix box must overlap a fresh detection this much to snap onto it (and
# inherit its keypoints); below that the fix box is embedded as drawn.
FIX_SNAP_IOU = 0.5


def _snap_to_detection(detections: list[dict], box: list[float]) -> PersonBox | None:
    """The stored detection a fix box refers to, matched by IoU."""
    best, best_iou = None, FIX_SNAP_IOU
    for d in detections:
        overlap = iou(d["box"], box)
        if overlap >= best_iou:
            best, best_iou = d, overlap
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
        # The raw detector box (display box is a superset): build_crop_prompt
        # needs it to exclude the chosen person from KPR's negative prompts.
        actor_box=[x0, y0, x1, y1],
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
    cap.release()

    source = build_keypoint_sources()[keypoints]
    records: list[dict] = []
    total = len(events)
    if on_progress:
        # The first detect() loads the detector — announce the stall.
        on_progress(0, total, f"loading detector ({keypoints})...")

    # A random seek costs more than detection (~55 vs ~27 ms/event), so a
    # decoder thread stays a few events ahead and the GPU never waits on
    # ffmpeg — the same producer/consumer split as tracking's dense pass.
    frame_q: queue.Queue = queue.Queue(maxsize=4)
    stop = threading.Event()
    decode_error: list[BaseException] = []

    def decode():
        cap = cv2.VideoCapture(str(video_path))
        try:
            for event in events:
                cap.set(cv2.CAP_PROP_POS_FRAMES, event["frame"])
                item = cap.read()  # (ok, frame)
                while not stop.is_set():
                    try:
                        frame_q.put(item, timeout=0.5)
                        break
                    except queue.Full:
                        continue
                if stop.is_set():
                    return
        except BaseException as exc:  # noqa: BLE001 — re-raised by the consumer
            decode_error.append(exc)
            stop.set()
        finally:
            cap.release()

    producer = threading.Thread(target=decode, name=f"reid-decode-{stem}", daemon=True)
    producer.start()

    # Cross-frame actor fixes crop from a different frame than the event's —
    # a lazy second capture serves those rare seeks without disturbing the
    # decoder thread.
    fix_cap = None

    def fix_frame_img(f: int):
        nonlocal fix_cap
        if fix_cap is None:
            fix_cap = cv2.VideoCapture(str(video_path))
        fix_cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = fix_cap.read()
        return img if ok else None

    try:
        for i, event in enumerate(events):
            while True:
                try:
                    ok, frame = frame_q.get(timeout=0.5)
                    break
                except queue.Empty:
                    if decode_error:
                        raise decode_error[0]
            xy = event.get("xy")
            event_visible = event.get("visible", True)
            record = {
                "id": event.get("id") or f"f{event['frame']}",
                "frame": event["frame"],
                "time": event.get("time"),
                "label": event.get("label"),
                "xy": xy,
                "status": "miss",
                "box": None,
                "score": None,
                "candidates": 0,
                "crop": None,
            }
            if not event_visible:
                record["visible"] = False
            if ok:
                pt = (xy[0] * frame_w, xy[1] * frame_h) if xy else None
                detections = source.detect(frame, focus=pt)
                # ALL person boxes, unfiltered — the UI's actor picker and the
                # future association training set both need the ones the
                # heuristic rejected.
                record["detections"] = _serialize_detections(detections, frame_w, frame_h)
                # Auto-association needs a contact point AND a visible actor —
                # an invisible event's point (if any) sits next to somebody
                # who did NOT perform it. Those stay miss until picked by hand.
                candidates = associate(detections, *pt) if pt and event_visible else []
                record["candidates"] = len(candidates)
                auto = candidates[0] if candidates else None
                fix = fixes.get(record["id"])
                if fix is None:
                    if auto is not None:
                        crop = _attach_person(record, frame, auto, *pt, frame_w, frame_h, out_crops)
                        if crop is not None:
                            record["status"] = "ok" if len(candidates) == 1 else "multi"
                else:
                    # Replay the user's actor fix; keep the auto pick alongside
                    # so the disagreement survives re-extraction.
                    record["box_source"] = "manual"
                    if auto is not None:
                        record["auto_box"] = list(_display_box(auto, *pt, frame_w, frame_h))
                    if fix.get("box"):
                        src = fix.get("frame")
                        if src is not None and src != record["frame"]:
                            # Cross-frame fix: the actor was undetected here —
                            # crop from the frame the user actually clicked,
                            # anchored on the box (the contact point belongs
                            # to the event frame, where the player isn't).
                            img = fix_frame_img(src)
                            person = PersonBox(xyxy=tuple(fix["box"]), score=0.0)
                            ax, ay = (person.xyxy[0] + person.xyxy[2]) / 2, (person.xyxy[1] + person.xyxy[3]) / 2
                            crop = _attach_person(record, img, person, ax, ay, frame_w, frame_h, out_crops) if img is not None else None
                            if crop is not None:
                                record["crop_frame"] = src
                                record["status"] = "ok"
                        else:
                            person = (
                                _snap_to_detection(record["detections"], fix["box"])
                                if fix.get("snap", True)
                                else None
                            ) or PersonBox(xyxy=tuple(fix["box"]), score=0.0)
                            ax, ay = pt if pt else ((person.xyxy[0] + person.xyxy[2]) / 2, (person.xyxy[1] + person.xyxy[3]) / 2)
                            crop = _attach_person(record, frame, person, ax, ay, frame_w, frame_h, out_crops)
                            if crop is not None:
                                record["status"] = "ok"
            records.append(record)
            if on_progress:
                on_progress(i + 1, total, f"event {i + 1}/{total}")
    finally:
        stop.set()
        producer.join(timeout=5)
        if fix_cap is not None:
            fix_cap.release()

    counts = {
        "events": total,
        "ok": sum(r["status"] == "ok" for r in records),
        "multi": sum(r["status"] == "multi" for r in records),
        "miss": sum(r["status"] == "miss" for r in records),
    }
    header = {
        "video": stem,
        "source": {"detector": DETECTOR_NAME, "keypoints": KEYPOINT_SOURCES[keypoints]},
        "frame_size": [frame_w, frame_h],
        "fps": fps,
        "created_at": time.time(),
        **counts,
    }
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(reid_path(stem), header, records)
    # Fresh extraction invalidates every model's rows — recompute them all,
    # continuing the same progress channel (the item bar restarts per phase).
    embed_video(stem, overwrite=True, on_progress=on_progress)
    return counts


# Crops per embed() call — small enough that progress moves visibly, large
# enough that per-call overhead (GPU launch, prompt stacking) stays amortized.
_EMBED_CHUNK = 32


def embed_video(
    stem: str, *, models: list[str] | None = None, overwrite: bool = False, on_progress: ProgressFn | None = None
) -> dict:
    """Crops on disk → one (n_records, dim) npy matrix per embedder.

    Reads the saved crop jpgs, so it needs only an extraction's output, never
    the video: registering a new embedder later means backfilling with this —
    not re-extracting. Rows align with the record order in the reid jsonl;
    records without a crop get NaN rows. ``models=None`` means every
    registered embedder; without ``overwrite`` existing matrices are kept.

    Every model embeds the same crops + prompts, so models stay A/B-comparable
    on identical inputs. Progress is per model (``done=0`` announces the
    model, including a first-use weight load). Returns
    ``{"models": [...], "crops": N}``.
    """
    import cv2
    import numpy as np

    _meta, records = read_jsonl(reid_path(stem))
    registry = build_embedders()
    unknown = set(models or ()) - set(registry)
    if unknown:
        raise ValueError(f"Unknown embedders: {', '.join(sorted(unknown))} (have: {', '.join(registry)})")
    targets = {
        name: embedder
        for name, embedder in registry.items()
        if (models is None or name in models)
        and (overwrite or not embedding_path(stem, name).exists())
    }
    if not targets:
        return {"models": [], "crops": 0}

    crops, owners, prompts = [], [], []
    cdir = crop_dir(stem)
    for i, record in enumerate(records):
        if not record.get("crop"):
            continue
        img = cv2.imread(str(cdir / record["crop"]))
        if img is None:
            continue
        crops.append(img)
        owners.append(i)
        prompts.append(build_crop_prompt(record, record.get("actor_box") or record["box"]))

    masked: list | None = None  # built once, shared by every masked variant
    for name, embedder in targets.items():
        inputs = crops
        if getattr(embedder, "masked_input", False):
            if masked is None:
                masked = _mask_crops(stem, crops, owners, records, on_progress)
            inputs = masked
        if on_progress:
            on_progress(0, len(inputs), f"loading {name} weights..." if not embedder.loaded else f"embedding ({name})...")
        parts = []
        for start in range(0, len(inputs), _EMBED_CHUNK):
            end = min(start + _EMBED_CHUNK, len(inputs))
            parts.append(embedder.embed(inputs[start:end], prompts=prompts[start:end]))
            if on_progress:
                on_progress(end, len(inputs), f"{name} · crop {end}/{len(inputs)}")
        matrix = np.concatenate(parts) if parts else embedder.embed([], prompts=[])
        full = np.full((len(records), matrix.shape[1]), np.nan, dtype=np.float32)
        if len(owners):
            full[owners] = matrix
        save_embedding_matrix(stem, name, full)
    return {"models": sorted(targets), "crops": len(crops)}


def _masked_record_crop(stem: str, record: dict, crop):
    """The crop with non-actor pixels greyed out (see reid/seg.py), persisted
    under crops-masked/ so the UI can show what the embedder saw. The actor's
    box comes back to crop coordinates via the display-box origin."""
    import cv2

    from yp_video.reid.seg import crop_masker
    from yp_video.reid.store import masked_crop_dir

    dx0, dy0 = record["box"][:2]
    bx = record.get("actor_box") or record["box"]
    masked = crop_masker().mask_crop(crop, [bx[0] - dx0, bx[1] - dy0, bx[2] - dx0, bx[3] - dy0])
    out_dir = masked_crop_dir(stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / record["crop"]), masked)
    return masked


def _mask_crops(stem: str, crops: list, owners: list[int], records: list[dict], on_progress: ProgressFn | None) -> list:
    from yp_video.reid.seg import crop_masker

    if on_progress:
        on_progress(0, len(crops), "loading rf-detr-seg-medium weights..." if not crop_masker().loaded else "masking crops...")
    out = []
    for i, (img, owner) in enumerate(zip(crops, owners)):
        out.append(_masked_record_crop(stem, records[owner], img))
        if on_progress:
            on_progress(i + 1, len(crops), f"masking · crop {i + 1}/{len(crops)}")
    return out


# Serializes apply_actor_fix's read-modify-write of the reid jsonl: two
# quick picks would otherwise interleave and one would be lost.
_actor_fix_lock = threading.Lock()


def apply_actor_fix(
    video_path: Path,
    event_id: str,
    box: list[float] | None,
    *,
    none: bool = False,
    frame: int | None = None,
    snap: bool = True,
) -> dict:
    """Re-point one extracted event at a user-chosen person, in place.

    Three modes: ``box`` given = manual pick (snapped by IoU onto a stored
    detection when possible, so keypoints carry over); ``none=True`` = nobody
    is the actor (crop/embedding cleared, so the event drops out of
    clustering and matching); neither = revert to the automatic pick, re-run
    from the stored detections. Persisting the fix into the players file is
    the caller's job — this only patches the derived jsonl.

    ``frame`` marks a CROSS-FRAME pick: the actor went undetected on the
    event frame, so the user clicked them on a nearby frame — the crop is
    cut from THAT frame (the pixels actually contain the actor) and no
    detection snap applies (stored detections belong to the event frame).

    ``snap=False`` embeds the box exactly as drawn — the client's mask
    arbitration ruled that no stored detection is this player, so an IoU
    snap could only attach an occluder.

    Returns the updated record without embeddings (the UI payload).
    """
    with _actor_fix_lock:
        return _apply_actor_fix(video_path, event_id, box, none=none, frame=frame, snap=snap)


def _apply_actor_fix(
    video_path: Path,
    event_id: str,
    box: list[float] | None,
    *,
    none: bool = False,
    frame: int | None = None,
    snap: bool = True,
) -> dict:
    import cv2

    stem = video_path.stem
    path = reid_path(stem)
    meta, records = read_jsonl(path)
    row = next((i for i, r in enumerate(records) if r["id"] == event_id), None)
    if row is None:
        raise KeyError(f"No ReID record for event {event_id}")
    record = records[row]

    frame_w, frame_h = meta.get("frame_size") or [0, 0]
    xy = record.get("xy")  # None for invisible / point-less events
    x, y = (xy[0] * frame_w, xy[1] * frame_h) if xy else (0.0, 0.0)
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
    record.update(status="miss", box=None, actor_box=None, score=None, crop=None, keypoints=None)
    record.pop("crop_frame", None)

    src_frame = frame if frame is not None else record["frame"]
    cross_frame = src_frame != record["frame"]
    person = None
    n_candidates = record.get("candidates", 0)
    if revert:
        # No contact point (invisible event) → there IS no automatic pick;
        # revert just clears back to miss.
        candidates = associate([_person_from_detection(d) for d in detections], x, y) if xy else []
        n_candidates = len(candidates)
        record["candidates"] = n_candidates
        person = candidates[0] if candidates else None
    elif box is not None:
        # No snapping for a cross-frame box (stored detections belong to the
        # event frame) or when the client's mask arbitration vetoed it.
        person = (
            _snap_to_detection(detections, box) if snap and not cross_frame else None
        ) or PersonBox(xyxy=tuple(box), score=0.0)

    crop = None
    if person is not None:
        cap = cv2.VideoCapture(str(video_path))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, src_frame)
            ok, frame_img = cap.read()
        finally:
            cap.release()
        if not ok:
            raise ValueError(f"Could not decode frame {src_frame} of {video_path.name}")
        bx0, by0 = int(person.xyxy[0]), int(person.xyxy[1])
        suffix = "" if revert else f"_fix_{src_frame}_{bx0}_{by0}"  # per-pick name busts browser cache
        # The display box unions the contact point — meaningless on another
        # frame (the player has moved) or when the event has none; anchor
        # those crops on the box itself.
        ax, ay = (x, y) if xy and not cross_frame else ((person.xyxy[0] + person.xyxy[2]) / 2, (person.xyxy[1] + person.xyxy[3]) / 2)
        crop = _attach_person(record, frame_img, person, ax, ay, frame_w, frame_h, crop_dir(stem), suffix=suffix)
        if crop is None:
            raise ValueError("Degenerate person box")
        if cross_frame:
            record["crop_frame"] = src_frame
        if revert:
            record["status"] = "ok" if n_candidates == 1 else "multi"
        else:
            record["status"] = "ok"

    write_jsonl(path, meta, records)
    _patch_embedding_row(stem, record, row, crop)
    return dict(record)


def _patch_embedding_row(stem: str, record: dict, row: int, crop) -> None:
    """Refresh one record's row in every embedding matrix on disk.

    ``crop=None`` (nobody is the actor) blanks the row to NaN; so does a
    matrix whose model is no longer registered — a stale embedding presented
    as current would silently corrupt that model's clusters.
    """
    import numpy as np

    registry = build_embedders()
    for name in embedded_models(stem):
        matrix = load_embedding_matrix(stem, name)
        embedder = registry.get(name)
        if crop is not None and embedder is not None:
            inp = _masked_record_crop(stem, record, crop) if getattr(embedder, "masked_input", False) else crop
            prompt = build_crop_prompt(record, record.get("actor_box") or record["box"])
            matrix[row] = embedder.embed([inp], prompts=[prompt])[0]
        else:
            matrix[row] = np.nan
        save_embedding_matrix(stem, name, matrix)
