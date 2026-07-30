"""Per-video extraction: action events → person crops → embeddings.

Two stages, coupled only through what extraction leaves on disk, and run as
two separate jobs on purpose:

- ``extract_video``: decode + detect + associate + crop. Writes the record
  jsonl and the crop jpgs (see extraction/store.py) — everything embedding
  needs, nothing more.
- ``embed_video``: crops → one npy matrix per embedder (see reid/store.py).
  Reads the saved jpgs (the embedder input IS the reviewable artifact), so a
  new embedder backfills old videos without touching the video file, and
  extraction cost no longer scales with the number of registered models.

The order is not an implementation detail. An embedding answers "who is this
person", and it can only be asked of a crop somebody has agreed contains the
right person — which is the actor review that happens between the two. When
extract_video chained embed_video, every actor fix during that review had to
re-cut the crop AND patch one row of every matrix; across the labelled corpus
that was 500 fixes, each re-embedded under four models, two of which cold-load
a ViT-L in a subprocess. Embedding after the review costs one pass instead.

A caller that wants both in one click composes them (see the extraction
router); nothing here hides the second stage inside the first.

Records keep the association outcome (ok / multi / miss) so downstream
matching and the UI can treat ambiguous events differently.

Every record also stores ``detections`` — ALL person boxes the detector found
on that frame, unfiltered by the association policy. The labeling UI needs
them so the user can re-point an event at the right person when the policy
picked the wrong one; those manual picks (see actor/labels.py) are replayed
here on re-extraction and stashed alongside the auto pick (``auto_box``), so
the auto/manual disagreement set is preserved as association training data.

This module is where the three packages meet: ``person`` finds the people,
``actor`` says which of them acted, ``reid`` turns the resulting crop into an
embedding.
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

from yp_video.actor.labels import ActorLabel, ActorVerdict
from yp_video.actor.resolution import ActorResolution, actor_resolution
from yp_video.actor.service import ActorAssociationService
from yp_video.core.jsonl import read_jsonl, read_jsonl_cached, write_jsonl
from yp_video.core.progress import ProgressFn
from yp_video.extraction.cropping import (
    clamp_box,
    cut,
    label_target,
    person_for,
)
from yp_video.extraction.store import (
    RECORDS_DIR,
    SKIP_LABELS,
    action_annotation_path,
    crop_dir,
    records_path,
)
from yp_video.person.detector import (
    DETECTOR_NAME,
    person_detector,
    person_from_detection,
)
from yp_video.reid.embedder import base_embedder_name, build_embedders
from yp_video.reid.store import (
    clear_embedding_refreshes,
    embedded_models,
    embedding_path,
    embedding_write_transaction,
    load_embedding_matrix,
    mark_actor_embedding_refreshed,
    mark_actor_embedding_stale,
    save_embedding_matrix,
)


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


def _serialize_detections(boxes, w: int, h: int) -> list[dict]:
    """All person detections of a frame as jsonl-friendly dicts, best first."""
    out = []
    for b in sorted(boxes, key=lambda b: -b.score):
        x0, y0, x1, y1 = clamp_box(b.xyxy, w, h)
        out.append({"box": [x0, y0, x1, y1], "score": round(float(b.score), 3)})
    return out


def detect_video(
    video_path: Path,
    *,
    on_progress: ProgressFn | None = None,
) -> dict:
    """Find every person on each annotated action frame. Decides nothing.

    Perception, not judgement: this is the sparse sibling of rally tracking —
    tracking detects every frame of every rally and links the results, this
    detects the ~300 frames an action actually happened on and keeps ALL the
    segmentation boxes. Which of those people acted is the association
    stage's answer (extraction/reassociate.py), and it re-decides among these
    boxes without ever needing the video again.

    It used to pick and crop here too, which made the first association pass
    a different code path from every later one — and made "who acted" a
    question you could only re-ask after paying for detection a second time.

    Records already on disk keep their association: a re-detect refreshes the
    candidate list, and re-deciding among the new one is the next stage's job.
    Returns the summary counts also written to the jsonl header.
    """
    import cv2

    stem = video_path.stem
    events = load_events(stem)
    if not events:
        raise ValueError(f"No action events for {video_path.name}")

    path = records_path(stem)
    previous: dict[str, dict] = {}
    if path.exists():
        _meta, existing = read_jsonl(path)
        previous = {str(r["id"]): r for r in existing}

    cap = cv2.VideoCapture(str(video_path))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    cap.release()

    detector = person_detector()
    records: list[dict] = []
    total = len(events)
    if on_progress:
        # The first detect() loads the detector — announce the stall.
        on_progress(0, total, f"loading detector ({DETECTOR_NAME})...")

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

    producer = threading.Thread(target=decode, name=f"detect-{stem}", daemon=True)
    producer.start()

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
            event_id = str(event.get("id") or f"f{event['frame']}")
            record = {
                "id": event_id,
                "frame": event["frame"],
                "xy": xy,
                # An event nobody has associated yet, which is what every
                # event is until the association stage runs.
                "status": "miss",
                "resolution": ActorResolution.UNRESOLVED.value,
                "box": None,
                "score": None,
                "candidates": 0,
                "crop": None,
            }
            # A pick already made survives a re-detect. Dropping it would
            # discard a human verdict that this stage has no opinion about.
            record.update(
                {
                    key: value
                    for key, value in previous.get(event_id, {}).items()
                    if key in _ASSOCIATION_FIELDS
                }
            )
            if ok:
                pt = (xy[0] * frame_w, xy[1] * frame_h) if xy else None
                # ALL person boxes, unfiltered — the actor picker and the
                # association training set both need the ones a policy would
                # reject.
                record["detections"] = _serialize_detections(
                    detector.detect(frame, focus=pt), frame_w, frame_h
                )
            records.append(record)
            if on_progress:
                on_progress(i + 1, total, f"event {i + 1}/{total}")
    finally:
        stop.set()
        producer.join(timeout=5)

    counts = {
        "events": total,
        "detections": sum(len(r.get("detections") or ()) for r in records),
        "undecodable": sum(1 for r in records if "detections" not in r),
    }
    header = {
        "video": stem,
        "source": {"detector": DETECTOR_NAME},
        "frame_size": [frame_w, frame_h],
        "fps": fps,
        "created_at": time.time(),
        **counts,
    }
    RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(path, header, records)
    return counts


#: Written by the association stage, not this one — carried across a
#: re-detect so refreshing the candidate list never erases an answer.
_ASSOCIATION_FIELDS = frozenset({
    "status", "resolution", "box", "actor_box", "score", "crop", "crop_schema",
    "candidates", "association", "auto_box", "track", "crop_frame",
    "actor_revision",
})


def embed_video(
    stem: str,
    *,
    models: list[str] | None = None,
    overwrite: bool = False,
    checkpoint: Path | None = None,
    on_progress: ProgressFn | None = None,
) -> dict:
    """Crops on disk → one (n_records, dim) npy matrix per embedder.

    Embedders consume the saved crop jpgs by path, so this needs only an
    extraction's output, never the video: registering a new embedder later
    means backfilling with this — not re-extracting. Rows align with the
    record order in the reid jsonl; records without a crop (and unreadable
    crop files) get NaN rows. ``models=None`` means every registered
    embedder; without ``overwrite`` existing matrices are kept.

    Every model embeds the same crops, so models stay A/B-comparable
    on identical inputs. Progress is per model (``done=0`` announces the
    model, including a first-use weight load). Returns
    ``{"models": [...], "crops": N}``.
    """
    import numpy as np

    _meta, records = read_jsonl(records_path(stem))
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

    cdir = crop_dir(stem)
    paths, owners = [], []
    for i, record in enumerate(records):
        if record.get("crop") and (cdir / record["crop"]).exists():
            paths.append(cdir / record["crop"])
            owners.append(i)

    masked_paths: list[Path] | None = None  # built once, shared by every masked variant
    for name, embedder in targets.items():
        inputs = paths
        if getattr(embedder, "masked_input", False):
            if masked_paths is None:
                masked_paths = _mask_crops(stem, paths, owners, records, on_progress)
            inputs = masked_paths
        if on_progress:
            on_progress(0, len(inputs), f"loading {name} weights..." if not embedder.loaded else f"embedding ({name})...")
        progress: ProgressFn | None = None
        if on_progress:
            def progress(done, total, msg, *, _name=name, _cb=on_progress):
                _cb(done, total, f"{_name} · {msg}")
        matrix = embedder.embed_paths(inputs, on_progress=progress, checkpoint=checkpoint)
        full = np.full((len(records), matrix.shape[1]), np.nan, dtype=np.float32)
        if len(owners):
            full[owners] = matrix
        with embedding_write_transaction():
            save_embedding_matrix(stem, name, full)
            clear_embedding_refreshes(stem, name)
    return {"models": sorted(targets), "crops": len(paths)}


def _masked_record_crop(stem: str, record: dict, crop):
    """The crop with non-actor pixels greyed out (see reid/seg.py), persisted
    under crops-masked/ so the UI can show what the embedder saw. The actor's
    box comes back to crop coordinates via the display-box origin."""
    import cv2

    from yp_video.extraction.store import masked_crop_dir
    from yp_video.person.seg import crop_masker

    dx0, dy0 = record["box"][:2]
    bx = record.get("actor_box") or record["box"]
    masked = crop_masker().mask_crop(crop, [bx[0] - dx0, bx[1] - dy0, bx[2] - dx0, bx[3] - dy0])
    out_dir = masked_crop_dir(stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / record["crop"]), masked)
    return masked


def _mask_crops(
    stem: str, paths: list[Path], owners: list[int], records: list[dict], on_progress: ProgressFn | None
) -> list[Path]:
    """Persist background-suppressed variants of *paths* and return their
    locations, aligned with the input. An unreadable source crop still yields
    its expected masked path — the file just won't exist, and the embedder
    turns that into a NaN row.

    A masked crop at least as new as the crop it was cut from is kept:
    backfilling one masked model used to re-run the segmenter over an entire
    video that had already been masked. Mtime and not mere existence, because
    an automatic pick's crop keeps its filename across a re-extraction — the
    pixels change underneath it, and only the timestamp says so.
    """
    import cv2

    from yp_video.extraction.store import masked_crop_dir
    from yp_video.person.seg import crop_masker

    mdir = masked_crop_dir(stem)

    def is_stale(path: Path, masked: Path) -> bool:
        return (
            not masked.exists()
            or masked.stat().st_mtime_ns < path.stat().st_mtime_ns
        )

    pending = [
        (path, owner)
        for path, owner in zip(paths, owners)
        if is_stale(path, mdir / records[owner]["crop"])
    ]
    if on_progress:
        on_progress(
            0,
            len(pending),
            "loading rf-detr-seg-medium weights..."
            if pending and not crop_masker().loaded
            else "masking crops...",
        )
    for i, (path, owner) in enumerate(pending):
        record = records[owner]
        img = cv2.imread(str(path))
        if img is not None:
            _masked_record_crop(stem, record, img)
        if on_progress:
            on_progress(i + 1, len(pending), f"masking · crop {i + 1}/{len(pending)}")
    return [mdir / records[owner]["crop"] for owner in owners]


# Serializes apply_actor_fix's read-modify-write of the record jsonl: two
# quick picks would otherwise interleave and one would be lost.
_actor_fix_lock = threading.RLock()


def apply_actor_fix(
    video_path: Path,
    event_id: str,
    label: ActorLabel | None,
    *,
    models: list[str],
) -> dict:
    """Re-point one extracted event at the person a human named, in place.

    The verdict drives everything: ``MANUAL`` crops the labeled box (snapped
    by IoU onto a stored segmentation detection when possible);
    ``OCCLUDED`` clears the crop and embedding, dropping the event out of
    clustering and matching; ``None`` reverts to the automatic pick, re-run
    from the stored detections. Persisting the label is the caller's job —
    this only patches the derived jsonl.

    ``label.frame`` marks a CROSS-FRAME pick: the actor went undetected on
    the event frame, so the user clicked them on a nearby frame — the crop is
    cut from THAT frame (the pixels actually contain the actor) and no
    detection snap applies (stored detections belong to the event frame).

    Only ``models`` are refreshed synchronously. Other existing matrices keep
    an explicit pending event in the refresh sidecar; the application service
    refreshes them after the response.

    Returns the updated record without embeddings (the UI payload).
    """
    stem = video_path.stem
    with embedding_write_transaction(), _actor_fix_lock:
        mark_actor_embedding_stale(stem, embedded_models(stem), event_id)
        record, row, crop = _apply_actor_fix(video_path, event_id, label)
    _patch_embedding_row(
        stem,
        record,
        row,
        crop,
        models=models,
        expected_revision=int(record["actor_revision"]),
    )
    return record


def _apply_actor_fix(
    video_path: Path,
    event_id: str,
    label: ActorLabel | None,
) -> tuple[dict, int, object | None]:
    import cv2

    stem = video_path.stem
    path = records_path(stem)
    meta, records = read_jsonl(path)
    row = next((i for i, r in enumerate(records) if r["id"] == event_id), None)
    if row is None:
        raise KeyError(f"No extraction record for event {event_id}")
    record = records[row]
    record["actor_revision"] = int(record.get("actor_revision") or 0) + 1

    frame_w, frame_h = meta.get("frame_size") or [0, 0]
    xy = record.get("xy")  # None for invisible / point-less events
    contact = (xy[0] * frame_w, xy[1] * frame_h) if xy else None
    detections = record.get("detections") or []

    revert = label is None
    human_picked = actor_resolution(record) in (
        ActorResolution.MANUAL,
        ActorResolution.OCCLUDED,
    )
    if revert:
        record.pop("auto_box", None)
        record["resolution"] = ActorResolution.UNRESOLVED.value
    else:
        if not human_picked:  # first fix stashes the auto pick
            record["auto_box"] = record.get("box")
        record["resolution"] = (
            ActorResolution.OCCLUDED.value
            if label.verdict is ActorVerdict.OCCLUDED
            else ActorResolution.MANUAL.value
        )

    # Clear the previous pick; each branch below re-fills what applies.
    record.update(status="miss", box=None, actor_box=None, score=None, crop=None)
    record.pop("crop_schema", None)
    record.pop("keypoints", None)
    record.pop("crop_frame", None)

    target = label_target(stem, record, label) if label is not None else None
    src_frame = target.frame if target is not None else record["frame"]
    person = None
    n_candidates = record.get("candidates", 0)
    if revert:
        # No contact point (invisible event) → there IS no automatic pick;
        # revert just clears back to miss.
        people = [person_from_detection(d) for d in detections]
        if contact is not None:
            association = ActorAssociationService().associate(people, *contact)
            candidates = association.production_candidates
            record["association"] = association.diagnostic()
        else:
            candidates = []
        n_candidates = len(candidates)
        record["candidates"] = n_candidates
        person = candidates[0] if candidates else None
    elif target is not None:
        person = person_for(record, target)

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
        crop = cut(
            record,
            frame_img,
            person,
            source_frame=src_frame,
            contact=contact,
            frame_size=(frame_w, frame_h),
            out_dir=crop_dir(stem),
            suffix=suffix,
        )
        if crop is None:
            raise ValueError("Degenerate person box")
        if revert:
            record["status"] = "ok" if n_candidates == 1 else "multi"
            record["resolution"] = ActorResolution.AUTO.value
        else:
            record["status"] = "ok"

    write_jsonl(path, meta, records)
    return dict(record), row, crop


_embedding_locks_guard = threading.Lock()
_embedding_locks: dict[tuple[str, str], threading.Lock] = {}


def _embedding_lock(stem: str, model: str) -> threading.Lock:
    with _embedding_locks_guard:
        return _embedding_locks.setdefault((stem, model), threading.Lock())


def _record_revision_is_current(
    stem: str, event_id: str, expected_revision: int
) -> bool:
    with _actor_fix_lock:
        _meta, records = read_jsonl_cached(records_path(stem))
    return any(
        record.get("id") == event_id
        and int(record.get("actor_revision") or 0) == expected_revision
        for record in records
    )


def refresh_actor_embeddings(
    stem: str,
    event_id: str,
    *,
    models: list[str],
    expected_revision: int,
) -> list[str]:
    """Refresh deferred models if this actor verdict is still current.

    A later pick supersedes this background task through ``actor_revision``.
    Each matrix is loaded only after inference and under its own lock, so
    concurrent fixes to different events merge instead of losing a row.
    """
    import cv2

    if not models or not _record_revision_is_current(
        stem, event_id, expected_revision
    ):
        return []
    with _actor_fix_lock:
        _meta, records = read_jsonl_cached(records_path(stem))
    row = next(
        (i for i, record in enumerate(records) if record["id"] == event_id),
        None,
    )
    if row is None:
        return []
    record = dict(records[row])
    crop = None
    if record.get("crop"):
        crop = cv2.imread(str(crop_dir(stem) / record["crop"]))
        if crop is None:
            raise ValueError(f"Actor crop is unreadable: {record['crop']}")
    return _patch_embedding_row(
        stem,
        record,
        row,
        crop,
        models=models,
        expected_revision=expected_revision,
    )


def _patch_embedding_row(
    stem: str,
    record: dict,
    row: int,
    crop,
    *,
    models: list[str],
    expected_revision: int,
) -> list[str]:
    """Refresh selected matrix rows, batching variants with shared weights.

    ``crop=None`` (nobody is the actor) blanks the row to NaN; so does a
    matrix whose model is no longer registered — a stale embedding presented
    as current would silently corrupt that model's clusters. Masked and
    unmasked variants of one base model are inferred in one batch, so the
    subprocess embedder loads its weights only once.
    """
    import numpy as np

    from yp_video.extraction.store import masked_crop_dir

    existing = set(embedded_models(stem))
    targets = sorted(set(models) & existing)
    if not targets:
        return []
    if not _record_revision_is_current(
        stem, str(record["id"]), expected_revision
    ):
        return []

    registry = build_embedders()
    normal_path = crop_dir(stem) / record["crop"] if crop is not None else None
    masked_path = None
    if crop is not None and any(name.endswith("-masked") for name in targets):
        masked_path = masked_crop_dir(stem) / record["crop"]
        if not masked_path.exists():
            # Crop creation participates in the same commit boundary as the
            # application-service rollback. A foreground failure must never
            # delete a crop concurrently produced by an older background job.
            with embedding_write_transaction():
                if not masked_path.exists():
                    _masked_record_crop(stem, record, crop)

    groups: dict[str, list[str]] = {}
    for name in targets:
        groups.setdefault(base_embedder_name(name), []).append(name)

    updated: list[str] = []
    for base_name, variants in groups.items():
        vectors: dict[str, np.ndarray] = {}
        embedder = registry.get(base_name)
        if crop is not None and embedder is not None:
            paths: list[Path] = []
            for name in variants:
                path = masked_path if name.endswith("-masked") else normal_path
                if path is None:
                    raise ValueError(f"Missing actor crop for {base_name}")
                paths.append(path)
            batch = embedder.embed_paths(paths)
            vectors = {name: batch[index] for index, name in enumerate(variants)}

        for name in variants:
            with (
                embedding_write_transaction(),
                _embedding_lock(stem, name),
                _actor_fix_lock,
            ):
                if not _record_revision_is_current(
                    stem, str(record["id"]), expected_revision
                ):
                    continue
                matrix = load_embedding_matrix(stem, name)
                matrix[row] = vectors.get(name, np.nan)
                save_embedding_matrix(stem, name, matrix)
                mark_actor_embedding_refreshed(
                    stem, name, str(record["id"])
                )
                updated.append(name)
    return updated
