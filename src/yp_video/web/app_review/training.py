"""Send a user's App match into training as a ``cuts-user-<angle>`` cut.

The source video is copied from the customer bucket into the pipeline bucket
under its match id, and what the App shows the user — the latest analysis
with their corrections applied — becomes the cut's rally and action
pre-annotations. Pre-annotations only: a human reviews them in Label like any
model output before anything trains on them.

A second send never rewrites a label: the video already in place answers
``exists``, and one under another angle is moved, its labels left as they
are (they are named by the match id, not the kind).
"""

from typing import Literal, TypedDict, get_args

from yp_video.config import (
    CUT_KINDS,
    RALLY_ANNOTATIONS_DIR,
    RALLY_SPOT_PRE_ANNOTATIONS_DIR,
    USER_CUT_KINDS,
)
from yp_video.core.ffmpeg import probe_video_metadata
from yp_video.core.jsonl import write_jsonl
from yp_video.core.rallies import annotation_name, number_rallies
from yp_video.web import action_annotations
from yp_video.web.r2_client import r2_client

from . import sources
from .models import Bundle, Event, Rally
from .projection import clip_key, kind

Angle = Literal["baseline", "corner", "high", "center"]
assert set(get_args(Angle)) == set(USER_CUT_KINDS)


class TrainingOutcome(TypedDict):
    video: str
    kind: str
    status: Literal["added", "exists", "moved"]
    rallies: int
    events: int


def prelabels(bundle: Bundle) -> tuple[list[dict], list[dict]]:
    """The App's corrected view of the match as ``(rally rows, action events)``:
    the same rally set and removals ``projection.project`` shows the user."""
    corrections = bundle.corrections
    source = bundle.result.rallies
    if bundle.library_rallies is not None:
        source = [r for r in bundle.library_rallies if r.deleted_at is None]
    deleted = set(corrections.deleted_rally_indices) if corrections else set()
    overrides = corrections.rally_winner_overrides if corrections else {}

    def row(rally: Rally) -> dict:
        winner = overrides.get(rally.index) or rally.winner
        return {"start": rally.start, "end": rally.end, "label": "rally", "winner": winner}

    rallies = [row(r) for r in sorted(source, key=lambda r: (r.start, r.index)) if r.index not in deleted]

    removed_actions = {c.key for c in corrections.actions if c.removed} if corrections else set()
    removed_scores = {c.key for c in corrections.scores if c.removed} if corrections else set()

    def kept(event: Event) -> bool:
        removed = removed_scores if kind(event) == "score" else removed_actions
        return clip_key(event) not in removed

    def fields(event: Event) -> dict:
        extra = event.model_extra or {}
        out = {"frame": event.frame, "label": event.label, "xy": extra.get("xy"), "visible": event.visible}
        if extra.get("score") is not None:
            out["score"] = extra["score"]
        return out

    return rallies, [fields(e) for e in bundle.result.action_events if kept(e)]


def _user_cut_key(kind_name: str, video: str) -> str:
    return f"{CUT_KINDS[kind_name].r2_category}/{video}"


def add_to_training(user: str, match: str, angle: Angle | None) -> TrainingOutcome:
    """Copy ``user``'s ``match`` into training under ``angle`` (the match's
    own angle when None). Raises ``ValueError`` / ``sources.NotFound`` for a
    match that cannot go in; R2 / admin API failures propagate as they are."""
    lib, row = sources.library_match(user, match)
    source_key = (row.get("source_video") or {}).get("r2_key")
    if not source_key:
        raise ValueError("This match has no source video")
    if any(source_key.startswith(f"{info.r2_category}/") for info in CUT_KINDS.values()):
        raise ValueError("This match plays a published cut — it is already training data")
    chosen = angle or row.get("angle")
    if chosen not in USER_CUT_KINDS:
        raise ValueError(f"Unknown camera angle {chosen!r}")
    kind_name = USER_CUT_KINDS[chosen]
    video = f"{match}.mp4"
    target = _user_cut_key(kind_name, video)

    if r2_client.object_exists(target):
        return {"video": video, "kind": kind_name, "status": "exists", "rallies": 0, "events": 0}
    for other in USER_CUT_KINDS.values():
        existing = _user_cut_key(other, video)
        if other != kind_name and r2_client.object_exists(existing):
            r2_client.copy_from(r2_client.bucket, existing, target)
            r2_client.delete_object(existing)
            return {"video": video, "kind": kind_name, "status": "moved", "rallies": 0, "events": 0}

    bundle, _ = sources.match_bundle(lib, row)
    rallies, events = prelabels(bundle)
    r2_client.copy_from(sources.customer.bucket, source_key, target)
    origin = {"type": "app", "user_id": row["owner_id"], "match_id": match, "job_id": bundle.result.job_id}
    _write_rallies(match, video, bundle.result.total_duration, rallies, origin)
    _write_actions(match, target, events, origin)
    return {"video": video, "kind": kind_name, "status": "added", "rallies": len(rallies), "events": len(events)}


def _write_rallies(stem: str, video: str, duration: float, rallies: list[dict], origin: dict) -> None:
    name = annotation_name(stem)
    path = RALLY_SPOT_PRE_ANNOTATIONS_DIR / name
    if path.exists() or (RALLY_ANNOTATIONS_DIR / name).exists():
        return
    rows, max_id = number_rallies(rallies)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(path, {"video": video, "duration": duration, "max_rally_id": max_id, "source": origin}, rows)
    r2_client.upload_file(path, f"rally-spot/pre-annotations/{name}")


def _write_actions(stem: str, cut_key: str, events: list[dict], origin: dict) -> None:
    path = action_annotations.pre_annotation_path(f"{stem}.mp4")
    if path.exists() or action_annotations.annotation_path(f"{stem}.mp4").exists():
        return
    meta = probe_video_metadata(r2_client.generate_presigned_url(cut_key))
    fps, num_frames = float(meta["fps"]), int(meta["num_frames"])
    normalized = action_annotations.normalize_events(stem, events, fps=fps, num_frames=num_frames, rallies=[])
    kept = [
        {
            **{k: e[k] for k in action_annotations.PERSISTED_EVENT_FIELDS},
            **({"score": e["score"]} if "score" in e else {}),
        }
        for e in normalized
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    action_annotations.write_annotation_atomic(path, {
        "video": stem, "num_frames": num_frames, "fps": fps, "num_events": len(kept),
        "source": origin, "events": kept,
    })
    r2_client.upload_file(path, f"action/pre-annotations/{path.name}")
