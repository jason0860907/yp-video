"""Send a user's App match into training as a ``cuts-user-<angle>`` cut.

The source video is copied from the customer bucket into the pipeline bucket
under its match id, and what the App shows the user — the latest analysis
with their corrections applied — becomes the cut's rally and action
pre-annotations, and the touchers its advanced analysis picked become actor
proposals (``agent`` / ``agent_occluded``, the user's "nobody" marks
included). Pre-annotations only: a human reviews them in Label like any
model output before anything trains on them. A picked box is the App's
detector's, not a 2XLarge one; Association Label's box check queues every
proposal no dense box matches, so a confirm still lands on a 2XLarge box.

The cut is named after the match's App title (``/`` -> ``⧸``, as yt-dlp
names the other cuts), with ``-<match id prefix>`` when another video or label
already uses that name. The video carries its match id as R2 metadata
(``match-id``): that, not the name, finds a match's cut again, so retitling
the match in the App never renames training data.

A second send never rewrites a label: the video already in place answers
``exists``, and one under another angle is moved under the same name, its
labels left as they are (they are named by the video, not the kind). Either
fills in a pre-label a failed earlier send never wrote.
"""

from typing import Literal, TypedDict, get_args

from yp_video.actor import labels as actor_labels
from yp_video.actor.labels import ActorLabel, ActorVerdict
from yp_video.config import (
    CUT_KINDS,
    RALLY_ANNOTATIONS_DIR,
    RALLY_SPOT_PRE_ANNOTATIONS_DIR,
    USER_CUT_KINDS,
)
from yp_video.core.ffmpeg import probe_video_metadata
from yp_video.core.frames import video_geometry
from yp_video.core.jsonl import read_jsonl, write_jsonl
from yp_video.core.rallies import annotation_name, number_rallies
from yp_video.web import action_annotations
from yp_video.web.r2_client import all_cut_paths, r2_client

from . import sources
from .models import IDENTIFICATION_VERSION, Bundle, Event, Rally
from .projection import clip_key, kind

Angle = Literal["baseline", "corner", "high", "center"]
assert set(get_args(Angle)) == set(USER_CUT_KINDS)


class TrainingOutcome(TypedDict):
    video: str
    kind: str
    status: Literal["added", "exists", "moved"]
    rallies: int
    events: int
    #: Actor proposals written (0 when the video already had actor labels, or
    #: the match has no advanced analysis).
    actors: int


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


#: User cut key -> its ``match-id`` metadata. A key's object is only ever
#: written by a send for that match, so an entry never goes stale.
_match_ids: dict[str, str | None] = {}


def _sent_cuts() -> dict[str, tuple[str, str]]:
    """Match id -> ``(kind, video)`` of every user cut that names its match."""
    out = {}
    for kind_name in USER_CUT_KINDS.values():
        prefix = f"{CUT_KINDS[kind_name].r2_category}/"
        for obj in r2_client.list_objects_cached(prefix):
            key = obj["key"]
            video = key.removeprefix(prefix)
            if not video.endswith(".mp4") or "/" in video:
                continue
            if key not in _match_ids:
                _match_ids[key] = r2_client.object_metadata(key).get("match-id")
            if _match_ids[key]:
                out[_match_ids[key]] = (kind_name, video)
    return out


class SentCut(TypedDict):
    kind: str
    #: The cut's file name (the App title it was sent under).
    video: str


def sent() -> dict[str, SentCut]:
    """Match id -> the training cut it became, for every match already sent."""
    return {match: {"kind": kind, "video": video} for match, (kind, video) in _sent_cuts().items()}


def video_name(title: str | None, match: str) -> str:
    """A new cut's file name: the App title, made unique with the match id."""
    stem = "".join(c for c in (title or "") if c.isprintable()).replace("/", "⧸").strip().strip(".")
    if not stem:
        return f"{match}.mp4"
    taken = {p.name for p in all_cut_paths()}
    if f"{stem}.mp4" in taken or _labels_exist(stem):
        stem = f"{stem}-{match[:8]}"
    return f"{stem}.mp4"


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
    metadata = {"match-id": match}

    status: Literal["added", "exists", "moved"] | None = None
    if match in (cuts := _sent_cuts()):
        sent_kind, video = cuts[match]
        target = _user_cut_key(kind_name, video)
        status = "exists"
        if sent_kind != kind_name:
            existing = _user_cut_key(sent_kind, video)
            r2_client.copy_from(r2_client.bucket, existing, target, metadata)
            r2_client.delete_object(existing)
            status = "moved"
    else:
        video = video_name(row.get("title"), match)
        target = _user_cut_key(kind_name, video)
    stem = video.removesuffix(".mp4")

    # A send that failed after its copy left the video without (some of) its
    # pre-labels; any send fills in what is missing and never rewrites a file.
    rallies = events = actors = 0
    if status is None or _labels_missing(stem):
        bundle, _ = sources.match_bundle(lib, row)
        if bundle.result.video_r2_key != source_key:
            raise ValueError("The latest analysis ran on another video than this match's source")
        rally_rows, event_rows = prelabels(bundle)
        if status is None:
            r2_client.copy_from(sources.customer.bucket, source_key, target, metadata)
            status = "added"
        origin = {"type": "app", "user_id": row["owner_id"], "match_id": match, "job_id": bundle.result.job_id}
        if _write_rallies(stem, video, bundle.result.total_duration, rally_rows, origin):
            rallies = len(rally_rows)
        if _write_actions(stem, target, event_rows, origin):
            events = len(event_rows)
        actors = _write_actors(stem, target, bundle, row)
    return {
        "video": video, "kind": kind_name, "status": status,
        "rallies": rallies, "events": events, "actors": actors,
    }


def _rally_paths(stem: str):
    name = annotation_name(stem)
    return RALLY_SPOT_PRE_ANNOTATIONS_DIR / name, RALLY_ANNOTATIONS_DIR / name


def _action_paths(stem: str):
    video = f"{stem}.mp4"
    return action_annotations.pre_annotation_path(video), action_annotations.annotation_path(video)


def _labels_missing(stem: str) -> bool:
    return not (
        all(any(p.exists() for p in pair) for pair in (_rally_paths(stem), _action_paths(stem)))
        and actor_labels.actors_path(stem).exists()
    )


def _labels_exist(stem: str) -> bool:
    return actor_labels.actors_path(stem).exists() or any(
        p.exists() for pair in (_rally_paths(stem), _action_paths(stem)) for p in pair
    )


def _write_rallies(stem: str, video: str, duration: float, rallies: list[dict], origin: dict) -> bool:
    path, human = _rally_paths(stem)
    if path.exists() or human.exists():
        return False
    rows, max_id = number_rallies(rallies)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(path, {"video": video, "duration": duration, "max_rally_id": max_id, "source": origin}, rows)
    r2_client.upload_file(path, f"rally-spot/pre-annotations/{path.name}")
    return True


def _write_actions(stem: str, cut_key: str, events: list[dict], origin: dict) -> bool:
    path, human = _action_paths(stem)
    if path.exists() or human.exists():
        return False
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
    return True


def actor_prelabels(bundle: Bundle, identification: dict, events: list[dict], size: tuple[int, int]) -> dict[str, ActorLabel]:
    """The App's toucher per event as proposals on the cut's ``events``.

    ``identification`` is the advanced analysis run (reid blob v6): its
    ``actors`` map an App event id to a normalized box on the event frame.
    The cut's events came from the same App events, so they join on
    ``(frame, label)``. A user's "nobody" mark for the run the corrections
    point at outranks the box, as it does in the App.
    """
    width, height = size
    ours = {(int(e["frame"]), e.get("label")): str(e["id"]) for e in events}
    pi = bundle.corrections.player_identification if bundle.corrections else None
    nobody = (
        {k for k, v in pi.event_overrides.items() if v is None}
        if pi and pi.result_id == identification.get("job_id")
        else set()
    )
    boxes = identification.get("actors") or {}
    out: dict[str, ActorLabel] = {}
    for event in bundle.result.action_events:
        event_id = ours.get((event.frame, event.label))
        if event_id is None:
            continue
        if event.id in nobody:
            out[event_id] = ActorLabel(ActorVerdict.AGENT_OCCLUDED)
        elif event.id in boxes:
            x0, y0, x1, y1 = boxes[event.id]
            out[event_id] = ActorLabel(
                ActorVerdict.AGENT, frame=event.frame, box=(x0 * width, y0 * height, x1 * width, y1 * height)
            )
    return out


def _write_actors(stem: str, cut_key: str, bundle: Bundle, row: dict) -> int:
    path = actor_labels.actors_path(stem)
    identification = _identification(bundle, row)
    action = next((p for p in reversed(_action_paths(stem)) if p.exists()), None)
    if path.exists() or identification is None or action is None:
        return 0
    geometry = video_geometry(r2_client.generate_presigned_url(cut_key))
    labels = actor_prelabels(bundle, identification, read_jsonl(action)[1], (geometry.width, geometry.height))
    if not actor_labels.create(stem, labels):
        return 0
    r2_client.upload_file(path, f"association/annotations/{path.name}")
    return len(labels)


def _identification(bundle: Bundle, row: dict) -> dict | None:
    """The advanced analysis run the App shows: the one the user's
    corrections map players on, else the match's latest. None without one,
    or in a version the App does not read either."""
    if bundle.identification is not None:
        return bundle.identification.model_dump()
    key = row["r2_keys"].get("reid")
    if not key:
        return None
    blob = sources.customer.read_json(key)
    return blob if blob.get("version") == IDENTIFICATION_VERSION else None
