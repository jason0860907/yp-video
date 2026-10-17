"""Immutable source snapshots, individual review decisions and explicit label edits.

App edits are feedback, not ground truth. Applying a false-positive verdict
requires an existing human annotation and an exact, unique source join. A
write-ahead intent makes a retried request safe after interruption.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock

from yp_video.config import RALLY_ANNOTATIONS_DIR, VIDEOS_DIR
from yp_video.core import label_done
from yp_video.core.jsonl import atomic_write, read_jsonl
from yp_video.core.rallies import annotation_name
from yp_video.web.action_annotations import annotation_path
from yp_video.web.annotation_lock import annotation_write_lock

from .models import Bundle
from .projection import clip_key, player_numbers, project

REVIEW_DIR = VIDEOS_DIR / "app-feedback"
_lock = RLock()


def fingerprint(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()


def save(path: Path, value: dict) -> None:
    with atomic_write(path) as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def path_for(review_id: str) -> Path:
    if len(review_id) != 64 or any(c not in "0123456789abcdef" for c in review_id):
        raise ValueError("Invalid review id")
    return REVIEW_DIR / f"{review_id}.json"


def load(review_id: str) -> dict:
    path = path_for(review_id)
    if not path.exists():
        raise ValueError("Review not found")
    return json.loads(path.read_text())


def snapshot_id(bundle: Bundle) -> str:
    """Reviews are keyed by the exact App snapshot they judge."""
    return fingerprint(canonical(bundle.model_dump(mode="json")))


def existing(bundle: Bundle) -> str | None:
    review_id = snapshot_id(bundle)
    return review_id if path_for(review_id).exists() else None


def create(bundle: Bundle, actor: str) -> dict:
    if not bundle.corrections and bundle.library_rallies is None:
        raise ValueError("A feedback review requires corrections or a library snapshot")
    if bundle.result.partial:
        raise ValueError("Partial analysis cannot be submitted for review")
    source = bundle.model_dump(mode="json")
    review_id = snapshot_id(bundle)
    with _lock:
        path = path_for(review_id)
        if path.exists():
            return load(review_id)
        data = {
            "id": review_id,
            "created_at": datetime.now(UTC).isoformat(),
            "created_by": actor,
            "bundle": source,
            "decisions": {},
            "application": None,
        }
        save(path, data)
        return data


def list_reviews(match: str) -> list[dict]:
    rows = []
    for path in REVIEW_DIR.glob("*.json"):
        data = json.loads(path.read_text())
        result = data["bundle"]["result"]
        if result["match_id"].lower() != match.lower():
            continue
        rows.append(
            {
                "id": data["id"],
                "match_id": result["match_id"],
                "job_id": result["job_id"],
                "created_at": data["created_at"],
                "reviewed": len(data["decisions"]),
                "total": len(candidates(Bundle.model_validate(data["bundle"]))),
            }
        )
    return sorted(rows, key=lambda r: r["created_at"], reverse=True)


def candidates(bundle: Bundle) -> list[dict]:
    c = bundle.corrections
    base = project(bundle, corrected=False)
    rows = []
    for scope, corrections in (
        ("actions", c.actions if c else []),
        ("scores", c.scores if c else []),
    ):
        clips = base[scope]
        for correction in corrections:
            matches = [clip for clip in clips if clip["key"] == correction.key]
            clip = matches[0] if len(matches) == 1 else None
            events = [
                e for e in bundle.result.action_events if clip_key(e) == correction.key
            ]
            rows.append(
                {
                    "id": f"{scope}:{correction.key}",
                    "scope": scope,
                    "value": correction.model_dump(),
                    "clip": clip,
                    "rally_index": clip["rally_index"] if clip else None,
                    "event": events[0].model_dump() if len(events) == 1 else None,
                    "can_remove_event": scope == "actions"
                    and correction.removed
                    and len(events) == 1
                    and clip is not None,
                    "reason": None
                    if clip
                    else "No unique clip found — check the analysis version and the Rally UUID mapping.",
                }
            )
    library = bundle.library_rallies or []
    deleted = set(c.deleted_rally_indices if c else []) | {
        r.index for r in library if r.deleted_at is not None
    }
    for index in sorted(deleted):
        rally = next((r for r in bundle.result.rallies if r.index == index), None)
        rows.append(
            {
                "id": f"rally:{index}",
                "scope": "rally",
                "value": {"deleted_rally_index": index},
                "rally_index": index,
                "clip": {"start": rally.start, "end": rally.end} if rally else None,
                "rally": rally.model_dump() if rally else None,
                "reason": None if rally else "The original analysis has no such rally.",
            }
        )
    for edited in library:
        original = next(
            (r for r in bundle.result.rallies if r.index == edited.index), None
        )
        if (
            edited.index in deleted
            or original is None
            or (edited.start, edited.end) == (original.start, original.end)
        ):
            continue
        rows.append(
            {
                "id": f"rally_bounds:{edited.index}",
                "scope": "rally_bounds",
                "value": {
                    "before": [original.start, original.end],
                    "after": [edited.start, edited.end],
                },
                "rally": original.model_dump(),
                "rally_index": edited.index,
                "bounds": [edited.start, edited.end],
                "clip": {
                    "start": min(original.start, edited.start),
                    "end": max(original.end, edited.end),
                },
                "reason": "An App trim is a playback preference; adopt it as a label only once it matches the full rally's start and end.",
            }
        )
    for index, side in sorted((c.rally_winner_overrides if c else {}).items()):
        rally = next((r for r in bundle.result.rallies if r.index == index), None)
        rows.append(
            {
                "id": f"winner:{index}",
                "scope": "winner",
                "value": {
                    "rally_index": index,
                    "before": rally.winner if rally else None,
                    "after": side,
                },
                "rally_index": index,
                "clip": {"start": rally.start, "end": rally.end} if rally else None,
                "reason": "Winner corrections are kept as review feedback; to train on one, change the winner in Label → Rally.",
            }
        )
    pi = c.player_identification if c else None
    if pi:
        events = {e.id: e for e in bundle.result.action_events}
        for event_id, number in player_numbers(bundle).items():
            event = events.get(event_id)
            rally = event and next(
                (r for r in bundle.result.rallies if r.start <= event.time <= r.end), None
            )
            rows.append(
                {
                    "id": f"player:{event_id}",
                    "scope": "player",
                    "value": {"event_id": event_id, "number": number},
                    "rally_index": rally.index if rally else None,
                    "clip": {"start": max(0, event.time - 1), "end": event.time + 1}
                    if event
                    else None,
                    "reason": "Player identity must be confirmed on the actual person crops in Association / ReID; only the review feedback is kept here.",
                }
            )
        # Preserve the run-scoped mapping, removed units and threshold even
        # when the identification artifact needed to expand it is absent.
        rows.append(
            {
                "id": "identification",
                "scope": "identification",
                "value": pi.model_dump(),
                "clip": None,
                "rally_index": None,
                "reason": "Person grouping and thresholds never become training labels directly.",
            }
        )
    if c and bundle.roster:
        rows.append(
            {
                "id": "roster",
                "scope": "roster",
                "value": [r.model_dump() for r in bundle.roster],
                "clip": None,
                "rally_index": None,
                "reason": "Rosters and cross-match identities stay feedback; names are never used to guess training identities.",
            }
        )
    return rows


def target_path(video: str, operation: str) -> Path:
    if not video or Path(video).name != video or Path(video).suffix.lower() != ".mp4":
        raise ValueError("Select a pipeline video basename ending in .mp4")
    return (
        annotation_path(video)
        if operation == "remove_event"
        else RALLY_ANNOTATIONS_DIR / annotation_name(Path(video).stem)
    )


def target_revision(video: str, operation: str) -> str | None:
    path = target_path(video, operation)
    return fingerprint(path.read_bytes()) if path.exists() else None


def prepare_patch(
    bundle: Bundle, candidate: dict, video: str, operation: str, revision: str
) -> tuple[Path, dict, list[dict]]:
    path = target_path(video, operation)
    if not path.exists():
        raise ValueError(
            "Fully review and save this video in Label first — feedback never promotes a whole model prediction to a human label."
        )
    if fingerprint(path.read_bytes()) != revision:
        raise ValueError("The labels changed — reload the target before reviewing.")
    meta, rows = read_jsonl(path)
    if operation == "remove_event":
        if not candidate.get("can_remove_event"):
            raise ValueError(
                "Only an unambiguous hidden Action can be reviewed as a false event"
            )
        source = candidate["event"]
        fps = float(meta.get("fps") or 0)
        frames = int(meta.get("num_frames") or 0)
        if fps <= 0 or abs(frames / fps - bundle.result.total_duration) > max(
            0.1, 2 / fps
        ):
            raise ValueError("Source duration does not match this pipeline video")
        if abs(source["frame"] / fps - source["time"]) > 0.5 / fps:
            raise ValueError(
                "Frame/time mismatch: videos may have different offsets or frame rates"
            )
        matches = [
            i
            for i, r in enumerate(rows)
            if r["frame"] == source["frame"] and r["label"] == source["label"]
        ]
    else:
        if candidate["scope"] != (
            "rally_bounds" if operation == "update_rally" else "rally"
        ) or not candidate.get("rally"):
            raise ValueError(
                "Only a resolved deleted Rally can be reviewed as a false rally"
            )
        if abs(float(meta.get("duration") or 0) - bundle.result.total_duration) > 0.1:
            raise ValueError("Source duration does not match this pipeline video")
        source = candidate["rally"]
        matches = [
            i
            for i, r in enumerate(rows)
            if r["start"] == source["start"]
            and r["end"] == source["end"]
            and r["label"] == "rally"
        ]
    if len(matches) != 1:
        raise ValueError(
            "No unique exact annotation match; open Label to review manually"
        )
    if operation == "update_rally":
        start, end = candidate["bounds"]
        if end <= start:
            raise ValueError("A training rally needs positive duration")
        if any(
            i != matches[0]
            and r["label"] == "rally"
            and start < r["end"]
            and end > r["start"]
            for i, r in enumerate(rows)
        ):
            raise ValueError("Reviewed bounds overlap another rally; resolve in Label")
        rows[matches[0]] = {**rows[matches[0]], "start": start, "end": end}
        return path, meta, rows
    kept = [r for i, r in enumerate(rows) if i != matches[0]]
    if operation == "remove_event" and "num_events" in meta:
        meta["num_events"] = len(kept)
    return path, meta, kept


def decide(
    review_id: str,
    candidate_id: str,
    decision: str,
    note: str,
    actor: str,
    *,
    video: str = "",
    revision: str = "",
    same_source: bool = False,
) -> tuple[dict, Path | None]:
    with _lock, annotation_write_lock:
        data = load(review_id)
        bundle = Bundle.model_validate(data["bundle"])
        candidate = next(
            (r for r in candidates(bundle) if r["id"] == candidate_id), None
        )
        if candidate is None:
            raise ValueError("Unknown candidate")
        prior = data["decisions"].get(candidate_id)
        if prior and prior["decision"] in {
            "remove_event",
            "remove_rally",
            "update_rally",
        }:
            raise ValueError(
                "This correction was already applied; change the annotation in Label"
            )
        if data.get("application"):
            raise ValueError(
                "An interrupted import needs recovery before another decision"
            )
        record = {
            "decision": decision,
            "note": note,
            "actor": actor,
            "at": datetime.now(UTC).isoformat(),
        }
        if decision in {"remove_event", "remove_rally", "update_rally"}:
            if not same_source or not note.strip():
                raise ValueError(
                    "Confirm identical source video and enter the visual evidence for this annotation verdict"
                )
            path, meta, rows = prepare_patch(
                bundle, candidate, video, decision, revision
            )
            record.update({"video": video, "before_sha256": revision})
            # Store the exact intended bytes in the journal before publishing
            # them. Recovery verifies before/after hashes and never re-applies
            # the patch to a different annotation revision.
            meta = {**meta, "_meta": True}
            payload = "".join(
                json.dumps(r, ensure_ascii=False) + "\n" for r in [meta, *rows]
            )
            data["application"] = {
                "candidate_id": candidate_id,
                "record": record,
                "payload": payload,
                "after_sha256": fingerprint(payload.encode()),
            }
            save(path_for(review_id), data)
            return recover(review_id)
        data["decisions"][candidate_id] = record
        save(path_for(review_id), data)
        return data, None


def recover(review_id: str) -> tuple[dict, Path | None]:
    with _lock, annotation_write_lock:
        data = load(review_id)
        intent = data.get("application")
        if not intent:
            return data, None
        record = intent["record"]
        path = target_path(record["video"], record["decision"])
        digest = fingerprint(path.read_bytes()) if path.exists() else None
        if digest not in {record["before_sha256"], intent["after_sha256"]}:
            raise ValueError(
                "Interrupted import conflicts with current annotations; review the saved application journal"
            )
        if digest != intent["after_sha256"]:
            with atomic_write(path) as file:
                file.write(intent["payload"])
        mode = "action" if record["decision"] == "remove_event" else "rally"
        label_done.set_done(Path(record["video"]).stem, mode, False)
        data["decisions"][intent["candidate_id"]] = {
            **record,
            "after_sha256": intent["after_sha256"],
        }
        data["application"] = None
        save(path_for(review_id), data)
        return data, path
