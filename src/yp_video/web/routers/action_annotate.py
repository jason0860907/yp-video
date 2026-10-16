"""SPOT-style point action annotator router."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Literal
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import Field, field_validator

from yp_video.action.attributes import attribute_defaults
from yp_video.config import (
    ACTION_ANNOTATIONS_DIR,
    CUT_R2_CATEGORIES,
)
from yp_video.contracts.action import (
    ACTION_LABELS_ORDERED,
    LABEL_FILE_GLOB,
    CourtSide,
    event_id,
)
from yp_video.core import label_done
from yp_video.core.jsonl import read_jsonl
from yp_video.core.rallies import load_rallies
from yp_video.web import audit, worklists
from yp_video.web.action_annotations import (
    annotation_path,
    load_annotation,
    normalize_events,
    persistable_events,
    pre_annotation_path,
    write_annotation_atomic,
)
from yp_video.web.action_waveform import audio_waveform, video_metadata
from yp_video.web.r2_client import (
    resolve_cut,
    serve_video_or_r2_redirect,
    sync_to_r2,
)
from yp_video.web.schemas import StrictModel

log = logging.getLogger(__name__)
router = APIRouter()

ACTION_LABELS = ACTION_LABELS_ORDERED


class ActionEvent(StrictModel):
    id: str | None = None
    rally_id: int | None = None
    frame: int = Field(ge=0)
    time: float | None = None
    relative_frame: int | None = None
    label: str
    xy: tuple[float, float]
    visible: bool = True
    side: CourtSide | None = None
    jump: bool | None = None

    @field_validator("label")
    @classmethod
    def validate_label(cls, value: str) -> str:
        if value not in ACTION_LABELS:
            raise ValueError(f"label must be one of: {', '.join(ACTION_LABELS)}")
        return value

    @field_validator("xy")
    @classmethod
    def validate_xy(cls, value: tuple[float, float]) -> tuple[float, float]:
        x, y = value
        if not (0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError("xy must be normalized to [0, 1]")
        return value


class SaveActionAnnotationsRequest(StrictModel):
    video: str
    fps: float = Field(gt=0)
    num_frames: int = Field(ge=0)
    events: list[ActionEvent]


def _load_rallies(video: Path) -> list[dict]:
    """This video's rally spans (see core/rallies.py for the source priority)."""
    return load_rallies(video.stem)


def _prior_events(output_path: Path) -> list[dict]:
    """The events currently on disk, or [] when there is no file yet."""
    if not output_path.exists():
        return []
    try:
        return read_jsonl(output_path)[1]
    except (OSError, ValueError, json.JSONDecodeError):
        # An unreadable prior file must not block the save; it only means
        # every event counts as new in the audit summary.
        return []


@router.get("/labels")
def labels() -> dict:
    return {"labels": list(ACTION_LABELS)}


@router.get("/videos")
def list_videos() -> list[dict]:
    return worklists.action_videos()


class DoneRequest(StrictModel):
    done: bool = True


@router.put("/done/{name:path}")
def set_done(name: str, req: DoneRequest) -> dict:
    """Persist the human "action labeling is finished" verdict for one video.

    Deliberately not tied to saving: Save writes the human store while this
    flag is the separate "I'm finished" claim.
    """
    video = resolve_cut(Path(unquote(name)).name)
    if video is None:
        raise HTTPException(404, "Video not found")
    flags = label_done.set_done(video.stem, "action", req.done)
    return {"done": flags["action"]}


@router.get("/annotations/{name:path}")
async def get_annotations(
    name: str,
    source: Literal["annotation", "pre-annotation"],
) -> dict:
    """One video's action annotation from exactly the requested store.

    Deterministic on purpose — the Source select means what it says, no
    automatic fallback. A store with no file is an empty editor
    (``loaded_source: null``), not an error: labeling can start from
    scratch on either view.
    """
    decoded = unquote(name)
    video = resolve_cut(Path(decoded).name)
    if video is None:
        raise HTTPException(404, "Video not found")

    meta = await asyncio.to_thread(video_metadata, video)
    rallies = await asyncio.to_thread(_load_rallies, video)
    path = annotation_path(video.name) if source == "annotation" else pre_annotation_path(video.name)
    forced = await asyncio.to_thread(load_annotation, path)
    if forced is not None:
        # Shallow copies throughout — the cached dict is shared; every key
        # below is reassigned wholesale, and normalize_events copies each
        # event.
        ann = dict(forced)
        # Which store this payload came from — the editor's "what am I
        # looking at" badge; the file's own provenance stays in `source`.
        ann["loaded_source"] = source
        ann.setdefault("video", video.stem)
        ann["source_video"] = video.name
        ann.setdefault("fps", meta["fps"])
        ann.setdefault("num_frames", meta["num_frames"])
        ann["rallies"] = rallies
        ann["events"] = normalize_events(
            video.stem,
            ann.get("events", []),
            fps=float(ann["fps"]),
            num_frames=int(ann["num_frames"]),
            rallies=rallies,
        )
        ann["num_events"] = len(ann["events"])
        ann["duration"] = meta["duration"]
        # What training assumes for an event with no stored side / jump —
        # shown beside the editor's own value, never saved.
        defaults, _ = attribute_defaults(rallies, ann["events"], float(ann["fps"]))
        ann["attribute_defaults"] = {
            event["id"]: default for event, default in zip(ann["events"], defaults)
        }
        return ann

    return {
        "video": video.stem,
        "source_video": video.name,
        "loaded_source": None,
        "duration": meta["duration"],
        "fps": meta["fps"],
        "num_frames": meta["num_frames"],
        "num_events": 0,
        "rallies": rallies,
        "events": [],
        "attribute_defaults": {},
    }


@router.get("/waveform/{name:path}")
async def get_waveform(name: str, points: int = Query(default=9600, ge=200, le=96000)) -> dict:
    decoded = unquote(name)
    video = resolve_cut(Path(decoded).name)
    if video is None:
        raise HTTPException(404, "Video not found")
    return await asyncio.to_thread(audio_waveform, video, points)


@router.post("/annotations")
async def save_annotations(req: SaveActionAnnotationsRequest) -> dict:
    video = resolve_cut(Path(req.video).name)
    if video is None:
        raise HTTPException(404, "Video not found")

    ACTION_ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    events = persistable_events(normalize_events(
        video.stem,
        [event.model_dump(mode="json") for event in req.events],
        fps=req.fps,
        num_frames=req.num_frames,
        rallies=[],
    ))
    data = {
        "video": video.stem,
        "num_frames": req.num_frames,
        "fps": req.fps,
        "source": {"type": "manual"},
        "num_events": len(events),
        "events": events,
    }
    output_path = annotation_path(video.name)
    before = await asyncio.to_thread(_prior_events, output_path)
    await asyncio.to_thread(write_annotation_atomic, output_path, data)
    sync_to_r2(output_path, "action/annotations")

    # Same autosave timer as the rally editor: an unchanged rewrite is not an
    # edit, so it leaves no audit row.
    audit.record_diff(
        target=video.stem,
        before=before,
        after=events,
        key=event_id,
        events=len(events),
    )
    return {"saved": str(output_path), "count": len(events)}


@router.get("/export")
def export_dataset() -> Response:
    ACTION_ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted(ACTION_ANNOTATIONS_DIR.glob(LABEL_FILE_GLOB)):
        data = load_annotation(path)
        if data is not None:
            # The export is self-contained: rallies and the rally-derived
            # event fields are joined fresh from the live rally store here,
            # never stored in the annotation file.
            stem = data.get("video", path.stem.removesuffix("_actions"))
            rallies = load_rallies(stem)
            events = normalize_events(
                stem,
                data.get("events", []),
                fps=float(data["fps"]),
                num_frames=int(data.get("num_frames") or 0),
                rallies=rallies,
            )
            records.append({
                "video": stem,
                "num_frames": data.get("num_frames", 0),
                "fps": data.get("fps", 0),
                "rallies": rallies,
                "num_events": len(events),
                "events": events,
            })
    lines = [
        json.dumps(
            {
                "_meta": True,
                "type": "spot_action_annotations",
                "num_videos": len(records),
                "num_events": sum(record["num_events"] for record in records),
            },
            ensure_ascii=False,
        )
    ]
    lines.extend(json.dumps(record, ensure_ascii=False) for record in records)
    return Response(
        "\n".join(lines) + "\n",
        media_type="application/x-ndjson",
        headers={"Content-Disposition": 'attachment; filename="spot_action_annotations.jsonl"'},
    )


@router.get("/video/{path:path}")
def stream_video(path: str):
    decoded_path = unquote(path)
    video_path = resolve_cut(Path(decoded_path).name)
    if video_path is None:
        raise HTTPException(404, "Video not found")
    response = serve_video_or_r2_redirect(video_path, CUT_R2_CATEGORIES)
    if response:
        return response
    raise HTTPException(404, "Video not found")
