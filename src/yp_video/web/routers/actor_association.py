"""Actor association: the labeling work list, the done flag, confirm and fix.

Serves the Association Label page — which video still has unreviewed actors,
and the one write that answers "this person performed this action". Player
identity is the ReID router's business; the only thing the two share is the
extraction records they both read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import unquote

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from yp_video.actor import labels as actor_labels
from yp_video.core import label_done
from yp_video.core.jsonl import read_jsonl_cached
from yp_video.extraction import actor_fix, links
from yp_video.extraction import done as extraction_done
from yp_video.extraction import store as extraction_store
from yp_video.tracklets import store as tracks_store
from yp_video.tracklets.geometry import TrackRef
from yp_video.web import audit, worklists
from yp_video.web.r2_client import (
    cut_frame_source,
    resolve_cut,
    sync_to_r2,
)
from yp_video.web.schemas import StrictModel

router = APIRouter()


@router.get("/videos")
def list_videos() -> list[dict]:
    return worklists.association_videos()


class DoneRequest(StrictModel):
    done: bool = True


@router.put("/done/{name:path}")
def set_done(name: str, req: DoneRequest) -> dict:
    """Persist the human "actor review is finished" verdict for one video.

    Done also sweep-confirms every current automatic answer (same write as
    the per-rally sweep, video-wide), and reassociation keeps honouring the
    flag afterwards — a predict re-run confirms the answers it invents
    instead of un-reviewing a finished video (extraction/done.py).
    """
    video = resolve_cut(Path(unquote(name)).name)
    if video is None:
        raise HTTPException(404, "Video not found")
    flags = label_done.set_done(video.stem, "association", req.done)
    confirmed = extraction_done.confirm_reviewed(video.stem) if req.done else 0
    return {"done": flags["association"], "confirmed": confirmed}


class ConfirmRequest(StrictModel):
    model_config = ConfigDict(extra="forbid")

    #: None = every automatic pick in the video that has no verdict yet.
    event_ids: list[str] | None = None


@router.post("/confirm/{name}")
def confirm(name: str, req: ConfirmRequest) -> dict:
    """Endorse the policy's picks: "it already got these right" — each lands
    as ``confirmed_auto`` (see actor/labels.confirmations_for).

    Purely an annotation write — the record, the crop and every embedding
    stay exactly as they are, because agreeing with a pick changes nothing
    about it. That is what separates this from a fix, and why it needs none
    of the fix's transaction machinery.

    Events already carrying a verdict are left alone (a human correction
    outranks a bulk confirmation), so the count reports what actually landed.
    """
    stem = Path(unquote(name)).stem
    path = extraction_store.records_path(stem)
    if not path.exists():
        raise HTTPException(404, f"No extraction records for {stem}")

    meta, records = read_jsonl_cached(path)
    records = extraction_store.labelable(
        records, stem, float(meta.get("fps") or 0)
    )
    confirmable = actor_labels.confirmations_for(records)
    if req.event_ids is not None:
        wanted = set(req.event_ids)
        unknown = sorted(wanted - set(confirmable))
        if unknown:
            # Silently dropping these would report a success that did not
            # happen; a miss needs a real verdict, not a confirmation.
            raise HTTPException(
                400,
                "Nothing to endorse — the policy picked nobody for: "
                f"{', '.join(unknown[:5])}"
                + (f" (+{len(unknown) - 5} more)" if len(unknown) > 5 else ""),
            )
        confirmable = {k: v for k, v in confirmable.items() if k in wanted}

    before = _actor_rows(actor_labels.load(stem))
    landed = actor_labels.confirm_auto(stem, confirmable)
    # A bulk endorsement writes many durable verdicts at once. Not folded into
    # a session: it is one click, not a stretch of work.
    audit.record_diff(
        target=stem,
        before=before,
        after=_actor_rows(actor_labels.load(stem)),
        key=lambda r: r["id"],
        confirmed=len(landed),
    )
    return {"confirmed": {event_id: confirmable[event_id].verdict.value for event_id in landed}}


class ActorFixBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1)


class PickActorRequest(ActorFixBase):
    mode: Literal["pick"]
    box: tuple[float, float, float, float]
    # The tracklet clicked, as "{rally_id}:{track_id}". When present the box
    # is only the anchor — the server re-resolves the tracklet to a croppable
    # detection itself, so the crop is reproducible from the label alone.
    track: str | None = Field(default=None, pattern=r"^\d+:\d+$")
    # Cross-frame pick: the box lives on this frame, not the event's — the
    # crop is cut from here (actor undetected on the event frame). Tracklet
    # picks do not need it; the tracklet already spans frames.
    frame: int | None = Field(default=None, ge=0)
    # False = no stored detection is this player, so embed the box as drawn
    # rather than IoU-snapping onto an occluder. Box picks only.
    snap: bool = True

    @property
    def command(self) -> actor_fix.PickActor:
        return actor_fix.PickActor(
            mode="pick",
            event_id=self.event_id,
            box=self.box,
            track=TrackRef.parse(self.track) if self.track else None,
            frame=self.frame,
            snap=self.snap,
        )


class OccludedActorRequest(ActorFixBase):
    mode: Literal["occluded"]

    @property
    def command(self) -> actor_fix.MarkOccluded:
        return actor_fix.MarkOccluded(mode="occluded", event_id=self.event_id)


class AutoActorRequest(ActorFixBase):
    mode: Literal["auto"]

    @property
    def command(self) -> actor_fix.RevertActor:
        return actor_fix.RevertActor(mode="auto", event_id=self.event_id)


ActorFixRequest = Annotated[
    PickActorRequest | OccludedActorRequest | AutoActorRequest,
    Field(discriminator="mode"),
]


def _actor_rows(labels) -> list[dict]:
    """The video's actor verdicts as records, for auditing.

    One per event that carries a human verdict. The payload is what actually
    lands on disk, so a diff of two of these is exactly what the save changed.
    """
    return [{"id": event_id, **label.payload()} for event_id, label in labels.items()]


@router.post("/fix/{name}")
def fix(
    name: str, req: ActorFixRequest, background_tasks: BackgroundTasks
) -> dict:
    """Re-point one event at the person the user clicked (or nobody / auto).

    The verdict lands in the video's actor labels (the durable human record,
    replayed on re-extraction) and is applied to the extraction record
    immediately: the chosen box is cropped. The crop is re-embedded in the
    background, so the identity clusters that read it follow shortly after.
    """
    # Cuts live in R2; the crop needs one frame, so an R2-only cut is read
    # over a presigned URL rather than downloaded.
    video_path = resolve_cut(Path(unquote(name)).name)
    if video_path is None:
        raise HTTPException(404, f"Video not found: {name}")
    stem = video_path.stem
    if not extraction_store.records_path(stem).exists():
        raise HTTPException(404, f"No extraction records for {stem}")

    before = _actor_rows(actor_labels.load(stem))
    command: actor_fix.ActorFixCommand = req.command
    try:
        result = actor_fix.apply(stem, cut_frame_source(video_path), command)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    sync_to_r2(actor_labels.actors_path(stem), "association/annotations")

    current = extraction_store.with_current_actions([result.record], stem)
    record = current[0] if current else result.record
    label = command.label
    record["actor_review"] = label.verdict.value if label else "unreviewed"
    # Sparse, recomputed AFTER the fix landed: present only when the fresh
    # label still resolves to no tracklet (see links.unresolved_labels) — a
    # box pick that lands on a tracked player clears the flag by resolving.
    if label is not None and req.event_id in links.unresolved_labels(stem):
        record["actor_review_unresolved"] = True
    track_link = None
    if tracks_store.tracks_path(stem).exists():
        ref = links.event_tracks(stem).get(req.event_id)
        track_link = ref.payload() if ref else None
    # Association is labeling work like the other three panels: one call per
    # event the reviewer re-points. Folded into a session (see audit's
    # _COALESCING) so an afternoon of it reads as hours worked rather than as
    # hundreds of instantaneous rows totalling nothing.
    audit.record_diff(
        target=stem,
        before=before,
        after=_actor_rows(actor_labels.load(stem)),
        key=lambda r: r["id"],
        event=req.event_id,
    )
    background_tasks.add_task(
        actor_fix.refresh_deferred,
        stem,
        req.event_id,
        models=result.refreshing_models,
        expected_revision=result.actor_revision,
    )
    return {
        "record": record,
        "track_link": track_link,
        "refreshing_models": result.refreshing_models,
    }
