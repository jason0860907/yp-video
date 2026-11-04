"""Browse any VolleyIQ user's App library and review their corrections."""

import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypedDict

from botocore.exceptions import BotoCoreError, ClientError
from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import Field

from yp_video.core.ffmpeg import FFmpegError, probe_video_metadata
from yp_video.web.access import current_actor
from yp_video.web.app_review import feedback, sources
from yp_video.web.app_review.feedback import (
    Candidate,
    DecisionKind,
    ReviewRecord,
    ReviewSummary,
)
from yp_video.web.app_review.models import Bundle, Window
from yp_video.web.app_review.projection import Preview, project
from yp_video.web.r2_client import all_cut_paths, sync_to_r2
from yp_video.web.schemas import StrictModel

router = APIRouter()


class AppUser(TypedDict):
    """A VolleyIQ App user as the Worker's admin API lists them."""

    id: str
    provider: str
    email: str | None
    display_name: str | None
    created_at: int
    #: Live (not deleted) matches.
    match_count: int
    updated_at: int


class MatchEditCounts(TypedDict):
    #: Candidate kind (``edit_kind``) -> count.
    edits: dict[str, int]
    #: When the user last edited (corrections.updated_at); None without any.
    edited_at: str | None
    warnings: list[str]


class MatchReadError(TypedDict):
    #: Why the match could not be read.
    error: str


class MatchDetail(TypedDict):
    #: The library row as the Worker's admin API serves it (owned there).
    match: dict[str, Any]
    preview: Preview
    candidates: list[Candidate]
    #: The review of exactly this snapshot, if one exists.
    review_id: str | None
    #: Every review of this match, newest first.
    reviews: list[ReviewSummary]


class VideoMeta(TypedDict):
    """The source's probe (core/ffmpeg.probe_video_metadata)."""

    fps: float
    duration: float
    num_frames: int
    start_time: float


class Review(ReviewRecord):
    """A review with its candidates and the preview it judges."""

    candidates: list[Candidate]
    preview: Preview


class AppReviewVideo(TypedDict):
    #: A pipeline cut's basename.
    name: str


class TargetRevisions(TypedDict):
    """Per label-edit operation: the target label file's fingerprint, or
    None when the file does not exist."""

    remove_event: str | None
    remove_rally: str | None
    update_rally: str | None


def checked(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except sources.AdminApiError as exc:
        raise HTTPException(502, str(exc)) from exc
    except sources.NotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    except (BotoCoreError, ClientError) as exc:
        raise HTTPException(
            502, "Could not read cloud data — check that the analysis / identification files exist and R2 is readable."
        ) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


def detail(data: ReviewRecord, mode: Window = "full_play", corrected: bool = True) -> Review:
    bundle = Bundle.model_validate(data["bundle"])
    return {
        **data,
        "candidates": feedback.candidates(bundle),
        "preview": project(bundle, mode=mode, corrected=corrected),
    }


@router.get("/users")
def users() -> list[AppUser]:
    return checked(sources.users)


@router.get("/users/{user}/library")
def library(user: str) -> dict[str, Any]:
    """The user's library as the Worker's admin API serves it — the App's
    /sync plus R2 keys and latest jobs. Its shape is owned there
    (upload-service src/admin), so it passes through undeclared."""
    return checked(sources.library, user)


@router.get("/users/{user}/overview")
def overview(user: str) -> dict[str, MatchEditCounts | MatchReadError]:
    """Per analysed match: the user's corrections counted by kind and when
    they last edited, or why that match could not be read."""
    lib = checked(sources.library, user)
    rows = [m for m in lib["matches"] if m["deleted_at"] is None and m["r2_keys"]["result"]]

    def summarize(row: dict) -> tuple[str, MatchEditCounts | MatchReadError]:
        try:
            bundle, notes = sources.match_bundle(lib, row)
        except (BotoCoreError, ClientError, ValueError) as exc:
            return row["id"], {"error": str(exc)}
        return row["id"], {
            "edits": Counter(edit_kind(c) for c in feedback.candidates(bundle)),
            "edited_at": bundle.corrections.updated_at if bundle.corrections else None,
            "warnings": notes,
        }

    with ThreadPoolExecutor(8) as pool:
        return dict(pool.map(summarize, rows))


def edit_kind(candidate: dict) -> str:
    """A candidate's scope, with player numbers split into the user's
    per-touch overrides and the touches a named person group numbers."""
    if candidate["scope"] != "player":
        return candidate["scope"]
    return "player_override" if candidate["value"]["override"] else "player_group"


@router.get("/users/{user}/matches/{match}")
def match(
    user: str, match: str, mode: Window = "full_play", corrected: bool = True
) -> MatchDetail:
    lib, row = checked(sources.library_match, user, match)
    bundle, notes = checked(sources.match_bundle, lib, row)
    preview = project(bundle, mode=mode, corrected=corrected)
    return {
        "match": row,
        "preview": {**preview, "warnings": notes + preview["warnings"]},
        "candidates": feedback.candidates(bundle),
        "review_id": feedback.existing(bundle),
        "reviews": feedback.list_reviews(row["id"]),
    }


@router.get("/users/{user}/matches/{match}/video")
def match_video(user: str, match: str):
    _, row = checked(sources.library_match, user, match)
    return RedirectResponse(checked(sources.video_url, row))


@router.get("/users/{user}/matches/{match}/video/meta")
def match_video_meta(user: str, match: str) -> VideoMeta:
    """``{fps, duration, num_frames, start_time}`` of the source: the App's
    result is in seconds, the player's frame clock needs the real rate."""
    _, row = checked(sources.library_match, user, match)
    try:
        return probe_video_metadata(checked(sources.video_url, row))
    except FFmpegError as exc:
        raise HTTPException(502, str(exc)) from exc


@router.post("/users/{user}/matches/{match}/review")
def start_review(user: str, match: str) -> Review:
    lib, row = checked(sources.library_match, user, match)
    bundle, _ = checked(sources.match_bundle, lib, row)
    return detail(checked(feedback.create, bundle, current_actor()))


@router.get("/reviews/{review_id}")
def review(review_id: str, mode: Window = "full_play", corrected: bool = True) -> Review:
    return detail(checked(feedback.load, review_id), mode=mode, corrected=corrected)


@router.get("/videos")
def videos() -> list[AppReviewVideo]:
    return [{"name": p.name} for p in sorted(all_cut_paths())]


@router.get("/target")
def target(video: str) -> TargetRevisions:
    return {
        op: checked(feedback.target_revision, video, op)
        for op in ("remove_event", "remove_rally", "update_rally")
    }


class DecisionRequest(StrictModel):
    candidate_id: str
    decision: DecisionKind
    note: str = Field(default="", max_length=4000)
    video: str = ""
    revision: str = ""
    same_source: bool = False


def after_apply(data: ReviewRecord, path) -> Review:
    if path:
        category = (
            "action/annotations"
            if path.name.endswith("_actions.jsonl")
            else "rally-spot/annotations"
        )
        sync_to_r2(path, category)
    return detail(data)


@router.post("/reviews/{review_id}/decisions")
async def decide(review_id: str, req: DecisionRequest) -> Review:
    data, path = await asyncio.to_thread(
        checked,
        feedback.decide,
        review_id,
        req.candidate_id,
        req.decision,
        req.note,
        current_actor(),
        video=req.video,
        revision=req.revision,
        same_source=req.same_source,
    )
    return after_apply(data, path)


@router.post("/reviews/{review_id}/recover")
async def recover(review_id: str) -> Review:
    data, path = await asyncio.to_thread(checked, feedback.recover, review_id)
    return after_apply(data, path)
