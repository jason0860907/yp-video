"""App-equivalent playback and explicit, attributable feedback review."""

import asyncio
from typing import Literal

from botocore.exceptions import BotoCoreError, ClientError
from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import Field

from yp_video.web.access import current_actor
from yp_video.web.app_review import feedback, sources
from yp_video.web.app_review.models import Bundle, Window
from yp_video.web.app_review.projection import project
from yp_video.web.r2_client import all_cut_paths, sync_to_r2
from yp_video.web.schemas import StrictModel

router = APIRouter()


def checked(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except (BotoCoreError, ClientError) as exc:
        raise HTTPException(
            502, "無法讀取雲端資料，請確認所選分析／辨識檔案存在及 R2 讀取權限。"
        ) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


def detail(data: dict, mode: Window = "full_play", corrected: bool = True) -> dict:
    bundle = Bundle.model_validate(data["bundle"])
    return {
        **data,
        "candidates": feedback.candidates(bundle),
        "preview": project(bundle, mode=mode, corrected=corrected),
    }


@router.get("/videos")
def videos():
    return [{"name": p.name} for p in sorted(all_cut_paths())]


@router.get("/local")
def local(video: str, mode: Window = "full_play"):
    bundle = checked(sources.local_bundle, video)
    return {"bundle": bundle.model_dump(), "preview": project(bundle, mode=mode)}


class PreviewRequest(StrictModel):
    bundle: Bundle
    mode: Window = "full_play"
    corrected: bool = True


@router.post("/preview")
def preview(req: PreviewRequest):
    return project(req.bundle, mode=req.mode, corrected=req.corrected)


@router.get("/cloud")
def cloud():
    return checked(sources.cloud_matches)


@router.get("/cloud/results")
def cloud_results(user: str, match: str):
    return checked(sources.cloud_results, user, match)


class CloudRequest(StrictModel):
    user: str
    match: str
    job: str


@router.post("/cloud/import")
def import_cloud(req: CloudRequest):
    bundle = checked(sources.cloud_bundle, req.user, req.match, req.job)
    return detail(checked(feedback.create, bundle, current_actor()))


@router.get("/reviews")
def reviews():
    return feedback.list_reviews()


@router.post("/reviews")
def create(req: Bundle):
    return detail(checked(feedback.create, req, current_actor()))


@router.get("/reviews/{review_id}")
def review(review_id: str, mode: Window = "full_play", corrected: bool = True):
    return detail(checked(feedback.load, review_id), mode=mode, corrected=corrected)


@router.get("/reviews/{review_id}/video")
def video(review_id: str):
    data = checked(feedback.load, review_id)
    key = data["bundle"]["result"]["video_r2_key"]
    if not key or not sources.customer.configured:
        raise HTTPException(
            404, "No customer video available; select the matching pipeline video"
        )
    return RedirectResponse(sources.customer.generate_presigned_url(key))


@router.get("/target")
def target(video: str):
    return {
        op: checked(feedback.target_revision, video, op)
        for op in ("remove_event", "remove_rally", "update_rally")
    }


class Decision(StrictModel):
    candidate_id: str
    decision: Literal[
        "accepted_feedback", "rejected", "remove_event", "remove_rally", "update_rally"
    ]
    note: str = Field(default="", max_length=4000)
    video: str = ""
    revision: str = ""
    same_source: bool = False


def after_apply(data: dict, path):
    if path:
        category = (
            "action/annotations"
            if path.name.endswith("_actions.jsonl")
            else "rally-spot/annotations"
        )
        sync_to_r2(path, category)
    return detail(data)


@router.post("/reviews/{review_id}/decisions")
async def decide(review_id: str, req: Decision):
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
async def recover(review_id: str):
    data, path = await asyncio.to_thread(checked, feedback.recover, review_id)
    return after_apply(data, path)
