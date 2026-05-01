"""YouTube playlist downloader router."""

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import yt_dlp
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from yp_video.config import RAW_VIDEOS_DIR
from yp_video.web.jobs import job_manager

router = APIRouter()

# Store active download sessions
download_sessions: dict[str, dict[str, Any]] = {}

# Global cap on concurrent yt-dlp downloads across all users.
# yt-dlp does demux/remux which is CPU-heavy, and multiple
# simultaneous downloads also saturate the VM's outbound bandwidth.
_DOWNLOAD_SEMAPHORE = asyncio.Semaphore(2)


class VideoInfo(BaseModel):
    id: str
    title: str
    duration: int | None
    url: str


class PlaylistInfo(BaseModel):
    title: str
    videos: list[VideoInfo]


class DownloadRequest(BaseModel):
    videos: list[VideoInfo]
    quality: str = "best"


class DownloadResponse(BaseModel):
    session_id: str


def get_format_string(quality: str) -> str:
    if quality == "best":
        return "bestvideo[vcodec^=avc1]+bestaudio[acodec^=mp4a]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
    else:
        return f"bestvideo[height<={quality}][vcodec^=avc1]+bestaudio[acodec^=mp4a]/bestvideo[height<={quality}][ext=mp4]+bestaudio[ext=m4a]/best[height<={quality}]"


def normalize_playlist_url(url: str) -> str:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    if "list" in query:
        playlist_id = query["list"][0]
        return f"https://www.youtube.com/playlist?list={playlist_id}"
    return url


@router.get("/playlist")
async def get_playlist(url: str) -> PlaylistInfo:
    normalized_url = normalize_playlist_url(url)

    ydl_opts = {
        "extract_flat": True,
        "quiet": True,
        "no_warnings": True,
    }

    try:
        loop = asyncio.get_event_loop()
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = await loop.run_in_executor(None, ydl.extract_info, normalized_url, False)

        if info is None:
            raise HTTPException(400, "Could not extract playlist info")

        if info.get("_type") != "playlist":
            videos = [VideoInfo(
                id=info.get("id", ""),
                title=info.get("title", "Unknown"),
                duration=info.get("duration"),
                url=info.get("webpage_url") or url,
            )]
            return PlaylistInfo(title=info.get("title", "Single Video"), videos=videos)

        videos = []
        for entry in info.get("entries", []):
            if entry is None:
                continue
            videos.append(VideoInfo(
                id=entry.get("id", ""),
                title=entry.get("title", "Unknown"),
                duration=entry.get("duration"),
                url=entry.get("url") or f"https://www.youtube.com/watch?v={entry.get('id', '')}",
            ))

        return PlaylistInfo(title=info.get("title", "Playlist"), videos=videos)

    except yt_dlp.DownloadError as e:
        raise HTTPException(400, f"Failed to fetch playlist: {str(e)}")
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")


@router.post("/start")
async def start_download(req: DownloadRequest) -> DownloadResponse:
    session_id = str(uuid.uuid4())

    n = len(req.videos)
    label = (
        f"YT download ({n} videos)" if n > 1
        else f"YT download — {req.videos[0].title}" if n == 1
        else "YT download"
    )
    job = job_manager.create_job("download", {"count": n, "quality": req.quality}, name=label)

    download_sessions[session_id] = {
        "videos": req.videos,
        "quality": req.quality,
        "queue": asyncio.Queue(),
        "cancelled": False,
        "task": None,
        "job_id": job.id,
    }

    session = download_sessions[session_id]
    task = asyncio.create_task(download_videos(session_id, req.videos, req.quality))
    session["task"] = task
    # Linking the task to the Job lets the Jobs page's cancel button reach
    # in via job_manager.cancel_job → task.cancel(); download_videos catches
    # CancelledError and mirrors it back into the session queue.
    job_manager.attach_task(job, task)

    return DownloadResponse(session_id=session_id)


async def download_videos(session_id: str, videos: list[VideoInfo], quality: str):
    session = download_sessions.get(session_id)
    if not session:
        return

    queue: asyncio.Queue = session["queue"]
    job_id: str | None = session.get("job_id")
    total = len(videos)
    failed = 0
    RAW_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    if job_id:
        await job_manager.update_job(
            job_id, status="running",
            message=f"Starting download of {total} video(s)...",
        )

    try:
        for i, video in enumerate(videos):
            if session.get("cancelled"):
                await queue.put({"type": "cancelled"})
                if job_id:
                    await job_manager.update_job(job_id, status="cancelled")
                return

            video_id = video.id
            base_progress = i / total
            if job_id:
                await job_manager.update_job(
                    job_id,
                    progress=base_progress,
                    message=f"({i + 1}/{total}) {video.title}",
                )
            await queue.put({
                "type": "start",
                "video_id": video_id,
                "title": video.title,
                "index": i,
                "total": total,
            })

            def make_progress_hook(vid_id: str, q: asyncio.Queue, loop, idx: int):
                # Throttle Job updates so a multi-megabyte file doesn't fire
                # hundreds of SSE events per second; the in-session queue still
                # gets every update for the Download page's byte-level UI.
                last_pushed = [0.0]
                def hook(d):
                    if d["status"] == "downloading":
                        downloaded = d.get("downloaded_bytes", 0)
                        total_b = d.get("total_bytes") or d.get("total_bytes_estimate", 0)
                        speed = d.get("speed", 0)
                        eta = d.get("eta", 0)
                        percent = (downloaded / total_b * 100) if total_b > 0 else 0

                        asyncio.run_coroutine_threadsafe(
                            q.put({
                                "type": "progress",
                                "video_id": vid_id,
                                "percent": percent,
                                "downloaded": downloaded,
                                "total": total_b,
                                "speed": speed,
                                "eta": eta,
                            }),
                            loop,
                        )
                        if job_id:
                            import time as _time
                            now = _time.monotonic()
                            if now - last_pushed[0] >= 1.0:
                                last_pushed[0] = now
                                # Sub-progress within the current video, scaled
                                # into its slice of the overall (idx..idx+1)/total.
                                frac = (idx + (percent / 100.0)) / total
                                speed_str = f" · {speed / 1e6:.1f} MB/s" if speed else ""
                                eta_str = f" · ETA {eta}s" if eta else ""
                                msg = f"({idx + 1}/{total}) {video.title} · {percent:.0f}%{speed_str}{eta_str}"
                                asyncio.run_coroutine_threadsafe(
                                    job_manager.update_job(job_id, progress=frac, message=msg),
                                    loop,
                                )
                    elif d["status"] == "finished":
                        asyncio.run_coroutine_threadsafe(
                            q.put({
                                "type": "finished",
                                "video_id": vid_id,
                                "filename": d.get("filename", ""),
                            }),
                            loop,
                        )
                return hook

            loop = asyncio.get_event_loop()
            ydl_opts = {
                "format": get_format_string(quality),
                "merge_output_format": "mp4",
                "outtmpl": str(RAW_VIDEOS_DIR / "%(title)s.%(ext)s"),
                "progress_hooks": [make_progress_hook(video_id, queue, loop, i)],
                "postprocessor_args": {"ffmpeg": ["-movflags", "+faststart"]},
                "cookiefile": str(Path.home() / "cookies.txt"),
                "js_runtimes": {"node": {}},
                "quiet": True,
                "no_warnings": True,
            }

            try:
                # Limit how many yt-dlp downloads run at once. Queued users
                # still see their session as "pending" rather than timing out.
                async with _DOWNLOAD_SEMAPHORE:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        await loop.run_in_executor(None, ydl.download, [video.url])

                await queue.put({"type": "complete", "video_id": video_id})
            except asyncio.CancelledError:
                raise
            except Exception as e:
                failed += 1
                await queue.put({"type": "error", "video_id": video_id, "error": str(e)})

        await queue.put({"type": "done"})
        if job_id:
            ok = total - failed
            if failed == 0:
                await job_manager.update_job(
                    job_id, status="completed", progress=1.0,
                    message=f"Downloaded {total} video(s)",
                )
            elif failed == total:
                await job_manager.update_job(
                    job_id, status="failed", progress=1.0,
                    message=f"All {total} downloads failed",
                )
            else:
                await job_manager.update_job(
                    job_id, status="completed", progress=1.0,
                    message=f"{ok}/{total} downloaded, {failed} failed",
                )
    except asyncio.CancelledError:
        # Job-page Cancel button calls task.cancel(), which lands here.
        # Mirror the cancellation back into the session queue so the
        # download page UI also reacts, then mark the Job cancelled.
        session["cancelled"] = True
        try:
            await queue.put({"type": "cancelled"})
        except Exception:
            pass
        if job_id:
            await job_manager.update_job(job_id, status="cancelled")
        raise


@router.get("/{session_id}/progress")
async def get_progress(session_id: str):
    session = download_sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    async def event_generator():
        queue: asyncio.Queue = session["queue"]
        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=30.0)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg["type"] in ("done", "cancelled"):
                    if session_id in download_sessions:
                        del download_sessions[session_id]
                    break
            except asyncio.TimeoutError:
                yield "data: {}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.post("/{session_id}/cancel")
async def cancel_download(session_id: str):
    session = download_sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    session["cancelled"] = True
    if session.get("task"):
        session["task"].cancel()

    return {"status": "cancelled"}
