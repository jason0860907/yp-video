#!/usr/bin/env python3
"""Test Qwen3.6-VL's ability to describe rally segments from rally-annotations.

For each rally segment in an annotations JSONL, extracts the clip with ffmpeg
and asks the VLM on the local vLLM server what happened.

Usage:
    # Pick the first annotation file found, describe up to 3 rallies:
    python scripts/describe_rally_segments.py

    # A specific file, first 5 rallies:
    python scripts/describe_rally_segments.py \
        --annotations "/home/jason_yp_wang/videos/rally-annotations/0104排島臨打 1_annotations.jsonl" \
        --limit 5

    # Skip ahead in the file:
    python scripts/describe_rally_segments.py --start-index 10 --limit 3
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import requests

from yp_video.config import ANNOTATIONS_DIR, VIDEOS_DIR, find_cut, load_tokens_env, load_vllm_env
from yp_video.core.ffmpeg import FFmpegError, extract_clip


DESCRIPTIONS_DIR = VIDEOS_DIR / "rally-descriptions"


DESCRIBE_PROMPT = """你是排球影片標註助手。請只描述「影片中清楚看得到」的 rally 事件,不要補完整劇情。

【硬性規則】
1. 只輸出 JSON,不要 Markdown,不要前言或後記。
2. 每個事件都必須有可見證據 evidence。沒有證據就不要寫該事件。
3. 看不清楚的欄位填 "不明",不要猜球員角色、攻擊路線、攔網人數或得分類型。
4. 如果只能確定球被觸碰,但不能確定技術類型,action 用 "touch",detail 用 "不明"。
5. 如果影片內沒有看到 rally 結束,ended_in_clip 填 false,end_reason 填 "影片內未結束"。
6. 如果看到 rally 結束,只填你能從畫面確認的 end_reason；不確定就填 "不明"。
7. 不要描述觀眾、氣氛、攝影機、計分板、廣告。

【允許的 action】
"serve","receive","set","attack","block","dig","free_ball","touch","end"

【信心 confidence】
"high": 畫面清楚看到球和觸球動作。
"medium": 大致看得到球或球員動作,但細節不完整。
"low": 只能從球路或球員反應推測,請保守使用。

【輸出格式】
{
  "events": [
    {
      "time": "0.0-1.0",
      "side": "left/right/不明",
      "action": "serve/receive/set/attack/block/dig/free_ball/touch/end",
      "detail": "繁體中文短句；不確定填不明",
      "confidence": "high/medium/low",
      "evidence": "畫面中可見的依據"
    }
  ],
  "ended_in_clip": true/false,
  "end_reason": "扣殺得分/攔網得分/球出界/觸網/對方失誤/發球得分/影片內未結束/不明"
}"""


def load_rally_segments(jsonl_path: Path) -> tuple[dict, list[dict]]:
    """Return (meta, [rally segments]) from a rally-annotations jsonl."""
    meta: dict = {}
    segments: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("_meta"):
                meta = row
                continue
            if row.get("label") == "rally":
                segments.append(row)
    return meta, segments


def resolve_video(annotations_path: Path, meta: dict) -> Path:
    """Find the source video for an annotations file."""
    stem = annotations_path.name.removesuffix(".jsonl").removesuffix("_annotations")
    cut = find_cut(f"{stem}.mp4")
    if cut is not None:
        return cut
    # Fall back to the path embedded in _meta.
    meta_video = meta.get("video")
    if meta_video and Path(meta_video).exists():
        return Path(meta_video)
    raise FileNotFoundError(f"Cannot locate source video for {annotations_path.name}")


_last_gemini_call_time: float = 0.0


def describe_clip_gemini(clip_path: Path, model: str, api_key: str, min_interval: float = 0.0, video_fps: float = 10.0) -> str:
    """Send a video clip to Gemini and return the text description.

    Uses inline base64 upload (fine for clips up to ~20MB; larger files would
    need the Files API).
    """
    # Lazy import so users without google-genai installed can still use the vllm path.
    from google import genai
    from google.genai import types

    global _last_gemini_call_time
    if min_interval > 0:
        wait = min_interval - (time.time() - _last_gemini_call_time)
        if wait > 0:
            time.sleep(wait)

    client = genai.Client(api_key=api_key)
    video_bytes = clip_path.read_bytes()
    _last_gemini_call_time = time.time()
    contents = types.Content(parts=[
        types.Part(
            inline_data=types.Blob(mime_type="video/mp4", data=video_bytes),
            video_metadata=types.VideoMetadata(fps=video_fps),
        ),
        types.Part(text=DESCRIBE_PROMPT),
    ])
    config = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=900,
        # Keep a small thinking budget for video grounding while avoiding long
        # speculative play-by-play traces.
        thinking_config=types.ThinkingConfig(thinking_budget=256),
    )
    # Retry on free-tier RPM throttling (5 req/min on gemini-2.5-flash).
    for attempt in range(4):
        try:
            response = client.models.generate_content(model=model, contents=contents, config=config)
            return (response.text or "").strip()
        except Exception as e:
            msg = str(e)
            if "429" not in msg and "RESOURCE_EXHAUSTED" not in msg:
                raise
            m = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", msg)
            wait = float(m.group(1)) + 1 if m else 15.0
            print(f"  [rate-limited, sleeping {wait:.0f}s]")
            time.sleep(wait)
    raise RuntimeError("Gemini rate-limit retries exhausted")


def describe_clip(clip_path: Path, server_url: str, model: str, fps: float) -> str:
    """Send a video clip to vLLM and return the text description.

    Passes both `media_io_kwargs.video.fps` (to override vLLM's video-loader
    default num_frames=32 cap) and `mm_processor_kwargs.fps` (so the HF Qwen3-VL
    processor uses the same rate when re-sampling).
    """
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": f"file://{clip_path}"}},
                    {"type": "text", "text": DESCRIBE_PROMPT},
                ],
            }
        ],
        "max_tokens": 900,
        "temperature": 0.0,
        # Anti-loop: frequency_penalty discourages re-using tokens the model has
        # already produced (so "右方攔網 → 左方接發 → 右方攔網 → 左方接發..." gets
        # penalised). presence_penalty bumps any token once it's appeared.
        # Both are needed because Qwen3.6 falls into 4-token cycles on volleyball
        # rallies and thinking-mode is too expensive to enable here.
        "frequency_penalty": 0.4,
        "presence_penalty": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
        "mm_processor_kwargs": {"fps": fps},
        # num_frames=-1 disables vLLM's video-loader cap (defaults to 32 via
        # limit_mm_per_prompt). Without this, fps=10 silently gets truncated
        # to ~32 frames regardless of clip length.
        # frame_recovery: forward-scan past broken frames to find next valid
        # frame — needed because extract_clip uses `-c:v copy` keyframe seek
        # which leaves many non-keyframe positions unreadable by opencv.
        "media_io_kwargs": {"video": {"num_frames": -1, "fps": fps, "frame_recovery": True}},
    }
    r = requests.post(
        f"{server_url}/v1/chat/completions",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=300,
    )
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    content = msg.get("content")
    if content:
        return content.strip()
    # Thinking burned through max_tokens without producing a final answer; surface
    # the reasoning so it isn't silently lost.
    reasoning = (msg.get("reasoning") or msg.get("reasoning_content") or "").strip()
    return f"[no final answer — reasoning only]\n{reasoning[:400]}"


def pick_default_annotations() -> Path:
    files = sorted(ANNOTATIONS_DIR.glob("*_annotations.jsonl"))
    if not files:
        raise FileNotFoundError(f"No annotations found in {ANNOTATIONS_DIR}")
    return files[0]


def build_chunk_ranges(start: float, duration: float, chunk_seconds: float) -> list[tuple[float, float]]:
    """Return absolute (chunk_start, chunk_duration) pairs for one rally."""
    if duration <= 0:
        return []
    if chunk_seconds <= 0 or duration <= chunk_seconds:
        return [(start, duration)]

    chunks: list[tuple[float, float]] = []
    offset = 0.0
    while offset < duration:
        chunk_duration = min(chunk_seconds, duration - offset)
        chunks.append((start + offset, chunk_duration))
        offset += chunk_seconds
    return chunks


def main() -> int:
    cfg = load_vllm_env()
    default_server = f"http://localhost:{cfg.get('VLLM_PORT', '8001')}"
    default_vllm_model = cfg.get("VLLM_MODEL", "")
    default_gemini_model = "gemini-2.5-flash"

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--annotations", "-a", type=Path, default=None,
                   help="Path to a rally-annotations JSONL. Defaults to first file in ~/videos/rally-annotations/.")
    p.add_argument("--limit", "-n", type=int, default=3, help="Max rally segments to describe (default: 3).")
    p.add_argument("--start-index", "-i", type=int, default=0, help="Skip this many rallies before starting (default: 0).")
    p.add_argument("--backend", choices=("vllm", "gemini"), default="vllm",
                   help="Which VLM backend to call (default: vllm).")
    p.add_argument("--server", "-s", default=default_server, help=f"vLLM URL (default: {default_server}, vllm backend only).")
    p.add_argument("--model", "-m", default=None,
                   help=f"Model name. Defaults: vllm={default_vllm_model!r}, gemini={default_gemini_model!r}.")
    p.add_argument("--fps", type=float, default=10.0, help="Frames per second sampled by vLLM (default: 10.0, vllm backend only).")
    p.add_argument("--gemini-rpm", type=float, default=4.5,
                   help="Throttle Gemini calls to this many requests/min (default: 4.5 = safe under free-tier 5 RPM). Set to 0 to disable.")
    p.add_argument("--gemini-video-fps", type=float, default=10.0,
                   help="Frames per second Gemini samples from the video (default: 10.0, max ~10).")
    p.add_argument("--max-seconds", type=float, default=20.0,
                   help="Cap on rally clip length sent to VLM (default: 20s). Longer rallies get truncated.")
    p.add_argument("--chunk-seconds", type=float, default=5.0,
                   help="Analyze each rally in chunks of this many seconds (default: 5.0). Set <=0 to send one clip.")
    p.add_argument("--output", "-o", type=Path, default=None,
                   help=f"Output JSONL path. Defaults to {DESCRIPTIONS_DIR}/<stem>_descriptions.jsonl.")
    p.add_argument("--append", action="store_true",
                   help="Append to the output file instead of overwriting (handy when running with --start-index in chunks).")
    args = p.parse_args()

    if args.model is None:
        args.model = default_gemini_model if args.backend == "gemini" else default_vllm_model

    gemini_api_key = ""
    if args.backend == "gemini":
        tokens = load_tokens_env()
        gemini_api_key = tokens.get("GEMINI_API_KEY", "").strip()
        if not gemini_api_key:
            print("error: GEMINI_API_KEY missing from tokens.env", file=sys.stderr)
            return 1

    annotations_path = args.annotations or pick_default_annotations()
    if not annotations_path.exists():
        print(f"error: annotations file not found: {annotations_path}", file=sys.stderr)
        return 1

    meta, segments = load_rally_segments(annotations_path)
    if not segments:
        print(f"error: no rally segments in {annotations_path}", file=sys.stderr)
        return 1

    video_path = resolve_video(annotations_path, meta)

    if args.output is None:
        stem = annotations_path.name.removesuffix(".jsonl").removesuffix("_annotations")
        suffix = "descriptions_gemini" if args.backend == "gemini" else "descriptions"
        output_path = DESCRIPTIONS_DIR / f"{stem}_{suffix}.jsonl"
    else:
        output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("─" * 70)
    print(f"Annotations: {annotations_path}")
    print(f"Video:       {video_path}")
    print(f"Server:      {args.server}")
    print(f"Model:       {args.model}")
    print(f"Output:      {output_path}{' (append)' if args.append else ''}")
    print(f"Rallies:     {len(segments)} total, describing {args.limit} starting at index {args.start_index}")
    print("─" * 70)

    selected = segments[args.start_index : args.start_index + args.limit]
    if not selected:
        print("error: start-index past end of rally list", file=sys.stderr)
        return 1

    out_fh = open(output_path, "a" if args.append else "w", encoding="utf-8")
    try:
        if not args.append:
            out_fh.write(json.dumps({
                "_meta": True,
                "video": str(video_path),
                "annotations": str(annotations_path),
                "model": args.model,
                "fps": args.fps,
                "max_seconds": args.max_seconds,
                "chunk_seconds": args.chunk_seconds,
            }, ensure_ascii=False) + "\n")
            out_fh.flush()
        with tempfile.TemporaryDirectory(dir=os.path.dirname(str(video_path))) as tmpdir:
            for offset, seg in enumerate(selected):
                idx = args.start_index + offset
                start = float(seg["start"])
                end = float(seg["end"])
                duration = min(end - start, args.max_seconds)
                chunks = build_chunk_ranges(start, duration, args.chunk_seconds)
                print(
                    f"\n[rally {idx}] {start:7.2f}s → {end:7.2f}s  "
                    f"(analyzing {duration:.1f}s in {len(chunks)} chunk(s))"
                )

                chunk_results = []
                failed = False
                for chunk_i, (chunk_start, chunk_duration) in enumerate(chunks, 1):
                    chunk_end = chunk_start + chunk_duration
                    clip_path = Path(tmpdir) / f"rally_{idx:05d}_chunk_{chunk_i:02d}.mp4"
                    print(f"  [chunk {chunk_i}/{len(chunks)}] {chunk_start:7.2f}s → {chunk_end:7.2f}s")
                    try:
                        extract_clip(str(video_path), chunk_start, chunk_duration, str(clip_path))
                    except FFmpegError as e:
                        print(f"    [extract failed] {e}")
                        failed = True
                        break

                    try:
                        if args.backend == "gemini":
                            min_interval = 60.0 / args.gemini_rpm if args.gemini_rpm > 0 else 0
                            description = describe_clip_gemini(
                                clip_path, args.model, gemini_api_key,
                                min_interval=min_interval, video_fps=args.gemini_video_fps,
                            )
                        else:
                            description = describe_clip(clip_path, args.server, args.model, args.fps)
                    except requests.HTTPError as e:
                        body = e.response.text[:300] if e.response is not None else ""
                        print(f"    [vLLM HTTP error] {e}\n    {body}")
                        failed = True
                        break
                    except requests.RequestException as e:
                        print(f"    [vLLM request error] {e}")
                        failed = True
                        break
                    except Exception as e:
                        print(f"    [{args.backend} error] {e}")
                        failed = True
                        break

                    for line in description.splitlines():
                        print(f"    {line}")

                    chunk_results.append({
                        "chunk_index": chunk_i,
                        "start": round(chunk_start, 3),
                        "end": round(chunk_end, 3),
                        "clip_seconds": round(chunk_duration, 3),
                        "description": description,
                    })

                if failed or not chunk_results:
                    continue

                out_fh.write(json.dumps({
                    "rally_index": idx,
                    "start": start,
                    "end": end,
                    "clip_seconds": duration,
                    "chunks": chunk_results,
                }, ensure_ascii=False) + "\n")
                out_fh.flush()
    finally:
        out_fh.close()

    print(f"\nWrote: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
