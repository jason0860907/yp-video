"""
Volleyball activity detection using Qwen3-VL via vLLM server.

Processes video with sliding window approach:
- 6 second clips
- 2 second sliding interval

Usage:
    python vlm_segment.py --video ~/videos/cuts/tpvl_set1.mp4 --output ~/videos/tpvl_set1_test_results.json
    python vlm_segment.py --video ~/videos/cuts/tpvl_set1.mp4 --server http://localhost:8000 --output tpvl_set1_test_results.json
"""

import argparse
import asyncio
import concurrent.futures
import json
import logging
import os
from pathlib import Path
import re
import tempfile
import time
from dataclasses import dataclass
from enum import Enum

import aiohttp
import requests
from tqdm import tqdm

from yp_video.core.ffmpeg import FFmpegError, extract_clip, get_video_duration
from yp_video.config import load_vllm_env, load_prompt

_VLLM_CONFIG = load_vllm_env()
VOLLEYBALL_SEGMENT_PROMPT = load_prompt("volleyball_segment.txt")


class VLLMServerError(RuntimeError):
    """Raised when the vLLM server is unreachable after retries."""


def check_server(server_url: str, retries: int = 5, backoff: float = 3.0) -> None:
    """Check if vLLM server is reachable, with retries and exponential backoff."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(f"{server_url}/v1/models", timeout=10)
            resp.raise_for_status()
            return
        except requests.exceptions.RequestException as e:
            if attempt < retries:
                wait = backoff * (2 ** (attempt - 1))  # 3, 6, 12, 24s
                print(f"WARNING: vLLM server not reachable (attempt {attempt}/{retries}), retrying in {wait:.0f}s... ({e})")
                time.sleep(wait)
            else:
                raise VLLMServerError(
                    f"vLLM server not reachable at {server_url} after {retries} attempts: {e}"
                ) from e


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS (e.g., 01:04)."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"




class ShotType(str, Enum):
    FULL_COURT = "full_court"
    CLOSE_UP = "close_up"


@dataclass
class ClipResult:
    """Result for a single video clip."""
    start_time: float
    end_time: float
    in_rally: bool
    shot_type: ShotType


def extract_json_from_response(content: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks.

    Args:
        content: Raw response content from LLM

    Returns:
        Parsed JSON dictionary, or fallback dict on parse failure
    """
    # Try to extract JSON from markdown code blocks
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        json_str = content.strip()

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {
            "in_rally": False,
            "shot_type": ShotType.CLOSE_UP,
        }


def save_results(output_file: str, video_path: str, clip_duration: float,
                 slide_interval: float, results: list[ClipResult]) -> None:
    """Save current results to JSONL file.

    Format:
        Line 1: metadata with _meta=true
        Line 2+: one clip result per line
    """
    rally_clips = [r for r in results if r.in_rally]

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        # First line: metadata
        meta = {
            "_meta": True,
            "video": video_path,
            "clip_duration": clip_duration,
            "slide_interval": slide_interval,
            "total_clips": len(results),
            "rally_clips": len(rally_clips),
        }
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

        # Subsequent lines: one result per line
        for r in results:
            result_dict = {
                "start_time": r.start_time,
                "end_time": r.end_time,
                "in_rally": r.in_rally,
                "shot_type": r.shot_type.value,
            }
            f.write(json.dumps(result_dict, ensure_ascii=False) + "\n")


def analyze_clip_with_vllm(
    video_path: str,
    server_url: str,
    model: str,
    fps: float = 4.0,
) -> dict:
    """Send video clip to vLLM server for analysis."""

    # Use file:// URL for local files (requires --allowed-local-media-path on server)
    video_url = f"file://{video_path}"

    prompt = VOLLEYBALL_SEGMENT_PROMPT

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": video_url
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "max_tokens": 256,
        "temperature": 0.1,
        "chat_template_kwargs": {"enable_thinking": False},
        "mm_processor_kwargs": {"fps": fps}
    }

    response = requests.post(
        f"{server_url}/v1/chat/completions",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=120
    )
    response.raise_for_status()

    result = response.json()
    content = result["choices"][0]["message"]["content"]

    return extract_json_from_response(content)


async def analyze_clip_async(
    session: aiohttp.ClientSession,
    video_path: str,
    server_url: str,
    model: str,
    fps: float = 4.0,
    max_retries: int = 3,
) -> dict:
    """Async version of analyze_clip_with_vllm with retry on transient errors."""
    video_url = f"file://{video_path}"

    prompt = VOLLEYBALL_SEGMENT_PROMPT

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": video_url
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "max_tokens": 256,
        "temperature": 0.1,
        "chat_template_kwargs": {"enable_thinking": False},
        "mm_processor_kwargs": {"fps": fps}
    }

    url = f"{server_url}/v1/chat/completions"
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            async with session.post(url, json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                content = result["choices"][0]["message"]["content"]
                return extract_json_from_response(content)
        except (asyncio.TimeoutError, aiohttp.ServerTimeoutError, aiohttp.ServerDisconnectedError) as e:
            last_error = e
            if attempt < max_retries:
                wait = 2 * attempt  # 2, 4s
                logging.getLogger(__name__).warning(
                    "vLLM request timeout (attempt %d/%d), retrying in %ds...", attempt, max_retries, wait
                )
                await asyncio.sleep(wait)
    raise last_error


def _parse_shot_type(value: str | ShotType) -> ShotType:
    """Convert string to ShotType enum."""
    if isinstance(value, ShotType):
        return value
    try:
        return ShotType(value)
    except ValueError:
        return ShotType.FULL_COURT


def process_single_clip(
    video_path: str,
    clip_path: str,
    clip_index: int,
    start_time: float,
    end_time: float,
    clip_duration: float,
    server_url: str,
    model: str
) -> ClipResult | None:
    """Process a single video clip: extract, analyze, and return result.

    Args:
        video_path: Source video path
        clip_path: Output path for extracted clip
        clip_index: Clip number for logging
        start_time: Clip start time in seconds
        end_time: Clip end time in seconds
        clip_duration: Duration of clip to extract
        server_url: vLLM server URL
        model: Model name

    Returns:
        ClipResult if successful, None if failed
    """
    print(f"\nProcessing clip {clip_index}: {start_time:.1f}s - {end_time:.1f}s")

    # Extract clip
    try:
        extract_clip(video_path, start_time, clip_duration, clip_path)
    except FFmpegError as e:
        print(f"  [ERROR] Failed to extract clip: {e}")
        return None

    # Analyze with vLLM
    try:
        analysis = analyze_clip_with_vllm(clip_path, server_url, model)

        result = ClipResult(
            start_time=start_time,
            end_time=end_time,
            in_rally=analysis.get("in_rally", False),
            shot_type=_parse_shot_type(analysis.get("shot_type", "full_court")),
        )

        status = "RALLY" if result.in_rally else "NO"
        print(f"  [{status}] {result.shot_type.value}")

        return result

    except requests.exceptions.ConnectionError:
        raise
    except requests.exceptions.RequestException as e:
        print(f"  [ERROR] API request failed: {e}")
        return None
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None


async def process_batch_async(
    batch_clips: list[tuple[str, int, float, float]],  # (clip_path, index, start, end)
    server_url: str,
    model: str,
    session: aiohttp.ClientSession,
    fps: float = 4.0,
) -> list[tuple[int, ClipResult | None]]:
    """Process a batch of clips concurrently using a shared session.

    Args:
        batch_clips: List of (clip_path, clip_index, start_time, end_time) tuples
        server_url: vLLM server URL
        model: Model name
        session: Shared aiohttp session (reused across batches)
        fps: Frames per second for VLM processing

    Returns:
        List of (clip_index, ClipResult or None) tuples
    """
    tasks = [
        analyze_clip_async(session, clip_path, server_url, model, fps=fps)
        for clip_path, _, _, _ in batch_clips
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    output = []
    for (clip_path, clip_index, start_time, end_time), result in zip(batch_clips, results):
        if isinstance(result, (aiohttp.ClientError, OSError)) and not isinstance(result, TimeoutError):
            raise ConnectionError(
                f"vLLM server unreachable: {result}"
            ) from result
        elif isinstance(result, Exception):
            output.append((clip_index, None))
        else:
            clip_result = ClipResult(
                start_time=start_time,
                end_time=end_time,
                in_rally=result.get("in_rally", False),
                shot_type=_parse_shot_type(result.get("shot_type", "full_court")),
            )
            output.append((clip_index, clip_result))

    return output


def build_clip_specs(
    total_duration: float, clip_duration: float = 6.0, slide_interval: float = 2.0
) -> list[tuple[int, float, float]]:
    """Build list of (clip_index, start_time, end_time) for a video.

    Single source of truth for clip windowing logic used by both
    count_clips() and process_video().
    """
    specs: list[tuple[int, float, float]] = []
    current_time = 0.0
    clip_index = 0

    while current_time + clip_duration <= total_duration:
        clip_index += 1
        specs.append((clip_index, current_time, current_time + clip_duration))
        current_time += slide_interval

    # Final partial clip if remaining time > slide_interval
    remaining = total_duration - current_time
    if remaining > slide_interval:
        clip_index += 1
        start_time = max(0, total_duration - clip_duration)
        specs.append((clip_index, start_time, total_duration))

    return specs


def count_clips(video_path: str, clip_duration: float = 6.0, slide_interval: float = 2.0) -> int:
    """Count how many clips a video will produce without processing it."""
    total_duration = get_video_duration(os.path.abspath(video_path))
    return len(build_clip_specs(total_duration, clip_duration, slide_interval))


def process_video(
    video_path: str,
    server_url: str = "http://localhost:8000",
    model: str = "Qwen/Qwen3-VL-8B-Instruct",
    clip_duration: float = 6.0,
    slide_interval: float = 2.0,
    output_file: str | None = None,
    batch_size: int = 16,
    max_concurrent: int = 16,
    on_progress: "Callable[[int, int], None] | None" = None,
    fps: float = 4.0,
    total_duration: float | None = None,
) -> list[ClipResult]:
    """Process video with sliding window approach using parallel batch processing.

    Args:
        video_path: Path to the video file
        server_url: vLLM server URL
        model: Model name
        clip_duration: Duration of each clip in seconds
        slide_interval: Sliding window interval in seconds
        output_file: Output JSON file path (optional)
        batch_size: Number of clips to process per batch
        max_concurrent: Max concurrent requests to vLLM (should match VLLM_MAX_NUM_SEQS)
        on_progress: Optional callback(clips_done, total_clips)
        fps: Frames per second for VLM processing
        total_duration: Pre-computed duration (skips ffprobe if provided)

    Returns:
        List of ClipResult objects
    """
    video_path = os.path.abspath(video_path)

    # Verify server is up before starting
    check_server(server_url)

    if total_duration is None:
        total_duration = get_video_duration(video_path)

    print(f"\n{'─' * 70}")
    print(f"Video: {video_path}")
    print(f"Duration: {format_time(int(total_duration))} | Clip: {clip_duration}s | Interval: {slide_interval}s | Batch: {batch_size}")

    clip_specs = build_clip_specs(total_duration, clip_duration, slide_interval)
    total_clips = len(clip_specs)

    results: list[ClipResult] = []
    total_inference_time = 0.0

    # Use directory next to video file for temp clips (for vLLM local file access)
    video_dir = os.path.dirname(video_path)

    # Single event loop + session for all batches (prevents resource leaks)
    loop = asyncio.new_event_loop()

    async def _create_session():
        connector = aiohttp.TCPConnector(limit=max_concurrent)
        return aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=120), connector=connector
        )

    session = loop.run_until_complete(_create_session())

    try:
        with tempfile.TemporaryDirectory(dir=video_dir) as tmpdir:
            pbar = tqdm(total=total_clips, desc="Processing", unit="clip")

            try:
                for batch_start in range(0, total_clips, batch_size):
                    batch_specs = clip_specs[batch_start:batch_start + batch_size]

                    # Step 1: Extract all clips in this batch (parallel)
                    batch_clips: list[tuple[str, int, float, float]] = []
                    extract_tasks = {}
                    with concurrent.futures.ThreadPoolExecutor(max_workers=min(batch_size, 8)) as executor:
                        for idx, start_time, end_time in batch_specs:
                            clip_path = os.path.join(tmpdir, f"clip_{idx}.mp4")
                            future = executor.submit(extract_clip, video_path, start_time, clip_duration, clip_path)
                            extract_tasks[future] = (clip_path, idx, start_time, end_time)

                        for future in concurrent.futures.as_completed(extract_tasks):
                            clip_path, idx, start_time, end_time = extract_tasks[future]
                            try:
                                future.result()
                                batch_clips.append((clip_path, idx, start_time, end_time))
                            except FFmpegError:
                                pass

                    if not batch_clips:
                        pbar.update(len(batch_specs))
                        continue

                    # Step 2: Analyze all clips in parallel (reuse session)
                    inference_start = time.time()
                    batch_results = loop.run_until_complete(
                        process_batch_async(batch_clips, server_url, model, session, fps=fps)
                    )
                    total_inference_time += time.time() - inference_start

                    # Step 3: Collect results (maintaining order)
                    batch_successes = 0
                    for clip_idx, clip_result in batch_results:
                        if clip_result:
                            results.append(clip_result)
                            batch_successes += 1

                    # If entire batch failed, server likely crashed — retry with backoff
                    if batch_successes == 0 and len(batch_clips) > 0:
                        print("\nWARNING: Entire batch failed. Checking server health...")
                        check_server(server_url)  # raises VLLMServerError if still down after retries

                    # Step 4: Save results after each batch
                    if output_file:
                        save_results(output_file, video_path, clip_duration, slide_interval, results)

                    # Update progress bar
                    pbar.update(len(batch_specs))

                    # Report progress via callback
                    if on_progress:
                        on_progress(pbar.n, total_clips)

                    # Clean up batch clips to free disk space
                    for clip_path, _, _, _ in batch_clips:
                        try:
                            os.remove(clip_path)
                        except OSError:
                            pass

            finally:
                pbar.close()
    finally:
        try:
            loop.run_until_complete(session.close())
        except Exception:
            logging.getLogger(__name__).debug("Error closing aiohttp session", exc_info=True)
        loop.close()

    # Summary
    rally_clips = [r for r in results if r.in_rally]
    print(f"Analyzed: {len(results)} clips | Rally: {len(rally_clips)} | Inference time: {total_inference_time:.1f}s")

    if output_file:
        print(f"Saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Detect volleyball activity in video using Qwen3-VL"
    )
    parser.add_argument(
        "--video", "-v",
        type=str,
        required=True,
        help="Path to video file"
    )
    _default_port = _VLLM_CONFIG["VLLM_PORT"]
    _default_model = _VLLM_CONFIG["VLLM_MODEL"]
    _default_server = f"http://localhost:{_default_port}"

    parser.add_argument(
        "--server", "-s",
        type=str,
        default=_default_server,
        help=f"vLLM server URL (default: {_default_server})"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=_default_model,
        help=f"Model name (default: {_default_model})"
    )
    parser.add_argument(
        "--clip-duration", "-d",
        type=float,
        default=6.0,
        help="Duration of each clip in seconds (default: 6.0)"
    )
    parser.add_argument(
        "--slide-interval", "-i",
        type=float,
        default=3.0,
        help="Sliding window interval in seconds (default: 3.0)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSONL file for results"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=4.0,
        help="Frames per second for VLM processing (default: 4.0)"
    )
    _default_max_seqs = int(_VLLM_CONFIG["VLLM_MAX_NUM_SEQS"])
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=_default_max_seqs,
        help=f"Number of clips to process per batch (default: {_default_max_seqs})"
    )

    args = parser.parse_args()

    # Default output to ~/videos/annotations/{video_basename}.jsonl
    if args.output is None:
        video_base = os.path.splitext(os.path.basename(args.video))[0]
        args.output = os.path.expanduser(f"~/videos/seg-annotations/{video_base}.jsonl")

    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return 1

    process_video(
        video_path=args.video,
        server_url=args.server,
        model=args.model,
        clip_duration=args.clip_duration,
        slide_interval=args.slide_interval,
        output_file=args.output,
        batch_size=args.batch_size,
        max_concurrent=_default_max_seqs,
        fps=args.fps,
    )

    return 0


if __name__ == "__main__":
    exit(main())
