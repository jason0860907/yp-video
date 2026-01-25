"""
Volleyball activity detection using Qwen3-VL via vLLM server.

Processes video with sliding window approach:
- 6 second clips
- 2 second sliding interval

Usage:
    python detect_volleyball.py --video ~/videos/cuts/tpvl_set1.mp4 --output ~/videos/tpvl_set1_test_results.json
    python detect_volleyball.py --video ~/videos/cuts/tpvl_set1.mp4 --server http://localhost:8000 --output tpvl_set1_test_results.json
"""

import argparse
import asyncio
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from enum import Enum

import aiohttp
import requests
from tqdm import tqdm

from utils.ffmpeg import FFmpegError, extract_clip, get_video_duration


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS (e.g., 01:04)."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"




class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ShotType(str, Enum):
    FULL_COURT = "full_court"
    CLOSE_UP = "close_up"


@dataclass
class ClipResult:
    """Result for a single video clip."""
    start_time: float
    end_time: float
    has_volleyball: bool
    confidence: Confidence
    shot_type: ShotType
    description: str


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
            "has_volleyball": False,
            "confidence": Confidence.LOW,
            "shot_type": ShotType.CLOSE_UP,
            "description": f"Failed to parse response: {content[:100]}"
        }


def save_results(output_file: str, video_path: str, clip_duration: float,
                 slide_interval: float, results: list[ClipResult]) -> None:
    """Save current results to JSON file."""
    volleyball_clips = [r for r in results if r.has_volleyball]
    output_data = {
        "video": video_path,
        "clip_duration": clip_duration,
        "slide_interval": slide_interval,
        "total_clips": len(results),
        "volleyball_clips": len(volleyball_clips),
        "results": [
            {
                "start_time": r.start_time,
                "end_time": r.end_time,
                "has_volleyball": r.has_volleyball,
                "confidence": r.confidence.value,
                "shot_type": r.shot_type.value,
                "description": r.description
            }
            for r in results
        ]
    }
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)


def analyze_clip_with_vllm(
    video_path: str,
    server_url: str,
    model: str
) -> dict:
    """Send video clip to vLLM server for analysis."""

    # Use file:// URL for local files (requires --allowed-local-media-path on server)
    video_url = f"file://{video_path}"

    prompt = """Analyze this video clip of a volleyball broadcast.

Determine:
1. Is active volleyball gameplay occurring? (rally in progress)
2. What is the camera shot type?

Shot types:
- "full_court": Can see the entire court, both teams, suitable for watching gameplay
- "close_up": Player close-up, celebration, interview, replay, or partial court view

Respond in this exact JSON format:
{
    "has_volleyball": true/false,
    "confidence": "high"/"medium"/"low",
    "shot_type": "full_court"/"close_up",
    "description": "Brief description"
}

Only output the JSON, no other text."""

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
        "temperature": 0.1
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
    model: str
) -> dict:
    """Async version of analyze_clip_with_vllm."""
    video_url = f"file://{video_path}"

    prompt = """Analyze this video clip of a volleyball broadcast.

Determine:
1. Is active volleyball gameplay occurring? (rally in progress)
2. What is the camera shot type?

Shot types:
- "full_court": Can see the entire court, both teams, suitable for watching gameplay
- "close_up": Player close-up, celebration, interview, replay, or partial court view

Respond in this exact JSON format:
{
    "has_volleyball": true/false,
    "confidence": "high"/"medium"/"low",
    "shot_type": "full_court"/"close_up",
    "description": "Brief description"
}

Only output the JSON, no other text."""

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
        "temperature": 0.1
    }

    url = f"{server_url}/v1/chat/completions"
    async with session.post(url, json=payload) as response:
        response.raise_for_status()
        result = await response.json()
        content = result["choices"][0]["message"]["content"]
        return extract_json_from_response(content)


def _parse_confidence(value: str | Confidence) -> Confidence:
    """Convert string to Confidence enum."""
    if isinstance(value, Confidence):
        return value
    try:
        return Confidence(value)
    except ValueError:
        return Confidence.LOW


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
            has_volleyball=analysis.get("has_volleyball", False),
            confidence=_parse_confidence(analysis.get("confidence", "low")),
            shot_type=_parse_shot_type(analysis.get("shot_type", "full_court")),
            description=analysis.get("description", "")
        )

        status = "VOLLEYBALL" if result.has_volleyball else "NO"
        print(f"  [{status}] ({result.confidence.value}) {result.description[:60]}...")

        return result

    except requests.exceptions.RequestException as e:
        print(f"  [ERROR] API request failed: {e}")
        return None
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None


async def process_batch_async(
    batch_clips: list[tuple[str, int, float, float]],  # (clip_path, index, start, end)
    server_url: str,
    model: str
) -> list[tuple[int, ClipResult | None]]:
    """Process a batch of clips concurrently.

    Args:
        batch_clips: List of (clip_path, clip_index, start_time, end_time) tuples
        server_url: vLLM server URL
        model: Model name

    Returns:
        List of (clip_index, ClipResult or None) tuples
    """
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            analyze_clip_async(session, clip_path, server_url, model)
            for clip_path, _, _, _ in batch_clips
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output = []
        for (clip_path, clip_index, start_time, end_time), result in zip(batch_clips, results):
            if isinstance(result, Exception):
                output.append((clip_index, None))
            else:
                clip_result = ClipResult(
                    start_time=start_time,
                    end_time=end_time,
                    has_volleyball=result.get("has_volleyball", False),
                    confidence=_parse_confidence(result.get("confidence", "low")),
                    shot_type=_parse_shot_type(result.get("shot_type", "full_court")),
                    description=result.get("description", "")
                )
                output.append((clip_index, clip_result))

        return output


def process_video(
    video_path: str,
    server_url: str = "http://localhost:8000",
    model: str = "Qwen/Qwen3-VL-8B-Instruct",
    clip_duration: float = 6.0,
    slide_interval: float = 3.0,
    output_file: str | None = None,
    batch_size: int = 8
) -> list[ClipResult]:
    """Process video with sliding window approach using parallel batch processing.

    Args:
        video_path: Path to the video file
        server_url: vLLM server URL
        model: Model name
        clip_duration: Duration of each clip in seconds
        slide_interval: Sliding window interval in seconds
        output_file: Output JSON file path (optional)
        batch_size: Number of clips to process in parallel (default: 4)

    Returns:
        List of ClipResult objects
    """
    video_path = os.path.abspath(video_path)
    total_duration = get_video_duration(video_path)

    print(f"Video: {video_path}")
    print(f"Duration: {format_time(int(total_duration))} | Clip: {clip_duration}s | Interval: {slide_interval}s | Batch: {batch_size}")

    # Build list of all clip specs: (clip_index, start_time, end_time)
    clip_specs: list[tuple[int, float, float]] = []
    current_time = 0.0
    clip_index = 0

    while current_time + clip_duration <= total_duration:
        clip_index += 1
        end_time = current_time + clip_duration
        clip_specs.append((clip_index, current_time, end_time))
        current_time += slide_interval

    # Add final partial clip if remaining time > slide_interval
    remaining = total_duration - current_time
    if remaining > slide_interval:
        clip_index += 1
        start_time = max(0, total_duration - clip_duration)
        clip_specs.append((clip_index, start_time, total_duration))

    total_clips = len(clip_specs)
    num_batches = (total_clips + batch_size - 1) // batch_size

    results: list[ClipResult] = []
    total_inference_time = 0.0

    # Use directory next to video file for temp clips (for vLLM local file access)
    video_dir = os.path.dirname(video_path)
    with tempfile.TemporaryDirectory(dir=video_dir) as tmpdir:
        # Process clips in batches with progress bar
        pbar = tqdm(total=total_clips, desc="Processing", unit="clip")

        for batch_start in range(0, total_clips, batch_size):
            batch_end = min(batch_start + batch_size, total_clips)
            batch_specs = clip_specs[batch_start:batch_end]

            # Step 1: Extract all clips in this batch
            batch_clips: list[tuple[str, int, float, float]] = []
            for idx, start_time, end_time in batch_specs:
                clip_path = os.path.join(tmpdir, f"clip_{idx}.mp4")
                try:
                    extract_clip(video_path, start_time, clip_duration, clip_path)
                    batch_clips.append((clip_path, idx, start_time, end_time))
                except FFmpegError:
                    pass

            if not batch_clips:
                pbar.update(len(batch_specs))
                continue

            # Step 2: Analyze all clips in parallel
            inference_start = time.time()
            batch_results = asyncio.run(
                process_batch_async(batch_clips, server_url, model)
            )
            total_inference_time += time.time() - inference_start

            # Step 3: Collect results (maintaining order)
            for clip_idx, clip_result in batch_results:
                if clip_result:
                    results.append(clip_result)

            # Step 4: Save results after each batch
            if output_file:
                save_results(output_file, video_path, clip_duration, slide_interval, results)

            # Update progress bar
            pbar.update(len(batch_specs))

            # Clean up batch clips to free disk space
            for clip_path, _, _, _ in batch_clips:
                try:
                    os.remove(clip_path)
                except OSError:
                    pass

        pbar.close()

    # Summary
    volleyball_clips = [r for r in results if r.has_volleyball]
    print(f"\nAnalyzed: {len(results)} clips | Volleyball: {len(volleyball_clips)} | Inference time: {total_inference_time:.1f}s")

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
    parser.add_argument(
        "--server", "-s",
        type=str,
        default="http://localhost:8000",
        help="vLLM server URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Model name (default: Qwen/Qwen3-VL-8B-Instruct)"
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
        help="Sliding window interval in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        help="Number of clips to process in parallel (default: 4)"
    )

    args = parser.parse_args()

    # Default output to ~/videos/{video_basename}.json
    if args.output is None:
        video_base = os.path.splitext(os.path.basename(args.video))[0]
        args.output = os.path.expanduser(f"~/videos/{video_base}.json")

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
        batch_size=args.batch_size
    )

    return 0


if __name__ == "__main__":
    exit(main())
