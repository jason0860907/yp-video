"""
Volleyball activity detection using Qwen3-VL via vLLM server.

Processes video with sliding window approach:
- 6 second clips
- 2 second sliding interval

Usage:
    python detect_volleyball.py --video path/to/video.mp4
    python detect_volleyball.py --video path/to/video.mp4 --server http://localhost:8000 --output results.json
"""

import argparse
import base64
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import requests

from utils.ffmpeg import get_video_duration, extract_clip


@dataclass
class ClipResult:
    """Result for a single video clip."""
    start_time: float
    end_time: float
    has_volleyball: bool
    confidence: str  # high, medium, low
    description: str


def encode_video_base64(video_path: str) -> str:
    """Encode video file to base64."""
    with open(video_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


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
                "confidence": r.confidence,
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

    prompt = """Analyze this video clip and determine if volleyball activity is occurring.

Look for:
- Volleyball court (indoor or outdoor/beach)
- Players in volleyball positions
- Ball being served, passed, set, or spiked
- Net visible
- Active gameplay or training

Respond in this exact JSON format:
{
    "has_volleyball": true/false,
    "confidence": "high"/"medium"/"low",
    "description": "Brief description of what you see"
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

    # Parse JSON from response
    try:
        # Handle potential markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        return json.loads(content.strip())
    except json.JSONDecodeError:
        return {
            "has_volleyball": False,
            "confidence": "low",
            "description": f"Failed to parse response: {content[:100]}"
        }


def process_video(
    video_path: str,
    server_url: str = "http://localhost:8000",
    model: str = "Qwen/Qwen3-VL-8B-Instruct",
    clip_duration: float = 6.0,
    slide_interval: float = 2.0,
    output_file: str = None
) -> list[ClipResult]:
    """Process video with sliding window approach."""

    video_path = os.path.abspath(video_path)
    total_duration = get_video_duration(video_path)

    print(f"Video: {video_path}")
    print(f"Duration: {total_duration:.2f}s")
    print(f"Clip duration: {clip_duration}s, Slide interval: {slide_interval}s")
    print("-" * 60)

    results = []
    current_time = 0.0
    clip_count = 0

    # Use directory next to video file for temp clips (for vLLM local file access)
    video_dir = os.path.dirname(video_path)
    with tempfile.TemporaryDirectory(dir=video_dir) as tmpdir:
        while current_time + clip_duration <= total_duration:
            clip_count += 1
            end_time = current_time + clip_duration

            print(f"\nProcessing clip {clip_count}: {current_time:.1f}s - {end_time:.1f}s")

            # Extract clip
            clip_path = os.path.join(tmpdir, f"clip_{clip_count}.mp4")
            if not extract_clip(video_path, current_time, clip_duration, clip_path):
                print(f"  [ERROR] Failed to extract clip")
                current_time += slide_interval
                continue

            # Analyze with vLLM
            try:
                analysis = analyze_clip_with_vllm(clip_path, server_url, model)

                result = ClipResult(
                    start_time=current_time,
                    end_time=end_time,
                    has_volleyball=analysis.get("has_volleyball", False),
                    confidence=analysis.get("confidence", "low"),
                    description=analysis.get("description", "")
                )
                results.append(result)

                # Progressive save after each clip
                if output_file:
                    save_results(output_file, video_path, clip_duration, slide_interval, results)

                status = "VOLLEYBALL" if result.has_volleyball else "NO"
                print(f"  [{status}] ({result.confidence}) {result.description[:60]}...")

            except requests.exceptions.RequestException as e:
                print(f"  [ERROR] API request failed: {e}")
            except Exception as e:
                print(f"  [ERROR] {e}")

            current_time += slide_interval

    # Process final partial clip if remaining time > slide_interval
    remaining = total_duration - current_time
    if remaining > slide_interval:
        clip_count += 1
        end_time = total_duration
        start_time = max(0, total_duration - clip_duration)

        print(f"\nProcessing final clip {clip_count}: {start_time:.1f}s - {end_time:.1f}s")

        with tempfile.TemporaryDirectory(dir=video_dir) as tmpdir:
            clip_path = os.path.join(tmpdir, f"clip_final.mp4")
            if extract_clip(video_path, start_time, clip_duration, clip_path):
                try:
                    analysis = analyze_clip_with_vllm(clip_path, server_url, model)
                    result = ClipResult(
                        start_time=start_time,
                        end_time=end_time,
                        has_volleyball=analysis.get("has_volleyball", False),
                        confidence=analysis.get("confidence", "low"),
                        description=analysis.get("description", "")
                    )
                    results.append(result)

                    # Progressive save after final clip
                    if output_file:
                        save_results(output_file, video_path, clip_duration, slide_interval, results)

                    status = "VOLLEYBALL" if result.has_volleyball else "NO"
                    print(f"  [{status}] ({result.confidence}) {result.description[:60]}...")
                except Exception as e:
                    print(f"  [ERROR] {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    volleyball_clips = [r for r in results if r.has_volleyball]
    print(f"Total clips analyzed: {len(results)}")
    print(f"Clips with volleyball: {len(volleyball_clips)}")

    if volleyball_clips:
        print("\nVolleyball activity detected at:")
        for r in volleyball_clips:
            print(f"  {r.start_time:.1f}s - {r.end_time:.1f}s ({r.confidence}): {r.description[:50]}...")

    # Note: Results are saved progressively after each clip
    if output_file:
        print(f"\nResults saved to: {output_file}")

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
        default=2.0,
        help="Sliding window interval in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON file for results"
    )

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return 1

    process_video(
        video_path=args.video,
        server_url=args.server,
        model=args.model,
        clip_duration=args.clip_duration,
        slide_interval=args.slide_interval,
        output_file=args.output
    )

    return 0


if __name__ == "__main__":
    exit(main())
