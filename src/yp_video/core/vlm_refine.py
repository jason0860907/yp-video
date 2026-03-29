"""VLM boundary refinement for rally annotations.

Uses existing binary VLM detection (in_rally) with fine-grained 1-second stride
around detected rally boundaries to achieve 1-second precision.

For each rally start/end:
- Generates 2s clips at 1s stride within ±window seconds of the boundary
- Calls VLM with same binary prompt as detection
- start: finds first in_rally=True clip → its start_time
- end:   finds last  in_rally=True clip → its end_time
- If no transition found (all True or all False) → keeps original time
"""

import argparse
import asyncio
import json
import os
import tempfile
from pathlib import Path

import aiohttp
from tqdm import tqdm

from yp_video.config import CUTS_DIR, PRE_ANNOTATIONS_DIR, REFINE_ANNOTATIONS_DIR, load_vllm_env
from yp_video.core.ffmpeg import FFmpegError, extract_clip, get_video_duration
from yp_video.core.vlm_segment import analyze_clip_async, check_server


def _read_pre_annotations(path: Path) -> tuple[dict, list[dict]]:
    meta: dict = {}
    rallies: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("_meta"):
                meta = record
            else:
                rallies.append(record)
    return meta, rallies


def _boundary_clip_times(
    center: float,
    window: float,
    clip_duration: float,
    video_duration: float,
    stride: float = 1.0,
) -> list[float]:
    """Clip start times at 1s stride within [center-window, center+window]."""
    seen: set[float] = set()
    times: list[float] = []
    t = center - window
    while t < center + window:
        clip_start = round(max(0.0, t), 1)
        if clip_start + clip_duration <= video_duration and clip_start not in seen:
            seen.add(clip_start)
            times.append(clip_start)
        t += stride
    return sorted(times)


def _find_refined_boundary(
    results: list[tuple[float, float, bool]],
    boundary_type: str,
) -> float | None:
    """Find refined boundary from (start, end, in_rally) results.

    Returns None if no transition found (keep original time).
    """
    if not results:
        return None
    flags = [r[2] for r in results]
    if all(flags) or not any(flags):
        return None
    if boundary_type == "start":
        for start_t, _, in_rally in results:
            if in_rally:
                return start_t
    else:
        last_end = None
        for _, end_t, in_rally in results:
            if in_rally:
                last_end = end_t
        return last_end
    return None


async def _analyze_batch(
    session: aiohttp.ClientSession,
    clips: list[tuple[str, float, float]],
    server_url: str,
    model: str,
) -> list[tuple[float, float, bool]]:
    """Analyze clips in parallel. clips = [(path, start_t, end_t)]."""
    tasks = [analyze_clip_async(session, path, server_url, model) for path, _, _ in clips]
    raw = await asyncio.gather(*tasks, return_exceptions=True)
    return [
        (start_t, end_t, bool((r or {}).get("in_rally", False)) if not isinstance(r, Exception) else False)
        for (_, start_t, end_t), r in zip(clips, raw)
    ]


def refine_video(
    video_path: Path,
    pre_annotations_path: Path,
    output_path: Path,
    server_url: str,
    model: str,
    window: float = 5.0,
    clip_duration: float = 2.0,
    batch_size: int = 16,
    on_progress: "callable | None" = None,
) -> int:
    """Refine rally start/end boundaries using fine-grained VLM detection.

    Returns number of rallies written.
    """
    meta, rallies = _read_pre_annotations(pre_annotations_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rallies:
        with open(output_path, "w") as f:
            f.write(json.dumps(meta) + "\n")
        return 0

    duration = get_video_duration(str(video_path))

    # Build boundary spec list: (rally_idx, boundary_type, clip_start)
    specs: list[tuple[int, str, float]] = []
    for i, rally in enumerate(rallies):
        for t in _boundary_clip_times(rally["start"], window, clip_duration, duration):
            specs.append((i, "start", t))
        for t in _boundary_clip_times(rally["end"], window, clip_duration, duration):
            specs.append((i, "end", t))

    total = len(specs)
    done = 0

    # Accumulate results per rally
    boundary_results: dict[int, dict[str, list]] = {
        i: {"start": [], "end": []} for i in range(len(rallies))
    }

    loop = asyncio.new_event_loop()

    async def _run():
        nonlocal done
        connector = aiohttp.TCPConnector(limit=batch_size)
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=120), connector=connector
        ) as session:
            with tempfile.TemporaryDirectory(dir=video_path.parent) as tmpdir:
                for batch_start in range(0, total, batch_size):
                    batch = specs[batch_start : batch_start + batch_size]

                    # Extract clips
                    extracted: list[tuple[str, int, str, float]] = []
                    for idx, (rally_idx, b_type, clip_start) in enumerate(batch):
                        clip_path = os.path.join(tmpdir, f"r{batch_start + idx}.mp4")
                        try:
                            extract_clip(str(video_path), clip_start, clip_duration, clip_path)
                            extracted.append((clip_path, rally_idx, b_type, clip_start))
                        except FFmpegError:
                            pass

                    if extracted:
                        clips_for_vlm = [
                            (path, start_t, start_t + clip_duration)
                            for path, _, _, start_t in extracted
                        ]
                        results = await _analyze_batch(session, clips_for_vlm, server_url, model)
                        for (_, rally_idx, b_type, _), (st, et, in_rally) in zip(extracted, results):
                            boundary_results[rally_idx][b_type].append((st, et, in_rally))

                        for path, *_ in extracted:
                            try:
                                os.remove(path)
                            except OSError:
                                pass

                    done += len(batch)
                    if on_progress:
                        on_progress(done, total)

    try:
        loop.run_until_complete(_run())
    finally:
        loop.close()

    # Apply refinements
    with open(output_path, "w") as f:
        f.write(json.dumps(meta) + "\n")
        for i, rally in enumerate(rallies):
            refined_start = _find_refined_boundary(boundary_results[i]["start"], "start")
            refined_end = _find_refined_boundary(boundary_results[i]["end"], "end")
            f.write(json.dumps({
                "start": refined_start if refined_start is not None else rally["start"],
                "end": refined_end if refined_end is not None else rally["end"],
                "label": rally.get("label", "rally"),
            }) + "\n")

    return len(rallies)


def _find_video(stem: str) -> Path | None:
    for ext in (".mp4", ".avi", ".mkv", ".mov", ".webm"):
        p = CUTS_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def main():
    _cfg = load_vllm_env()
    parser = argparse.ArgumentParser(description="Refine rally boundaries using VLM")
    parser.add_argument("--video", type=str, default=None, help="Video stem (default: all)")
    parser.add_argument("--window", type=float, default=5.0, help="Boundary window in seconds")
    parser.add_argument("--batch-size", type=int, default=16)
    _default_port = _cfg["VLLM_PORT"]
    parser.add_argument("--server", type=str, default=f"http://localhost:{_default_port}")
    parser.add_argument("--model", type=str, default=_cfg["VLLM_MODEL"])
    args = parser.parse_args()

    check_server(args.server)
    REFINE_ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    if args.video:
        stems = [args.video]
    else:
        if not PRE_ANNOTATIONS_DIR.exists():
            print("No pre-annotations directory found.")
            return
        stems = [
            p.stem.removesuffix("_annotations")
            for p in sorted(PRE_ANNOTATIONS_DIR.glob("*_annotations.jsonl"))
        ]

    for stem in tqdm(stems, desc="Refining videos"):
        pre_path = PRE_ANNOTATIONS_DIR / f"{stem}_annotations.jsonl"
        if not pre_path.exists():
            print(f"Skipping {stem}: pre-annotations not found")
            continue
        video_path = _find_video(stem)
        if video_path is None:
            print(f"Skipping {stem}: video not found in {CUTS_DIR}")
            continue

        output_path = REFINE_ANNOTATIONS_DIR / f"{stem}_annotations.jsonl"
        print(f"Refining {stem}...")
        count = refine_video(
            video_path, pre_path, output_path,
            args.server, args.model,
            window=args.window, batch_size=args.batch_size,
        )
        print(f"  {count} rallies → {output_path.name}")


if __name__ == "__main__":
    main()
