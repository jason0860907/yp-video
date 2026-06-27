"""Run yp-spot action spotting and write an action pre-annotation JSONL.

Router-free entry point shared by the web dashboard and the selfhost GPU
worker. Given a video it shells out to the yp-spot model — which lives in its
own repo + venv and is reached across a subprocess boundary (see
``contracts/action.py``) — and writes a ``*_actions.jsonl`` file in the exact
shape ``yp_video.tad.action_trim`` consumes: a ``_meta`` header line (carrying
``fps``) followed by one action event per line.

This keeps the SPOT→JSONL orchestration in one place so the worker does not
duplicate (and drift from) the web router's flow.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections import deque
from collections.abc import Callable
from pathlib import Path

from yp_video.action import prelabel
from yp_video.config import SPOT_DIR
from yp_video.contracts.action import (
    ACTION_CONTRACT_VERSION,
    ACTION_CONTRACT_VERSION_ENV,
    SPOT_PROGRESS_PREFIX,
)
from yp_video.core.ffmpeg import FFmpegError, probe_video_metadata
from yp_video.core.jsonl import write_jsonl


class SpotInferenceError(RuntimeError):
    """yp-spot was unavailable or its inference subprocess failed.

    Callers that want graceful degradation (e.g. fall back to untrimmed TAD
    rallies) should catch this and continue rather than aborting.
    """


def _spot_progress_ratio(line: str) -> float | None:
    """Parse a yp-spot ``SPOT_PROGRESS`` stdout line to a fraction in [0, 1].

    Returns ``None`` for any non-progress line. Parsing lives in ``prelabel`` so
    the web dashboard and this worker path share one implementation.
    """
    if not line.startswith(SPOT_PROGRESS_PREFIX):
        return None
    data = prelabel.parse_spot_progress(line[len(SPOT_PROGRESS_PREFIX):])
    return prelabel.spot_progress_fraction(data) if data is not None else None


def _probe_fps_frames(video_path: Path) -> tuple[float, int]:
    """Return ``(fps, num_frames)`` for ``video_path`` via ffprobe."""
    try:
        meta = probe_video_metadata(video_path)
    except FFmpegError as exc:
        raise SpotInferenceError(str(exc)) from exc
    return meta["fps"], meta["num_frames"]


def predict_actions_to_jsonl(
    video_path: Path,
    output_path: Path,
    *,
    checkpoint_path: Path | None = None,
    batch_size: int = 8,
    num_workers: int = 4,
    clip_len: int = 64,
    use_amp: bool = True,
    min_score: float = 0.0,
    on_message: Callable[[str], None] | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> Path:
    """Run yp-spot action spotting on ``video_path`` and write the action JSONL.

    Args:
        video_path: Input video.
        output_path: Destination ``*_actions.jsonl`` (``_meta`` line + events).
        checkpoint_path: Explicit SPOT ``.pt`` checkpoint. When ``None`` the
            newest checkpoint under ``~/videos/action-checkpoints`` is used.
        min_score: Drop predicted events below this confidence.
        on_message: Optional status callback for step-level progress.
        on_progress: Optional ``(fraction) -> None`` callback fired per SPOT
            progress tick (0..1 of inference). Lets long-running callers push
            live sub-progress while the subprocess streams.

    Returns:
        ``output_path``.

    Raises:
        SpotInferenceError: yp-spot is not installed, has no checkpoint, or its
            inference subprocess failed / produced no output.
    """
    def _msg(text: str) -> None:
        if on_message:
            on_message(text)

    if not prelabel.spot_available():
        raise SpotInferenceError(
            f"yp-spot not available (looked under {SPOT_DIR}); "
            "set YP_SPOT_DIR / YP_SPOT_PYTHON and install its venv"
        )

    try:
        # resolve_checkpoint handles VIDEOS_DIR-relative refs, existence, and the
        # ~/videos/action-checkpoints containment check for both the explicit and
        # default (None) cases.
        checkpoint = prelabel.resolve_checkpoint(checkpoint_path)
    except (FileNotFoundError, ValueError) as exc:
        raise SpotInferenceError(f"SPOT checkpoint unavailable: {exc}") from exc

    _msg("Reading video metadata...")
    fps, num_frames = _probe_fps_frames(video_path)

    with tempfile.TemporaryDirectory(prefix="yp-spot-infer-") as tmp_root:
        pred_file = Path(tmp_root) / "predictions.json"
        cmd = prelabel.build_command(
            video_path=video_path,
            checkpoint_path=checkpoint,
            save_dir=pred_file.parent,
            batch_size=batch_size,
            num_workers=num_workers,
            clip_len=clip_len,
            use_amp=use_amp,
        )
        env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            ACTION_CONTRACT_VERSION_ENV: ACTION_CONTRACT_VERSION,
        }
        _msg("Running SPOT action inference...")
        # Stream stdout so progress ticks surface live; merge stderr in so a
        # single reader can't deadlock and the error tail is captured too.
        proc = subprocess.Popen(
            cmd,
            cwd=SPOT_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        tail: deque[str] = deque(maxlen=20)
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            if not line:
                continue
            ratio = _spot_progress_ratio(line)
            if ratio is not None:
                if on_progress:
                    on_progress(ratio)
            else:
                tail.append(line)
        rc = proc.wait()
        if rc != 0:
            raise SpotInferenceError(
                f"yp-spot inference failed (rc={rc}): " + " | ".join(list(tail)[-5:])
            )
        if not pred_file.exists():
            raise SpotInferenceError(f"yp-spot produced no predictions at {pred_file}")

        predictions = prelabel.load_predictions(pred_file)

    data = prelabel.predictions_to_annotation(
        predictions,
        video_path=video_path,
        metadata={"fps": fps, "num_frames": num_frames},
        checkpoint_path=checkpoint,
        min_score=min_score,
    )

    meta = {k: v for k, v in data.items() if k != "events"}
    write_jsonl(output_path, meta, data.get("events", []))
    _msg(f"Wrote {data.get('num_events', 0)} action events to {output_path.name}")
    return output_path
