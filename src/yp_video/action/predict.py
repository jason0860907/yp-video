"""The yp-spot inference subprocess: one decode pass over a video, every
requested spotting head, streamed progress and partial events.

yp-spot lives in its own repo + venv and is reached across a subprocess
boundary (see ``contracts/action.py``). This module owns that boundary and
nothing else: building the command, reading its stdout protocol, loading the
per-head ``predictions.json`` files. What to do with the predictions — merge
rallies, cut actions to rally spans, write annotation files — is
``yp_video.action.spot_pass``.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path

from yp_video.action import prelabel
from yp_video.config import SPOT_DIR
from yp_video.contracts.action import (
    ACTION_CONTRACT_VERSION,
    ACTION_CONTRACT_VERSION_ENV,
    SPOT_PARTIAL_PREFIX,
    SPOT_PROGRESS_PREFIX,
)


class SpotInferenceError(RuntimeError):
    """yp-spot was unavailable or its inference subprocess failed.

    Callers that want graceful degradation (e.g. omit highlight data when
    action spotting fails) should catch this and continue rather than aborting.
    """


def _spot_progress_ratio(line: str) -> float | None:
    """Parse a yp-spot ``SPOT_PROGRESS`` stdout line to a fraction in [0, 1].

    Returns ``None`` for any non-progress line. Parsing lives in ``prelabel`` so
    the web dashboard and this worker path share one implementation.
    """
    # tqdm redraws its bar on stderr without a newline, so with stderr merged
    # into stdout the prefix can sit after a bar fragment on the same segment
    # — locate it rather than anchoring at column 0.
    at = line.find(SPOT_PROGRESS_PREFIX)
    if at < 0:
        return None
    data = prelabel.parse_spot_progress(line[at + len(SPOT_PROGRESS_PREFIX):])
    return prelabel.spot_progress_fraction(data) if data is not None else None


def _spot_partial_payload(line: str) -> tuple[str, bool, list[dict]] | None:
    """Parse a yp-spot ``SPOT_PARTIAL`` stdout line to ``(task, cumulative, events)``.

    ``task`` names the head the events belong to — a joint pass interleaves
    the heads' lines on one stream. ``cumulative=True`` (postprocessed point
    heads such as action) means ``events`` is the head's full settled prefix
    and replaces everything streamed before for it; ``cumulative=False``
    (dense segment heads such as rally) means it is that batch's delta.
    Returns ``None`` for any non-partial line and for a malformed payload —
    the authoritative event set still arrives via ``predictions.json``.
    """
    at = line.find(SPOT_PARTIAL_PREFIX)
    if at < 0:
        return None
    try:
        payload = json.loads(line[at + len(SPOT_PARTIAL_PREFIX):])
    except ValueError:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("task"), str):
        return None
    events = payload.get("events")
    return payload["task"], bool(payload.get("cumulative")), (events if isinstance(events, list) else [])


def run_spot_inference(
    video_path: Path | str,
    *,
    checkpoint: Path,
    tasks: Sequence[str],
    batch_size: int = 8,
    num_workers: int = 0,
    clip_len: int = 64,
    use_amp: bool = True,
    segments: Sequence[tuple[float, float]] | None = None,
    on_progress: Callable[[float], None] | None = None,
    on_events: Callable[[str, list[dict]], None] | None = None,
) -> dict[str, list[dict]]:
    """Run one yp-spot inference subprocess — one decode pass over the video
    for every head in ``tasks`` — and return ``{task: predictions}``.

    Streams stdout so progress ticks surface live; merges stderr in so a
    single reader can't deadlock and the error tail is captured too. yp-spot
    postprocesses point heads (score filter + NMS) and leaves segment heads
    dense, so callers never say which. ``num_workers`` is the decode thread
    count handed to ffmpeg (0 = auto).

    ``on_events(task, events)`` is progressive delivery: fired per settled
    batch with that head's cumulative event list so far (deltas accumulate,
    cumulative payloads replace) — one list per head, never merged.

    Raises:
        SpotInferenceError: yp-spot is not installed, or its inference
            subprocess failed / produced no output.
    """
    if not prelabel.spot_available():
        raise SpotInferenceError(
            f"yp-spot not available (looked under {SPOT_DIR}); "
            "set YP_SPOT_DIR / YP_SPOT_PYTHON and install its venv"
        )

    with tempfile.TemporaryDirectory(prefix="yp-spot-infer-") as tmp_root:
        save_dir = Path(tmp_root)
        pred_files = {task: save_dir / task / "predictions.json" for task in tasks}
        cmd = prelabel.build_command(
            video_source=str(video_path),
            checkpoint_path=checkpoint,
            tasks=tasks,
            save_dir=save_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            clip_len=clip_len,
            use_amp=use_amp,
            segments=segments,
        )
        env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            ACTION_CONTRACT_VERSION_ENV: ACTION_CONTRACT_VERSION,
        }
        # Ask yp-spot to stream settled foreground events only when a consumer
        # wants them. Leaving it off otherwise keeps runs from emitting
        # SPOT_PARTIAL lines nobody reads.
        if on_events is not None:
            env["SPOT_EMIT_PARTIAL"] = "1"
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
        partial: dict[str, list[dict]] = {}
        assert proc.stdout is not None
        for raw in proc.stdout:
            # tqdm redraws end in \r without a newline; merged into stdout they
            # glue onto the next SPOT_PROGRESS print. Split on \r as well so
            # the progress prefix always sits at the start of its segment
            # (stream_subprocess does the same for the web jobs).
            for line in raw.rstrip("\n").split("\r"):
                if not line:
                    continue
                ratio = _spot_progress_ratio(line)
                if ratio is not None:
                    if on_progress:
                        on_progress(ratio)
                    continue
                if on_events is not None:
                    parsed = _spot_partial_payload(line)
                    if parsed is not None:
                        task, cumulative, batch = parsed
                        if cumulative:
                            partial[task] = batch
                        else:
                            partial.setdefault(task, []).extend(batch)
                        on_events(task, partial[task])
                        continue
                tail.append(line)
        rc = proc.wait()
        if rc != 0:
            raise SpotInferenceError(
                f"yp-spot inference failed (rc={rc}): " + " | ".join(list(tail)[-5:])
            )
        missing = [task for task, path in pred_files.items() if not path.exists()]
        if missing:
            raise SpotInferenceError(
                f"yp-spot produced no predictions for {missing} under {save_dir}"
            )
        return {task: prelabel.load_predictions(path) for task, path in pred_files.items()}
