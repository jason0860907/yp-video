"""Ask the yp-spot actor head who performed each action, for one video.

The model needs frames and a GPU, both of which live in the other repo behind
its own venv, so this is a subprocess call — the same shape as
``action/prelabel.py`` for the spotting model. What crosses the boundary is
the candidate set going in and a choice per event coming back; neither side
imports the other.

The candidates are built here, from tracking, and deliberately by the SAME
function that builds them for training. If inference offered a different
candidate set than training saw, the model would be answering a question
nobody taught it.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from yp_video.action import actor_labels, prelabel
from yp_video.action.frames import ensure_action_frame_cache
from yp_video.actor.spot_predictions import (
    ACTOR_PREDICTIONS_DIR,
    SpotAnswer,
    predictions_path,
)
from yp_video.config import (
    ACTION_ANNOTATIONS_DIR,
    ACTION_AUDIO_DIR,
    ACTION_FRAMES_DIR,
    ACTION_PRE_ANNOTATIONS_DIR,
    SPOT_DIR,
    SPOT_PYTHON,
)
from yp_video.contracts.action import (
    ACTION_CONTRACT_VERSION,
    ACTION_CONTRACT_VERSION_ENV,
    ACTOR_FILE_SUFFIX,
)
from yp_video.core.jsonl import atomic_write, read_jsonl, write_jsonl

SPOT_ASSOCIATE_MODULE = "yp_spot.associate"


def action_label_path(stem: str) -> Path | None:
    """The action labels this video's events come from, manual winning."""
    for directory in (ACTION_ANNOTATIONS_DIR, ACTION_PRE_ANNOTATIONS_DIR):
        path = directory / f"{stem}_actions.jsonl"
        if path.exists():
            return path
    return None


def run(
    video: Path,
    checkpoint: Path,
    *,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, SpotAnswer]:
    """Score one video's events and persist the answers; returns them too."""
    stem = video.stem
    labels = action_label_path(stem)
    if labels is None:
        raise FileNotFoundError(f"No action labels for {stem} — run Action Predict first")

    if on_progress is not None:
        on_progress(0, 3, "building candidates from tracking...")
    _meta, events = read_jsonl(labels)
    rows = actor_labels.candidates_only(stem, events)
    if not rows:
        raise RuntimeError(
            f"{stem} has no tracked candidates — run Rally Tracking first"
        )

    # The frame cache is what the model was trained on; extraction may never
    # have needed it for this video.
    if on_progress is not None:
        on_progress(1, 3, "ensuring the frame cache...")
    ensure_action_frame_cache(video, cache_root=ACTION_FRAMES_DIR)

    with tempfile.TemporaryDirectory() as scratch:
        scratch_path = Path(scratch)
        candidates_dir = scratch_path / "candidates"
        candidates_dir.mkdir()
        write_jsonl(
            candidates_dir / f"{stem}{ACTOR_FILE_SUFFIX}",
            {"video": stem, "num_events": len(rows)},
            rows,
        )
        answers_file = scratch_path / "answers.json"

        if on_progress is not None:
            on_progress(2, 3, f"scoring {len(rows)} events...")
        result = subprocess.run(
            [
                str(SPOT_PYTHON),
                "-m", SPOT_ASSOCIATE_MODULE,
                "--checkpoint_path", str(checkpoint),
                "--frame_dir", str(ACTION_FRAMES_DIR),
                "--label_file", str(labels),
                "--actor_candidates", str(candidates_dir),
                "--audio_dir", str(ACTION_AUDIO_DIR),
                "--out", str(answers_file),
            ],
            cwd=SPOT_DIR,
            capture_output=True,
            text=True,
            env={
                **_spot_env(),
                ACTION_CONTRACT_VERSION_ENV: ACTION_CONTRACT_VERSION,
            },
        )
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "").strip().splitlines()
            raise RuntimeError(
                f"yp-spot associate failed (rc={result.returncode}): "
                + (tail[-1] if tail else "no output")
            )
        payload = json.loads(answers_file.read_text(encoding="utf-8"))

    ACTOR_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    with atomic_write(predictions_path(stem)) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)

    if on_progress is not None:
        on_progress(3, 3, f"{len(payload.get('events', []))} events decided")
    return {
        str(row["id"]): SpotAnswer(
            track=_track(row.get("track")),
            confidence=float(row.get("confidence") or 0.0),
            kind=str(row.get("kind") or "untracked"),
        )
        for row in payload.get("events", [])
    }


def _track(value):
    from yp_video.tracklets.geometry import TrackRef

    return TrackRef.parse(value) if value else None


def _spot_env() -> dict:
    import os

    return dict(os.environ)


def rejection(checkpoint: Path) -> str | None:
    """Why this checkpoint cannot answer "who acted", or None.

    Checked once, here, rather than discovered per video after the job has
    already started: a spotting checkpoint has no actor head, and the weights
    are the only honest place to ask.
    """
    try:
        import torch  # noqa: PLC0415 — optional at import time for the web app
    except ImportError:
        return None
    try:
        state = torch.load(str(checkpoint), weights_only=True, map_location="cpu")
    except Exception as exc:  # noqa: BLE001
        return f"Cannot read {checkpoint.name}: {exc}"
    if not any("_pred_actor" in key for key in state):
        return (
            f"{checkpoint.parent.name}/{checkpoint.name} has no actor head — "
            "train one with --predict_actor"
        )
    return None


def list_actor_checkpoints() -> list[dict]:
    """Action checkpoints that carry an actor head, for the picker.

    Read from each package's ``config.json`` rather than its weights: this
    feeds a status poll, and torch-loading every checkpoint on every poll to
    answer "does it have the head" would cost seconds. The weights remain the
    authority at submit time — see ``rejection``.
    """
    out: list[dict] = []
    for entry in prelabel.list_checkpoints():
        package = prelabel.resolve_checkpoint_path(entry["path"]).parent
        config = package / "config.json"
        if not config.exists():
            continue
        try:
            declared = json.loads(config.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not declared.get("predict_actor"):
            continue
        manifest = package / "manifest.json"
        summary = {}
        if manifest.exists():
            try:
                summary = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                summary = {}
        out.append(
            {
                "path": entry["path"],
                "name": package.name,
                "epoch": entry.get("epoch"),
                "mtime": entry.get("mtime"),
                "holdout": summary.get("holdout"),
                "metrics": summary.get("holdout_metrics") or {},
                "note": summary.get("note"),
            }
        )
    out.sort(key=lambda row: row.get("mtime") or 0, reverse=True)
    return out
