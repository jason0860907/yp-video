"""Ask the independent yp-association model who acted in each known event.

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

from yp_video.action import prelabel
from yp_video.action.frames import ensure_action_frame_cache
from yp_video.actor import candidates
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
    FUSION_PACKAGE_TYPE,
)
from yp_video.core.checkpoints import checkpoint_ref
from yp_video.core.jsonl import atomic_write, read_jsonl, write_jsonl

INDEPENDENT_ASSOCIATE_MODULE = "yp_spot.association.predict"
LEGACY_ASSOCIATE_MODULE = "yp_spot.associate"
LEGACY_ACTOR_FORMAT = "legacy-actor-head"
INDEPENDENT_FORMAT = "yp-association-v1"


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
    family = checkpoint_family(checkpoint)
    if family is None:
        raise ValueError(
            f"{checkpoint.parent.name}/{checkpoint.name} is not an "
            "Association Predict model"
        )
    labels = action_label_path(stem)
    if labels is None:
        raise FileNotFoundError(f"No action labels for {stem} — run Action Predict first")

    total_steps = 4 if family == LEGACY_ACTOR_FORMAT else 3
    if on_progress is not None:
        on_progress(0, total_steps, "building candidates from tracking...")
    _meta, events = read_jsonl(labels)
    rows = candidates.candidates_only(stem, events)
    if not rows:
        raise RuntimeError(
            f"{stem} has no tracked candidates — run Rally Tracking first"
        )

    # The frame cache is what the model was trained on; extraction may never
    # have needed it for this video.
    if on_progress is not None:
        on_progress(1, total_steps, "ensuring the frame cache...")
    ensure_action_frame_cache(video, cache_root=ACTION_FRAMES_DIR)

    audio_dir = None
    if family == LEGACY_ACTOR_FORMAT:
        if on_progress is not None:
            on_progress(2, total_steps, "ensuring legacy Log-mel audio...")
        audio_dir = _ensure_legacy_audio(video, labels, checkpoint)

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
        if family == INDEPENDENT_FORMAT:
            command = [
                str(SPOT_PYTHON),
                "-m", INDEPENDENT_ASSOCIATE_MODULE,
                "--checkpoint-path", str(checkpoint),
                "--frame-dir", str(ACTION_FRAMES_DIR),
                "--label-file", str(labels),
                "--actor-candidates", str(candidates_dir),
                "--out", str(answers_file),
            ]
        else:
            command = [
                str(SPOT_PYTHON),
                "-m", LEGACY_ASSOCIATE_MODULE,
                "--checkpoint_path", str(checkpoint),
                "--frame_dir", str(ACTION_FRAMES_DIR),
                "--label_file", str(labels),
                "--actor_candidates", str(candidates_dir),
                "--out", str(answers_file),
            ]
            if audio_dir is not None:
                command.extend(["--audio_dir", str(audio_dir)])

        if on_progress is not None:
            on_progress(total_steps - 1, total_steps, f"scoring {len(rows)} events...")
        result = subprocess.run(
            command,
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

    # Stamp WHO answered. The file is overwritten by whichever run predicted
    # last, so without this an evaluator comparing two heads has no way to
    # say which one it is reading — and would silently score a mix.
    payload["checkpoint"] = checkpoint.parent.name

    ACTOR_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    with atomic_write(predictions_path(stem)) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)

    if on_progress is not None:
        on_progress(
            total_steps, total_steps,
            f"{len(payload.get('events', []))} events decided",
        )
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



def _read_package_json(checkpoint: Path, filename: str) -> dict:
    path = checkpoint.parent / filename
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def checkpoint_family(checkpoint: Path) -> str | None:
    """Architecture contract declared by the checkpoint package."""
    config = _read_package_json(checkpoint, "config.json")
    manifest = _read_package_json(checkpoint, "manifest.json")
    if (
        config.get("task") == "association"
        and config.get("checkpoint_format") == INDEPENDENT_FORMAT
    ):
        return INDEPENDENT_FORMAT
    if (
        config.get("predict_actor") is True
        and manifest.get("type") == FUSION_PACKAGE_TYPE
    ):
        return LEGACY_ACTOR_FORMAT
    return None


def _ensure_legacy_audio(
    video: Path, labels: Path, checkpoint: Path
) -> Path | None:
    config = _read_package_json(checkpoint, "config.json")
    backend = str(config.get("audio_backend") or "none")
    if backend == "none":
        return None
    audio_dir = ACTION_AUDIO_DIR / backend
    audio_file = audio_dir / f"{video.stem}.npy"
    if audio_file.is_file():
        return audio_dir
    audio_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            str(SPOT_PYTHON),
            "-m",
            "yp_spot.audio.precompute",
            "--label-file",
            str(labels),
            "--video-root",
            str(video.parent),
            "--out",
            str(audio_dir),
            "--backend",
            backend,
        ],
        cwd=SPOT_DIR,
        capture_output=True,
        text=True,
        env=_spot_env(),
    )
    if result.returncode != 0 or not audio_file.is_file():
        tail = (result.stderr or result.stdout or "").strip().splitlines()
        raise RuntimeError(
            f"yp-spot audio precompute failed (rc={result.returncode}): "
            + (tail[-1] if tail else f"did not create {audio_file.name}")
        )
    return audio_dir


def rejection(checkpoint: Path) -> str | None:
    """Why this checkpoint cannot answer "who acted", or None.

    The package metadata chooses the inference contract; the weights then
    prove that the package really contains the corresponding head. Checked
    once here rather than discovered per video after the job has started.
    """
    family = checkpoint_family(checkpoint)
    if family is None:
        return (
            f"{checkpoint.parent.name}/{checkpoint.name} is not an "
            "Association Predict model"
        )
    try:
        import torch  # noqa: PLC0415 — optional at import time for the web app
    except ImportError:
        return None
    try:
        state = torch.load(str(checkpoint), weights_only=True, map_location="cpu")
    except Exception as exc:  # noqa: BLE001
        return f"Cannot read {checkpoint.name}: {exc}"
    if family == INDEPENDENT_FORMAT:
        if isinstance(state, dict) and state.get("format") == INDEPENDENT_FORMAT:
            return None
        return (
            f"{checkpoint.parent.name}/{checkpoint.name} is not an independent "
            "yp-association-v1 model — train it in Association Train"
        )
    if isinstance(state, dict) and any("_pred_actor" in key for key in state):
        return None
    return (
        f"{checkpoint.parent.name}/{checkpoint.name} declares a legacy actor "
        "head, but its weights do not contain one"
    )


def list_association_checkpoints() -> list[dict]:
    """Association checkpoints, including supported legacy actor heads.

    Read from each package's ``config.json`` rather than its weights: this
    feeds a status poll, and torch-loading every checkpoint on every poll to
    answer "does it have the head" would cost seconds. The weights remain the
    authority at submit time — see ``rejection``.
    """
    out: list[dict] = []
    for entry in prelabel.list_checkpoints():
        checkpoint = prelabel.resolve_checkpoint_path(entry["path"])
        package = checkpoint.parent
        family = checkpoint_family(checkpoint)
        if family is None:
            continue
        manifest = package / "manifest.json"
        summary = {}
        if manifest.exists():
            try:
                summary = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                summary = {}
        training = summary.get("training") or {}
        label_summary = training.get("label_summary") or {}
        holdout = summary.get("holdout")
        validation_videos = label_summary.get("val_videos") or (
            [holdout] if holdout else []
        )
        if holdout is None and len(validation_videos) == 1:
            holdout = validation_videos[0]
        best = summary.get("best") or {}
        # A fusion package selects its headline epoch by ACTION mAP; the
        # actor head's own best lives in the manifest's per-task record.
        # Answering "who acted" with the action-best epoch quietly serves a
        # compromised actor head, so this row points at the actor-best file.
        path, epoch = entry["path"], entry.get("epoch")
        actor_best = (summary.get("best_per_task") or {}).get("actor") or {}
        if family == LEGACY_ACTOR_FORMAT and actor_best.get("file"):
            candidate = package / actor_best["file"]
            if candidate.is_file():
                path = checkpoint_ref(candidate)
                epoch = actor_best.get("epoch", epoch)
        actor_quality = actor_best.get("metrics") or (
            ((best.get("task_metrics") or {}).get("actor") or {}).get(
                "validation"
            )
            or {}
        ).get("metrics") or {}
        metrics = (
            best.get("metrics")
            or (
                {
                    "player_top1": actor_quality.get("player_top1"),
                    "overall_exact": actor_quality.get("overall_top1"),
                    "occluded_recall": actor_quality.get("occluded_recall"),
                    "untracked_recall": actor_quality.get("untracked_recall"),
                }
                if actor_quality
                else {}
            )
            or summary.get("holdout_metrics")
            or {}
        )
        out.append(
            {
                "path": path,
                "name": package.name,
                "family": family,
                "epoch": epoch,
                "mtime": entry.get("mtime"),
                "holdout": holdout,
                "validation_videos": validation_videos,
                "actor_targets": (
                    label_summary.get("actor_targets")
                    or summary.get("actor_targets")
                    or {}
                ),
                "best": summary.get("best"),
                "metrics": metrics,
                "note": summary.get("note"),
            }
        )
    out.sort(key=lambda row: row.get("mtime") or 0, reverse=True)
    return out
