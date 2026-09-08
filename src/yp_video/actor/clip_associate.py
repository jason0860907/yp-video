"""Ask the per-player clip classifier who performed each known event.

The classifier lives in yp-spot (``yp_spot.clips``) behind its own venv and
GPU, so this is a subprocess call, the same shape as ``spot_associate``.
What crosses the boundary: the crops of every candidate going in (cut by
``actor/clips.py``, the SAME exporter training's crops came from), and per
candidate the class probabilities plus a contact point coming back.

Deciding is then a comparison, not a threshold: the event's label says
what happened, so the actor is the candidate the classifier rates most
likely to have done THAT — and only if it is also that candidate's own
best answer. If nobody on the frame reads as a spiker, nobody is named.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from yp_video.action import prelabel
from yp_video.actor import clips
from yp_video.actor.spot_associate import action_label_path
from yp_video.config import ASSOCIATION_DIR, SPOT_CHECKPOINTS_DIR, SPOT_DIR, SPOT_PYTHON
from yp_video.contracts.action import CLIP_PACKAGE_TYPE
from yp_video.core.checkpoints import checkpoint_ref
from yp_video.core.jsonl import atomic_write, read_jsonl
from yp_video.extraction.store import SKIP_LABELS
from yp_video.tracklets.geometry import TrackRef

CLIP_PREDICT_MODULE = "yp_spot.clips.predict"
#: Where a video's last answers land, for inspection and evaluation.
CLIP_PREDICTIONS_DIR = ASSOCIATION_DIR / "clips"
#: What a run directory must contribute to a package.
_PACKAGE_FILES = ("checkpoint_best.pt", "config.json", "metrics.jsonl")


@dataclass(frozen=True)
class ClipAnswer:
    """One event's answer: who, how sure, and where they touched the ball
    (source-frame pixels)."""

    track: TrackRef | None
    confidence: float
    contact_px: tuple[float, float] | None


# ------------------------------------------------------------------ packages


def package_run(run_dir: Path, root: Path = SPOT_CHECKPOINTS_DIR) -> Path:
    """Copy a ``yp_spot.clips.train`` run into the checkpoint layout with a
    manifest, so the Inference page can list and serve it."""
    missing = [name for name in _PACKAGE_FILES if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{run_dir} lacks {', '.join(missing)}")
    package = root / run_dir.name
    package.mkdir(parents=True, exist_ok=True)
    for name in _PACKAGE_FILES:
        shutil.copy2(run_dir / name, package / name)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    best = _best_epoch(run_dir / "metrics.jsonl")
    manifest = {
        "type": CLIP_PACKAGE_TYPE,
        "created_at": time.time(),
        "run_name": run_dir.name,
        "source_run_dir": str(run_dir),
        "checkpoint": "checkpoint_best.pt",
        "classes": config["classes"],
        "feature_arch": config["feature_arch"],
        "best": best,
    }
    (package / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return package


def _best_epoch(metrics: Path) -> dict:
    """``{epoch, metric, value}`` of the epoch the trainer kept."""
    best: dict = {}
    with metrics.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            value = record["val"]["macro_f1_actions"]
            if not best or value > best["value"]:
                best = {"epoch": record["epoch"], "metric": "macro_f1_actions", "value": value}
    return best


def list_checkpoints(root: Path = SPOT_CHECKPOINTS_DIR) -> list[dict]:
    """Clip classifier packages under ``root``, newest first."""
    rows = []
    for package in sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []:
        manifest_path = package / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if manifest.get("type") != CLIP_PACKAGE_TYPE:
            continue
        path = package / manifest.get("checkpoint", "checkpoint_best.pt")
        if not path.is_file():
            continue
        best = manifest.get("best") or {}
        stat = path.stat()
        rows.append({
            "path": checkpoint_ref(path),
            "name": f"{package.name}/{path.name}",
            "experiment": package.name,
            "epoch": int(best.get("epoch", -1)),
            "is_best": True,
            "best_metric": best.get("metric"),
            "best_value": best.get("value"),
            "mtime": stat.st_mtime,
            "size_mb": stat.st_size / (1024 * 1024),
        })
    rows.sort(key=lambda c: c["mtime"], reverse=True)
    return rows


def default_checkpoint(root: Path = SPOT_CHECKPOINTS_DIR) -> str:
    rows = list_checkpoints(root)
    return rows[0]["path"] if rows else ""


def resolve_checkpoint(value: str, root: Path = SPOT_CHECKPOINTS_DIR) -> Path:
    """The clip classifier weights behind ``value`` (the newest package when
    empty), verified to be a clip package."""
    ref = value or default_checkpoint(root)
    if not ref:
        raise FileNotFoundError(
            f"No clip classifier package under {root}; train one with "
            "yp_spot.clips.train and package it with yp-clip-package"
        )
    checkpoint = prelabel.resolve_checkpoint(ref, root)
    manifest = checkpoint.parent / "manifest.json"
    try:
        kind = json.loads(manifest.read_text(encoding="utf-8")).get("type")
    except (OSError, ValueError):
        kind = None
    if kind != CLIP_PACKAGE_TYPE:
        raise ValueError(f"{checkpoint.parent.name} is not a clip classifier package")
    return checkpoint


# ------------------------------------------------------------------ deciding


def decide_event(label: str, scored: Sequence[dict]) -> ClipAnswer | None:
    """The candidate most likely to have performed ``label``, if that is also
    its own top class; None when nobody on the frame reads as doing it."""
    best = None
    for row in scored:
        probs = row["probs"]
        if label not in probs:
            return None
        if best is None or probs[label] > best["probs"][label]:
            best = row
    if best is None:
        return None
    probs = best["probs"]
    if max(probs, key=probs.get) != label:
        return None
    contact = best.get("contact_px")
    return ClipAnswer(
        track=TrackRef.parse(best["track"]),
        confidence=float(probs[label]),
        contact_px=(float(contact[0]), float(contact[1])) if contact else None,
    )


def decide(events: Iterable[dict], scored: Iterable[dict]) -> dict[str, ClipAnswer]:
    by_event: dict[str, list[dict]] = defaultdict(list)
    for row in scored:
        by_event[row["event_id"]].append(row)
    answers: dict[str, ClipAnswer] = {}
    for event in events:
        rows = by_event.get(clips.event_id(event))
        if not rows:
            continue
        answer = decide_event(str(event["label"]), rows)
        if answer is not None:
            answers[clips.event_id(event)] = answer
    return answers


# ------------------------------------------------------------------- running


def run(
    video: Path,
    checkpoint: Path,
    *,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, ClipAnswer]:
    """Score one video's events with the clip classifier and persist the
    answers; returns them too. ``video`` must be readable on disk."""
    stem = video.stem
    labels = action_label_path(stem)
    if labels is None:
        raise FileNotFoundError(f"No action labels for {stem} — run Action Predict first")
    _meta, all_events = read_jsonl(labels)
    events = [e for e in all_events if e.get("frame") is not None and e.get("label") not in SKIP_LABELS]

    total_steps = 3
    if on_progress is not None:
        on_progress(0, total_steps, "cutting candidate clips...")
    samples = clips.plan_candidates(stem, events)
    if not samples:
        raise RuntimeError(f"{stem} has no tracked candidates — run Rally Tracking first")

    with tempfile.TemporaryDirectory() as scratch:
        scratch_path = Path(scratch)
        clips.export_video(stem, video, scratch_path, samples)
        predictions = scratch_path / "predictions.jsonl"
        if on_progress is not None:
            on_progress(1, total_steps, f"scoring {len(samples)} clips...")
        result = subprocess.run(
            [
                str(SPOT_PYTHON), "-m", CLIP_PREDICT_MODULE,
                "--checkpoint", str(checkpoint),
                "--clips_dir", str(scratch_path),
                "--out", str(predictions),
            ],
            cwd=SPOT_DIR, capture_output=True, text=True,
        )
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "").strip().splitlines()
            raise RuntimeError(
                f"yp-spot clips.predict failed (rc={result.returncode}): "
                + (tail[-1] if tail else "no output")
            )
        with predictions.open(encoding="utf-8") as f:
            scored = [json.loads(line) for line in f]

    answers = decide(events, scored)
    CLIP_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    with atomic_write(CLIP_PREDICTIONS_DIR / f"{stem}_actor_predictions.json") as handle:
        json.dump(
            {
                "video": stem,
                "checkpoint": checkpoint.parent.name,
                "events": [
                    {
                        "id": event_id, "track": answer.track.key,
                        "confidence": round(answer.confidence, 4),
                        "contact_px": answer.contact_px,
                    }
                    for event_id, answer in answers.items()
                ],
                "clips": scored,
            },
            handle, ensure_ascii=False,
        )
    if on_progress is not None:
        on_progress(total_steps, total_steps, f"{len(answers)} of {len(events)} events decided")
    return answers


def main() -> None:
    """``yp-clip-package <run_dir>``: package a clip classifier run."""
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("run_dir", type=Path, help="a yp_spot.clips.train run directory (exp/<run>)")
    args = p.parse_args()
    package = package_run(args.run_dir.resolve())
    print(f"packaged → {package}")


if __name__ == "__main__":
    main()
