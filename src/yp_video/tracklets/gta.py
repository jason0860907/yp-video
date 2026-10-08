"""GTA tracklet refinement, run by the yp-track package (contracts/track.py).

ByteTrack links boxes by overlap alone, which is where a volleyball rally
breaks it: a player lost behind someone at the net comes back as a new
tracklet, and two players crossing can trade one. GTA repairs a rally's
tracklets after the fact by appearance (OSNet trained on SportsMOT): it
splits a tracklet at an ID switch and joins tracklets that never share a
frame (yp_track/gta.py).

Measured 2026-10-08 on six human-named sideline videos (same-rally pairwise
F1 of tracklet ≡ player): ByteTrack .477 → ByteTrack + GTA .633, against
McByte++'s .586 in about half its time — so McByte++ was removed.

This module only crosses the process boundary: yp-track answers with which
input detections each refined tracklet holds, and every per-detection field
(box, score, mask) is carried across from the input here.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections import deque
from pathlib import Path

import numpy as np

from yp_video.config import TRACK_GTA_MODULE, TRACK_PKG_DIR, TRACK_PYTHON
from yp_video.contracts.track import (
    TRACK_CONTRACT_VERSION,
    TRACK_CONTRACT_VERSION_ENV,
    TRACK_PROGRESS_PREFIX,
)
from yp_video.core.progress import ProgressFn

#: Appended to the tracker named in a tracks header.
SOURCE_SUFFIX = f" + GTA (yp-track contract {TRACK_CONTRACT_VERSION})"


class GTAError(RuntimeError):
    """yp-track failed; the message carries the tail of its output."""


def refine(
    video: Path,
    records: list[dict],
    masks: dict[str, np.ndarray] | None,
    *,
    on_progress: ProgressFn | None = None,
) -> tuple[list[dict], dict[str, np.ndarray] | None]:
    """GTA over ``records`` (``{rally_id, track_id, frames, boxes, scores}``,
    boxes in frame pixels); ``masks`` (key ``"rally:track"`` → rows aligned
    with that tracklet's frames) follow their detections, when given.

    Returns the refined records, track ids renumbered from 1 per rally.
    """
    if not records:
        return records, masks
    if not TRACK_PYTHON.exists():
        raise GTAError(f"yp-track is not installed at {TRACK_PKG_DIR} (uv sync there)")
    by_key = {(r["rally_id"], r["track_id"]): r for r in records}
    with tempfile.TemporaryDirectory(prefix="yp-track-") as tmp:
        source, out = Path(tmp) / "tracks.jsonl", Path(tmp) / "refined.jsonl"
        source.write_text("".join(
            json.dumps({"rally_id": r["rally_id"], "track_id": r["track_id"],
                        "frames": r["frames"], "boxes": r["boxes"]}) + "\n"
            for r in records
        ))
        _run([str(TRACK_PYTHON), "-m", TRACK_GTA_MODULE, "--video", str(video),
              "--tracks", str(source), "--out", str(out), "--threads", str(os.cpu_count() or 1)],
             on_progress)
        refined = [json.loads(line) for line in out.read_text().splitlines() if line]

    out_records: list[dict] = []
    out_masks: dict[str, np.ndarray] | None = {} if masks is not None else None
    covered = 0
    for t in refined:
        members = [(by_key[(t["rally_id"], track)], index) for track, index in t["members"]]
        covered += len(members)
        out_records.append({
            "rally_id": t["rally_id"],
            "track_id": t["track_id"],
            "frames": [r["frames"][i] for r, i in members],
            "boxes": [r["boxes"][i] for r, i in members],
            "scores": [r["scores"][i] for r, i in members],
        })
        if out_masks is not None:
            out_masks[f"{t['rally_id']}:{t['track_id']}"] = np.stack(
                [masks[f"{r['rally_id']}:{r['track_id']}"][i] for r, i in members]
            )
    if covered != sum(len(r["frames"]) for r in records):
        raise GTAError("GTA lost or duplicated detections")
    return out_records, out_masks


def _run(cmd: list[str], on_progress: ProgressFn | None) -> None:
    env = {**os.environ, "PYTHONUNBUFFERED": "1", TRACK_CONTRACT_VERSION_ENV: TRACK_CONTRACT_VERSION}
    tail: deque[str] = deque(maxlen=20)
    proc = subprocess.Popen(cmd, cwd=TRACK_PKG_DIR, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.strip()
            if line.startswith(TRACK_PROGRESS_PREFIX.strip()):
                p = json.loads(line[len(TRACK_PROGRESS_PREFIX.strip()):])
                if on_progress:
                    on_progress(p["done"], p["total"], f"GTA frame {p['done']}/{p['total']}")
            elif line:
                tail.append(line)
        if proc.wait():
            raise GTAError(f"yp-track gta rc={proc.returncode}: " + " | ".join(list(tail)[-5:]))
    except BaseException:
        proc.kill()
        proc.wait()
        raise
