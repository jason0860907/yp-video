"""Contract for tracklet refinement through the yp-track package (GTA).

yp-video owns detection, ByteTrack and the tracks layout; yp-track owns the
refinement whose dependencies (OSNet via deep-person-reid) cannot live in
this environment. yp-track lives in a separate repo + venv, so the two cannot
share Python at runtime.

This module is the single authoritative definition on the yp-video side;
yp-track mirrors the constants in ``yp_track/contract.py``. A version
handshake keeps the copies honest: yp-video exports
``TRACK_CONTRACT_VERSION`` through ``YP_TRACK_CONTRACT_VERSION`` when it
spawns yp-track, and the consumer fails loud on a mismatch. Bump the version
whenever any layout below changes — and update both sides.

``python -m yp_track.gta --video <mp4> --tracks <jsonl> --out <jsonl> --threads K``

- ``--tracks``: jsonl, one input tracklet per line ``{rally_id, track_id,
  frames, boxes}`` (no header): native frames, boxes ``x0 y0 x1 y1`` in frame
  pixels. ``(rally_id, track_id)`` is unique; rallies are independent.
- ``--out``: jsonl, one refined tracklet per line ``{rally_id, track_id,
  members}`` (no header). ``members`` lists ``[input track_id, index]`` — the
  input detections it holds, in frame order — so every per-detection field
  (box, score, mask) is carried across by the caller. Every input detection
  lands in exactly one refined tracklet; ``track_id`` restarts at 1 per rally.
- ``--threads``: CPU threads for each of the process's pools (torch, OpenCV).
- stdout: progress lines ``TRACK_PROGRESS {"done": int, "total": int}`` over
  the frames read for embedding.
"""

TRACK_CONTRACT_VERSION = "2.0.0"
TRACK_CONTRACT_VERSION_ENV = "YP_TRACK_CONTRACT_VERSION"
TRACK_PROGRESS_PREFIX = "TRACK_PROGRESS "
