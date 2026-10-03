"""Contract for tracking through the yp-track package (McByte++).

yp-video owns detection and the tracks layout; yp-track owns trackers whose
dependencies cannot live in this environment. yp-track lives in a separate
repo + venv, so the two cannot share Python at runtime.

This module is the single authoritative definition on the yp-video side;
yp-track mirrors the constants in ``yp_track/contract.py``. A version
handshake keeps the copies honest: yp-video exports
``TRACK_CONTRACT_VERSION`` through ``YP_TRACK_CONTRACT_VERSION`` when it
spawns yp-track, and the consumer fails loud on a mismatch. Bump the version
whenever any layout below changes — and update both sides.

``python -m yp_track.mcbyte --video <mp4> --detections <npz> --spans <json>
--out <jsonl> --stride N --track-thresh T --low-thresh L --min-frames M
--threads K --cmc orb|none``

- ``--track-thresh``: first association above T, new tracks above T + 0.1;
  ``--low-thresh``: the second association recovers lost tracks above L.
- ``--min-frames``: tracklets with fewer frames are dropped (yp-video's
  ``tracking.MIN_TRACK_FRAMES``, the one floor every tracker shares).
- ``--threads``: CPU threads for each of the process's pools (torch,
  OpenCV, BLAS), so parallel workers share the cores.

- ``--detections``: npz, key ``"<native frame>"`` → ``(n, 5)`` float32
  ``x0 y0 x1 y1 score`` in frame pixels. A frame without a key has none.
- ``--spans``: JSON ``[[rally_id, first_frame, last_frame], ...]``, native
  and inclusive. One independent tracker per span; frames ``first``,
  ``first + stride``, ... are tracked, so detections are only read there.
- ``--out``: jsonl, one tracklet per line ``{rally_id, track_id, frames,
  boxes, scores, det_index}`` (no header). ``boxes[i]`` is the matched
  detection, rounded; ``det_index[i]`` indexes ``frames[i]``'s detection
  array, which is how per-detection data (masks) is carried across.
- stdout: progress lines ``TRACK_PROGRESS {"done": int, "total": int}``.
"""

TRACK_CONTRACT_VERSION = "1.2.0"
TRACK_CONTRACT_VERSION_ENV = "YP_TRACK_CONTRACT_VERSION"
TRACK_PROGRESS_PREFIX = "TRACK_PROGRESS "
