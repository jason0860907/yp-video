"""The yp-track process boundary, and masks riding back on McByte++'s det indices."""

import sys
import textwrap

import numpy as np
import pytest

from yp_video.contracts.track import TRACK_CONTRACT_VERSION
from yp_video.tracklets import mcbyte, tracking

FAKE = textwrap.dedent('''
    import argparse, json, os, sys
    import numpy as np
    p = argparse.ArgumentParser()
    for a in ("--video", "--detections", "--spans", "--out", "--stride", "--track-thresh", "--low-thresh",
              "--min-frames", "--threads", "--cmc"):
        p.add_argument(a)
    a = p.parse_args()
    if os.environ.get("YP_TRACK_CONTRACT_VERSION") != "%s":
        sys.exit("bad contract")
    (rally, f0, f1), = json.load(open(a.spans))
    name = os.path.basename(a.video)
    if name == "broken.mp4" or (name == "one_broken.mp4" and rally == 1):
        print("Traceback: engine exploded"); sys.exit(3)
    if name in ("one_broken.mp4", "slow.mp4"):
        import time; time.sleep(60)
    dets = np.load(a.detections)
    frames = list(range(f0, f1 + 1, int(a.stride)))
    print("TRACK_PROGRESS " + json.dumps({"done": len(frames), "total": len(frames)}), flush=True)
    rec = {"rally_id": rally, "track_id": 1, "frames": frames,
           "boxes": [dets[str(f)][0, :4].round().tolist() for f in frames],
           "scores": [0.9] * len(frames), "det_index": [0] * len(frames),
           "args": [a.track_thresh, a.low_thresh, a.min_frames, a.cmc]}
    open(a.out, "w").write(json.dumps(rec) + "\\n")
''' % TRACK_CONTRACT_VERSION)


@pytest.fixture
def fake_yp_track(tmp_path, monkeypatch):
    (tmp_path / "fake_mcbyte.py").write_text(FAKE)
    monkeypatch.setattr(mcbyte, "TRACK_PYTHON", type(tmp_path)(sys.executable))
    monkeypatch.setattr(mcbyte, "TRACK_PKG_DIR", tmp_path)
    monkeypatch.setattr(mcbyte, "TRACK_MCBYTE_MODULE", "fake_mcbyte")
    return tmp_path


def test_track_round_trips_detections_spans_and_progress(fake_yp_track, tmp_path):
    dets = {f: np.array([[f, 0, f + 10, 20, 0.9]], np.float32) for f in range(0, 9, 2)}
    progress = []
    out = mcbyte.track(
        tmp_path / "match.mp4", dets, [(5, 0, 8)], stride=2, track_thresh=0.4, low_thresh=0.05, min_frames=5,
        cmc=False, on_progress=lambda d, t, m: progress.append((d, t)),
    )
    assert out[0]["frames"] == [0, 2, 4, 6, 8] and out[0]["boxes"][1] == [2, 0, 12, 20]
    assert out[0]["args"] == ["0.4", "0.05", "5", "none"]
    assert progress == [(5, 5)]


def test_failure_surfaces_the_engine_output(fake_yp_track, tmp_path):
    with pytest.raises(mcbyte.McByteError, match="engine exploded"):
        mcbyte.track(tmp_path / "broken.mp4", {}, [(1, 0, 2)], stride=1, track_thresh=0.6, low_thresh=0.1,
                     min_frames=5, cmc=True)


def test_one_failure_stops_the_other_workers(fake_yp_track, tmp_path):
    import time

    started = time.monotonic()
    with pytest.raises(mcbyte.McByteError, match="worker 0 rc=3: Traceback: engine exploded") as err:
        mcbyte.track(tmp_path / "one_broken.mp4", {}, [(1, 0, 900), (2, 0, 10), (3, 0, 20)], stride=1,
                     track_thresh=0.4, low_thresh=0.05, min_frames=5, cmc=False)
    assert time.monotonic() - started < 30  # the sleeping siblings were killed, not awaited
    assert "worker 1" not in str(err.value)  # stopped workers carry no blame


def test_no_worker_outlives_a_failing_progress_callback(fake_yp_track, tmp_path):
    def explode(*_):
        raise RuntimeError("callback broke")

    with pytest.raises(RuntimeError, match="callback broke"):
        mcbyte.track(tmp_path / "match.mp4", {f: np.zeros((1, 5), np.float32) for f in range(3)}, [(1, 0, 2)],
                     stride=1, track_thresh=0.4, low_thresh=0.05, min_frames=5, cmc=False, on_progress=explode)


def test_an_interrupted_call_kills_its_workers(fake_yp_track, tmp_path, monkeypatch):
    import subprocess

    procs = []
    real_popen = subprocess.Popen

    def popen(*a, **k):
        procs.append(real_popen(*a, **k))
        return procs[-1]

    def interrupt(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(mcbyte.subprocess, "Popen", popen)
    monkeypatch.setattr(mcbyte.time, "sleep", interrupt)
    with pytest.raises(KeyboardInterrupt):
        mcbyte.track(tmp_path / "slow.mp4", {}, [(1, 0, 5)], stride=1, track_thresh=0.4, low_thresh=0.05,
                     min_frames=5, cmc=False)
    assert procs and all(p.poll() is not None for p in procs)


def test_rallies_are_dealt_longest_first_to_the_least_loaded_worker():
    spans = [(1, 0, 100), (2, 0, 900), (3, 0, 300), (4, 0, 500), (5, 0, 50)]
    groups = mcbyte._balanced(spans, 3)
    assert sorted(r for g in groups for r, _, _ in g) == [1, 2, 3, 4, 5]
    assert [[r for r, _, _ in g] for g in groups] == [[2], [4], [3, 1, 5]]
    assert mcbyte._balanced(spans[:2], 3) == [[(2, 0, 900)], [(1, 0, 100)]]


def test_rfdetr_masks_follow_the_matched_detection(monkeypatch, tmp_path):
    masks = {0: [np.full(4, 1, np.uint8), np.full(4, 2, np.uint8)], 2: [np.full(4, 3, np.uint8)]}
    seen = {}

    def track(*_a, **kw):
        seen.update(kw)
        return [{"rally_id": 1, "track_id": 7, "frames": [0, 2], "boxes": [[0] * 4] * 2,
                 "scores": [0.9, 0.8], "det_index": [1, 0]}]

    monkeypatch.setattr(tracking.mcbyte, "track", track)
    records, store = tracking._mcbyte_tracks(
        tmp_path / "m.mp4", [(1, 0, 2)], 2, {}, masks, moving_camera=False, on_progress=None,
    )
    # The dense pass's own floor is McByte++'s second-association floor.
    assert seen["low_thresh"] == tracking.TRACK_SCORE_THRESHOLD
    assert seen["track_thresh"] == tracking.RFDETR_MCBYTE_TRACK_THRESH
    assert seen["min_frames"] == tracking.MIN_TRACK_FRAMES and seen["cmc"] is False
    assert "det_index" not in records[0]
    assert store["1:7"].tolist() == [[2] * 4, [3] * 4]
