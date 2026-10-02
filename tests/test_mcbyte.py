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
    for a in ("--video", "--detections", "--spans", "--out", "--stride", "--track-thresh", "--cmc"):
        p.add_argument(a)
    a = p.parse_args()
    if os.environ.get("YP_TRACK_CONTRACT_VERSION") != "%s":
        sys.exit("bad contract")
    if a.video.endswith("broken.mp4"):
        print("Traceback: engine exploded"); sys.exit(3)
    dets = np.load(a.detections)
    (rally, f0, f1), = json.load(open(a.spans))
    frames = list(range(f0, f1 + 1, int(a.stride)))
    print("TRACK_PROGRESS " + json.dumps({"done": len(frames), "total": len(frames)}), flush=True)
    rec = {"rally_id": rally, "track_id": 1, "frames": frames,
           "boxes": [dets[str(f)][0, :4].round().tolist() for f in frames],
           "scores": [0.9] * len(frames), "det_index": [0] * len(frames),
           "args": [a.track_thresh, a.cmc]}
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
        tmp_path / "match.mp4", dets, [(5, 0, 8)], stride=2, track_thresh=0.4, cmc=False,
        on_progress=lambda d, t, m: progress.append((d, t)),
    )
    assert out[0]["frames"] == [0, 2, 4, 6, 8] and out[0]["boxes"][1] == [2, 0, 12, 20]
    assert out[0]["args"] == ["0.4", "none"]
    assert progress == [(5, 5)]


def test_failure_surfaces_the_engine_output(fake_yp_track, tmp_path):
    with pytest.raises(mcbyte.McByteError, match="engine exploded"):
        mcbyte.track(tmp_path / "broken.mp4", {}, [(1, 0, 2)], stride=1, track_thresh=0.6, cmc=True)


def test_rfdetr_masks_follow_the_matched_detection(monkeypatch, tmp_path):
    masks = {0: [np.full(4, 1, np.uint8), np.full(4, 2, np.uint8)], 2: [np.full(4, 3, np.uint8)]}
    monkeypatch.setattr(tracking.mcbyte, "track", lambda *a, **k: [
        {"rally_id": 1, "track_id": 7, "frames": [0, 2], "boxes": [[0] * 4] * 2,
         "scores": [0.9, 0.8], "det_index": [1, 0]},
    ])
    records, store = tracking._mcbyte_tracks(
        tmp_path / "cuts-sideline" / "m.mp4", [(1, 0, 2)], 2, {}, masks, on_progress=None,
    )
    assert "det_index" not in records[0]
    assert store["1:7"].tolist() == [[2] * 4, [3] * 4]
