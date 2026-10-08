"""The yp-track process boundary: GTA answers with member indices, and every
per-detection field (box, score, mask) rides back on them."""

import sys
import textwrap

import numpy as np
import pytest

from yp_video.contracts.track import TRACK_CONTRACT_VERSION
from yp_video.tracklets import gta

# Joins every rally's tracklets into one, in frame order — or misbehaves on cue.
FAKE = textwrap.dedent('''
    import argparse, json, os, sys
    p = argparse.ArgumentParser()
    for a in ("--video", "--tracks", "--out", "--threads"):
        p.add_argument(a)
    a = p.parse_args()
    if os.environ.get("YP_TRACK_CONTRACT_VERSION") != "%s":
        sys.exit("bad contract")
    if os.path.basename(a.video) == "broken.mp4":
        print("Traceback: osnet exploded"); sys.exit(3)
    rallies = {}
    for line in open(a.tracks):
        t = json.loads(line)
        for i, f in enumerate(t["frames"]):
            rallies.setdefault(t["rally_id"], []).append((f, t["track_id"], i))
    print("TRACK_PROGRESS " + json.dumps({"done": 1, "total": 1}), flush=True)
    drop = os.path.basename(a.video) == "lossy.mp4"
    with open(a.out, "w") as out:
        for rally, members in rallies.items():
            members = sorted(members)[drop:]
            out.write(json.dumps({"rally_id": rally, "track_id": 1, "members": [[t, i] for _, t, i in members]}) + "\\n")
''' % TRACK_CONTRACT_VERSION)

RECORDS = [
    {"rally_id": 2, "track_id": 5, "frames": [10, 12], "boxes": [[1, 1, 5, 9], [2, 1, 6, 9]], "scores": [0.9, 0.8]},
    {"rally_id": 2, "track_id": 7, "frames": [16, 18], "boxes": [[3, 1, 7, 9], [4, 1, 8, 9]], "scores": [0.7, 0.6]},
]


@pytest.fixture
def fake_yp_track(tmp_path, monkeypatch):
    (tmp_path / "fake_gta.py").write_text(FAKE)
    monkeypatch.setattr(gta, "TRACK_PYTHON", type(tmp_path)(sys.executable))
    monkeypatch.setattr(gta, "TRACK_PKG_DIR", tmp_path)
    monkeypatch.setattr(gta, "TRACK_GTA_MODULE", "fake_gta")
    return tmp_path


def test_members_carry_boxes_scores_and_masks_across(fake_yp_track):
    masks = {"2:5": np.array([[1], [2]], np.uint8), "2:7": np.array([[3], [4]], np.uint8)}
    progress = []
    records, out_masks = gta.refine(fake_yp_track / "match.mp4", RECORDS, masks,
                                    on_progress=lambda d, n, msg: progress.append((d, n)))
    assert records == [{"rally_id": 2, "track_id": 1, "frames": [10, 12, 16, 18],
                        "boxes": [[1, 1, 5, 9], [2, 1, 6, 9], [3, 1, 7, 9], [4, 1, 8, 9]],
                        "scores": [0.9, 0.8, 0.7, 0.6]}]
    assert out_masks["2:1"].ravel().tolist() == [1, 2, 3, 4]
    assert progress == [(1, 1)]


def test_box_only_records_refine_without_masks(fake_yp_track):
    records, masks = gta.refine(fake_yp_track / "match.mp4", RECORDS, None)
    assert masks is None and records[0]["frames"] == [10, 12, 16, 18]


def test_failures_and_lost_detections_surface(fake_yp_track):
    with pytest.raises(gta.GTAError, match="rc=3: Traceback: osnet exploded"):
        gta.refine(fake_yp_track / "broken.mp4", RECORDS, None)
    with pytest.raises(gta.GTAError, match="lost or duplicated"):
        gta.refine(fake_yp_track / "lossy.mp4", RECORDS, None)


def test_nothing_to_refine_skips_yp_track(tmp_path, monkeypatch):
    monkeypatch.setattr(gta, "TRACK_PYTHON", tmp_path / "missing")
    assert gta.refine(tmp_path / "match.mp4", [], None) == ([], None)
