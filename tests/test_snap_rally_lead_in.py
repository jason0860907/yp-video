"""Boundary edits preserve identities and handle ambiguous/out-of-video spans."""
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("snap_rally_lead_in", SCRIPTS / "snap_rally_lead_in.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.path.pop(0)
snap = module.snap


def test_preserves_identity_and_extra_fields_without_mutation():
    row = dict(rally_id=27, start=8.5, end=21.5, label="rally", winner="near", custom="keep")
    result = snap(row, [(10, "serve"), (20, "score")], 100)
    assert result == {**row, "start": 9, "end": 21}
    assert row["start"] == 8.5 and row["end"] == 21.5


def test_duplicate_serve_after_score_is_reported_even_when_trimmed():
    row = dict(rally_id=1, start=8.5, end=13.5)
    events = [(10, "serve"), (12, "score"), (13.3, "serve")]
    result = snap(row, events, 100)
    assert result == {**row, "start": 9, "end": 13}
    assert "多個 serve" in module.problems("test", row, events, 100)
    assert "多個 serve" not in module.problems("test", result, events, 100)
    assert snap(result, events, 100) == result
    assert len(events) == 3


def test_duplicate_score_uses_last_and_clamps_to_video():
    row = dict(rally_id=1, start=-1, end=11)
    result = snap(row, [(0.2, "serve"), (8, "score"), (9.7, "score")], 10)
    assert result == {**row, "start": 0, "end": 10}


def test_missing_anchor_keeps_only_that_edge():
    row = dict(rally_id=1, start=8, end=20)
    assert snap(row, [(10, "serve"), (18, "spike")], 100) == {**row, "start": 9}
    assert snap(row, [(10, "receive"), (18, "score")], 100) == {**row, "end": 19}
    assert snap(row, [], 100) == row


def test_reversed_anchors_fail_before_writing():
    row = dict(rally_id=1, start=0, end=20)
    with pytest.raises(ValueError):
        snap(row, [(2, "score"), (18, "serve")], 100)
