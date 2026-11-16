"""The match sets rule held to the shared fixture.

The fixture lives with the App (``VolleyIQTests/Fixtures/match_sets.json``);
the Swift and Worker ports read the same file, so a rule change on any side
fails until all three agree again.
"""

import json
from pathlib import Path

from yp_video.web.app_review.models import Corrections
from yp_video.web.app_review.sets import assign

FIXTURE = Path(__file__).resolve().parents[2] / "VolleyIQ/VolleyIQTests/Fixtures/match_sets.json"


def test_rule_matches_fixture():
    for case in json.loads(FIXTURE.read_text())["cases"]:
        rallies = [(r["index"], r["start"]) for r in case["rallies"]]
        got = assign(rallies, case["starts"], case["ends"])
        assert {str(i): got.get(i) for i, _ in rallies} == case["expected"], case["name"]


def corrections(**extra):
    return Corrections.model_validate({
        "schema_version": "7.0", "match_id": "m", "updated_at": "t", "roster": [],
        "actions": [], "scores": [], "deleted_rally_indices": [], "rally_winner_overrides": {},
        **extra,
    })


def test_corrections_carry_sets():
    assert corrections(sets={"starts": [4], "ends": []}).sets.starts == [4]
    assert corrections().sets is None
