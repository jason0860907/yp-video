"""App Review shows a match the way the App does once the user marked sets:
rallies outside every set — and the touches in them — are gone, and the
rest score set by set."""

from yp_video.web.app_review.models import Bundle
from yp_video.web.app_review.projection import project

MATCH = "00000000-0000-4000-8000-000000000001"


def bundle(sets: dict | None) -> Bundle:
    rallies = [
        {"index": i, "set": 1, "start": (i - 1) * 10.0, "end": (i - 1) * 10.0 + 6, "winner": "near"}
        for i in range(1, 5)
    ]
    events = [
        {"id": f"e{i}", "label": "spike", "time": (i - 1) * 10.0 + 3, "frame": ((i - 1) * 10 + 3) * 30, "visible": True}
        for i in range(1, 5)
    ]
    corrections = {
        "schema_version": "7.0", "match_id": MATCH, "updated_at": "t", "roster": [],
        "actions": [], "scores": [], "deleted_rally_indices": [], "rally_winner_overrides": {},
    }
    if sets is not None:
        corrections["sets"] = sets
    return Bundle.model_validate({
        "result": {
            "job_id": "job", "user_id": "user", "match_id": MATCH, "video_r2_key": "",
            "total_duration": 100, "action_events": events, "rallies": rallies,
            "attacks": [], "rally_outcomes": [],
        },
        "library_rallies": [{**r, "id": f"00000000-0000-4000-8000-{r['index']:012d}", "match_id": MATCH} for r in rallies],
        "corrections": corrections,
    })


def test_rallies_outside_sets_leave_the_preview():
    preview = project(bundle({"starts": [2, 4], "ends": [2]}))
    assert [(r["index"], r["set"]) for r in preview["rallies"]] == [(2, 1), (4, 2)]
    assert [c["anchor_time"] for c in preview["actions"]] == [13.0, 33.0]
    assert [c["set"] for c in preview["actions"]] == [1, 2]
    assert [c["rally_index"] for c in preview["scores"]] == [2, 4]
    assert [t["set"] for t in preview["court_totals"]] == [1, 2]


def test_uncorrected_view_and_no_marks_keep_every_rally():
    marked = bundle({"starts": [2], "ends": []})
    assert len(project(marked, corrected=False)["rallies"]) == 4
    assert len(project(bundle(None))["rallies"]) == 4
