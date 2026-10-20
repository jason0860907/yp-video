"""分析 → 摘要 held to the App's answer.

The fixture lives with the App (``VolleyIQTests/Fixtures/analysis_summary.json``)
and ``AnalysisSummaryFixtureTests`` runs it through ``Match.stats``; this runs
the same file through ``app_review/stats.py``, so a change to the summary on
either side fails one of them until the fixture is regenerated from the App.
"""

import json
from pathlib import Path

from yp_video.web.app_review.models import Bundle
from yp_video.web.app_review.projection import project

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "VolleyIQ/VolleyIQTests/Fixtures/analysis_summary.json"
)


def load() -> dict:
    assert FIXTURE.exists(), (
        f"{FIXTURE} is missing: this test reads the App's fixture, so the "
        "VolleyIQ repo must be checked out next to yp-video."
    )
    return json.loads(FIXTURE.read_text())


def fixture_bundle(f: dict) -> Bundle:
    events = [{**e, "frame": round(e["time"] * 30), "visible": True} for e in f["events"]]
    position = {e["time"]: i for i, e in enumerate(events)}
    rallies = [
        {"index": r["index"], "set": 1, "start": r["start"], "end": r["end"], "winner": r["winner"]}
        for r in f["rallies"]
    ]
    return Bundle.model_validate(
        {
            "result": {
                "job_id": "job",
                "user_id": "user",
                "match_id": "match",
                "video_r2_key": "",
                "total_duration": 100,
                "action_events": events,
                "rallies": rallies,
                "attacks": [
                    {"rally_index": int(r), "event_indices": [position[t] for t in a]}
                    for r, attacks in f["attacks"].items()
                    for a in attacks
                ],
                "rally_outcomes": [
                    {
                        "rally_index": r["index"],
                        "serve_point": None,
                        "score_event_index": position.get(f["whistles"].get(str(r["index"]))),
                        "deciding_event_indices": [
                            position[t] for t in f["deciding"].get(str(r["index"]), [])
                        ],
                    }
                    for r in f["rallies"]
                ],
            },
            "library_rallies": [
                {**rally, "id": r["id"], "match_id": "match"}
                for rally, r in zip(rallies, f["rallies"], strict=True)
            ],
            "corrections": {
                "schema_version": "7.0",
                "match_id": "match",
                "updated_at": "now",
                "roster": [{**p, "position": ""} for p in f["roster"]],
                "actions": [
                    {"key": key, "removed": True, "tag_ids": []} for key in f["removed_actions"]
                ],
                "scores": [
                    {"key": key, "removed": False, "tag_ids": [], "loss_reason": code}
                    for key, code in f["loss_reasons"].items()
                ],
                "deleted_rally_indices": [],
                "rally_winner_overrides": {
                    r["index"]: r["winner_override"] for r in f["rallies"] if r.get("winner_override")
                },
                "player_identification": {
                    "result_id": "identify",
                    "unit_roster": f["unit_roster"],
                    "removed_units": f["removed_units"],
                    "event_overrides": f["event_overrides"],
                },
                "team_sides": f["team_sides"],
            },
            "identification": {
                "version": 5,
                "job_id": "identify",
                "user_id": "user",
                "match_id": "match",
                "units": [{"key": u["id"], "events": u["events"]} for u in f["units"]],
            },
        }
    )


def actual(f: dict) -> dict:
    stats = project(fixture_bundle(f))["stats"]
    return {
        "rally_count": stats["rally_count"],
        "in_rally_seconds": stats["in_rally_seconds"],
        "team_score": stats["team_score"],
        "has_teams": stats["has_teams"],
        "progression": stats["progression"],
        "players": [
            {
                "number": p["player"]["number"],
                "touches": {k: n for k, n in p["touches"].items() if n},
                "deciding": p["deciding"],
            }
            for p in stats["players"]
        ],
        "losses": stats["losses"],
        # By value, not the App's wording: its tests may run in English.
        "findings": [
            {"id": "loss", "team": f["team"], "reason": f["reason"], "count": f["count"]}
            for f in stats["findings"]
        ],
    }


def test_python_summary_matches_the_app():
    f = load()
    assert actual(f) == f["expected"]
