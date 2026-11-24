"""board.with_verdicts: an event detection never boxed draws its label's box."""

from yp_video.actor.labels import ActorLabel, ActorVerdict
from yp_video.extraction import board


def test_a_boxless_row_takes_its_labels_box(monkeypatch):
    labels = {
        "a": ActorLabel(ActorVerdict.AGENT, 10, (1.0, 2.0, 3.0, 4.0)),
        "b": ActorLabel(ActorVerdict.AGENT_OCCLUDED, None, None),
    }
    monkeypatch.setattr(board.actor_labels, "load", lambda stem: labels)
    rows = [
        {"id": "a", "frame": 10, "box": None},
        {"id": "b", "frame": 20, "box": None},
        {"id": "c", "frame": 30, "box": None},
        {"id": "d", "frame": 40, "box": [5, 6, 7, 8]},
    ]
    out = board.with_verdicts("v", rows)
    assert [r["box"] for r in out] == [[1.0, 2.0, 3.0, 4.0], None, None, [5, 6, 7, 8]]
    assert [r["actor_review"] for r in out] == ["agent", "agent_occluded", "unreviewed", "unreviewed"]
    assert rows[0]["box"] is None  # the cached rows are never mutated
