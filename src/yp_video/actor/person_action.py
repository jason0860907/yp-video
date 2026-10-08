"""Who touched the ball at each event, as an actor policy.

Two sources, one model (the fusion package's actor head):

* Standard identify reads the SPOT pass's own picks (core/actor_picks.py):
  the head already scored the person head's boxes at every event it
  spotted, so the answer is a box and nothing re-runs.
* Advanced identify hands the head each event's tracklets
  (``actor/candidates.boxes_on``) across the SPOT boundary and answers with
  the tracklet it picked, so every later step — crop, window embedding,
  units — stands on that tracklet rather than re-matching a box onto one.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from yp_video.actor.candidates import boxes_on, normalized_paths
from yp_video.actor.policy import ActorPick
from yp_video.config import SPOT_DIR, SPOT_PYTHON
from yp_video.contracts.action import event_id
from yp_video.core.actor_picks import SpotActorPick
from yp_video.core.progress import ProgressFn
from yp_video.tracklets.geometry import TrackRef

SPOT_PASS_POLICY = "fusion-spot-pass"
TRACKLET_POLICY = "fusion-person-action:tracklet"


class PersonActionPolicy:
    """The person/action head's answers, decided and validated up front."""

    def __init__(self, picks: dict[str, ActorPick], *, name: str, needs_tracklets: bool):
        self.picks = picks
        self.name = name
        #: Tracklet answers resolve through the tracks and their masks.
        self.needs_tracklets = needs_tracklets

    def decide(self, event_id: str) -> ActorPick:
        return self.picks[event_id]


def policy_from_spot_picks(events: list[dict], picks: Mapping[tuple[int, str], SpotActorPick], *,
                           width: int, height: int) -> PersonActionPolicy:
    """Standard identify's policy: each event takes the SPOT pass's pick at its
    frame and label. An event the pass did not spot that way (a label changed
    since, a hand-added touch) gets no pick — the user assigns it by hand."""
    decided = {}
    for event in events:
        pick = picks.get((int(event["frame"]), event["label"]))
        if pick is None:
            decided[event_id(event)] = ActorPick(diagnostic={"source": SPOT_PASS_POLICY, "status": "not_spotted"})
            continue
        box = pick.box
        decided[event_id(event)] = ActorPick(
            box=(box[0] * width, box[1] * height, box[2] * width, box[3] * height) if box else None,
            candidates=pick.candidates,
            diagnostic={"source": SPOT_PASS_POLICY, "status": "selected" if box else "no_candidate"},
        )
    return PersonActionPolicy(decided, name=SPOT_PASS_POLICY, needs_tracklets=False)


def build_policy(video: Path, checkpoint: Path, events: list[dict], *,
                 width: int, height: int,
                 tracks: Mapping[str, Mapping[int, Sequence[float]]],
                 on_progress: ProgressFn | None = None) -> PersonActionPolicy:
    """Advanced identify's policy: the actor head over each event's tracklets
    (``actor/candidates.track_paths``), following them across its window.
    ``checkpoint`` is a fusion package's weights carrying the actor head."""
    rows = [{"id": event_id(e), "frame": int(e["frame"]), "label": e["label"]}
            for e in events]
    keys = {}
    for row in rows:
        near = boxes_on(tracks, row["frame"], width, height)
        keys[row["id"]] = [key for key, _ in near]
        row["candidates"] = [{"track": key, "box": box} for key, box in near]
    with tempfile.TemporaryDirectory(prefix="fusion-association-") as scratch:
        source, output = Path(scratch) / "events.json", Path(scratch) / "answers.json"
        source.write_text(json.dumps(rows))
        paths = Path(scratch) / "tracks.json"
        named = {key for options in keys.values() for key in options}
        paths.write_text(json.dumps(normalized_paths(tracks, sorted(named), width, height)))
        command = [str(SPOT_PYTHON), "-m", "yp_spot.person_action.associate",
                   "--checkpoint", str(checkpoint), "--video", str(video),
                   "--events", str(source), "--out", str(output), "--tracks", str(paths)]
        tail = []
        with subprocess.Popen(command, cwd=SPOT_DIR, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True) as process:
            try:
                for line in process.stdout:
                    tail.append(line.rstrip())
                    tail = tail[-20:]
                    if line.startswith("PROGRESS ") and on_progress:
                        _, done, total, message = line.strip().split(maxsplit=3)
                        on_progress(int(done), int(total), message)
                if process.wait():
                    raise RuntimeError("Person/action inference failed: " + "\n".join(tail))
            except BaseException:
                process.kill()
                process.wait()
                raise
        answers = json.loads(output.read_text())["events"]
    return policy_from_answers(rows, answers, keys=keys)


def policy_from_answers(rows: list[dict], answers: list[dict], *,
                        keys: Mapping[str, list[str]]) -> PersonActionPolicy:
    """Validate the subprocess boundary before any crop or embedding is made.

    ``keys`` is event id → the tracklet behind each candidate, in order: the
    answer's ``pick`` names one of those tracklets.
    """
    if len(answers) != len(rows) or {a["id"] for a in answers} != {r["id"] for r in rows}:
        raise ValueError("Person/action output does not match the requested events")
    expected = {r["id"]: r for r in rows}
    picks = {}
    for answer in answers:
        row = expected[answer["id"]]
        if any(answer[key] != row[key] for key in ("frame", "label")):
            raise ValueError("Person/action inference changed an existing event")
        pick, count = answer["pick"], answer["num_candidates"]
        if pick is not None and not (type(pick) is int and 0 <= pick < count):
            raise ValueError("Person/action pick does not name a candidate")
        if (pick is None) != (answer["status"] == "no_candidate"):
            raise ValueError("Person/action status disagrees with its pick")
        options = keys[answer["id"]]
        if count != len(options):
            raise ValueError("Person/action scored a different candidate set")
        picks[answer["id"]] = ActorPick(
            track=TrackRef.parse(options[pick]) if pick is not None else None, candidates=count,
            diagnostic={"source": TRACKLET_POLICY, "status": answer["status"]},
        )
    return PersonActionPolicy(picks, name=TRACKLET_POLICY, needs_tracklets=True)
