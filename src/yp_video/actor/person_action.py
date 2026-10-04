"""Joint person/action inference at existing events, across the SPOT boundary.

Standard identify lets the model score its own person proposals and answers
with a box. Advanced identify hands it each event's tracklets instead
(``actor/candidates.boxes_on``) and answers with the tracklet it picked, so
every later step — crop, window embedding, units — stands on that tracklet
rather than re-matching a box onto one.
"""
from __future__ import annotations

import json
import math
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from yp_video.actor.candidates import boxes_on, normalized_paths
from yp_video.actor.policy import ActorPick, EventContext
from yp_video.config import SPOT_DIR, SPOT_PYTHON
from yp_video.contracts.action import event_id
from yp_video.core.progress import ProgressFn
from yp_video.tracklets.geometry import TrackRef


class PersonActionPolicy:
    """The person/action head's answers, decided and validated up front."""

    def __init__(self, picks: dict[str, ActorPick], *, name: str, needs_tracklets: bool):
        self.picks = picks
        self.name = name
        #: Tracklet answers resolve through the tracks and their masks.
        self.needs_tracklets = needs_tracklets

    def decide(self, context: EventContext) -> ActorPick:
        return self.picks[context.event_id]


def build_policy(video: Path, checkpoint: Path, events: list[dict], *,
                 width: int, height: int,
                 tracks: Mapping[str, Mapping[int, Sequence[float]]] | None = None,
                 on_progress: ProgressFn | None = None):
    """``tracks`` (``actor/candidates.track_paths``) switches the candidates
    from the model's own proposals to each event's tracklets, which the head
    follows across its window (their paths go along as ``--tracks``)."""
    rows = [{"id": event_id(e), "frame": int(e["frame"]), "label": e["label"]}
            for e in events]
    keys = None
    if tracks is not None:
        keys = {}
        for row in rows:
            near = boxes_on(tracks, row["frame"], width, height)
            keys[row["id"]] = [key for key, _ in near]
            row["candidates"] = [{"track": key, "box": box} for key, box in near]
    with tempfile.TemporaryDirectory(prefix="fusion-association-") as scratch:
        source, output = Path(scratch) / "events.json", Path(scratch) / "answers.json"
        source.write_text(json.dumps(rows))
        command = [str(SPOT_PYTHON), "-m", "yp_spot.person_action.associate",
                   "--checkpoint", str(checkpoint), "--video", str(video),
                   "--events", str(source), "--out", str(output)]
        if keys is not None:
            paths = Path(scratch) / "tracks.json"
            named = {key for options in keys.values() for key in options}
            paths.write_text(json.dumps(normalized_paths(tracks, sorted(named), width, height)))
            command += ["--tracks", str(paths)]
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
    return policy_from_answers(rows, answers, width=width, height=height, keys=keys)


def policy_from_answers(rows: list[dict], answers: list[dict], *, width: int, height: int,
                        keys: Mapping[str, list[str]] | None = None):
    """Validate the subprocess boundary before any crop or embedding is made.

    ``keys`` (event id → the tracklet behind each candidate, in order) marks
    a tracklet call: the answer's ``pick`` names one of those tracklets and
    its box is not used. Otherwise the box is the answer.
    """
    if len(answers) != len(rows) or {a["id"] for a in answers} != {r["id"] for r in rows}:
        raise ValueError("Person/action output does not match the requested events")
    expected = {r["id"]: r for r in rows}
    by_tracklet = keys is not None
    name = "fusion-person-action:tracklet" if by_tracklet else "fusion-person-action"
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
        track = box = None
        if by_tracklet:
            options = keys[answer["id"]]
            if count != len(options):
                raise ValueError("Person/action scored a different candidate set")
            track = TrackRef.parse(options[pick]) if pick is not None else None
        elif pick is not None:
            box = answer["box"]
            if (box is None or len(box) != 4
                    or not all(math.isfinite(v) and 0 <= v <= 1 for v in box)
                    or box[0] >= box[2] or box[1] >= box[3]):
                raise ValueError("Invalid person/action box")
            box = (box[0] * width, box[1] * height, box[2] * width, box[3] * height)
        picks[answer["id"]] = ActorPick(
            box=box, track=track, candidates=count,
            diagnostic={"source": name, "status": answer["status"]},
        )
    return PersonActionPolicy(picks, name=name, needs_tracklets=by_tracklet)
