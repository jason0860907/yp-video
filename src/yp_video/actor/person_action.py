"""Joint person/action inference at existing events, across the SPOT boundary."""
from __future__ import annotations

import json
import math
import subprocess
import tempfile
from pathlib import Path

from yp_video.actor.policy import ActorPick, EventContext
from yp_video.config import SPOT_DIR, SPOT_PYTHON
from yp_video.contracts.action import event_id
from yp_video.core.progress import ProgressFn


class PersonActionPolicy:
    name = "fusion-person-action"
    needs_tracklets = False

    def __init__(self, answers: dict[str, dict]):
        self.answers = answers

    def decide(self, context: EventContext) -> ActorPick:
        answer = self.answers[context.event_id]
        return ActorPick(
            box=tuple(answer["box"]) if answer["box"] is not None else None,
            candidates=answer["candidates"],
            diagnostic={"source": self.name, "status": answer["status"]},
        )


def build_policy(video: Path, checkpoint: Path, events: list[dict], *,
                 width: int, height: int, on_progress: ProgressFn | None = None):
    rows = [{"id": event_id(e), "frame": int(e["frame"]), "label": e["label"]}
            for e in events]
    with tempfile.TemporaryDirectory(prefix="fusion-association-") as scratch:
        source, output = Path(scratch) / "events.json", Path(scratch) / "answers.json"
        source.write_text(json.dumps(rows))
        command = [str(SPOT_PYTHON), "-m", "yp_spot.person_action.associate",
                   "--checkpoint", str(checkpoint), "--video", str(video),
                   "--events", str(source), "--out", str(output)]
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
    return policy_from_answers(rows, answers, width=width, height=height)


def policy_from_answers(rows: list[dict], answers: list[dict], *, width: int, height: int):
    """Validate the subprocess boundary before any crop or embedding is made."""
    if len(answers) != len(rows) or {a["id"] for a in answers} != {r["id"] for r in rows}:
        raise ValueError("Person/action output does not match the requested events")
    expected = {r["id"]: r for r in rows}
    for answer in answers:
        row = expected[answer["id"]]
        if any(answer[key] != row[key] for key in ("frame", "label")):
            raise ValueError("Person/action inference changed an existing event")
        box = answer["box"]
        if box is not None:
            if (len(box) != 4 or not all(math.isfinite(v) and 0 <= v <= 1 for v in box)
                    or box[0] >= box[2] or box[1] >= box[3]):
                raise ValueError("Invalid person/action box")
            answer["box"] = [box[0] * width, box[1] * height,
                             box[2] * width, box[3] * height]
    return PersonActionPolicy({a["id"]: a for a in answers})
