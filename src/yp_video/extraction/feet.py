"""Where each touch's actor stood, in the frame — the input court positions
stand on (court/positions.py).

One implementation for the labeling web page and for identify, which ships
these points in its result so the production court job can place touches
long after the extraction records are gone.
"""

from __future__ import annotations

from collections.abc import Iterable

from yp_video.contracts.action import event_id
from yp_video.core.jsonl import read_jsonl_cached
from yp_video.core.rallies import load_rallies
from yp_video.extraction import links
from yp_video.extraction import store as extraction_store
from yp_video.tracklets.store import tracklet_index, tracks_path

#: Contacts made in the air. At the contact the feet are off the floor, and
#: the floor homography puts raised feet far behind where the player stands —
#: twice as far for a 0.4 m jump under a camera 0.8 m up. Their feet come
#: from the takeoff or the landing instead.
AIRBORNE = ("spike", "block", "serve")
#: How far either side of an airborne contact grounded feet are looked for.
#: Both sides: tracking often loses a jumper mid-air, and a server's track
#: only starts once the rally does, just after the toss.
GROUNDED_WINDOW_S = 0.7


class NotReady(LookupError):
    pass


def _grounded_box(tracklet: dict, frame: int, window: int) -> list[float] | None:
    """The actor's box with feet on the floor around an airborne contact.

    Feet lowest in the frame over the window: a jump only raises them, so the
    lowest is the takeoff or the landing (the camera stands above the floor,
    so lower in the frame is lower in the world). A jumper lands about where
    they took off, so either one places them.
    """
    boxes = [
        box
        for f, box in zip(tracklet["frames"], tracklet["boxes"])
        if abs(f - frame) <= window
    ]
    return max(boxes, key=lambda box: box[3]) if boxes else None


def _rally_of(time: float, spans: list[dict]) -> int | None:
    for span in spans:
        if span["start"] <= time <= span["end"]:
            return span["rally_id"]
    return None


def actor_feet(stem: str, events: Iterable[dict]) -> dict[str, list[float] | str]:
    """Event id → the actor's feet (normalized bottom-centre of their box), or
    why there are none: outside rally, no detection, occluded, no
    association, no takeoff. Score events name nobody and are skipped.

    Raises NotReady without extraction records or tracks.
    """
    path = extraction_store.records_path(stem)
    if not path.exists():
        raise NotReady("No extraction records — run Player Detection and Association first")
    meta, records = read_jsonl_cached(path)
    width, height = meta["frame_size"]
    fps = float(meta.get("fps") or 0)
    if not fps:
        raise NotReady("Extraction records carry no fps")
    if not tracks_path(stem).exists():
        raise NotReady("No tracks — run Player Tracking first")
    spans = load_rallies(stem)
    actors = links.event_tracks(stem)
    index = tracklet_index(stem)
    window = round(GROUNDED_WINDOW_S * fps)
    placeable = {r["id"]: r for r in extraction_store.labelable(records, stem, fps)}

    feet: dict[str, list[float] | str] = {}
    for event in events:
        if event.get("frame") is None or event.get("label") in extraction_store.SKIP_LABELS:
            continue
        eid, frame = event_id(event), int(event["frame"])
        r = placeable.get(eid)
        if r is None:
            # labelable drops what lies in no rally; the rest never got a record.
            feet[eid] = "outside rally" if _rally_of(frame / fps, spans) is None else "no detection"
            continue
        if r.get("resolution") == "occluded":
            feet[eid] = "occluded"
            continue
        box = r.get("box")
        if not box or r.get("resolution") == "unresolved":
            feet[eid] = "no association"
            continue
        if r.get("label") in AIRBORNE:
            ref = actors.get(eid)
            tracklet = index.tracklet(ref) if ref else None
            box = _grounded_box(tracklet, frame, window) if tracklet else None
            if box is None:
                feet[eid] = "no takeoff"
                continue
        feet[eid] = [(box[0] + box[2]) / 2 / width, box[3] / height]
    return feet
