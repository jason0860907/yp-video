"""The four label work lists and the union tally the sidebar polls.

Routers stay HTTP surfaces (tests/test_layering.py): the list builders live
here so annotate / action-annotate / association / reid each serve theirs
while app.py warms them and /label/stats counts them, without any router
importing another.

Every row carries a ``status`` from the one vocabulary every mode speaks —
unlabeled (nothing for this mode yet), pre-annotate (machine output only),
in-progress (a human started but has not claimed to be finished), done (the
stored Done verdict, core/label_done.py — never derived from counts). The
frontend renders these verbatim (lib/labelStatus.ts); status is decided here
and nowhere else, so the sidebar counters and the Label page filters cannot
disagree.
"""

from __future__ import annotations

import logging
from pathlib import Path

from yp_video.action.frames import inspect_action_frame_cache
from yp_video.actor import review as actor_review
from yp_video.config import cut_kind_of, find_cut, iter_all_cuts
from yp_video.core import label_done
from yp_video.core.jsonl import read_jsonl_header
from yp_video.core.rallies import RALLY_SOURCES, rally_sources
from yp_video.extraction import links
from yp_video.extraction import pipeline as extraction_pipeline
from yp_video.extraction import store as extraction_store
from yp_video.extraction.prerequisites import prerequisites
from yp_video.reid import store as reid_store
from yp_video.web.action_annotations import annotation_state
from yp_video.web.r2_client import r2_client

log = logging.getLogger(__name__)

#: Pipeline order — also the row order the sidebar renders.
MODES = ("rally", "action", "association", "reid")
STATUSES = ("unlabeled", "pre-annotate", "in-progress", "done")


def _rally_stem(result_name: str) -> str:
    # Strip the conventional "_annotations.jsonl" suffix to get the cut stem.
    return result_name.removesuffix(".jsonl").removesuffix("_annotations")


def _rally_status_of(tags: set[str] | list[str], done: bool) -> str:
    # A manual annotation file means a human started; done is the stored
    # verdict; any machine store alone is pre-annotate.
    if done:
        return "done"
    if "annotation" in tags:
        return "in-progress"
    return "pre-annotate" if tags else "unlabeled"


def rally_status(stem: str) -> str:
    """Rally status for one cut from its local stores — the same ladder
    ``rally_results`` climbs, for listings keyed by cut instead of by file
    (the rally predict pages)."""
    return _rally_status_of(rally_sources(stem), label_done.is_done(stem, "rally"))


def rally_results() -> list[dict]:
    """Rally annotation files across every source store, local and R2."""
    files: dict[str, set[str]] = {}  # name -> set of source tags
    for source in RALLY_SOURCES:
        if source.directory.exists():
            for f in source.directory.glob("*.jsonl"):
                files.setdefault(f.name, set()).add(source.tag)
    # Include R2-only files
    if r2_client.configured:
        try:
            for source in RALLY_SOURCES:
                for obj in r2_client.list_objects_cached(prefix=f"{source.r2_category}/"):
                    files.setdefault(Path(obj["key"]).name, set()).add(source.tag)
        except Exception:  # noqa: BLE001 — R2 down must not take the page down
            log.warning("R2 listing failed; remote annotations will look absent")

    def _kind(stem: str) -> str:
        cut = find_cut(f"{stem}.mp4")
        return cut_kind_of(cut) if cut else "broadcast"

    def _row(name: str, tags: set[str]) -> dict:
        stem = _rally_stem(name)
        done = label_done.is_done(stem, "rally")
        status = _rally_status_of(tags, done)
        return {
            "name": name,
            "source": sorted(tags),
            "kind": _kind(stem),
            "done": done,
            "status": status,
        }

    return sorted(
        [_row(k, v) for k, v in files.items()],
        key=lambda x: x["name"],
    )


def action_videos() -> list[dict]:
    """Every cut and where its action labeling stands."""
    results = []
    for video in sorted(iter_all_cuts(), key=lambda p: p.name):
        state = annotation_state(video.name)
        has_active = state.active is not None or state.active_error is not None
        # -1 marks a file that exists but fails to parse.
        event_count = -1 if state.active_error else len(state.active["events"]) if state.active else 0
        done = label_done.is_done(video.stem, "action")
        # A file in the human store means started (provenance by store); done
        # is the stored verdict, never derived from saving.
        status = ("done" if done
                  else "in-progress" if state.human
                  else "pre-annotate" if has_active
                  else "unlabeled")
        results.append({
            "name": video.name,
            "kind": cut_kind_of(video),
            "rally_sources": rally_sources(video.stem),
            "has_action_annotation": has_active,
            "has_action_pre_annotation": has_active and not state.human,
            "has_action_final_annotation": state.human,
            "done": done,
            "status": status,
            "event_count": event_count,
            "frame_cache": inspect_action_frame_cache(video),
        })
    return results


def association_videos() -> list[dict]:
    """Extracted videos and how much of their actor review is left.

    Action annotations own event membership and labels. Extraction records
    only say which of those events have detector output, and actor labels say
    which current, labelable ids a human reviewed.

    A video missing anything association is built on is left out entirely
    rather than listed as a row with nothing in it: actions own which events
    exist, rallies own which of them are in play (and namespace every tracklet
    an answer can name), and records hold the detections a pick chooses among.
    Producing any of the three is another page's job, so a row here would be a
    dead end — the pipeline chips on those pages are where the gap belongs.
    """
    results = []
    for path in sorted(iter_all_cuts(), key=lambda p: p.name):
        # Cheapest gate first: this walks every cut on every page load, and
        # only a minority have been extracted at all.
        records = extraction_store.records_path(path.stem)
        if not records.exists():
            continue
        pipeline = prerequisites(path.stem)
        if not (pipeline.rally_sources and pipeline.has_action):
            continue
        header = read_jsonl_header(records)
        progress = actor_review.review_progress(
            path.stem, float(header.get("fps") or 0)
        )
        done = label_done.is_done(path.stem, "association")
        results.append(
            {
                "name": path.name,
                "kind": cut_kind_of(path),
                "event_count": progress.event_count,
                "reviewed": progress.reviewed,
                "unreviewed": progress.unreviewed,
                "verdicts": progress.verdicts,
                # The re-pick worklist (links.unresolved_labels): labels no
                # tracklet can be derived for today, whatever their verdict.
                "unresolved": len(links.unresolved_labels(path.stem)),
                # The human "I'm finished" flag — a verdict counts can't
                # derive, same as ReID's (see core/label_done.py).
                "done": done,
                # A listed row exists only once extraction ran, and the auto
                # policy's picks are machine pre-annotation awaiting review.
                "status": ("done" if done
                           else "in-progress" if progress.reviewed > 0
                           else "pre-annotate"),
                # The automatic policy's own outcome, for context on how much
                # of the remainder is likely to just need confirming. These
                # are detector-run diagnostics; unlike progress above they
                # deliberately describe that immutable run.
                "auto_counts": {
                    key: int(header.get(key) or 0)
                    for key in ("ok", "multi", "miss")
                },
                "pipeline": pipeline.payload(),
            }
        )
    return results


def reid_videos() -> list[dict]:
    """Extracted videos and how far their player naming has got."""
    results = []
    for f in sorted(iter_all_cuts(), key=lambda p: p.name):
        events = extraction_pipeline.load_events(f.stem)
        if not events:
            continue
        players = reid_store.load_players(f.stem)
        embedded = reid_store.embedded_models(f.stem)
        player_count = len(
            set(players.tracks.values()) | set(players.assignments.values())
        )
        done = label_done.is_done(f.stem, "reid")
        # Embeddings are the machine's prep work — computed, nobody grouped.
        status = ("done" if done
                  else "in-progress" if player_count > 0
                  else "pre-annotate" if embedded
                  else "unlabeled")
        results.append({
            "name": f.name,
            "kind": cut_kind_of(f),
            "event_count": len(events),
            "embedded_models": embedded,
            "stale_embedding_models": reid_store.stale_embedding_models(f.stem),
            "player_count": player_count,
            "done": done,
            "status": status,
            "pipeline": prerequisites(f.stem).payload(),
        })
    return results


def label_stats() -> dict[str, dict[str, int]]:
    """Per-mode status tally of the union video list — videos, not events.

    The union mirrors the frontend's (lib/useUnionVideos.ts): action lists
    every cut and is the base; association/reid rows always match an action
    row; rally annotations whose stem matches no cut (R2-only) still count.
    """
    by_mode: dict[str, dict[str, str]] = {
        "rally": {_rally_stem(r["name"]): r["status"] for r in rally_results()},
        "action": {Path(r["name"]).stem: r["status"] for r in action_videos()},
        "association": {Path(r["name"]).stem: r["status"] for r in association_videos()},
        "reid": {Path(r["name"]).stem: r["status"] for r in reid_videos()},
    }
    stems = set(by_mode["action"]) | set(by_mode["rally"])
    counts = {mode: dict.fromkeys(STATUSES, 0) for mode in MODES}
    for stem in stems:
        for mode in MODES:
            counts[mode][by_mode[mode].get(stem, "unlabeled")] += 1
    return counts
