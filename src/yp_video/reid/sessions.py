"""Recording sessions, inferred from shared player names.

Identities only have to be consistent WITHIN a session: the product question
is "which of the ~12 players on court did this", not "is this the same person
as three weeks ago". Nothing on disk records which cuts form a session —
filenames are inconsistent enough that prefix parsing is guesswork — so the
grouping is inferred from the labels themselves: two videos belong to the
same session iff they share at least one assigned player name.

That makes the grouping self-correcting. Per-video placeholder names (P1…P12)
leave each video in its own group, which is the honest reading of labels that
were never linked; naming a session's cuts consistently merges them with no
configuration anywhere.

It is deliberately fragile in one direction: a generic name reused across two
real sessions ("7號女") merges them wrongly. ``SessionGroup.shared`` therefore
carries the evidence for every merge so the UI can show which names caused it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from yp_video.config import REID_ANNOTATIONS_DIR
from yp_video.reid.identity import load_assignments


@dataclass(frozen=True)
class SessionGroup:
    """One recording session: the videos sharing a player name-space."""

    id: str
    stems: tuple[str, ...]
    players: tuple[str, ...]
    #: player name -> assigned event count across the whole group
    counts: dict[str, int] = field(default_factory=dict)
    #: merge evidence — player name -> the stems carrying it (only names in >1)
    shared: dict[str, list[str]] = field(default_factory=dict)
    n_assigned: int = 0

    @property
    def is_isolated(self) -> bool:
        """No name links this video to any other — usually a session whose
        cuts were labeled under different naming schemes."""
        return len(self.stems) == 1


def labeled_stems() -> list[str]:
    """Video stems with at least one player assignment, sorted."""
    if not REID_ANNOTATIONS_DIR.exists():
        return []
    suffix = "_players.json"
    stems = [p.name[: -len(suffix)] for p in REID_ANNOTATIONS_DIR.glob(f"*{suffix}")]
    return sorted(s for s in stems if load_assignments(s))


def build_sessions(stems: Sequence[str] | None = None) -> list[SessionGroup]:
    """Group videos by shared player names (union-find).

    Groups come back ordered by descending assignment count, so the
    richest session is ``g0``. That ordering is NOT stable across relabeling —
    exports snapshot their resolved membership rather than trusting the id.
    """
    stems = list(stems) if stems is not None else labeled_stems()
    assignments = {s: load_assignments(s) for s in stems}

    # name -> stems that use it; a name in two stems links them.
    by_name: dict[str, list[str]] = {}
    for stem in stems:
        for name in set(assignments[stem].values()):
            by_name.setdefault(name, []).append(stem)

    parent = {s: s for s in stems}

    def find(s: str) -> str:
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s

    for members in by_name.values():
        for other in members[1:]:
            ra, rb = find(members[0]), find(other)
            if ra != rb:
                parent[rb] = ra

    clusters: dict[str, list[str]] = {}
    for stem in stems:
        clusters.setdefault(find(stem), []).append(stem)

    groups: list[SessionGroup] = []
    for members in clusters.values():
        members.sort()
        counts: dict[str, int] = {}
        for stem in members:
            for name in assignments[stem].values():
                counts[name] = counts.get(name, 0) + 1
        shared = {
            name: sorted(carriers)
            for name, carriers in by_name.items()
            if len(carriers) > 1 and carriers[0] in members
        }
        groups.append(
            SessionGroup(
                id="",  # assigned below, once the ordering is known
                stems=tuple(members),
                players=tuple(sorted(counts)),
                counts=counts,
                shared=dict(sorted(shared.items())),
                n_assigned=sum(counts.values()),
            )
        )

    groups.sort(key=lambda g: (-g.n_assigned, g.stems))
    return [
        SessionGroup(
            id=f"g{i}",
            stems=g.stems,
            players=g.players,
            counts=g.counts,
            shared=g.shared,
            n_assigned=g.n_assigned,
        )
        for i, g in enumerate(groups)
    ]


def group_of(stem: str, groups: Sequence[SessionGroup]) -> SessionGroup | None:
    """The group a stem belongs to, or None when it carries no assignments."""
    return next((g for g in groups if stem in g.stems), None)
