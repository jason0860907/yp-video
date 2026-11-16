"""Which set each rally plays in, from the user's 局開頭 / 局結尾 marks.

A port of the App's ``MatchSets.assign``; the shared fixture
(``VolleyIQ/VolleyIQTests/Fixtures/match_sets.json``) keeps the App, the
Worker and this in step.

Rallies run in play order (start, then index). A start opens the next set;
with any start marked, the rallies before the first are warm-up. An end
closes the set at that rally, and the rallies after it wait for the next
start. A mark on a rally that is gone is ignored.
"""

from collections.abc import Iterable


def assign(
    rallies: Iterable[tuple[int, float]], starts: Iterable[int], ends: Iterable[int]
) -> dict[int, int]:
    """Rally index → its set number; a rally that does not count is absent."""
    order = [index for index, _ in sorted(rallies, key=lambda r: (r[1], r[0]))]
    present = set(order)
    starts, ends = set(starts) & present, set(ends) & present
    count = 0 if starts else 1
    current: int | None = None if starts else 1
    numbers: dict[int, int] = {}
    for index in order:
        if index in starts:
            count += 1
            current = count
        if current is not None:
            numbers[index] = current
            if index in ends:
                current = None
    return numbers
