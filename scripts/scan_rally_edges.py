"""Audit: rally spans whose edges do not land on the action that defines them.

A rally opens on a `serve` and closes on a `score`. Both audits ask the same
question of opposite ends, so they are one scan with one set of categories:

- ``邊界切偏``   the action exists just outside the span. The annotation is
                there, the boundary stops short of it, and the fix is the
                boundary.
- ``疑似漏標``   the action is nowhere near the span in either direction. The
                fix is labelling one.

Appendices carry the near misses: the action is inside but not at the edge
(a serve with something before it, a score with something after it), the span
holds two of them, or it holds no action at all and no rule can speak to it.

Membership is the rule every reader uses (extraction/store._within):
``start <= frame / fps <= end``, inclusive at both ends.

Read-only. It writes both reports and nothing else, and re-running it after
the annotations change reproduces them — which the first serve audit could
not, having been a throwaway script that never reached the repo.

    uv run python scripts/scan_rally_edges.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date

from yp_video.config import (
    ACTION_ANNOTATIONS_DIR,
    PROJECT_ROOT,
    RALLY_ANNOTATIONS_DIR,
)
from yp_video.contracts.action import LABEL_FILE_SUFFIX
from yp_video.core.jsonl import read_jsonl
from yp_video.core.rallies import annotation_name

#: How close an action outside the span has to be for the boundary — not the
#: labelling — to be the thing that is wrong.
NEAR_S = 3.0
#: How many actions of the sequence to show, from the edge being audited.
WINDOW = 6


@dataclass(frozen=True)
class Edge:
    """One end of a rally and the action that should sit on it."""

    label: str
    #: True for the opening edge (serve at `start`), False for the close.
    opening: bool
    filename: str
    title: str
    #: What a mis-set boundary is called at this end.
    boundary: str
    #: What "inside, but not at the edge" is called.
    displaced: str

    @property
    def edge_word(self) -> str:
        return "開頭" if self.opening else "結尾"

    @property
    def seq_word(self) -> str:
        return "前" if self.opening else "後"


SERVE = Edge(
    label="serve",
    opening=True,
    filename="rally-missing-serve.md",
    title="Rally 開頭沒有 serve 的清單",
    boundary="開頭切太晚",
    displaced="serve 不在最前",
)
SCORE = Edge(
    label="score",
    opening=False,
    filename="rally-missing-score.md",
    title="Rally 結尾沒有 score 的清單",
    boundary="結尾切太早",
    displaced="score 不在最後",
)


def ts(seconds: float) -> str:
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


@dataclass(frozen=True)
class Row:
    stem: str
    rally_id: int
    start: float
    end: float
    #: Action labels at the audited edge, always in time order.
    seq: tuple[str, ...]
    #: Seconds to the nearest matching action outside the span, or None.
    outside: float | None
    #: Distance from the edge to the matching action inside it, or None.
    inside_gap: float | None
    #: How many actions sit between the edge and the matching one inside.
    displaced_by: int
    matches: int
    #: Closest gap between two matching actions inside the span. Under half a
    #: second means one action annotated twice, not two of them.
    match_gap: float | None = None


def scan(edge: Edge) -> tuple[list[Row], list[Row], list[Row], list[Row], int, int]:
    """(unlabelled, boundary, displaced, empty, videos, rallies)."""
    unlabelled: list[Row] = []
    boundary: list[Row] = []
    displaced: list[Row] = []
    empty: list[Row] = []
    extra: list[Row] = []
    videos = seen = 0

    for rally_path in sorted(RALLY_ANNOTATIONS_DIR.glob(annotation_name("*"))):
        stem = rally_path.name[: -len(annotation_name(""))]
        action_path = ACTION_ANNOTATIONS_DIR / f"{stem}{LABEL_FILE_SUFFIX}"
        if not action_path.exists():
            continue
        videos += 1
        meta, events = read_jsonl(action_path)
        fps = float(meta.get("fps") or 30.0) or 30.0
        timeline = sorted(
            (int(e["frame"]) / fps, str(e.get("label") or ""))
            for e in events
            if e.get("frame") is not None
        )
        hits = [t for t, label in timeline if label == edge.label]
        _, rows = read_jsonl(rally_path)

        for row in rows:
            seen += 1
            start, end = float(row["start"]), float(row["end"])
            rid = int(row.get("rally_id") or 0)
            inside = [x for x in timeline if start <= x[0] <= end]
            # Always read the sequence from the edge under audit.
            window = inside[:WINDOW] if edge.opening else inside[-WINDOW:]
            seq = tuple(label for _, label in window)
            at = [t for t, label in inside if label == edge.label]
            n = len(at)
            gap = min((b - a for a, b in zip(at, at[1:])), default=None)

            def make(outside=None, inside_gap=None, displaced_by=0) -> Row:
                return Row(stem, rid, start, end, seq, outside, inside_gap,
                           displaced_by, n, gap)

            if not inside:
                empty.append(make())
                continue
            if n == 0:
                after = [t - end for t in hits if t > end]
                before = [start - t for t in hits if t < start]
                near = [min(x) for x in (after, before) if x]
                nearest = min(near) if near else None
                target = boundary if nearest is not None and nearest <= NEAR_S else unlabelled
                target.append(make(outside=nearest))
                continue
            if n > 1:
                extra.append(make())
            at_edge = inside[0] if edge.opening else inside[-1]
            if at_edge[1] != edge.label:
                picked = at[0] if edge.opening else at[-1]
                gap = picked - start if edge.opening else end - picked
                between = sum(
                    1 for t, _ in inside
                    if (t < picked if edge.opening else t > picked)
                )
                displaced.append(make(inside_gap=round(gap, 1), displaced_by=between))
    return unlabelled, boundary, displaced, empty, extra, videos, seen


def render(edge: Edge, found, videos: int, seen: int) -> str:
    unlabelled, boundary, displaced, empty, extra = found
    missing = len(unlabelled) + len(boundary) + len(empty)
    at = "最前" if edge.opening else "最後"

    out = [
        f"# {edge.title}",
        "",
        f"掃描日期：{date.today():%Y-%m-%d} · 資料：`videos/rally-spot/annotations` × "
        f"`videos/action/annotations`（{videos} 支影片、{seen:,} rallies）",
        "",
        "重跑：`uv run python scripts/scan_rally_edges.py`",
        "",
        f"判定：rally span `[start, end]` 內{at}的動作事件不是 `{edge.label}`。"
        f"共 {missing + len(displaced)} 筆。",
        "",
        f"分類：`{edge.boundary}` = span 外 {NEAR_S:.0f} 秒內就有 `{edge.label}`，"
        f"標註在、是邊界偏了；`{edge.displaced}` = span 內有 `{edge.label}`，"
        f"但{'前' if edge.opening else '後'}面還有別的動作；"
        f"`疑似漏標` = 前後都找不到鄰近的 `{edge.label}`。",
        "",
    ]

    def table(rows: list[Row], gap_col: str | None) -> None:
        head = f"| 影片 | Rally | 起 | 訖 | {at}動作 |"
        sep = "|---|---:|---:|---:|---|"
        if gap_col:
            head += f" {gap_col} |"
            sep += "---:|"
        out.extend([head + f" 動作序列（{edge.seq_word} {WINDOW}） |", sep + "---|"])
        for r in rows:
            edge_action = (r.seq[0] if edge.opening else r.seq[-1]) if r.seq else "—"
            cells = [r.stem, str(r.rally_id), ts(r.start), ts(r.end), edge_action]
            if gap_col:
                value = r.outside if r.inside_gap is None else r.inside_gap
                cells.append(f"{value:.1f}s" if value is not None else "—")
            cells.append(" → ".join(r.seq) if r.seq else "—")
            out.append("| " + " | ".join(cells) + " |")
        out.append("")

    key = lambda r: (r.stem, r.rally_id)  # noqa: E731 — one sort key, used thrice
    if unlabelled:
        out += [f"## 疑似漏標 — {len(unlabelled)} 筆", ""]
        table(sorted(unlabelled, key=key), None)
    if displaced:
        out += [
            f"## {edge.displaced} — {len(displaced)} 筆",
            "",
            f"span 內有 `{edge.label}`，但它不是{at}的事件。",
            "",
        ]
        table(sorted(displaced, key=key), f"{edge.label} 距{edge.edge_word}")
    if boundary:
        out += [
            f"## {edge.boundary} — {len(boundary)} 筆",
            "",
            f"`{edge.label}` 就在 span 外不到 {NEAR_S:.0f} 秒 —— 標註本身在，"
            f"要動的是 `{'start' if edge.opening else 'end'}`。",
            "",
        ]
        table(sorted(boundary, key=key), f"{edge.label} 距{edge.edge_word}")
    if extra:
        out += [
            f"## 附錄 A：span 內有 2 個以上 `{edge.label}` — {len(extra)} 筆",
            "",
            f"間隔不到 0.5 秒的，是同一個 `{edge.label}` 被標了兩次，不是兩個。",
            "",
            f"| 影片 | Rally | 起 | 訖 | {edge.label} 數 | 最小間隔 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for r in sorted(extra, key=key):
            gap = f"{r.match_gap:.2f}s" if r.match_gap is not None else "—"
            out.append(
                f"| {r.stem} | {r.rally_id} | {ts(r.start)} | {ts(r.end)} | "
                f"{r.matches} | {gap} |"
            )
        out.append("")
    if empty:
        out += [
            f"## 附錄 B：span 內完全沒有動作 — {len(empty)} 筆",
            "",
            "沒有事件可以判定，兩份清單裡的是同一批。",
            "",
            "| 影片 | Rally | 起 | 訖 |",
            "|---|---:|---:|---:|",
        ]
        for r in sorted(empty, key=key):
            out.append(f"| {r.stem} | {r.rally_id} | {ts(r.start)} | {ts(r.end)} |")
        out.append("")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    docs = PROJECT_ROOT / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    for edge in (SERVE, SCORE):
        unlabelled, boundary, displaced, empty, extra, videos, seen = scan(edge)
        path = docs / edge.filename
        path.write_text(
            render(edge, (unlabelled, boundary, displaced, empty, extra), videos, seen),
            encoding="utf-8",
        )
        print(f"{edge.label}: {videos} video(s), {seen:,} rallies")
        print(f"  疑似漏標            {len(unlabelled)}")
        print(f"  {edge.displaced:<18s}{len(displaced)}")
        print(f"  {edge.boundary:<18s}{len(boundary)}")
        print(f"  span 內無動作        {len(empty)}")
        print(f"  2 個以上            {len(extra)}")
        print(f"  → {path.relative_to(PROJECT_ROOT)}\n")


if __name__ == "__main__":
    main()
