"""Audit: which rally spans hold no `score` event, and why.

A rally ends when somebody scores, so the span should close on a `score`. Most
do — this finds the ones that do not and separates the two reasons, because
they are different work:

- ``結尾切太早``  a score sits just past ``end``. The annotation is there; the
                span stops before it, so the fix is the boundary.
- ``疑似漏標``    no score anywhere near the span in either direction. The fix
                is labelling one.

Two appendices carry the mirror-image problems — a span that closes on
something after its score, and a span holding more than one — plus the spans
with no actions at all, which no rule can classify.

Membership is the rule every reader uses (extraction/store._within):
``start <= frame / fps <= end``, inclusive at both ends.

Read-only. It writes the report and nothing else, and re-running it after the
annotations change reproduces the report — which the previous rally audit
could not, having been a throwaway script that never reached the repo.

    uv run python scripts/scan_rally_score.py
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

SCORE = "score"
#: How close a score outside the span has to be for the span — not the
#: labelling — to be the thing that is wrong.
NEAR_S = 3.0
#: How many trailing actions to show; the interesting end of a rally is its end.
TAIL = 6
OUT = PROJECT_ROOT / "docs" / "rally-missing-score.md"


def ts(seconds: float) -> str:
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


@dataclass(frozen=True)
class Finding:
    stem: str
    rally_id: int
    start: float
    end: float
    #: Trailing action labels, oldest first.
    tail: tuple[str, ...]
    #: Seconds from ``end`` to the nearest score outside the span, or None.
    outside: float | None

    @property
    def kind(self) -> str:
        if not self.tail:
            return "無動作"
        if self.outside is not None and self.outside <= NEAR_S:
            return "結尾切太早"
        return "疑似漏標"

    @property
    def sort_key(self) -> tuple:
        return (self.stem, self.rally_id)


def scan() -> tuple[list[Finding], list[tuple], list[tuple], int, int]:
    """(missing, late_score, extra_score, videos, rallies)."""
    missing: list[Finding] = []
    late: list[tuple] = []
    extra: list[tuple] = []
    videos = rallies_seen = 0

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
        scores = [t for t, label in timeline if label == SCORE]
        _, rows = read_jsonl(rally_path)

        for row in rows:
            rallies_seen += 1
            start, end = float(row["start"]), float(row["end"])
            rally_id = int(row.get("rally_id") or 0)
            inside = [x for x in timeline if start <= x[0] <= end]
            n = sum(1 for _, label in inside if label == SCORE)
            tail = tuple(label for _, label in inside[-TAIL:])

            if n == 0:
                after = [t - end for t in scores if t > end]
                before = [start - t for t in scores if t < start]
                nearest = min(
                    [d for d in ([min(after)] if after else []) + ([min(before)] if before else [])],
                    default=None,
                )
                missing.append(Finding(stem, rally_id, start, end, tail, nearest))
                continue
            if n > 1:
                extra.append((stem, rally_id, start, end, n))
            if inside and inside[-1][1] != SCORE:
                last_score = max(t for t, label in inside if label == SCORE)
                late.append((stem, rally_id, start, end, tail,
                             round(end - last_score, 1),
                             sum(1 for t, _ in inside if t > last_score)))
    return missing, late, extra, videos, rallies_seen


def render(missing, late, extra, videos, rallies_seen) -> str:
    by_kind: dict[str, list[Finding]] = {}
    for f in sorted(missing, key=lambda f: f.sort_key):
        by_kind.setdefault(f.kind, []).append(f)
    cut, unlabelled, empty = (
        by_kind.get("結尾切太早", []),
        by_kind.get("疑似漏標", []),
        by_kind.get("無動作", []),
    )

    out = [
        "# Rally 結尾沒有 score 的清單",
        "",
        f"掃描日期：{date.today():%Y-%m-%d} · 資料：`videos/rally-spot/annotations` × "
        f"`videos/action/annotations`（{videos} 支影片、{rallies_seen:,} rallies）",
        "",
        "重跑：`uv run python scripts/scan_rally_score.py`",
        "",
        f"判定：rally span `[start, end]` 內沒有 `score` 動作事件。共 {len(missing)} 筆。",
        "",
        f"分類：`結尾切太早` = span 外 {NEAR_S:.0f} 秒內就有 score，標註在、是邊界停早了；"
        "`疑似漏標` = 前後都找不到鄰近的 score。",
        "",
    ]

    def table(rows: list[Finding], with_gap: bool) -> None:
        head = "| 影片 | Rally | 起 | 訖 | 末個動作 |"
        sep = "|---|---:|---:|---:|---|"
        if with_gap:
            head += " score 距結尾 |"
            sep += "---:|"
        out.extend([head + " 動作序列（後 6） |", sep + "---|"])
        for f in rows:
            cells = [
                f.stem,
                str(f.rally_id),
                ts(f.start),
                ts(f.end),
                f.tail[-1] if f.tail else "—",
            ]
            if with_gap:
                cells.append(f"{f.outside:.1f}s" if f.outside is not None else "—")
            cells.append(" → ".join(f.tail) if f.tail else "—")
            out.append("| " + " | ".join(cells) + " |")
        out.append("")

    if unlabelled:
        out += [f"## 疑似漏標 — {len(unlabelled)} 筆", ""]
        table(unlabelled, with_gap=False)
    if cut:
        out += [f"## 結尾切太早 — {len(cut)} 筆", ""]
        table(cut, with_gap=True)

    if late:
        out += [
            f"## 附錄 A：score 之後還有動作 — {len(late)} 筆",
            "",
            "span 內有 score，但它不是最後一個事件 —— 與「serve 不在最前」對稱。",
            "",
            "| 影片 | Rally | 起 | 訖 | score 距結尾 | score 後動作數 | 動作序列（後 6） |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for stem, rid, start, end, tail, gap, n_after in sorted(late):
            out.append(
                f"| {stem} | {rid} | {ts(start)} | {ts(end)} | {gap}s | {n_after} | "
                f"{' → '.join(tail)} |"
            )
        out.append("")

    if extra:
        out += [
            f"## 附錄 B：span 內有 2 個以上 score — {len(extra)} 筆",
            "",
            "| 影片 | Rally | 起 | 訖 | score 數 |",
            "|---|---:|---:|---:|---:|",
        ]
        for stem, rid, start, end, n in sorted(extra):
            out.append(f"| {stem} | {rid} | {ts(start)} | {ts(end)} | {n} |")
        out.append("")

    if empty:
        out += [
            f"## 附錄 C：span 內完全沒有動作 — {len(empty)} 筆",
            "",
            "沒有事件可以判定，和 serve 那份清單裡的是同一批。",
            "",
            "| 影片 | Rally | 起 | 訖 |",
            "|---|---:|---:|---:|",
        ]
        for f in empty:
            out.append(f"| {f.stem} | {f.rally_id} | {ts(f.start)} | {ts(f.end)} |")
        out.append("")

    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    missing, late, extra, videos, rallies_seen = scan()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(missing, late, extra, videos, rallies_seen), encoding="utf-8")

    kinds: dict[str, int] = {}
    for f in missing:
        kinds[f.kind] = kinds.get(f.kind, 0) + 1
    print(f"{videos} video(s), {rallies_seen:,} rallies")
    print(f"  no score: {len(missing)}")
    for kind, n in sorted(kinds.items(), key=lambda kv: -kv[1]):
        print(f"    {kind}: {n}")
    print(f"  score not last: {len(late)}")
    print(f"  more than one score: {len(extra)}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
