"""Set human rally spans to first serve - 1s / last score + 1s.

Missing anchors leave that edge unchanged. Clamp to the video duration.
Preserve every annotation field and action; report ambiguous/missing events.
No track fingerprints are re-stamped: moved spans must be checked/retracked.

    uv run python scripts/snap_rally_lead_in.py
    uv run python scripts/snap_rally_lead_in.py --apply
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scan_rally_edges import SCORE_VERDICTS, SERVE_VERDICTS  # noqa: E402

from yp_video.config import (  # noqa: E402
    ACTION_ANNOTATIONS_DIR,
    PROJECT_ROOT,
    RALLY_ANNOTATIONS_DIR,
)
from yp_video.contracts.action import LABEL_FILE_SUFFIX  # noqa: E402
from yp_video.core.jsonl import read_jsonl, write_jsonl  # noqa: E402
from yp_video.core.rallies import annotation_name, resolve_rally_ids  # noqa: E402

PADDING_S = 1.0


def inside_events(row: dict, events: list[tuple[float, str]]) -> list[tuple[float, str]]:
    return [e for e in events if row["start"] <= e[0] <= row["end"]]


def snap(row: dict, events: list[tuple[float, str]], duration: float) -> dict:
    """Use the original span for both anchors; never mutate the input row."""
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("A positive video duration is required")
    if not 0 <= row["start"] < row["end"] or row["end"] > duration:
        # Old spans can extend beyond the video; their event membership still
        # defines the anchors. Invalid/empty order cannot define a rally.
        if not all(math.isfinite(row[e]) for e in ("start", "end")) or row["start"] >= row["end"]:
            raise ValueError(f"Invalid rally span: {row}")
    inside = inside_events(row, events)
    serves = [t for t, label in inside if label == "serve"]
    scores = [t for t, label in inside if label == "score"]
    new = dict(row)
    if serves:
        new["start"] = round(min(serves) - PADDING_S, 3)
    if scores:
        new["end"] = round(max(scores) + PADDING_S, 3)
    new["start"] = max(0.0, new["start"])
    new["end"] = min(duration, new["end"])
    if new["start"] >= new["end"]:
        raise ValueError(f"Anchors produce an empty/reversed span: {row}")
    # Every chosen anchor must remain in the span, including pathological
    # annotations where the serve is after the score.
    anchors = ([min(serves)] if serves else []) + ([max(scores)] if scores else [])
    if any(not new["start"] <= t <= new["end"] for t in anchors):
        raise ValueError(f"Conflicting or out-of-video anchors: {row}")
    return new


def problems(stem: str, row: dict, events: list[tuple[float, str]], duration: float) -> dict[str, str]:
    inside = inside_events(row, events)
    found = {}
    for label, opening, verdicts in (
        ("serve", True, SERVE_VERDICTS), ("score", False, SCORE_VERDICTS),
    ):
        hits = [t for t, action in inside if action == label]
        verdict = verdicts.get((stem, row["rally_id"]))
        note = f"；既有人工紀錄：{verdict}（本次未重看影片）" if verdict else ""
        if not hits:
            found[f"缺少 {label}"] = f"span 內沒有 {label}" + note
        elif len(hits) > 1:
            found[f"多個 {label}"] = f"{len(hits)} 個：" + ", ".join(f"{t:.3f}s" for t in hits)
        if hits and inside and (inside[0] if opening else inside[-1])[1] != label:
            found[f"{label} 不在{'最前' if opening else '最後'}"] = (
                "事件序列：" + ", ".join(f"{action}@{t:.3f}s" for t, action in inside) + note
            )
    if not inside:
        found["完全沒有動作"] = "span 內沒有任何 action 事件"
    if row["start"] < 0:
        found["start 小於 0"] = f"start={row['start']:.3f}s"
    if row["end"] > duration:
        found["end 超出影片"] = f"end={row['end']:.3f}s；metadata duration={duration:.3f}s"
    return found


def write_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def build_plans():
    plans, issues, missing, changes = [], [], [], []
    counts = Counter()
    for path in sorted(RALLY_ANNOTATIONS_DIR.glob(annotation_name("*"))):
        stem = path.name[:-len(annotation_name(""))]
        meta, rows = read_jsonl(path)
        resolve_rally_ids(rows)
        counts["all_videos"] += 1
        counts["all_rallies"] += len(rows)
        action_path = ACTION_ANNOTATIONS_DIR / f"{stem}{LABEL_FILE_SUFFIX}"
        if not action_path.exists():
            missing.append(dict(id=f"U{len(missing)+1:04}", video=stem, rallies=len(rows),
                                reason="無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改"))
            continue
        action_meta, actions = read_jsonl(action_path)
        fps = float(action_meta["fps"])
        duration = float(meta["duration"])
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid fps: {stem}")
        events = sorted((int(e["frame"]) / fps, str(e["label"])) for e in actions)
        counts["paired_videos"] += 1
        counts["paired_rallies"] += len(rows)
        updated = []
        for row in rows:
            new = snap(row, events, duration)
            updated.append(new)
            before = problems(stem, row, events, duration)
            after = problems(stem, new, events, duration)
            old_events = inside_events(row, events)
            new_events = inside_events(new, events)
            excluded = [e for e in old_events if not new["start"] <= e[0] <= new["end"]]
            admitted = [e for e in new_events if not row["start"] <= e[0] <= row["end"]]
            for kind in sorted(before.keys() | after.keys()):
                status = "調整後仍存在" if kind in before and kind in after else (
                    "調整後出現" if kind in after else "調整後未出現；原始問題保留供複查")
                issues.append(dict(video=stem, rally_id=row["rally_id"], start=new["start"], end=new["end"],
                                   kind=kind, status=status, detail=after.get(kind, before.get(kind))))
            for kind, group in (("調整後移出 span 的動作", excluded), ("調整後納入 span 的動作", admitted)):
                if group:
                    issues.append(dict(video=stem, rally_id=row["rally_id"], start=new["start"], end=new["end"],
                                       kind=kind, status="需複查；action 標註保留",
                                       detail=", ".join(f"{label}@{t:.3f}s" for t, label in group)))
            clamped = []
            for edge, label, opening in (("start", "serve", True), ("end", "score", False)):
                hits = [t for t, action in old_events if action == label]
                if not hits:
                    counts[f"missing_{label}"] += 1
                    continue
                anchor = min(hits) if opening else max(hits)
                target = anchor - 1 if opening else anchor + 1
                if target < 0 or target > duration:
                    clamped.append(edge)
                    counts[f"clamped_{edge}"] += 1
                else:
                    assert abs(new[edge] - target) <= 0.000501
                    counts[f"exact_{edge}"] += 1
            if clamped:
                issues.append(dict(video=stem, rally_id=row["rally_id"], start=new["start"], end=new["end"],
                                   kind="影片頭尾不足 1 秒", status="已截在影片範圍",
                                   detail=f"{', '.join(clamped)} 無法留滿 1 秒；duration={duration:.3f}s"))
            if new != row:
                counts["changed_rallies"] += 1
                counts["changed_edges"] += sum(new[e] != row[e] for e in ("start", "end"))
                changes.append(dict(video=stem, rally_id=row["rally_id"], old_start=row["start"],
                                    old_end=row["end"], new_start=new["start"], new_end=new["end"]))
        ordered = sorted(updated, key=lambda r: r["start"])
        for a, b in zip(ordered, ordered[1:]):
            if b["start"] < a["end"]:
                issues.append(dict(video=stem, rally_id=b["rally_id"], start=b["start"], end=b["end"],
                                   kind="相鄰 rally 重疊", status="需複查",
                                   detail=f"與 r{a['rally_id']} 重疊 {a['end']-b['start']:.3f}s"))
        # Plan validation: only boundaries may change, IDs/metadata/events survive.
        assert [{k: v for k, v in r.items() if k not in ("start", "end")} for r in rows] == [
            {k: v for k, v in r.items() if k not in ("start", "end")} for r in updated]
        # A second application must not silently change anchors again.
        assert updated == [snap(r, events, duration) for r in updated], f"Non-idempotent: {stem}"
        plans.append((path, meta, rows, updated, hashlib.sha256(path.read_bytes()).hexdigest()))
    for i, issue in enumerate(issues, 1):
        issue["id"] = f"Q{i:04}"
    counts["changed_videos"] = sum(old != new for _, _, old, new, _ in plans)
    return plans, issues, missing, changes, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write annotations and sync configured R2")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "docs" / "rally-one-second")
    args = parser.parse_args()
    plans, issues, missing, changes, counts = build_plans()
    output = args.report_dir
    output.mkdir(parents=True, exist_ok=True)
    changed = [p for p in plans if p[2] != p[3]]
    sync = "未執行（dry run）"
    if args.apply:
        backup = RALLY_ANNOTATIONS_DIR.with_name(
            "annotations.before-one-second-" + datetime.now().strftime("%Y%m%dT%H%M%S")
        )
        backup.mkdir()
        for path, _, _, _, digest in plans:
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Annotations changed during planning: {path}")
        for path, meta, _, new, _ in changed:
            shutil.copy2(path, backup / path.name)
            write_jsonl(path, meta, new)
            assert read_jsonl(path) == (meta, new)
        from yp_video.web.r2_client import r2_client
        if r2_client.configured:
            failures = []
            for i, (path, _, _, _, _) in enumerate(changed, 1):
                try:
                    r2_client.upload_file(path, f"rally-spot/annotations/{path.name}")
                except Exception as exc:
                    failures.append(dict(file=path.name, error=type(exc).__name__))
                if i % 20 == 0:
                    print(f"R2 sync {i}/{len(changed)}", flush=True)
            sync = f"{len(changed)-len(failures)}/{len(changed)} 檔案成功"
            (output / "sync-failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2))
        else:
            sync = "R2 未設定；僅更新本機"
    counts["issues"] = len(issues)
    counts["issue_rallies"] = len({(r["video"], r["rally_id"]) for r in issues})
    write_csv(output / "issues.csv", issues)
    write_csv(output / "unverified-videos.csv", missing)
    write_csv(output / "changes.csv", changes)
    summary = dict(counts=counts, sync=sync, applied=args.apply,
                   issue_types=dict(Counter(r["kind"] for r in issues)),
                   remaining_issue_types=dict(Counter(r["kind"] for r in issues
                       if r["status"] in ("調整後仍存在", "調整後出現"))))
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    lines = ["# Rally 前後各 1 秒：調整結果與編號問題清單", "",
             f"產生時間：{datetime.now():%Y-%m-%d %H:%M:%S}；狀態：{'已套用' if args.apply else '預覽，尚未修改'}。", "",
             "## 規則與範圍", "",
             "以原 span 內第一個 serve − 1 秒、最後一個 score + 1 秒設定起訖（毫秒精度）。"
             "缺少某端動作時保留該端原值；邊界限制在 [0, metadata duration]。"
             "所有動作標註、rally ID、winner 與其他欄位均保留。多個動作的問題仍須人工複查。", "",
             f"全部 {counts['all_videos']} 支 / {counts['all_rallies']:,} rallies；"
             f"有人工 action 標註 {counts['paired_videos']} 支 / {counts['paired_rallies']:,} rallies。"
             f"{'已修改' if args.apply else '預計修改'} {counts['changed_videos']} 支 / {counts['changed_rallies']:,} rallies / {counts['changed_edges']:,} 個邊界。", "",
             f"可留滿 1 秒：start {counts['exact_start']:,}、end {counts['exact_end']:,}；"
             f"受影片頭尾限制：start {counts['clamped_start']}、end {counts['clamped_end']}；"
             f"缺少 serve {counts['missing_serve']}、缺少 score {counts['missing_score']}。", "",
             f"R2 同步：{sync}。調整過的 rally fingerprint 會改變，未偽造 tracking 新鮮度；"
             "既有 tracking 可能需要重新執行。", "",
             "本次為標註資料檢查，未重新觀看影片；既有人工判定只作參考。"
             "問題按影片/rally/類型固定排序編號；同一 rally 可有多個問題。"
             "保留調整前與調整後問題的聯集，避免縮短範圍後把問題隱藏。", "",
             "## 問題統計", "", "| 問題 | 筆數 |", "|---|---:|"]
    lines.extend(f"| {kind} | {n} |" for kind, n in Counter(r["kind"] for r in issues).most_common())
    lines += ["", "## Rally 問題清單", "", "時間以影片內秒數表示。", "",
              "| 編號 | 影片 | Rally | 調整後起訖 | 問題 | 狀態 | 詳細資料 |", "|---|---|---:|---|---|---|---|"]
    for r in issues:
        cells = [r["id"], r["video"], str(r["rally_id"]), f"{r['start']:.3f}–{r['end']:.3f}",
                 r["kind"], r["status"], r["detail"]]
        lines.append("| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in cells) + " |")
    lines += ["", "## 無 action 標註：未調整的影片", "",
              f"共 {len(missing)} 支 / {sum(r['rallies'] for r in missing):,} rallies。"
              "這些影片無法得知正確 serve/score 時間，不能宣稱已改為前後 1 秒。", "",
              "| 編號 | 影片 | Rally 數 | 處理結果 |", "|---|---|---:|---|"]
    lines.extend(f"| {r['id']} | {r['video'].replace('|', chr(92)+'|')} | {r['rallies']} | {r['reason']} |" for r in missing)
    lines += ["", "## 附件", "", "- [編號問題 CSV](issues.csv)",
              "- [未驗證影片 CSV](unverified-videos.csv)", "- [逐筆邊界修改 CSV](changes.csv)",
              "- [摘要 JSON](summary.json)", "", "重跑：`uv run python scripts/snap_rally_lead_in.py`（預覽）。"]
    (output / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Report: {output / 'README.md'}")


if __name__ == "__main__":
    main()
