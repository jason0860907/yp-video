#!/usr/bin/env python3
"""Derive rally segments from SPOT action predictions and visualize them.

The action (SPOT) model emits per-frame point events (serve / receive / set /
spike / block / score). A volleyball rally starts on a *serve* and ends on a
*score*. This script turns those points into rally segments with a
serve-anchored heuristic, then renders a self-contained HTML timeline so you
can eyeball the result before wiring it into the TAD predict pipeline.

It reads the existing predictions in ~/videos/action-pre-annotations/*.jsonl
(produced by the action checkpoint) — it does NOT re-run the model and does NOT
touch any rally annotation files.

Usage:
    # First few videos -> /tmp/action_rallies.html
    python scripts/viz_action_rallies.py

    # Specific videos by stem substring, custom output:
    python scripts/viz_action_rallies.py --match 小窩季打 --limit 6 -o /tmp/x.html

    # Tune the heuristic:
    python scripts/viz_action_rallies.py --end-pad 0.5 --max-rally 60 --min-rally 2
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

ACTION_PRE_DIR = Path.home() / "videos" / "action-pre-annotations"
# TAD (ActionFormer) first-pass rally segments live here, one *_annotations.jsonl
# per video (first line is a {"_meta": true, ...} header, rest are rally rows).
RALLY_PRE_DIR = Path.home() / "videos" / "rally-pre-annotations"

# Matches the overlay colors used in action-annotate.js
COLORS = {
    "serve": "#38BDF8",
    "receive": "#22C55E",
    "set": "#A78BFA",
    "spike": "#F97316",
    "block": "#EF4444",
    "score": "#FBBF24",
}


def load_action_file(path: Path) -> tuple[dict, list[dict]]:
    """Return (meta, events) from an action pre-annotation JSONL."""
    meta: dict = {}
    events: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if i == 0 and (obj.get("_meta") or "events" not in obj and "frame" not in obj):
                meta = obj
                continue
            events.append(obj)
    events.sort(key=lambda e: e.get("frame", 0))
    return meta, events


def derive_rallies(
    events: list[dict],
    fps: float,
    *,
    end_pad: float = 0.5,
    max_rally_s: float = 60.0,
    min_rally_s: float = 2.0,
    next_serve_gap_s: float = 0.7,
) -> list[dict]:
    """Serve-anchored rally derivation.

    Each ``serve`` opens a rally. It closes on the first ``score`` that falls
    before the next serve; if there is none, it falls back to ending just
    before the next serve (or a max-length cap for the final serve).

    ``via`` records how each rally ended so the viz can flag fallbacks:
      - "score"      : closed on a real score event (high confidence)
      - "next_serve" : no score found, truncated at the next serve
      - "cap"        : last serve with no score, capped at max_rally_s
    """
    if fps <= 0:
        fps = 30.0
    serves = [e for e in events if e.get("label") == "serve"]
    scores = sorted(e["frame"] for e in events if e.get("label") == "score")

    rallies: list[dict] = []
    for i, s in enumerate(serves):
        fs = int(s["frame"])
        next_serve = int(serves[i + 1]["frame"]) if i + 1 < len(serves) else None

        end_frame = None
        via = None
        for sc in scores:
            if sc > fs and (next_serve is None or sc < next_serve):
                end_frame = sc + int(round(end_pad * fps))
                via = "score"
                break
        if end_frame is None:
            if next_serve is not None:
                end_frame = next_serve - int(round(next_serve_gap_s * fps))
                via = "next_serve"
            else:
                end_frame = fs + int(round(max_rally_s * fps))
                via = "cap"

        # Clamp length.
        end_frame = min(end_frame, fs + int(round(max_rally_s * fps)))
        if end_frame <= fs:
            continue
        start_s = fs / fps
        end_s = end_frame / fps
        if end_s - start_s < min_rally_s:
            continue
        rallies.append({
            "rally_id": len(rallies) + 1,
            "start": round(start_s, 2),
            "end": round(end_s, 2),
            "label": "rally",
            "via": via,
        })
    return rallies


def load_tad_rallies(path: Path) -> list[dict]:
    """Return the TAD first-pass rally segments (seconds) from a rally
    pre-annotation JSONL, skipping the ``_meta`` header line."""
    rallies: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("_meta"):
                continue
            if "start" in obj and "end" in obj:
                rallies.append(obj)
    rallies.sort(key=lambda r: r.get("start", 0.0))
    return rallies


def _event_time(e: dict, fps: float) -> float:
    if "time" in e and e["time"] is not None:
        return float(e["time"])
    return float(e.get("frame", 0)) / fps


def trim_tad_with_actions(
    tad_rallies: list[dict],
    events: list[dict],
    fps: float,
    *,
    serve_pad: float = 1.0,
    score_pad: float = 1.0,
    max_gap_s: float = 3.0,
    min_rally_s: float = 1.0,
) -> list[dict]:
    """Refine TAD first-pass rallies using SPOT action events.

    A volleyball rally truly starts on a *serve* and ends on a *score*. TAD
    tends to over-shoot both ends, so for each TAD segment we look for the
    serve/score that anchors each end and trim to it:

      - take the *first* serve in ``[start - max_gap_s, end]`` and trim the
        start to ``serve - serve_pad`` (1 s of lead-in by default), and
      - take the *last* score in ``[start, end + max_gap_s]`` and trim the end
        to ``score + score_pad`` (1 s of tail by default).

    ``max_gap_s`` lets the anchor sit slightly *outside* the TAD segment: the
    true serve/score is often detected just before/after the (imperfect) TAD
    boundary, so a small window (3 s) recovers it. The cap matters because when
    the event was genuinely not detected, the nearest serve/score belongs to a
    neighbouring rally tens of seconds away — beyond the cap we treat it as
    missing and keep the original TAD boundary. ``max_gap_s=0`` reproduces the
    strict "inside the segment only" behaviour.

    ``via`` records how each end was set so the viz can flag fallbacks:
      - "trimmed"   : both serve and score found within the search window
      - "no_serve"  : no serve found -> original TAD start kept
      - "no_score"  : no score found -> original TAD end kept
      - "none"      : neither found -> TAD segment passed through unchanged
    """
    serves = sorted(_event_time(e, fps) for e in events if e.get("label") == "serve")
    scores = sorted(_event_time(e, fps) for e in events if e.get("label") == "score")

    out: list[dict] = []
    for r in tad_rallies:
        s0, e0 = float(r["start"]), float(r["end"])
        srv = next((t for t in serves if s0 - max_gap_s <= t <= e0), None)
        scr = next((t for t in reversed(scores) if s0 <= t <= e0 + max_gap_s), None)

        if srv is not None and scr is not None:
            via = "trimmed"
        elif srv is None and scr is None:
            via = "none"
        elif srv is None:
            via = "no_serve"
        else:
            via = "no_score"

        start = max(0.0, srv - serve_pad) if srv is not None else s0
        end = scr + score_pad if scr is not None else e0
        if end - start < min_rally_s:
            continue
        out.append({
            "rally_id": len(out) + 1,
            "start": round(start, 2),
            "end": round(end, 2),
            "tad_start": round(s0, 2),
            "tad_end": round(e0, 2),
            "label": "rally",
            "via": via,
        })
    return out


VIA_COLOR = {
    # action-mode (serve-anchored) vias
    "score": "#22C55E", "next_serve": "#F59E0B", "cap": "#EF4444",
    # tad-trim vias
    "trimmed": "#22C55E",   # serve+score both found -> fully refined
    "no_serve": "#F59E0B",  # start fell back to TAD
    "no_score": "#F97316",  # end fell back to TAD
    "none": "#EF4444",      # passed through unchanged
}


def _track_svg(duration: float, events, derived, reference, width=1400,
               tad_underlay=None) -> str:
    """Render one video's timeline as an SVG string.

    ``tad_underlay`` (optional) are the TAD first-pass segments; when given they
    are drawn as a faint gray bar behind the derived row so the trimming done by
    the serve/score events is visible at a glance.
    """
    h_evt, h_der, h_ref, gap = 34, 22, 16, 8
    total_h = h_evt + h_der + h_ref + gap * 4 + 24
    pad_l, pad_r = 8, 8
    inner = width - pad_l - pad_r
    dur = max(duration, 1e-6)

    def x(t):
        return pad_l + inner * (min(max(t, 0), dur) / dur)

    parts = [f'<svg width="{width}" height="{total_h}" '
             f'style="background:#0b0f14;border-radius:8px">']

    # time grid every 30s
    y_top, y_bot = 18, total_h - 6
    t = 0
    while t <= dur:
        xx = x(t)
        parts.append(f'<line x1="{xx:.1f}" y1="{y_top}" x2="{xx:.1f}" y2="{y_bot}" '
                     f'stroke="#1e293b" stroke-width="1"/>')
        parts.append(f'<text x="{xx + 2:.1f}" y="12" fill="#475569" '
                     f'font-size="9" font-family="monospace">{int(t)}s</text>')
        t += 30

    # Row 1: action event ticks
    y1 = y_top + 4
    for e in events:
        lbl = e.get("label", "")
        c = COLORS.get(lbl, "#64748b")
        xx = x(e.get("time", e.get("frame", 0) / 30.0))
        emphasize = lbl in ("serve", "score")
        wpx = 2.5 if emphasize else 1.2
        op = 1.0 if emphasize else 0.5
        parts.append(f'<rect x="{xx - wpx/2:.1f}" y="{y1}" width="{wpx:.1f}" '
                     f'height="{h_evt}" fill="{c}" opacity="{op}"/>')
        if emphasize:
            mark = "S" if lbl == "serve" else "•"
            parts.append(f'<text x="{xx:.1f}" y="{y1 - 1}" fill="{c}" font-size="9" '
                         f'text-anchor="middle" font-family="monospace">{mark}</text>')

    # Row 2: derived rallies (with optional TAD first-pass underlay)
    y2 = y1 + h_evt + gap
    for r in (tad_underlay or []):
        x0, x1 = x(r["start"]), x(r["end"])
        parts.append(f'<rect x="{x0:.1f}" y="{y2 - 2:.1f}" width="{max(x1 - x0,1):.1f}" '
                     f'height="{h_der + 4}" fill="#475569" opacity="0.25" '
                     f'stroke="#64748b" stroke-width="0.8" stroke-dasharray="3 2" rx="2"/>')
    for r in derived:
        x0, x1 = x(r["start"]), x(r["end"])
        c = VIA_COLOR.get(r.get("via"), "#22C55E")
        parts.append(f'<rect x="{x0:.1f}" y="{y2}" width="{max(x1 - x0,1):.1f}" '
                     f'height="{h_der}" fill="{c}" opacity="0.35" stroke="{c}" '
                     f'stroke-width="1" rx="2"/>')

    # Row 3: reference rallies (from existing rally annotations, if any)
    y3 = y2 + h_der + gap
    for r in reference:
        x0, x1 = x(r["start"]), x(r["end"])
        parts.append(f'<rect x="{x0:.1f}" y="{y3}" width="{max(x1 - x0,1):.1f}" '
                     f'height="{h_ref}" fill="#60a5fa" opacity="0.3" stroke="#60a5fa" '
                     f'stroke-width="1" rx="2"/>')

    parts.append('</svg>')
    return "".join(parts)


def render_html(blocks: list[dict], out_path: Path, params: dict):
    legend = " ".join(
        f'<span style="color:{c}">■ {name}</span>' for name, c in COLORS.items()
    )
    via_legend = (
        '<span style="color:#22C55E">■ ended on score</span> '
        '<span style="color:#F59E0B">■ ended on next-serve (no score)</span> '
        '<span style="color:#EF4444">■ capped</span> '
        '<span style="color:#60a5fa">■ reference rally (existing annotation)</span>'
    )
    rows = []
    for b in blocks:
        n_score = sum(1 for r in b["derived"] if r["via"] == "score")
        rows.append(f"""
        <div style="margin:18px 0">
          <div style="font:600 14px sans-serif;color:#e2e8f0;margin-bottom:4px">
            {html.escape(b['title'])}
            <span style="color:#94a3b8;font-weight:400;font-size:12px">
              — {len(b['derived'])} rallies derived ({n_score} on score),
              {len(b['reference'])} reference, {b['n_serve']} serves / {b['n_score']} scores
            </span>
          </div>
          {b['svg']}
          <div style="font:11px monospace;color:#64748b;margin-top:2px">
            row1 = action events &nbsp; row2 = derived rallies &nbsp; row3 = reference
          </div>
        </div>""")

    doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Action -> Rally derivation</title></head>
<body style="background:#020617;color:#cbd5e1;font-family:sans-serif;padding:20px">
<h2 style="color:#f1f5f9">SPOT action → rally segment derivation</h2>
<div style="font-size:12px;color:#94a3b8;margin-bottom:6px">
  heuristic: serve-anchored, end on first score before next serve.
  params: {html.escape(json.dumps(params))}
</div>
<div style="font-size:12px;margin-bottom:4px">events: {legend}</div>
<div style="font-size:12px;margin-bottom:12px">rallies: {via_legend}</div>
{''.join(rows)}
</body></html>"""
    out_path.write_text(doc, encoding="utf-8")


def render_html_trim(blocks: list[dict], out_path: Path, params: dict):
    """Same dark timeline style as ``render_html``, but for the TAD->trim
    experiment: row2 shows the trimmed rallies (with the TAD first-pass drawn
    faintly behind), row3 keeps the reference rallies."""
    legend = " ".join(
        f'<span style="color:{c}">■ {name}</span>' for name, c in COLORS.items()
    )
    via_legend = (
        '<span style="color:#22C55E">■ trimmed (serve+score found)</span> '
        '<span style="color:#F59E0B">■ no serve inside (start = TAD)</span> '
        '<span style="color:#F97316">■ no score inside (end = TAD)</span> '
        '<span style="color:#EF4444">■ passthrough (neither found)</span> '
        '<span style="color:#64748b">▢ TAD first-pass (before trim)</span> '
        '<span style="color:#60a5fa">■ reference rally (existing annotation)</span>'
    )
    rows = []
    for b in blocks:
        n_trim = sum(1 for r in b["derived"] if r["via"] == "trimmed")
        rows.append(f"""
        <div style="margin:18px 0">
          <div style="font:600 14px sans-serif;color:#e2e8f0;margin-bottom:4px">
            {html.escape(b['title'])}
            <span style="color:#94a3b8;font-weight:400;font-size:12px">
              — {len(b['tad'])} TAD → {len(b['derived'])} trimmed ({n_trim} fully refined),
              {len(b['reference'])} reference, {b['n_serve']} serves / {b['n_score']} scores
            </span>
          </div>
          {b['svg']}
          <div style="font:11px monospace;color:#64748b;margin-top:2px">
            row1 = action events &nbsp; row2 = trimmed rallies (TAD underlay) &nbsp; row3 = reference
          </div>
        </div>""")

    doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>TAD -> serve/score trim</title></head>
<body style="background:#020617;color:#cbd5e1;font-family:sans-serif;padding:20px">
<h2 style="color:#f1f5f9">TAD first-pass rally → serve/score trim</h2>
<div style="font-size:12px;color:#94a3b8;margin-bottom:6px">
  heuristic: per TAD segment, start = first serve − serve_pad, end = last score + score_pad.
  params: {html.escape(json.dumps(params))}
</div>
<div style="font-size:12px;margin-bottom:4px">events: {legend}</div>
<div style="font-size:12px;margin-bottom:12px">rallies: {via_legend}</div>
{''.join(rows)}
</body></html>"""
    out_path.write_text(doc, encoding="utf-8")


def _find_chromium() -> str | None:
    """Locate a Chromium/Chrome binary (system or Playwright cache)."""
    import shutil
    for name in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable"):
        p = shutil.which(name)
        if p:
            return p
    cache = Path.home() / ".cache" / "ms-playwright"
    cands = sorted(cache.glob("chromium-*/chrome-linux*/chrome"), reverse=True)
    cands += sorted(cache.glob("chromium_headless_shell-*/chrome-linux*/headless_shell"),
                    reverse=True)
    for c in cands:
        if c.exists():
            return str(c)
    return None


def _crop_trailing_bg(png_path: Path, bg=(2, 6, 23), margin: int = 24):
    """Chromium captures a fixed-height window, leaving a band of page
    background below the content. Crop those trailing rows so the PNG is
    content-fit (matching the original screenshots)."""
    import numpy as np
    import matplotlib.image as mpimg
    img = mpimg.imread(png_path)  # HxWx{3,4} float[0..1]
    rgb = (img[:, :, :3] * 255).round().astype(int)
    bg_arr = np.array(bg)
    # a row is "content" if any pixel differs from the page background
    is_content = (np.abs(rgb - bg_arr).sum(axis=2) > 12).any(axis=1)
    if not is_content.any():
        return
    last = int(np.where(is_content)[0].max())
    cut = min(img.shape[0], last + 1 + margin)
    if cut < img.shape[0]:
        mpimg.imsave(png_path, img[:cut])


def html_to_png(html_path: Path, png_path: Path, *, width: int = 1480, scale: int = 2):
    """Rasterize an HTML file to a content-fit PNG via headless Chromium."""
    import subprocess
    chrome = _find_chromium()
    if not chrome:
        raise RuntimeError("No Chromium/Chrome binary found for HTML->PNG conversion")
    cmd = [
        chrome, "--headless", "--no-sandbox", "--hide-scrollbars",
        "--disable-gpu", "--default-background-color=00000000",
        f"--force-device-scale-factor={scale}",
        f"--window-size={width},4000",
        f"--screenshot={png_path}",
        f"file://{html_path}",
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    _crop_trailing_bg(png_path)
    return chrome


def run_tad_trim_experiment(args):
    """TAD first-pass rally -> serve/score trim, rendered HTML then PNG."""
    stems = sorted(
        p.name[: -len("_annotations.jsonl")]
        for p in RALLY_PRE_DIR.glob("*_annotations.jsonl")
    )
    if args.match:
        stems = [s for s in stems if args.match in s]
    # keep only videos that also have action predictions
    stems = [s for s in stems if (ACTION_PRE_DIR / f"{s}_actions.jsonl").exists()]
    stems = stems[: args.limit]
    if not stems:
        print("No videos with both TAD rally and action predictions"
              + (f" matching {args.match!r}" if args.match else ""))
        return

    params = {"serve_pad": args.serve_pad, "score_pad": args.score_pad,
              "max_gap_s": args.max_gap, "min_rally_s": args.min_rally}

    blocks = []
    for stem in stems:
        tad = load_tad_rallies(RALLY_PRE_DIR / f"{stem}_annotations.jsonl")
        meta, events = load_action_file(ACTION_PRE_DIR / f"{stem}_actions.jsonl")
        fps = float(meta.get("fps") or 30.0)
        num_frames = int(meta.get("num_frames") or 0)
        duration = num_frames / fps if num_frames else max(
            [r["end"] for r in tad] + [e.get("time", 0) for e in events] + [1.0])
        trimmed = trim_tad_with_actions(tad, events, fps, **params)
        reference = meta.get("rallies") or []
        n_serve = sum(1 for e in events if e.get("label") == "serve")
        n_score = sum(1 for e in events if e.get("label") == "score")
        svg = _track_svg(duration, events, trimmed, reference, tad_underlay=tad)
        blocks.append({
            "title": stem, "svg": svg, "tad": tad, "derived": trimmed,
            "reference": reference, "n_serve": n_serve, "n_score": n_score,
        })
        n_trim = sum(1 for r in trimmed if r["via"] == "trimmed")
        print(f"{stem}: {len(tad)} TAD -> {len(trimmed)} trimmed "
              f"({n_trim} fully refined), {n_serve} serves / {n_score} scores")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".png":
        html_path = args.output.with_suffix(".html")
        render_html_trim(blocks, html_path, params)
        chrome = html_to_png(html_path, args.output)
        print(f"\nWrote {html_path}\nWrote {args.output}  (via {Path(chrome).name})")
    else:
        render_html_trim(blocks, args.output, params)
        print(f"\nWrote {args.output}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match", default="", help="only videos whose stem contains this")
    ap.add_argument("--limit", type=int, default=6, help="max videos to render")
    ap.add_argument("-o", "--output", type=Path, default=Path("/tmp/action_rallies.html"))
    ap.add_argument("--mode", choices=["action", "tad-trim"], default="action",
                    help="'action' = serve-anchored from actions only (HTML); "
                         "'tad-trim' = TAD first-pass rallies trimmed by serve/score (PNG)")
    ap.add_argument("--end-pad", type=float, default=0.5)
    ap.add_argument("--max-rally", type=float, default=60.0)
    ap.add_argument("--min-rally", type=float, default=2.0)
    ap.add_argument("--next-serve-gap", type=float, default=0.7)
    ap.add_argument("--serve-pad", type=float, default=1.0,
                    help="[tad-trim] seconds of lead-in kept before the serve")
    ap.add_argument("--score-pad", type=float, default=1.0,
                    help="[tad-trim] seconds of tail kept after the score")
    ap.add_argument("--max-gap", type=float, default=3.0,
                    help="[tad-trim] how far outside the TAD segment a serve/score "
                         "anchor may sit; beyond this it's treated as not detected")
    args = ap.parse_args()

    if args.mode == "tad-trim":
        run_tad_trim_experiment(args)
        return

    files = sorted(ACTION_PRE_DIR.glob("*_actions.jsonl"))
    if args.match:
        files = [f for f in files if args.match in f.stem]
    files = files[: args.limit]
    if not files:
        print(f"No action pre-annotation files found in {ACTION_PRE_DIR}"
              + (f" matching {args.match!r}" if args.match else ""))
        return

    params = {
        "end_pad": args.end_pad, "max_rally_s": args.max_rally,
        "min_rally_s": args.min_rally, "next_serve_gap_s": args.next_serve_gap,
    }

    blocks = []
    for path in files:
        meta, events = load_action_file(path)
        fps = float(meta.get("fps") or 30.0)
        num_frames = int(meta.get("num_frames") or 0)
        duration = num_frames / fps if num_frames else (
            max((e.get("frame", 0) for e in events), default=0) / fps)
        derived = derive_rallies(events, fps, **params)
        reference = meta.get("rallies") or []
        n_serve = sum(1 for e in events if e.get("label") == "serve")
        n_score = sum(1 for e in events if e.get("label") == "score")
        svg = _track_svg(duration, events, derived, reference)
        blocks.append({
            "title": meta.get("video") or path.stem,
            "svg": svg, "derived": derived, "reference": reference,
            "n_serve": n_serve, "n_score": n_score,
        })
        print(f"{path.stem}: {n_serve} serves, {n_score} scores -> "
              f"{len(derived)} rallies ({sum(1 for r in derived if r['via']=='score')} on score)")

    render_html(blocks, args.output, params)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
