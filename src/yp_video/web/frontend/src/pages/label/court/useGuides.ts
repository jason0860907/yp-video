/** Guide lines: straight lines drawn along the court paint, whose crossings
 *  locate intersections the eye cannot — including ones past the frame edge.
 *  A marking aid, not a label: they live only in this panel's state. */

import { useMemo, useState } from 'react';
import type { PointerEvent as ReactPointerEvent } from 'react';
import { intersect, type Point, type Segment } from './geometry';
import { trackPointer } from './pointer';

/** How close (fraction of frame width) a mark must land to snap onto a
 *  guide crossing. */
const SNAP = 0.015;
/** Shorter drags are clicks, not lines. */
const MIN_LENGTH = 0.01;

const round4 = (v: number) => Math.round(v * 1e4) / 1e4;

/**
 * @param toPoint client coordinates → frame point (null off the frame)
 * @param reach   how far past the frame edge a crossing still counts
 * @param aspect  frame width / height, so the snap radius is round on screen
 */
export function useGuides(
  toPoint: (cx: number, cy: number) => Point | null,
  reach: number,
  aspect: number,
) {
  const [guides, setGuides] = useState<Segment[]>([]);
  const [draft, setDraft] = useState<Segment | null>(null);

  const crossings = useMemo(
    () =>
      guides.flatMap((a, i) =>
        guides.slice(i + 1).flatMap((b) => {
          const p = intersect(a, b, reach);
          return p ? [p] : [];
        }),
      ),
    [guides, reach],
  );

  /** The nearest crossing within the snap radius, else p itself. */
  const snap = (p: Point): Point => {
    let best: Point | null = null;
    let bestD = SNAP;
    for (const c of crossings) {
      const d = Math.hypot(c[0] - p[0], (c[1] - p[1]) / aspect);
      if (d < bestD) [best, bestD] = [c, d];
    }
    return best ? [round4(best[0]), round4(best[1])] : p;
  };

  const draw = (e: ReactPointerEvent) => {
    const start = toPoint(e.clientX, e.clientY);
    if (!start) return;
    let seg: Segment = [start, start];
    setDraft(seg);
    trackPointer(
      e,
      (ev) => {
        const p = toPoint(ev.clientX, ev.clientY);
        if (p) setDraft((seg = [start, p]));
      },
      () => {
        setDraft(null);
        if (Math.hypot(seg[1][0] - seg[0][0], seg[1][1] - seg[0][1]) > MIN_LENGTH)
          setGuides((g) => [...g, seg]);
      },
    );
  };

  /** Nudge one end of a guide; the crossings follow live. */
  const dragEnd = (e: ReactPointerEvent, index: number, end: 0 | 1) => {
    e.preventDefault();
    e.stopPropagation();
    trackPointer(e, (ev) => {
      const p = toPoint(ev.clientX, ev.clientY);
      if (!p) return;
      setGuides((gs) =>
        gs.map((g, i) => (i === index ? ((end === 0 ? [p, g[1]] : [g[0], p]) as Segment) : g)),
      );
    });
  };

  return {
    guides,
    draft,
    crossings,
    snap,
    draw,
    dragEnd,
    remove: (index: number) => setGuides((gs) => gs.filter((_, i) => i !== index)),
    undo: () => setGuides((gs) => gs.slice(0, -1)),
    clear: () => setGuides([]),
  };
}

export type Guides = ReturnType<typeof useGuides>;
