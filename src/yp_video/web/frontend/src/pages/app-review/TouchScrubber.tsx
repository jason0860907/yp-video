/** The App's detail progress bar (TouchScrubber.swift): a track over the
 *  clip's window with a dot per touch in its kind's colour — hollow when the
 *  ball was not seen — the kind glyph above each, and the player's name only
 *  on the touch nearest the playhead. Dragging snaps onto a touch within
 *  8 px. `children` hang under the track on the same axis (attack bands). */

import { useRef, useState, type ReactNode } from 'react';
import { displayName, kindOf } from './appTerms';
import type { Touch } from './types';

const SNAP = 8;
const short = (name: string) => (name.length > 5 ? `${name.slice(0, 5)}…` : name);

export function TouchScrubber({
  touches,
  start,
  end,
  time,
  onSeek,
  children,
}: {
  touches: Touch[];
  /** The window, in source-video seconds. */
  start: number;
  end: number;
  /** The playhead, in source-video seconds. */
  time: number;
  onSeek: (time: number) => void;
  children?: ReactNode;
}) {
  const track = useRef<HTMLDivElement>(null);
  const [dragging, setDragging] = useState(false);
  const duration = Math.max(0.1, end - start);
  const frac = (t: number) => Math.min(1, Math.max(0, (t - start) / duration));
  const head = frac(time);
  const nearest = touches.reduce<Touch | null>(
    (best, t) => (!best || Math.abs(t.time - time) < Math.abs(best.time - time) ? t : best),
    null,
  );

  const seekAt = (clientX: number) => {
    const el = track.current;
    if (!el) return;
    const { left, width } = el.getBoundingClientRect();
    const x = clientX - left;
    const near = touches.reduce<Touch | null>(
      (best, t) => (!best || Math.abs(frac(t.time) * width - x) < Math.abs(frac(best.time) * width - x) ? t : best),
      null,
    );
    if (near && Math.abs(frac(near.time) * width - x) <= SNAP) onSeek(near.time);
    else onSeek(start + Math.min(1, Math.max(0, x / Math.max(1, width))) * duration);
  };

  return (
    <div className="select-none space-y-1">
      <div
        ref={track}
        className="relative h-9 cursor-pointer touch-none"
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId);
          setDragging(true);
          seekAt(e.clientX);
        }}
        onPointerMove={(e) => dragging && seekAt(e.clientX)}
        onPointerUp={() => setDragging(false)}
      >
        <div className="absolute inset-x-0 top-[24px] h-1 rounded-full bg-ink/10" />
        <div className="absolute left-0 top-[24px] h-1 rounded-full bg-text-muted/60" style={{ width: `${head * 100}%` }} />
        {touches.map((t) => {
          const k = kindOf(t.kind);
          return (
            <div key={t.event_id} className="absolute top-0" style={{ left: `${frac(t.time) * 100}%` }}>
              <span
                className="absolute -translate-x-1/2 whitespace-nowrap text-[10px] font-bold leading-none"
                style={{ color: k.color, top: 4 }}
              >
                {k.short}
                {t === nearest && t.player && ` ${short(displayName(t.player))}`}
              </span>
              <span
                className="absolute h-2.5 w-2.5 -translate-x-1/2 rounded-full border-2"
                style={{
                  top: 21,
                  borderColor: t.visible ? 'var(--surface-50)' : k.color,
                  background: t.visible ? k.color : 'var(--surface-50)',
                }}
              />
            </div>
          );
        })}
        <span
          className="absolute h-3.5 w-3.5 -translate-x-1/2 rounded-full bg-white shadow"
          style={{ left: `${head * 100}%`, top: 19 }}
        />
      </div>
      {children}
      <div className="flex justify-between font-mono text-[10px] tabular-nums text-text-muted">
        <span>{(head * duration).toFixed(1)}</span>
        <span>{duration.toFixed(1)}s</span>
      </div>
    </div>
  );
}
