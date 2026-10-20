/** The App's detail progress bar (TouchScrubber.swift): a track over the
 *  clip's window, filled to the playhead, with a dot per touch in its kind's
 *  colour — hollow when the ball was not seen — the kind glyph above each,
 *  and the player's name only on the touch nearest the playhead. The
 *  playhead is a thin bar with its elapsed time under it. Dragging snaps
 *  onto a touch within 8 px. `children` hang under the track on the same
 *  axis (attack bands). */

import { useRef, useState, type ReactNode } from 'react';
import { APP, displayName, kindOf } from './appTerms';
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
    <div className="select-none space-y-1.5">
      <div
        ref={track}
        className="relative mx-2 h-12 cursor-pointer touch-none"
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId);
          setDragging(true);
          seekAt(e.clientX);
        }}
        onPointerMove={(e) => dragging && seekAt(e.clientX)}
        onPointerUp={() => setDragging(false)}
      >
        <div className="absolute inset-x-0 top-[25px] h-[3px] rounded-full bg-ink/10" />
        <div className="absolute left-0 top-[25px] h-[3px] rounded-full" style={{ width: `${head * 100}%`, background: APP.primary }} />
        {touches.map((t) => {
          const k = kindOf(t.kind);
          return (
            <div key={t.event_id} className="absolute top-0" style={{ left: `${frac(t.time) * 100}%` }}>
              <span
                className="absolute -translate-x-1/2 whitespace-nowrap text-[11px] font-bold leading-none"
                style={{ color: k.color, top: 6 }}
              >
                {k.short}
                {t === nearest && t.player && ` ${short(displayName(t.player))}`}
              </span>
              <span
                className="absolute h-[11px] w-[11px] -translate-x-1/2 rounded-full border-2"
                style={{
                  top: 21,
                  borderColor: t.visible ? 'var(--surface-100)' : k.color,
                  background: t.visible ? k.color : 'var(--surface-100)',
                }}
              />
            </div>
          );
        })}
        <div className="absolute top-[17px] -translate-x-1/2" style={{ left: `${head * 100}%` }}>
          <span className="block h-[19px] w-[2px] rounded-full bg-text-primary" />
          <span className="absolute left-1/2 top-[22px] -translate-x-1/2 font-mono text-[9px] tabular-nums text-text-muted">
            {(head * duration).toFixed(1)}s
          </span>
        </div>
      </div>
      {children}
    </div>
  );
}
