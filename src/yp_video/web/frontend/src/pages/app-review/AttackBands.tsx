/** A rally's attacks (組織進攻) under its TouchScrubber (AttackBands.swift):
 *  one band per attack across its own touches, the spike colour on its start
 *  edge, lit while the playhead is inside; overlapping attacks stack into
 *  another lane. A click seeks to the attack. */

import { displayName, KIND } from './appTerms';
import type { Attack } from './types';

const LANE = 24;
const GAP = 4;

export function AttackBands({
  attacks,
  start,
  end,
  time,
  onSeek,
}: {
  attacks: Attack[];
  /** The scrubber's window, in source-video seconds. */
  start: number;
  end: number;
  time: number;
  onSeek: (time: number) => void;
}) {
  if (!attacks.length) return null;
  const duration = Math.max(0.1, end - start);
  const frac = (t: number) => Math.min(1, Math.max(0, (t - start) / duration));
  const laneEnds: number[] = [];
  const placed = attacks.map((a, i) => {
    const lo = frac(a.start);
    const hi = Math.max(lo, frac(a.end));
    let lane = laneEnds.findIndex((e) => e <= lo);
    if (lane < 0) lane = laneEnds.push(hi) - 1;
    else laneEnds[lane] = hi;
    return { attack: a, number: i + 1, lo, hi, lane };
  });
  const lanes = Math.max(...placed.map((p) => p.lane)) + 1;
  const spike = KIND.spike.color;

  return (
    <div className="relative mx-2" style={{ height: lanes * LANE + (lanes - 1) * GAP }}>
      {placed.map(({ attack, number, lo, hi, lane }) => {
        const playing = time >= attack.start && time <= attack.end;
        const title = `組織進攻 ${number}`;
        return (
          <button
            key={attack.key}
            type="button"
            title="點一下跳到這段"
            onClick={() => onSeek(attack.start)}
            className="absolute flex items-center overflow-hidden rounded-md px-1.5 text-[10px] font-bold"
            style={{
              // Reaches 8 px past its first and last touch so their dots sit inside.
              left: `calc(${lo * 100}% - 8px)`,
              width: `max(24px, calc(${(hi - lo) * 100}% + 16px))`,
              top: lane * (LANE + GAP),
              height: LANE,
              borderLeft: `3px solid ${spike}`,
              background: playing ? `${spike}47` : 'rgb(var(--fg-rgb) / 0.08)',
              color: playing ? 'var(--text-primary)' : 'var(--text-secondary)',
            }}
          >
            <span className="truncate">{attack.player ? `${title} · ${displayName(attack.player)}` : title}</span>
          </button>
        );
      })}
    </div>
  );
}
