/** The top-down court card: the action at the playhead (or every action),
 *  the tracked players, and the current rally's flights and ball. */

import { useState } from 'react';
import { actionColor } from '@/lib/actionColors';
import { cn } from '@/lib/cn';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { CourtMap, type MapDot } from './CourtMap';
import type { Arc, CourtPositions, CourtState, Point3 } from './geometry';

type Position = CourtPositions['events'][number];

export function MapCard({
  state,
  positions,
  current,
  players,
  arcs,
  ball,
}: {
  state: CourtState;
  positions: Position[];
  current: Position | null;
  players: MapDot[];
  arcs: Arc[];
  ball: Point3 | null;
}) {
  const [scope, setScope] = useState<'current' | 'all'>('current');
  const shown = scope === 'all' ? positions : current ? [current] : [];
  const events: MapDot[] = shown.map((p) => ({
    key: `e${p.id}`,
    at: p.court_xy,
    color: actionColor(p.label),
    kind: 'event',
    title: `${p.label ?? ''} · frame ${p.frame} · (${p.court_xy[0].toFixed(2)}, ${p.court_xy[1].toFixed(2)}) m`,
    selected: p.id === current?.id,
  }));

  return (
    <>
      <div className="mb-2.5 flex items-center justify-between">
        <SectionLabel className="mb-0">Court</SectionLabel>
        <div className="inline-flex items-center gap-2 text-[11px]">
          {current && (
            <span style={{ color: actionColor(current.label) }}>
              {current.label}
              {ball ? ` · ball ${ball[2].toFixed(1)} m` : ''}
            </span>
          )}
          <div className="inline-flex overflow-hidden rounded-md border border-border">
            {(['current', 'all'] as const).map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => setScope(s)}
                className={cn(
                  'px-2 py-0.5',
                  scope === s ? 'bg-primary/20 text-text-primary' : 'text-text-muted',
                )}
              >
                {s === 'current' ? 'Current' : 'All'}
              </button>
            ))}
          </div>
        </div>
      </div>
      <CourtMap state={state} dots={[...events, ...players]} arcs={arcs} ball={ball} />
    </>
  );
}
