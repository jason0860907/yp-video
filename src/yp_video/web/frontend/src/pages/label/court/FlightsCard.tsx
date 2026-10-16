/** The app's ball-path statistics, run here first (court/flights.ts): one
 *  kind of touch at a time, every count a filter, the filtered touches a
 *  list to step through on the video — the iOS "tap a number, watch those
 *  plays" flow, with the raw numbers and anything implausible flagged. */

import { useEffect, useMemo, useState } from 'react';
import { actionColor } from '@/lib/actionColors';
import { cn } from '@/lib/cn';
import { formatActionTime } from '@/lib/actionEditorModel';
import { Button } from '@/components/ui/Button';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { Segmented } from '@/components/ui/Segmented';
import { touchFlights, ZONES, type Half, type TouchFlight } from './flights';
import type { CourtPositions } from './geometry';

type Kind = 'serve' | 'receive' | 'set' | 'spike';
/** A facet of the kind's counts, e.g. `zone:5`, `pass:poor`, `out`. */
type Facet = string | null;

const KINDS: { key: Kind; label: string }[] = [
  { key: 'serve', label: 'Serve' },
  { key: 'receive', label: 'Receive' },
  { key: 'set', label: 'Set' },
  { key: 'spike', label: 'Spike' },
];

/** A touch is played from a second before it. */
const LEAD_S = 1;

function facetsOf(f: TouchFlight): string[] {
  const out: string[] = [];
  if (f.landing != null) out.push(`zone:${f.landing}`);
  else if (f.route && !f.blocked) out.push('out');
  if (f.blocked) out.push('blocked');
  if (f.direction) out.push(`dir:${f.direction}`);
  if (f.pass) out.push(`pass:${f.pass}`);
  if (f.setZone) out.push(`set:${f.setZone}`);
  return out;
}

/** What the flight came to, in the words its kind is judged by. */
function outcome(f: TouchFlight): string {
  if (f.pass) return f.pass;
  if (f.setZone) return `set ${f.setZone}`;
  if (f.blocked) return 'blocked';
  const where = f.landing != null ? `zone ${f.landing}` : f.route ? 'out' : f.crossed ? 'over' : 'own side';
  return f.direction ? `${f.direction} · ${where}` : where;
}

const n = (v: number | null | undefined, digits: number) => (v == null ? '–' : v.toFixed(digits));

export function FlightsCard({
  positions,
  netHeight,
  time,
  onSeek,
}: {
  positions: CourtPositions | undefined;
  netHeight: number;
  /** The playhead, seconds. */
  time: number;
  onSeek: (seconds: number) => void;
}) {
  const [kind, setKind] = useState<Kind>('serve');
  const [half, setHalf] = useState<Half | 'both'>('both');
  const [facet, setFacet] = useState<Facet>(null);
  const [suspectOnly, setSuspectOnly] = useState(false);

  const flights = useMemo(
    () => (positions ? touchFlights(positions, netHeight) : []),
    [positions, netHeight],
  );
  const ofKind = flights.filter((f) => f.arc.label === kind && (half === 'both' || f.half === half));
  const shown = ofKind.filter(
    (f) => (facet == null || facetsOf(f).includes(facet)) && (!suspectOnly || f.suspect.length > 0),
  );
  const counts = new Map<string, number>();
  for (const f of ofKind) for (const key of facetsOf(f)) counts.set(key, (counts.get(key) ?? 0) + 1);

  // The listed touch the playhead is in: the last one started.
  const currentIndex = shown.filter((f) => f.event.time - LEAD_S <= time + 0.05).length - 1;
  const step = (offset: number) => {
    const target = shown[Math.max(0, Math.min(shown.length - 1, currentIndex + offset))];
    if (target) onSeek(Math.max(0, target.event.time - LEAD_S));
  };
  // [ and ] walk the list, like the app's previous / next clip.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.key === ']') step(1);
      else if (e.key === '[') step(-1);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  const chip = (key: string, label: string) => {
    const count = counts.get(key) ?? 0;
    return (
      <button
        key={key}
        type="button"
        disabled={count === 0}
        onClick={() => setFacet(facet === key ? null : key)}
        className={cn(
          'rounded-md border px-2 py-1 text-[11px] tabular-nums transition-colors disabled:opacity-40',
          facet === key ? 'border-primary bg-primary/20 text-text-primary' : 'border-border text-text-secondary',
        )}
      >
        {label} <span className="font-semibold">{count}</span>
      </button>
    );
  };

  const values = (pick: (f: TouchFlight) => number | null) => {
    const v = shown.map(pick).filter((x): x is number => x != null);
    return v.length ? { mean: v.reduce((a, b) => a + b, 0) / v.length, max: Math.max(...v) } : null;
  };
  const speed = values((f) => f.stats.speed);
  const contact = values((f) => f.stats.contact);
  const clearance = values((f) => f.stats.clearance);

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <Segmented options={KINDS} value={kind} onChange={(k) => { setKind(k); setFacet(null); }} />
        <Segmented
          options={[
            { key: 'both', label: 'Both ends' },
            { key: 'left', label: 'From left' },
            { key: 'right', label: 'From right' },
          ]}
          value={half}
          onChange={setHalf}
        />
      </div>

      {(kind === 'serve' || kind === 'spike') && (
        <div className="space-y-1.5">
          <SectionLabel className="mb-0">Landing · receiver&apos;s zones, net at top</SectionLabel>
          <div className="grid w-fit grid-cols-3 gap-1">
            {ZONES.map((z) => chip(`zone:${z}`, `${z}`))}
          </div>
          <div className="flex flex-wrap gap-1">
            {chip('out', 'Out')}
            {kind === 'spike' && chip('blocked', 'Blocked')}
            {kind === 'spike' && (['line', 'cross', 'tip'] as const).map((d) => chip(`dir:${d}`, d))}
          </div>
        </div>
      )}
      {kind === 'receive' && (
        <div className="flex flex-wrap gap-1">
          {(['perfect', 'playable', 'poor'] as const).map((g) => chip(`pass:${g}`, g))}
        </div>
      )}
      {kind === 'set' && (
        <div className="flex flex-wrap gap-1">
          {(['left', 'middle', 'right'] as const).map((z) => chip(`set:${z}`, z))}
        </div>
      )}

      <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] text-text-secondary">
        <span className="tabular-nums">
          {shown.length} shown · speed {n(speed?.mean, 0)} / {n(speed?.max, 0)} km/h · contact{' '}
          {n(contact?.mean, 2)} / {n(contact?.max, 2)} m
          {clearance ? ` · clearance ${n(clearance.mean, 2)} / ${n(clearance.max, 2)} m` : ''}
        </span>
        <label className="inline-flex items-center gap-1.5">
          <input type="checkbox" checked={suspectOnly} onChange={(e) => setSuspectOnly(e.target.checked)} />
          Suspect only ({ofKind.filter((f) => f.suspect.length > 0).length})
        </label>
      </div>

      <div className="flex items-center gap-2">
        <Button size="sm" onClick={() => step(-1)} disabled={!shown.length}>
          ‹ Prev
        </Button>
        <Button size="sm" onClick={() => step(1)} disabled={!shown.length}>
          Next ›
        </Button>
        <span className="text-[11px] tabular-nums text-text-muted">
          {shown.length ? `${Math.max(currentIndex, 0) + 1} / ${shown.length}` : 'Nothing matches'}
        </span>
      </div>

      <div className="max-h-[24rem] space-y-0.5 overflow-y-auto pr-1">
        <div className="grid grid-cols-[3rem_2.2rem_3rem_2.8rem_2.8rem_minmax(0,1fr)] gap-1 px-1.5 text-[10px] text-text-muted">
          <span>time</span>
          <span>rally</span>
          <span className="text-right">km/h</span>
          <span className="text-right">z m</span>
          <span className="text-right">net m</span>
          <span className="pl-2">outcome</span>
        </div>
        {shown.map((f, i) => (
          <button
            key={f.event.id}
            type="button"
            onClick={() => onSeek(Math.max(0, f.event.time - LEAD_S))}
            title={f.suspect.join(' · ') || undefined}
            className={cn(
              'grid w-full grid-cols-[3rem_2.2rem_3rem_2.8rem_2.8rem_minmax(0,1fr)] items-center gap-1 rounded-md px-1.5 py-1 text-left font-mono text-[11px] tabular-nums',
              i === currentIndex ? 'bg-primary/15' : 'hover:bg-ink/5',
              f.suspect.length > 0 && 'text-amber-400',
            )}
          >
            <span className="flex items-center gap-1">
              <span
                className={cn('h-2 w-2 rounded-full', f.arc.source === 'touches' && 'border')}
                style={
                  f.arc.source === 'ball'
                    ? { background: actionColor(f.arc.label) }
                    : { borderColor: actionColor(f.arc.label) }
                }
                title={f.arc.source === 'ball' ? 'Fitted to the detected ball' : 'Inferred from the touches'}
              />
              {formatActionTime(f.event.time)}
            </span>
            <span>{f.event.rally_id ?? '–'}</span>
            <span className="text-right">{n(f.stats.speed, 0)}</span>
            <span className="text-right">{n(f.stats.contact, 2)}</span>
            <span className="text-right">{n(f.stats.clearance, 2)}</span>
            <span className="truncate pl-2 font-sans">{outcome(f)}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
