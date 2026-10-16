/** The app's ball-path statistics, run here first (court/flights.ts): one
 *  kind of touch at a time on one top-down court (FlightChart), every count
 *  a filter — the zones and spots on the court itself, the rest as rows —
 *  and the filtered touches listed to step through on the video: the iOS
 *  "tap a number, watch those plays" flow, with the raw numbers and anything
 *  implausible flagged. Attacks are counted the DataVolley way — from where,
 *  how fast a set, which way, how hard, and what came of it — and the
 *  filters combine, one pick per row. The touch being watched is reported up
 *  so the video draws only its flight. */

import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { actionColor } from '@/lib/actionColors';
import { cn } from '@/lib/cn';
import { formatActionTime } from '@/lib/actionEditorModel';
import { Button } from '@/components/ui/Button';
import { Segmented } from '@/components/ui/Segmented';
import { FlightChart, type ZoneCell } from './FlightChart';
import { touchFlights, ZONES, type AttackResult, type Half, type TouchFlight } from './flights';
import type { Arc, CameraView, CourtPositions } from '@/pages/label/court/geometry';

type Kind = 'serve' | 'receive' | 'set' | 'spike';
/** A filter row — one value picked per row, rows combined. */
type Group = 'zone' | 'pass' | 'result' | 'spot' | 'tempo' | 'direction' | 'style';
type Picks = Partial<Record<Group, string>>;

const KINDS: { key: Kind; label: string }[] = [
  { key: 'serve', label: 'Serve' },
  { key: 'receive', label: 'Receive' },
  { key: 'set', label: 'Set' },
  { key: 'spike', label: 'Spike' },
];

/** The attackers' good-to-bad scale, blue to red through a neutral gray —
 *  a diverging scale, readable with red–green colour blindness. The ends
 *  are the lightest, so on the dark court the decisive outcomes stand out
 *  and a plain dig recedes. */
const SCALE = {
  best: '#9fcbff',
  good: '#3f86f5',
  neutral: '#9a9ea6',
  neutralLight: '#d4d7dc',
  bad: '#ef5953',
  worst: '#ffbdb7',
};

/** A result the video doesn't settle: the rally's end wasn't seen. */
const UNKNOWN = 'rgba(255, 255, 255, 0.3)';

/** DataVolley's evaluation, best first, with its symbol. */
const RESULTS: { key: AttackResult; label: string; symbol: string; color: string; hint: string }[] =
  [
    {
      key: 'kill',
      label: 'Kill',
      symbol: '#',
      color: SCALE.best,
      hint: 'Point: down untouched, or the dig could not keep it up',
    },
    { key: 'tough', label: 'Tough', symbol: '+', color: SCALE.good, hint: 'Dug, but off target' },
    {
      key: 'dug',
      label: 'Dug',
      symbol: '−',
      color: SCALE.neutral,
      hint: 'Dug well enough to attack back',
    },
    {
      key: 'covered',
      label: 'Covered',
      symbol: '!',
      color: SCALE.neutralLight,
      hint: 'Blocked and played again by the attackers',
    },
    {
      key: 'blocked',
      label: 'Blocked',
      symbol: '/',
      color: SCALE.bad,
      hint: 'Stuffed: off the block onto the attackers’ court',
    },
    { key: 'error', label: 'Error', symbol: '=', color: SCALE.worst, hint: 'Into the net or out' },
  ];
const RESULT = Object.fromEntries(RESULTS.map((r) => [r.key, r]));
const PASSES: [string, string, string][] = [
  ['perfect', 'Perfect', SCALE.best],
  ['playable', 'Playable', SCALE.neutral],
  ['poor', 'Poor', SCALE.bad],
];
const PASS_COLOR = Object.fromEntries(PASSES.map(([key, , color]) => [key, color]));

/** The teams named the way the rally model names sides: by which court
 *  end — `Half`, the court's own left or right end — is near the camera. */
function teamNames(view: CameraView): Record<Half, string> {
  return view.near === 'left'
    ? { left: 'Near team', right: 'Far team' }
    : { left: 'Far team', right: 'Near team' };
}

/** The attack rows: their values and labels. */
const ATTACK_ROWS: { group: Group; title: string; values: [string, string][] }[] = [
  {
    group: 'spot',
    title: 'From',
    values: [
      ['left', 'Left · 4'],
      ['middle', 'Middle · 3'],
      ['right', 'Right · 2'],
      ['back', 'Back row'],
    ],
  },
  {
    group: 'tempo',
    title: 'Set',
    values: [
      ['quick', 'Quick'],
      ['fast', 'Fast'],
      ['high', 'High'],
    ],
  },
  {
    group: 'direction',
    title: 'Direction',
    values: [
      ['line', 'Line'],
      ['cross', 'Cross'],
      ['sharp', 'Sharp'],
    ],
  },
  {
    group: 'style',
    title: 'Hit',
    values: [
      ['hard', 'Hard'],
      ['soft', 'Soft'],
      ['tip', 'Tip'],
    ],
  },
];
const LABEL = Object.fromEntries(
  ATTACK_ROWS.flatMap((row) =>
    row.values.map(([value, label]) => [`${row.group}:${value}`, label]),
  ),
);

const HEAT = '#f59e0b';
/** The same scale for a share from 0 (bad) to 1 (good): red through the
 *  neutral gray to blue. */
function goodColor(share: number): string {
  const rgb = (hex: string) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
  const [from, to, t] =
    share < 0.5
      ? [SCALE.bad, SCALE.neutral, share * 2]
      : [SCALE.neutral, SCALE.good, share * 2 - 1];
  const a = rgb(from);
  const b = rgb(to);
  return `rgb(${a.map((v, i) => Math.round(v + (b[i]! - v) * t)).join(' ')})`;
}

/** A touch is played from a second before it. */
const LEAD_S = 1;

function keysOf(f: TouchFlight): string[] {
  const out: string[] = [];
  if (f.landing != null) out.push(`zone:${f.landing}`);
  else if (f.route && !f.blocked) out.push('zone:out');
  if (f.pass) out.push(`pass:${f.pass}`);
  if (f.servedZone != null) out.push(`zone:${f.servedZone}`);
  // A set is filed under the attack it led to.
  const a = f.attack ?? f.setFor;
  if (a) {
    if (a.result) out.push(`result:${a.result}`);
    if (a.spot) out.push(`spot:${a.spot}`);
    if (a.tempo) out.push(`tempo:${a.tempo}`);
    out.push(`direction:${a.direction}`, `style:${a.style}`);
  }
  return out;
}

const matches = (keys: string[], picks: Picks, skip?: Group) =>
  Object.entries(picks).every(
    ([group, value]) => group === skip || keys.includes(`${group}:${value}`),
  );

/** What the flight came to, in the words its kind is judged by. */
function outcome(f: TouchFlight): string {
  if (f.pass) return f.pass;
  if (f.setFor) {
    const a = f.setFor;
    return [
      a.spot ? `to ${LABEL[`spot:${a.spot}`]}` : 'to ?',
      a.tempo && `${LABEL[`tempo:${a.tempo}`]} ${a.setTime!.toFixed(2)}s`,
      a.result && `attack: ${RESULT[a.result]!.label}`,
    ]
      .filter(Boolean)
      .join(' · ');
  }
  const where =
    f.landing != null
      ? `Zone ${f.landing}`
      : f.route && !f.blocked
        ? 'off court'
        : f.crossed
          ? 'over'
          : 'own side';
  const a = f.attack;
  if (!a) return where;
  return [
    a.spot && LABEL[`spot:${a.spot}`],
    a.tempo && `${LABEL[`tempo:${a.tempo}`]} ${a.setTime!.toFixed(2)}s`,
    `${LABEL[`direction:${a.direction}`]} ${Math.round(a.angle)}°`,
    LABEL[`style:${a.style}`],
    f.blocked ? 'block' : where,
    a.result && `${RESULT[a.result]!.label}${a.error ? ` (${a.error})` : ''}`,
  ]
    .filter(Boolean)
    .join(' · ');
}

const n = (v: number | null | undefined, digits: number) => (v == null ? '–' : v.toFixed(digits));

/** One row's values as a joined switch, each with its count; picking the
 *  picked one clears it. */
function Choice({
  values,
  picked,
  count,
  onPick,
}: {
  values: [string, string][];
  picked: string | undefined;
  count: (value: string) => number;
  onPick: (value: string) => void;
}) {
  return (
    <div className="inline-flex rounded-lg border border-border bg-surface-50 p-0.5">
      {values.map(([value, label]) => {
        const c = count(value);
        const on = picked === value;
        return (
          <button
            key={value}
            type="button"
            aria-pressed={on}
            disabled={c === 0 && !on}
            onClick={() => onPick(value)}
            className={cn(
              'rounded-md px-2.5 py-1 text-xs font-medium tabular-nums transition-colors disabled:opacity-35',
              on ? 'bg-primary text-on-primary' : 'text-text-secondary hover:bg-ink/[0.04]',
            )}
          >
            {label}
            <span className={cn('ml-1.5 text-[10px]', on ? 'opacity-80' : 'text-text-muted')}>
              {c}
            </span>
          </button>
        );
      })}
    </div>
  );
}

/** The share of each outcome as one bar, with a legend that filters. */
function Breakdown({
  items,
  picked,
  count,
  onPick,
}: {
  items: { key: string; label: string; color: string; title?: string }[];
  picked: string | undefined;
  count: (key: string) => number;
  onPick: (key: string) => void;
}) {
  const total = items.reduce((sum, i) => sum + count(i.key), 0);
  return (
    <div className="space-y-2">
      <div className="flex h-2.5 overflow-hidden rounded-full bg-surface-50">
        {items.map((i) => (
          <div
            key={i.key}
            className="transition-opacity"
            style={{
              width: total ? `${(100 * count(i.key)) / total}%` : 0,
              background: i.color,
              opacity: picked && picked !== i.key ? 0.25 : 1,
            }}
          />
        ))}
      </div>
      <div className="grid grid-cols-2 gap-1">
        {items.map((i) => {
          const c = count(i.key);
          const on = picked === i.key;
          return (
            <button
              key={i.key}
              type="button"
              title={[i.title, total ? `${Math.round((100 * c) / total)}%` : '']
                .filter(Boolean)
                .join(' · ')}
              disabled={c === 0 && !on}
              onClick={() => onPick(i.key)}
              className={cn(
                'flex items-center gap-1.5 rounded-lg px-1.5 py-1.5 text-left text-xs transition-colors disabled:opacity-35',
                on ? 'bg-ink/[0.08] ring-1 ring-inset ring-text-secondary' : 'hover:bg-ink/[0.04]',
              )}
            >
              <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: i.color }} />
              <span className="truncate text-text-secondary">{i.label}</span>
              <span className="ml-auto font-semibold tabular-nums text-text-primary">{c}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function Stat({ label, value, sub }: { label: string; value: ReactNode; sub?: string }) {
  return (
    <div className="rounded-lg bg-surface-50 px-2.5 py-1.5">
      <div className="text-[10px] uppercase tracking-wide text-text-muted">{label}</div>
      <div className="text-sm font-semibold tabular-nums text-text-primary">
        {value}
        {sub && <span className="ml-1 text-[10px] font-normal text-text-muted">{sub}</span>}
      </div>
    </div>
  );
}

export function FlightsCard({
  view,
  positions,
  netHeight,
  time,
  onSeek,
  onFocus,
}: {
  view: CameraView;
  positions: CourtPositions | undefined;
  netHeight: number;
  /** The playhead, seconds. */
  time: number;
  onSeek: (seconds: number) => void;
  /** The listed flight the playhead is in, or null. */
  onFocus: (arc: Arc | null) => void;
}) {
  const names = teamNames(view);
  // Side by side in the order their balls land: the near team's into the
  // top half, then the far team's into the bottom.
  const halves: Half[] = view.near === 'left' ? ['left', 'right'] : ['right', 'left'];
  const [kind, setKind] = useState<Kind>('spike');
  const [picks, setPicks] = useState<Picks>({});
  const [suspectOnly, setSuspectOnly] = useState(false);

  const flights = useMemo(
    () =>
      (positions ? touchFlights(positions, netHeight) : []).map((f) => ({ f, keys: keysOf(f) })),
    [positions, netHeight],
  );
  // Receive means the serve's: a dig is judged as the attack's result.
  const ofKind = flights.filter(({ f }) => f.arc.label === kind && (kind !== 'receive' || f.pass));
  const shown = ofKind
    .filter(({ f, keys }) => matches(keys, picks) && (!suspectOnly || f.suspect.length > 0))
    .map(({ f }) => f);
  /** A value's count among the touches — one team's, or both — the other
   *  rows' picks leave. */
  const count = (group: Group, team?: Half) => (value: string | number) =>
    ofKind.filter(
      ({ f, keys }) =>
        (!team || f.half === team) &&
        matches(keys, picks, group) &&
        keys.includes(`${group}:${value}`),
    ).length;
  const toggle = (group: Group) => (value: string | number) =>
    setPicks(({ [group]: current, ...rest }) =>
      current === `${value}` ? rest : { ...rest, [group]: `${value}` },
    );

  // The listed touch the playhead is in: the last one started.
  const currentIndex = shown.filter((f) => f.event.time - LEAD_S <= time + 0.05).length - 1;
  const current = shown[currentIndex] ?? null;
  const focus = current?.arc ?? null;
  useEffect(() => onFocus(focus), [focus, onFocus]);
  useEffect(() => () => onFocus(null), [onFocus]);
  const play = (f: TouchFlight) => onSeek(Math.max(0, f.event.time - LEAD_S));
  const step = (offset: number) => {
    const target = shown[Math.max(0, Math.min(shown.length - 1, currentIndex + offset))];
    if (target) play(target);
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

  const resultOf = (f: TouchFlight) => (f.attack ?? f.setFor)?.result;
  const colorOf = (f: TouchFlight) =>
    kind === 'spike' || kind === 'set'
      ? resultOf(f)
        ? RESULT[resultOf(f)!]!.color
        : UNKNOWN
      : f.pass
        ? PASS_COLOR[f.pass]!
        : actionColor(f.arc.label);
  const row = (title: string, body: ReactNode) => (
    <div key={title} className="flex items-center gap-3">
      <span className="w-16 shrink-0 text-[10px] font-medium uppercase tracking-wide text-text-muted">
        {title}
      </span>
      {body}
    </div>
  );
  const landed = kind === 'serve' || kind === 'spike';
  const offCourt = count('zone')('out');

  /** Where a team's balls came down: a count per zone, heat by share. */
  const landings = (half: Half) => {
    const counts = new Map(ZONES.map((z) => [z, count('zone', half)(z)]));
    const most = Math.max(1, ...counts.values());
    return (zone: string): ZoneCell | null => {
      const c = counts.get(Number(zone)) ?? 0;
      return c ? { value: `${c}`, color: HEAT, strength: c / most } : null;
    };
  };
  /** How well a team passed serves that came down in each zone of its half:
   *  the share passed perfect or playable, red to green. */
  const passing =
    (half: Half) =>
    (zone: string): ZoneCell | null => {
      const passes = ofKind.filter(
        ({ f, keys }) =>
          f.half === half && matches(keys, picks, 'zone') && keys.includes(`zone:${zone}`),
      );
      if (!passes.length) return null;
      const good = passes.filter(({ f }) => f.pass !== 'poor').length / passes.length;
      return {
        value: `${Math.round(good * 100)}%`,
        sub: [`${passes.length} passed`],
        color: goodColor(good),
        strength: 0.7,
      };
    };

  /** Where a team's setter sent the ball: each spot's share of the sets,
   *  coloured by how well the attacks from there went — DataVolley's
   *  efficiency, red at −50 % to green at +50 %. */
  const distribution = (half: Half) => {
    const sets = ofKind.filter(
      ({ f, keys }) => f.half === half && f.setFor?.spot && matches(keys, picks, 'spot'),
    );
    return (spot: string): ZoneCell | null => {
      const here = sets
        .filter(({ keys }) => keys.includes(`spot:${spot}`))
        .map(({ f }) => f.setFor!);
      if (!here.length) return null;
      const tally = (r: AttackResult) => here.filter((a) => a.result === r).length;
      const efficiency = (tally('kill') - tally('error') - tally('blocked')) / here.length;
      return {
        value: `${Math.round((100 * here.length) / sets.length)}%`,
        sub: [`${here.length} sets`, `eff ${Math.round(efficiency * 100)}%`],
        color: goodColor(Math.min(1, Math.max(0, efficiency + 0.5))),
        strength: 0.7,
      };
    };
  };

  /** One team's summary: its outcomes and its numbers. */
  const team = (half: Half) => {
    const mine = shown.filter((f) => f.half === half);
    const values = (pick: (f: TouchFlight) => number | null) => {
      const v = mine.map(pick).filter((x): x is number => x != null);
      return v.length
        ? { mean: v.reduce((a, b) => a + b, 0) / v.length, max: Math.max(...v) }
        : null;
    };
    const speed = values((f) => f.stats.speed);
    const contact = values((f) => f.stats.contact);
    const clearance = values((f) => f.stats.clearance);
    // DataVolley's attack efficiency: (kills − errors − stuff blocks) / attacks.
    const tally = (r: AttackResult) => mine.filter((f) => resultOf(f) === r).length;
    const efficiency = mine.length
      ? (tally('kill') - tally('error') - tally('blocked')) / mine.length
      : null;
    return (
      <div key={half} className="min-w-0 space-y-2.5 rounded-xl border border-border p-3">
        <div>
          <h4 className="text-sm font-semibold text-text-primary">{names[half]}</h4>
          <div className="text-[11px] tabular-nums text-text-muted">
            {half === view.near ? '↑ from the bottom' : '↓ from the top'} · {mine.length} of{' '}
            {ofKind.filter(({ f }) => f.half === half).length}
          </div>
        </div>
        {(kind === 'spike' || kind === 'set') && (
          <Breakdown
            items={RESULTS.map((r) => ({
              key: r.key,
              label: r.label,
              color: r.color,
              title: `${r.symbol}  ${r.hint}`,
            }))}
            picked={picks.result}
            count={count('result', half)}
            onPick={toggle('result')}
          />
        )}
        {kind === 'receive' && (
          <Breakdown
            items={PASSES.map(([key, label, color]) => ({ key, label, color }))}
            picked={picks.pass}
            count={count('pass', half)}
            onPick={toggle('pass')}
          />
        )}
        {kind === 'set' ? (
          <div className="grid grid-cols-2 gap-1.5">
            <Stat
              label="Attack eff."
              value={efficiency == null ? '–' : `${Math.round(efficiency * 100)}%`}
            />
            <Stat
              label="Set time"
              value={n(values((f) => f.setFor?.setTime ?? null)?.mean, 2)}
              sub="s avg"
            />
            <Stat label="Peak" value={n(values((f) => f.stats.peak)?.mean, 2)} sub="m avg" />
            <Stat label="Contact" value={n(contact?.mean, 2)} sub="m avg" />
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-1.5">
            {kind === 'spike' ? (
              <Stat
                label="Efficiency"
                value={efficiency == null ? '–' : `${Math.round(efficiency * 100)}%`}
              />
            ) : (
              <Stat label="Over net" value={n(clearance?.mean, 2)} sub="m" />
            )}
            <Stat label="Speed" value={n(speed?.mean, 0)} sub={`max ${n(speed?.max, 0)} km/h`} />
            <Stat label="Contact" value={n(contact?.mean, 2)} sub={`max ${n(contact?.max, 2)} m`} />
            {kind === 'spike' && <Stat label="Over net" value={n(clearance?.mean, 2)} sub="m" />}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="space-y-4">
      <Segmented
        options={KINDS}
        value={kind}
        onChange={(k) => {
          setKind(k);
          setPicks({});
        }}
      />

      {kind === 'spike' && (
        <div className="space-y-2">
          {ATTACK_ROWS.map((r) =>
            row(
              r.title,
              <Choice
                values={r.values}
                picked={picks[r.group]}
                count={count(r.group)}
                onPick={toggle(r.group)}
              />,
            ),
          )}
        </div>
      )}
      {kind === 'set' &&
        ATTACK_ROWS.filter((r) => r.group === 'tempo' || r.group === 'spot').map((r) =>
          row(
            r.group === 'spot' ? 'To' : r.title,
            <Choice
              values={r.values}
              picked={picks[r.group]}
              count={count(r.group)}
              onPick={toggle(r.group)}
            />,
          ),
        )}

      {/* Each team's balls on their own court: the left-end team plays from
          the top half, so its balls land in the bottom one. */}
      <div className="grid grid-cols-2 gap-3">
        {halves.map(team)}
        {halves.map((half) => (
          <div key={half} className="min-w-0">
            <FlightChart
              view={view}
              sender={half}
              flights={shown.filter((f) => f.half === half)}
              focus={current}
              colorOf={colorOf}
              names={names}
              zones={
                kind === 'set'
                  ? {
                      regions: 'spots',
                      sender: half,
                      cell: distribution(half),
                      picked: picks.spot ?? null,
                      onPick: toggle('spot'),
                    }
                  : {
                      regions: 'zones',
                      // A pass is played in its own half: where the other team's serve came down.
                      sender: landed ? half : half === 'left' ? 'right' : 'left',
                      cell: landed ? landings(half) : passing(half),
                      picked: picks.zone ?? null,
                      onPick: toggle('zone'),
                    }
              }
              target={kind === 'receive'}
              onPlay={play}
            />
          </div>
        ))}
      </div>

      <div className="flex items-center justify-between gap-2 text-[11px] text-text-muted">
        <span>
          {landed
            ? 'Tap a zone to filter, a line to play it'
            : kind === 'receive'
              ? 'Serve receptions · each zone: share passed perfect or playable from serves landing there · rings 1.5 m and 3 m from the setter’s target'
              : 'Each spot: share of the sets sent there, coloured by the attacks’ efficiency · tap a spot to filter, a line to play it'}
        </span>
        {landed && (
          <button
            type="button"
            disabled={offCourt === 0 && picks.zone !== 'out'}
            onClick={() => toggle('zone')('out')}
            className={cn(
              'shrink-0 rounded-md px-2 py-0.5 tabular-nums transition-colors disabled:opacity-35',
              picks.zone === 'out'
                ? 'bg-primary text-on-primary'
                : 'border border-border text-text-secondary',
            )}
          >
            Off court <span className="font-semibold">{offCourt}</span>
          </button>
        )}
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
        <label className="ml-auto inline-flex items-center gap-1.5 text-[11px] text-text-secondary">
          <input
            type="checkbox"
            checked={suspectOnly}
            onChange={(e) => setSuspectOnly(e.target.checked)}
          />
          Suspect only ({ofKind.filter(({ f }) => f.suspect.length > 0).length})
        </label>
      </div>

      <div className="max-h-[24rem] space-y-0.5 overflow-y-auto pr-1">
        <div className="grid grid-cols-[4.2rem_2.2rem_3rem_2.8rem_2.8rem_minmax(0,1fr)] gap-1 px-1.5 text-[10px] text-text-muted">
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
            onClick={() => play(f)}
            title={[outcome(f), ...f.suspect].join(' · ')}
            className={cn(
              'grid w-full grid-cols-[4.2rem_2.2rem_3rem_2.8rem_2.8rem_minmax(0,1fr)] items-start gap-1 rounded-md px-1.5 py-1 text-left font-mono text-[11px] tabular-nums',
              i === currentIndex ? 'bg-primary/15' : 'hover:bg-ink/5',
              f.suspect.length > 0 && 'text-amber-400',
            )}
          >
            <span className="flex items-center gap-1">
              <span
                className={cn('h-2 w-2 rounded-full', f.arc.source === 'touches' && 'border')}
                style={
                  f.arc.source === 'ball' ? { background: colorOf(f) } : { borderColor: colorOf(f) }
                }
                title={
                  f.arc.source === 'ball'
                    ? 'Fitted to the detected ball'
                    : 'Inferred from the touches'
                }
              />
              {formatActionTime(f.event.time)}
              <span className="text-text-muted" title={names[f.half]}>
                {names[f.half][0]}
              </span>
            </span>
            <span>{f.event.rally_id ?? '–'}</span>
            <span className="text-right">{n(f.stats.speed, 0)}</span>
            <span className="text-right">{n(f.stats.contact, 2)}</span>
            <span className="text-right">{n(f.stats.clearance, 2)}</span>
            <span className="flex min-w-0 items-start gap-1.5 pl-2 font-sans">
              {f.attack?.result && (
                <span
                  className="shrink-0 rounded px-1 font-mono text-[10px] font-semibold text-black"
                  style={{ background: RESULT[f.attack.result]!.color }}
                >
                  {RESULT[f.attack.result]!.symbol}
                </span>
              )}
              <span className="leading-snug">{outcome(f)}</span>
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
