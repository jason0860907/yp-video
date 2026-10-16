/** One team's flights on the whole court top-down, as the camera faces it:
 *  the near team's half at the bottom, the far team's on top, the net
 *  across the middle, left and right as in the frame. One half's nine zones
 *  carry a number each — where balls came down, or how well they were
 *  passed there — and a zone is a filter. A pass draws the team's target. */

import type { ReactNode } from 'react';
import { COURT } from '@/lib/actionColors';
import { cn } from '@/lib/cn';
import { zoneOf, type Half, type TouchFlight } from './flights';
import type { CameraView } from '@/pages/label/court/geometry';

const W = 9;
const NET = 9;
/** Free zone drawn past the end lines and the sidelines, metres. */
const END = 11;
const SIDE = 1.5;
/** Where a setter wants the pass (flights.ts: a metre off the net, 6 m
 *  from the passer's left), in court metres per half, and its grade rings. */
const TARGETS: Record<Half, [number, number]> = { left: [8, 3], right: [10, 6] };
const RINGS = [1.5, 3];

/** Court metres to the drawing: the width across, the length down, the
 *  near end at the bottom. */
const drawing =
  ({ near, yRight }: CameraView) =>
  ([x, y]: readonly number[]) => ({
    x: yRight ? y! : W - y!,
    y: near === 'right' ? x! - NET : NET - x!,
  });

type Drawing = ReturnType<typeof drawing>;
interface Region {
  key: string;
  label: string;
  x: number;
  y: number;
  w: number;
  h: number;
}

/** A court rectangle, corners in court metres, as drawn. */
function region(
  key: string,
  label: string,
  at: Drawing,
  p: [number, number],
  q: [number, number],
): Region {
  const a = at(p);
  const b = at(q);
  return {
    key,
    label,
    x: Math.min(a.x, b.x),
    y: Math.min(a.y, b.y),
    w: Math.abs(b.x - a.x),
    h: Math.abs(b.y - a.y),
  };
}

/** The zones of the half `sender`'s balls land in: each cell's centre
 *  names the zone the way flights.ts lands the ball. */
function zoneRegions(sender: Half, at: Drawing): Region[] {
  return [0, 1, 2].flatMap((i) =>
    [0, 1, 2].map((j) => {
      const x = sender === 'left' ? NET + i * 3 : i * 3;
      const y = j * 3;
      const zone =
        sender === 'left'
          ? zoneOf(y + 1.5, x + 1.5 - NET)!
          : zoneOf(W - (y + 1.5), NET - (x + 1.5))!;
      return region(`${zone}`, `${zone}`, at, [x, y], [x + 3, y + 3]);
    }),
  );
}

/** The attack spots on `sender`'s own half (flights.ts Spot): the front
 *  row's three columns as the team faces the net, and the back row. */
function spotRegions(sender: Half, at: Drawing): Region[] {
  // Sender coordinates (lateral from their left, along from the net) to court.
  const court = (lateral: number, along: number): [number, number] =>
    sender === 'left' ? [NET + along, W - lateral] : [NET - along, lateral];
  return [
    region('left', '4', at, court(0, 0), court(3, -3)),
    region('middle', '3', at, court(3, 0), court(6, -3)),
    region('right', '2', at, court(6, 0), court(9, -3)),
    region('back', 'Back row', at, court(0, -3), court(9, -9)),
  ];
}

/** Text laid out at 20× and scaled down: glyphs a fraction of a unit tall
 *  get hinted to whole pixels first, and pile up. */
const TEXT_SCALE = 20;
function Label({
  x,
  y,
  size,
  opacity,
  bold = false,
  halo = false,
  anchor = 'middle',
  children,
}: {
  x: number;
  y: number;
  size: number;
  opacity: number;
  bold?: boolean;
  /** A dark outline, for text drawn over lines. */
  halo?: boolean;
  anchor?: 'start' | 'middle' | 'end';
  children: ReactNode;
}) {
  return (
    <text
      transform={`translate(${x} ${y}) scale(${1 / TEXT_SCALE})`}
      fontSize={size * TEXT_SCALE}
      fontWeight={bold ? 600 : undefined}
      textAnchor={anchor}
      fill="white"
      fillOpacity={opacity}
      stroke={halo ? 'rgba(0,0,0,0.55)' : undefined}
      strokeWidth={halo ? size * TEXT_SCALE * 0.18 : undefined}
      paintOrder="stroke"
      strokeLinejoin="round"
      pointerEvents="none"
    >
      {children}
    </text>
  );
}

export interface ZoneCell {
  value: string;
  /** Small lines under the value. */
  sub?: string[];
  color: string;
  /** How strongly to fill, 0–1. */
  strength: number;
}

export interface Zones {
  /** The landing zones of the half `sender`'s balls come down in, or the
   *  attack spots on `sender`'s own half. */
  regions: 'zones' | 'spots';
  sender: Half;
  /** What a region shows, by zone number or spot; null when it's empty. */
  cell: (key: string) => ZoneCell | null;
  picked: string | null;
  onPick: (key: string) => void;
}

export function FlightChart({
  view,
  sender,
  flights,
  focus,
  colorOf,
  names,
  zones,
  target: showTarget,
  onPlay,
}: {
  view: CameraView;
  /** The team whose flights these are. */
  sender: Half;
  flights: TouchFlight[];
  focus: TouchFlight | null;
  colorOf: (f: TouchFlight) => string;
  /** Each team's name, drawn at its end. */
  names: Record<Half, string>;
  zones?: Zones;
  /** Draw the team's pass target and its grade rings. */
  target?: boolean;
  onPlay: (f: TouchFlight) => void;
}) {
  const at = drawing(view);
  const cells = !zones
    ? []
    : (zones.regions === 'zones' ? zoneRegions : spotRegions)(zones.sender, at);
  const target = at(TARGETS[sender]);
  const far: Half = view.near === 'left' ? 'right' : 'left';
  // The focus may be the other team's flight: then nothing here is dimmed.
  const mine = focus && flights.includes(focus) ? focus : null;
  const ordered = mine ? [...flights.filter((f) => f !== mine), mine] : flights;
  return (
    <svg
      viewBox={`${-SIDE} ${-END} ${W + 2 * SIDE} ${2 * END}`}
      className="w-full select-none rounded-xl"
      style={{ background: COURT.freeZone }}
    >
      <rect x={0} y={-9} width={W} height={18} fill={COURT.surface} />

      {zones &&
        cells.map(({ key, x, y, w, h }) => {
          const c = zones.cell(key);
          const picked = zones.picked === key;
          return (
            <g
              key={key}
              className={cn(c && 'cursor-pointer')}
              onClick={() => c && zones.onPick(key)}
            >
              <rect
                x={x}
                y={y}
                width={w}
                height={h}
                fill={c?.color ?? 'none'}
                fillOpacity={c ? 0.1 + 0.55 * c.strength : 0}
                stroke={picked ? 'white' : 'none'}
                strokeWidth={0.12}
              />
            </g>
          );
        })}

      {/* Lines: the outline, the attack lines, the zone grid, the net. */}
      <g stroke="white" fill="none" pointerEvents="none">
        <rect x={0} y={-9} width={W} height={18} strokeWidth={0.08} strokeOpacity={0.85} />
        <line x1={0} y1={-3} x2={W} y2={-3} strokeWidth={0.06} strokeOpacity={0.7} />
        <line x1={0} y1={3} x2={W} y2={3} strokeWidth={0.06} strokeOpacity={0.7} />
        {zones && (
          <g strokeWidth={0.03} strokeOpacity={0.3} strokeDasharray="0.2 0.2">
            <line x1={0} y1={-6} x2={W} y2={-6} />
            <line x1={0} y1={6} x2={W} y2={6} />
            <line x1={3} y1={-9} x2={3} y2={9} />
            <line x1={6} y1={-9} x2={6} y2={9} />
          </g>
        )}
        <line x1={-0.6} y1={0} x2={W + 0.6} y2={0} strokeWidth={0.16} />
      </g>

      {showTarget &&
        [target].map((t) => (
          <g key={sender} pointerEvents="none">
            {RINGS.map((r) => (
              <circle
                key={r}
                cx={t.x}
                cy={t.y}
                r={r}
                fill="none"
                stroke="white"
                strokeOpacity={0.35}
                strokeWidth={0.05}
                strokeDasharray="0.2 0.15"
              />
            ))}
            <circle cx={t.x} cy={t.y} r={0.15} fill="white" fillOpacity={0.6} />
          </g>
        ))}

      {ordered.map((f) => {
        const a = at(f.arc.points[0]!);
        const b = at(f.arc.points[f.arc.points.length - 1]!);
        const bold = f === mine;
        const dim = mine != null && !bold;
        const color = colorOf(f);
        return (
          <g key={f.event.id} className="cursor-pointer" onClick={() => onPlay(f)}>
            <line
              x1={a.x}
              y1={a.y}
              x2={b.x}
              y2={b.y}
              stroke={color}
              strokeWidth={bold ? 0.14 : 0.06}
              strokeOpacity={dim ? 0.3 : 0.75}
              strokeLinecap="round"
            />
            <circle
              cx={a.x}
              cy={a.y}
              r={bold ? 0.16 : 0.1}
              fill={color}
              fillOpacity={dim ? 0.3 : 0.6}
            />
            <circle
              cx={b.x}
              cy={b.y}
              r={bold ? 0.3 : 0.2}
              fill={color}
              fillOpacity={dim ? 0.3 : 1}
              stroke={bold ? 'white' : 'none'}
              strokeWidth={0.07}
            />
            <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="transparent" strokeWidth={0.6} />
          </g>
        );
      })}

      {/* The zones' numbers over the lines, haloed, so a busy zone still reads. */}
      {zones &&
        cells.map(({ key, label, x, y, w, h }) => {
          const c = zones.cell(key);
          return (
            <g key={key}>
              <Label x={x + 0.25} y={y + 0.65} size={0.5} opacity={0.55} anchor="start">
                {label}
              </Label>
              <Label
                x={x + w / 2}
                y={y + h / 2 + (c?.sub ? -0.1 : 0.4)}
                size={c?.sub ? 0.95 : 1.1}
                bold
                halo
                opacity={c ? 1 : 0.3}
              >
                {c?.value ?? '0'}
              </Label>
              {c?.sub?.map((line, i) => (
                <Label
                  key={i}
                  x={x + w / 2}
                  y={y + h / 2 + 0.6 + i * 0.55}
                  size={0.5}
                  halo
                  opacity={0.85}
                >
                  {line}
                </Label>
              ))}
            </g>
          );
        })}

      <Label x={W / 2} y={-END + 0.9} size={0.6} bold opacity={0.7}>
        {names[far]}
      </Label>
      <Label x={W / 2} y={END - 0.4} size={0.6} bold opacity={0.7}>
        {names[view.near]}
      </Label>
    </svg>
  );
}
