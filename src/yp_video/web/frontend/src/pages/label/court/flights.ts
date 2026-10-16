/** Every touch's outgoing flight, classified the way the iOS app counts it
 *  (VolleyIQ Models/TouchRecord.swift): where it went in the receiver's
 *  zones, a pass's grade, an attack's direction, a set's zone. Mirrored here
 *  so the rules can be checked against the video before the app ships them. */

import { flightStats, type Arc, type CourtPositions, type FlightStats } from './geometry';

const WIDTH = 9;
const NET = 9;
const LINE_ANGLE = 20;
const TIP_SPEED = 30;
/** Where a setter wants the pass: a metre off the net, zones 3–2. */
const PASS_TARGET = { lateral: 6, along: -1 };
const PERFECT_PASS = 1.5;
const PLAYABLE_PASS = 3;

/** A point as the sender faces the net: `lateral` from their left sideline,
 *  `along` from the net towards the far end — negative on their own side. */
export interface SidePoint {
  lateral: number;
  along: number;
}

export type PassGrade = 'perfect' | 'playable' | 'poor';
export type Direction = 'line' | 'cross' | 'tip';
export type SetZone = 'left' | 'middle' | 'right';
/** Which end of the frame the ball was sent from (court x below or above the net). */
export type Half = 'left' | 'right';

export interface TouchFlight {
  /** The touch that sent the ball. */
  event: CourtPositions['events'][number];
  arc: Arc;
  stats: FlightStats;
  half: Half;
  from: SidePoint;
  to: SidePoint;
  crossed: boolean;
  /** A spike met by a block. */
  blocked: boolean;
  direction: Direction | null;
  pass: PassGrade | null;
  setZone: SetZone | null;
  /** The receiver's zone 1–6 where a serve or attack came down; null when
   *  blocked, short or out. */
  landing: number | null;
  /** A serve that crossed, or an attack that crossed or met a block. */
  route: boolean;
  /** Numbers no real flight has — a reconstruction to look at. */
  suspect: string[];
}

/** The six zones of one half as its team faces the net. */
export const ZONES = [4, 3, 2, 5, 6, 1] as const;

export function zoneOf(lateral: number, depth: number): number | null {
  if (lateral < 0 || lateral > WIDTH || depth < 0 || depth > NET) return null;
  const column = Math.min(Math.floor(lateral / (WIDTH / 3)), 2);
  return (depth < 3 ? [4, 3, 2] : [5, 6, 1])[column]!;
}

export function touchFlights(positions: CourtPositions, netHeight: number): TouchFlight[] {
  const events = new Map(positions.events.map((e) => [e.id, e]));
  const servesMet = new Set<string>();
  const ends = positions.arcs.flatMap((arc) => {
    const p = arc.points[0];
    const q = arc.points[arc.points.length - 1];
    if (!p || !q || arc.end <= arc.start) return [];
    const fromLeft = p[0] < NET;
    const seen = ([x, y]: [number, number, number]): SidePoint =>
      fromLeft ? { lateral: WIDTH - y, along: x - NET } : { lateral: y, along: NET - x };
    const e = { arc, from: seen(p), to: seen(q), half: (fromLeft ? 'left' : 'right') as Half };
    if (arc.label === 'serve' && e.to.along > 0) servesMet.add(arc.to);
    return [e];
  });

  const sent = new Set<string>();
  const flights = ends.flatMap(({ arc, from, to, half }) => {
    const event = events.get(arc.from);
    const stats = flightStats(arc, NET, netHeight);
    if (!event || !stats || sent.has(arc.from)) return [];
    sent.add(arc.from);
    const next = events.get(arc.to)?.label;
    const crossed = to.along > 0;
    const blocked = arc.label === 'spike' && next === 'block';

    let direction: Direction | null = null;
    if (arc.label === 'spike' && (crossed || blocked)) {
      const angle = (Math.atan2(Math.abs(to.lateral - from.lateral), Math.abs(to.along - from.along)) * 180) / Math.PI;
      direction = stats.speed < TIP_SPEED ? 'tip' : angle < LINE_ANGLE ? 'line' : 'cross';
    }
    let pass: PassGrade | null = null;
    if (arc.label === 'receive' && servesMet.has(arc.from)) {
      const off = Math.hypot(to.lateral - PASS_TARGET.lateral, to.along - PASS_TARGET.along);
      pass = crossed || off > PLAYABLE_PASS ? 'poor' : off > PERFECT_PASS ? 'playable' : 'perfect';
    }
    let setZone: SetZone | null = null;
    if (arc.label === 'set' && !crossed && next === 'spike') {
      setZone = to.lateral < WIDTH / 3 ? 'left' : to.lateral < (2 * WIDTH) / 3 ? 'middle' : 'right';
    }
    const route = arc.label === 'serve' ? crossed : arc.label === 'spike' ? crossed || blocked : false;
    // The receiver faces the sender: their left is the sender's right.
    const landing = route && crossed && !blocked ? zoneOf(WIDTH - to.lateral, to.along) : null;

    const suspect: string[] = [];
    if (stats.speed > 130) suspect.push('speed > 130 km/h');
    if (stats.contact < 0 || stats.contact > 4.2) suspect.push('contact outside 0–4.2 m');
    if (stats.peak > 15) suspect.push('peak > 15 m');
    if (arc.label === 'serve' && stats.clearance != null && stats.clearance < 0) suspect.push('serve under the net');

    return [
      { event, arc, stats, half, from, to, crossed, blocked, direction, pass, setZone, landing, route, suspect },
    ];
  });
  return flights.sort((a, b) => a.event.time - b.event.time);
}
