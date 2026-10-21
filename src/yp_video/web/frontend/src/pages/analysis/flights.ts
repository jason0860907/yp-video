/** Every touch's outgoing flight, classified the way the iOS app counts it
 *  (VolleyIQ Models/TouchRecord.swift): where it went in the receiver's
 *  zones, a pass's grade, the attack a set led to, and for an attack the DataVolley-style
 *  record — where it was hit from, the set's tempo, the direction, how hard,
 *  and what came of it. Mirrored here so the rules can be checked against the
 *  video before the app ships them. */

import { flightStats, type Arc, type CourtPositions, type FlightStats } from '@/pages/label/court/geometry';

const WIDTH = 9;
const NET = 9;
const LENGTH = 18;
const ROW = 3;
/** Where a setter wants the pass: a metre off the net, at its middle. */
const PASS_TARGET = { lateral: 4.5, along: -1 };
const PERFECT_PASS = 1.5;
const PLAYABLE_PASS = 3;
/** Attack direction, degrees off straight across the net. */
const LINE_ANGLE = 15;
const SHARP_ANGLE = 45;
/** Attack speed, km/h: under TIP is a tip, under SOFT a roll shot. */
const TIP_SPEED = 30;
const SOFT_SPEED = 45;
/** The set's flight time, s: up to QUICK a quick, up to FAST a fast set. */
const QUICK_SET = 0.6;
const FAST_SET = 1.0;
/** Contact this far off the net, m, is a back-row attack: the 3 m line plus
 *  the metre a hitter drifts forward in the air — and the reconstruction,
 *  which places a jumping hitter's contact up to a metre deep. */
const BACK_ROW = 4;

type Event = CourtPositions['events'][number];

/** A point as the sender faces the net: `lateral` from their left sideline,
 *  `along` from the net towards the far end — negative on their own side. */
export interface SidePoint {
  lateral: number;
  along: number;
}

export type PassGrade = 'perfect' | 'playable' | 'poor';
/** Which end of the frame the ball was sent from (court x below or above the net). */
export type Half = 'left' | 'right';

/** Where the attacker hit from, as they face the net. */
export type Spot = 'left' | 'middle' | 'right' | 'back';
export type Tempo = 'quick' | 'fast' | 'high';
export type Direction = 'line' | 'cross' | 'sharp';
export type Style = 'hard' | 'soft' | 'tip';
/** DataVolley's attack evaluation: # + - ! / =. */
export type AttackResult = 'kill' | 'tough' | 'dug' | 'covered' | 'blocked' | 'error';
export type AttackError = 'net' | 'out side' | 'out long';

export interface Attack {
  /** The attacker's zone 1–9 on their own half. */
  zone: number | null;
  spot: Spot | null;
  /** The set's flight time, s; null with no set flight into this attack. */
  setTime: number | null;
  tempo: Tempo | null;
  /** Degrees off straight across: positive towards the far sideline. */
  angle: number;
  direction: Direction;
  style: Style;
  result: AttackResult | null;
  /** How an error was made: into the net, or out wide or long. */
  error: AttackError | null;
}

export interface TouchFlight {
  /** The touch that sent the ball. */
  event: Event;
  arc: Arc;
  stats: FlightStats;
  half: Half;
  from: SidePoint;
  to: SidePoint;
  crossed: boolean;
  /** A spike met by a block. */
  blocked: boolean;
  pass: PassGrade | null;
  /** For a serve reception, the zone the serve came down in — where the
   *  passer played it. */
  servedZone: number | null;
  /** For a set that was attacked, that attack — where it was hit from is
   *  where the set went. */
  setFor: Attack | null;
  /** For a set, how well the ball was passed to the setter: the grade of
   *  the receive or dig it came off. */
  passed: PassGrade | null;
  /** The receiver's zone 1–9 where a serve or attack came down or was
   *  played; null when blocked, short or out. */
  landing: number | null;
  /** A serve that crossed, or any attack. */
  route: boolean;
  attack: Attack | null;
  /** Numbers no real flight has — a reconstruction to look at. */
  suspect: string[];
}

/** The nine zones of one half as its team faces the net, net at the top. */
const ROWS = [
  [4, 3, 2],
  [7, 8, 9],
  [5, 6, 1],
];
export const ZONES = ROWS.flat();

export function zoneOf(lateral: number, depth: number): number | null {
  if (lateral < 0 || lateral > WIDTH || depth < 0 || depth > NET) return null;
  const column = Math.min(Math.floor(lateral / ROW), 2);
  return ROWS[Math.min(Math.floor(depth / ROW), 2)]![column]!;
}

const grade = (to: SidePoint, crossed: boolean): PassGrade => {
  const off = Math.hypot(to.lateral - PASS_TARGET.lateral, to.along - PASS_TARGET.along);
  return crossed || off > PLAYABLE_PASS ? 'poor' : off > PERFECT_PASS ? 'playable' : 'perfect';
};

/** Which half a touch was made on — its feet, else the ball. */
const halfOf = (e: Event): Half | null => {
  const x = e.court_xy?.[0] ?? e.ball_3d?.[0];
  return x == null ? null : x < NET ? 'left' : 'right';
};

const inCourt = ([x, y]: [number, number]) => x >= 0 && x <= LENGTH && y >= 0 && y <= WIDTH;

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
    if (arc.label === 'serve' && e.to.along > 0 && arc.to) servesMet.add(arc.to);
    return [e];
  });

  const sent = new Set<string>();
  const flights = ends.flatMap(({ arc, from, to, half }): TouchFlight[] => {
    const event = events.get(arc.from);
    const stats = flightStats(arc, NET, netHeight);
    if (!event || !stats || sent.has(arc.from)) return [];
    sent.add(arc.from);
    const next = arc.to ? events.get(arc.to)?.label : undefined;
    const crossed = to.along > 0;
    const blocked = arc.label === 'spike' && next === 'block';

    const pass = arc.label === 'receive' && servesMet.has(arc.from) ? grade(to, crossed) : null;
    const route = arc.label === 'serve' ? crossed : arc.label === 'spike';
    // The receiver faces the sender: their left is the sender's right.
    const landing = route && crossed && !blocked ? zoneOf(WIDTH - to.lateral, to.along) : null;

    const suspect: string[] = [];
    if (stats.speed > 130) suspect.push('speed > 130 km/h');
    if (stats.contact < 0 || stats.contact > 4.2) suspect.push('contact outside 0–4.2 m');
    if (stats.peak > 15) suspect.push('peak > 15 m');
    if (crossed && stats.clearance != null && stats.clearance < 0) suspect.push('crossed under the net');

    return [
      { event, arc, stats, half, from, to, crossed, blocked, pass, servedZone: null, setFor: null, passed: null, landing, route, attack: null, suspect },
    ];
  });

  const byEvent = new Map(flights.map((f) => [f.event.id, f]));
  const serveInto = new Map(flights.filter((f) => f.arc.label === 'serve').map((f) => [f.arc.to, f]));
  for (const f of flights) if (f.pass) f.servedZone = serveInto.get(f.event.id)?.landing ?? null;
  // A set: how well the receive or dig it came off reached the setter.
  for (const f of flights) {
    if (f.arc.label !== 'receive' || !f.arc.to || events.get(f.arc.to)?.label !== 'set') continue;
    const setter = byEvent.get(f.arc.to);
    if (setter) setter.passed = f.pass ?? grade(f.to, f.crossed);
  }
  const setInto = new Map(positions.arcs.filter((a) => a.label === 'set').map((a) => [a.to, a]));
  const rallies = new Map<number, Event[]>();
  for (const e of [...positions.events].sort((a, b) => a.time - b.time)) {
    if (e.rally_id != null) rallies.set(e.rally_id, [...(rallies.get(e.rally_id) ?? []), e]);
  }
  for (const f of flights) {
    if (f.arc.label !== 'spike') continue;
    const rally = f.event.rally_id != null ? (rallies.get(f.event.rally_id) ?? []) : [];
    const at = rally.findIndex((e) => e.id === f.event.id);
    const set = setInto.get(f.event.id) ?? null;
    f.attack = attackOf(f, at >= 0 ? rally.slice(at + 1) : [], set, byEvent);
    const setter = set && byEvent.get(set.from);
    if (setter) setter.setFor = f.attack;
  }
  return flights.sort((a, b) => a.event.time - b.event.time);
}

function attackOf(f: TouchFlight, after: Event[], set: Arc | null, byEvent: Map<string, TouchFlight>): Attack {
  const depth = -f.from.along;
  const zone = zoneOf(f.from.lateral, Math.max(depth, 0));
  const column = Math.min(Math.max(Math.floor(f.from.lateral / ROW), 0), 2);
  const spot: Spot | null = zone == null ? null : depth > BACK_ROW ? 'back' : (['left', 'middle', 'right'] as const)[column]!;

  const setTime = set ? set.end - set.start : null;
  const tempo: Tempo | null =
    setTime == null ? null : setTime <= QUICK_SET ? 'quick' : setTime <= FAST_SET ? 'fast' : 'high';

  // Positive away from the hitter's own sideline: to the right for one on
  // the left, to the left for one on the right, either way from the middle.
  const across = Math.max(f.to.along - f.from.along, 1e-6);
  const raw = (Math.atan2(f.to.lateral - f.from.lateral, across) * 180) / Math.PI;
  const angle = column === 2 ? -raw : column === 1 ? Math.abs(raw) : raw;
  const direction: Direction = angle < LINE_ANGLE ? 'line' : angle < SHARP_ANGLE ? 'cross' : 'sharp';
  const style: Style = f.stats.speed < TIP_SPEED ? 'tip' : f.stats.speed < SOFT_SPEED ? 'soft' : 'hard';

  const [result, error] = resultOf(f, after, byEvent);
  return { zone, spot, setTime, tempo, angle, direction, style, result, error };
}

/** What came of the attack, from what happened next in the rally. */
function resultOf(
  f: TouchFlight,
  after: Event[],
  byEvent: Map<string, TouchFlight>,
): [AttackResult | null, AttackError | null] {
  const [next, then] = after;
  if (!next) return [null, null];
  const ours = (e: Event) => halfOf(e) === f.half;

  if (next.label === 'score') {
    if (!f.crossed) return ['error', 'net'];
    const at = next.court_xy;
    if (!at) return [null, null];
    if (!inCourt(at)) return ['error', at[1] < 0 || at[1] > WIDTH ? 'out side' : 'out long'];
    return ours(next) ? [null, null] : ['kill', null];
  }
  if (next.label === 'block') {
    if (!then) return [null, null];
    // Off the block and down on the attackers' own court is a stuff block;
    // anywhere else the block touched it out or it fell on their side.
    if (then.label === 'score') {
      return then.court_xy && inCourt(then.court_xy) && ours(then) ? ['blocked', null] : ['kill', null];
    }
    return ours(then) ? ['covered', null] : ['dug', null];
  }
  // Played by the other side: a dig that ended the rally is a kill, a poor
  // one put them under pressure, a good one dug it.
  if (ours(next)) return [null, null];
  if (then?.label === 'score') return ['kill', null];
  const dig = byEvent.get(next.id);
  if (!dig) return [null, null];
  return [grade(dig.to, dig.crossed) === 'poor' ? 'tough' : 'dug', null];
}
