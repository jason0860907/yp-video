/** Client side of the court calibration: applying the homographies the
 *  server solved (court/geometry.py). Solving stays server-side — one solver,
 *  which the iOS port will mirror — so nothing here inverts or fits. */

type Row = [number, number, number];
export type Matrix = [Row, Row, Row];
type Row4 = [number, number, number, number];
export type Point3 = [number, number, number];

/** The solved pinhole camera (court/camera.py). */
export interface Camera {
  /** 3×4: court (x, y, z, 1) → normalized image (x, y, 1), up to scale. */
  projection: [Row4, Row4, Row4];
  center: Point3;
  focal: number;
  /** RMS reprojection error over every mark, in frame heights. */
  rmse: number;
  off_floor: number;
}

/** Project a court point in metres into the normalized frame. */
export function project3([a, b, c]: Camera['projection'], [x, y, z]: Point3): Point | null {
  const w = c[0] * x + c[1] * y + c[2] * z + c[3];
  if (w <= 1e-9) return null;
  return [(a[0] * x + a[1] * y + a[2] * z + a[3]) / w, (b[0] * x + b[1] * y + b[2] * z + b[3]) / w];
}
export type Point = [number, number];

export interface CourtState {
  points: Record<string, Point>;
  net_height_m: number;
  fit: { image_to_court: Matrix; court_to_image: Matrix; rmse_m: number } | null;
  fit_error: string | null;
  camera: Camera | null;
  camera_error: string | null;
  landmarks: Record<string, Point>;
  /** Net-top marks: their floor position; they stand net_height_m above it. */
  net_landmarks: Record<string, Point>;
  lines: [number, number, number, number][];
  court: { length: number; width: number };
  /** How far past the frame edge a mark may sit (fraction of the frame). */
  outside_frame: number;
  /** Longest gap between touches still drawn as one flight, seconds. */
  max_flight_s: number;
}

/** Apply a 3×3 homography; null for a point on or behind the horizon. */
export function apply([a, b, c]: Matrix, [x, y]: Point): Point | null {
  const w = c[0] * x + c[1] * y + c[2];
  if (w <= 1e-9) return null;
  return [(a[0] * x + a[1] * y + a[2]) / w, (b[0] * x + b[1] * y + b[2]) / w];
}

/** Bottom-centre of a pixel xyxy box, normalized — where the feet touch the
 *  floor, the one part of a person the homography can place. */
export function footOf(box: [number, number, number, number], [w, h]: [number, number]): Point {
  return [(box[0] + box[2]) / 2 / w, box[3] / h];
}

/** Display names, left-to-right along the court as seen from the camera. */
export const LANDMARK_LABELS: Record<string, string> = {
  left_near: 'Left end · near',
  left_far: 'Left end · far',
  left_attack_near: 'Left attack · near',
  left_attack_far: 'Left attack · far',
  center_near: 'Center · near',
  center_far: 'Center · far',
  right_attack_near: 'Right attack · near',
  right_attack_far: 'Right attack · far',
  right_near: 'Right end · near',
  right_far: 'Right end · far',
  net_top_near: 'Net top · near',
  net_top_far: 'Net top · far',
};

export type Segment = [Point, Point];

/** GET /court/positions — each action's actor feet, in court metres. */
export interface CourtPositions {
  video: string;
  events: {
    id: string;
    frame: number;
    time: number;
    rally_id: number | null;
    label: string | null;
    /** The actor's feet in the frame; null for a score or when unplaced. */
    foot_image: Point | null;
    /** The annotated ball point in the frame; null when not visible. */
    ball_image: Point | null;
    /** Actor's feet — or, for a score, where the ball landed. */
    /** null when it cannot be placed — `reason` says why. */
    court_xy: Point | null;
    in_court: boolean;
    /** The ball at the touch, metres; null without a camera or ball point. */
    ball_3d: Point3 | null;
    /** Why there is no court position: occluded, no association, no
     *  detection, outside rally, no takeoff, off court or ball hidden. */
    reason: string | null;
  }[];
  /** Ballistic flights between consecutive touches, sampled evenly in time. */
  arcs: Arc[];
}

export interface Arc {
  rally_id: number;
  /** The touch that sent the ball — the flight takes its action colour. */
  label: string | null;
  from: string;
  to: string;
  start: number;
  end: number;
  points: Point3[];
}

/** The arcs of the rally playing at time t (one flight's slack either side). */
export function rallyArcs(arcs: Arc[], t: number): Arc[] {
  const byRally = new Map<number, Arc[]>();
  for (const a of arcs) byRally.set(a.rally_id, [...(byRally.get(a.rally_id) ?? []), a]);
  for (const group of byRally.values()) {
    const first = group[0]!;
    const last = group[group.length - 1]!;
    if (t >= first.start - 1 && t <= last.end + 1) return group;
  }
  return [];
}

/** Where the reconstructed ball is at time t, if a flight covers it. */
export function ballAt(arcs: Arc[], t: number): Point3 | null {
  const a = arcs.find((x) => x.start <= t && t <= x.end);
  if (!a || a.points.length < 2) return null;
  const f = ((t - a.start) / (a.end - a.start)) * (a.points.length - 1);
  const i = Math.min(Math.floor(f), a.points.length - 2);
  const p = a.points[i]!;
  const q = a.points[i + 1]!;
  const k = f - i;
  return [p[0] + (q[0] - p[0]) * k, p[1] + (q[1] - p[1]) * k, p[2] + (q[2] - p[2]) * k];
}

/** The infinite line through a segment, clipped to the frame grown by
 *  `margin` — a guide drawn along a short stretch of paint should reach every
 *  line it crosses, off-screen ones included. */
export function extendToFrame([[x1, y1], [x2, y2]]: Segment, margin: number): Segment | null {
  const lo = -margin;
  const hi = 1 + margin;
  const dx = x2 - x1;
  const dy = y2 - y1;
  const hits: [number, Point][] = [];
  const push = (t: number) => {
    const p: Point = [x1 + t * dx, y1 + t * dy];
    if (p[0] >= lo - 1e-9 && p[0] <= hi + 1e-9 && p[1] >= lo - 1e-9 && p[1] <= hi + 1e-9)
      hits.push([t, p]);
  };
  if (Math.abs(dx) > 1e-9) [lo, hi].forEach((x) => push((x - x1) / dx));
  if (Math.abs(dy) > 1e-9) [lo, hi].forEach((y) => push((y - y1) / dy));
  if (hits.length < 2) return null;
  hits.sort((a, b) => a[0] - b[0]);
  return [hits[0]![1], hits[hits.length - 1]![1]];
}

/** Where two guides' lines cross, if within `margin` of the frame. */
export function intersect(
  [[x1, y1], [x2, y2]]: Segment,
  [[x3, y3], [x4, y4]]: Segment,
  margin: number,
): Point | null {
  const d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
  if (Math.abs(d) < 1e-9) return null;
  const a = x1 * y2 - y1 * x2;
  const b = x3 * y4 - y3 * x4;
  const p: Point = [(a * (x3 - x4) - (x1 - x2) * b) / d, (a * (y3 - y4) - (y1 - y2) * b) / d];
  const inside = (v: number) => v >= -margin && v <= 1 + margin;
  return inside(p[0]) && inside(p[1]) ? p : null;
}
