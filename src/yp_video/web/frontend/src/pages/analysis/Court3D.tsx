/** The rally in three dimensions: the court, the net at its height, every
 *  reconstructed flight of the rally under the playhead with its shadow on
 *  the floor, the ball now, the action's lift from the feet to the ball and
 *  the tracked players. Drag to turn it, double-click to face it again.
 *
 *  A plain SVG projection — a turn about the vertical, a tilt, a little
 *  perspective — no 3D library, like the video overlay. */

import { useRef, useState, type PointerEvent as ReactPointerEvent } from 'react';
import { actionColor, COURT } from '@/lib/actionColors';
import { arcParts, type Arc, type CourtState, type Point3 } from '@/pages/label/court/geometry';
import type { Position } from '@/pages/label/court/usePlays';
import type { MapDot } from './CourtMap';

const CENTRE: Point3 = [9, 4.5, 0];
/** Distance to the eye, metres: how strong the perspective is. */
const EYE = 32;
/** Facing the near sideline, looking down at about 30°. */
const HOME = { yaw: 0, tilt: 0.5 };
const MIN_TILT = 0.05;
const MAX_TILT = Math.PI / 2;
/** Radians per pixel dragged. */
const TURN = 0.008;

type View = typeof HOME;

/** Court metres to the drawing: x across, y down the page. */
function projector({ yaw, tilt }: View) {
  const [cy, sy, ct, st] = [Math.cos(yaw), Math.sin(yaw), Math.cos(tilt), Math.sin(tilt)];
  return ([x, y, z]: readonly number[]): [number, number] => {
    const px = x! - CENTRE[0];
    const py = y! - CENTRE[1];
    const across = px * cy - py * sy;
    const away = px * sy + py * cy;
    const depth = away * ct - z! * st;
    const k = EYE / (EYE + depth);
    return [across * k, (-z! * ct - away * st) * k];
  };
}

export function Court3D({
  state,
  arcs,
  ball,
  current,
  players,
}: {
  state: CourtState;
  arcs: Arc[];
  ball: Point3 | null;
  current: Position | null;
  players: MapDot[];
}) {
  const [view, setView] = useState<View>(HOME);
  const drag = useRef<{ x: number; y: number; from: View } | null>(null);
  const at = projector(view);
  const path = (points: readonly (readonly number[])[]) =>
    points.map((p) => at(p).join(',')).join(' ');

  const onDown = (e: ReactPointerEvent<SVGSVGElement>) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    drag.current = { x: e.clientX, y: e.clientY, from: view };
  };
  const onMove = (e: ReactPointerEvent<SVGSVGElement>) => {
    const d = drag.current;
    if (!d) return;
    setView({
      yaw: d.from.yaw + (e.clientX - d.x) * TURN,
      tilt: Math.min(MAX_TILT, Math.max(MIN_TILT, d.from.tilt + (e.clientY - d.y) * TURN)),
    });
  };

  const h = state.net_height_m;
  // The net between its posts — the marked net tops' floor points, or the
  // court's sidelines — its top band a metre deep.
  const posts = Object.values(state.net_landmarks);
  const [y0, y1] = posts.length === 2 ? [posts[0]![1], posts[1]![1]] : [-0.5, 9.5];
  const net = [
    [9, y0, h - 1],
    [9, y1, h - 1],
    [9, y1, h],
    [9, y0, h],
  ];
  const lift =
    current?.ball_3d && current.court_xy ? [[...current.court_xy, 0], current.ball_3d] : null;

  return (
    <div className="space-y-1">
      <svg
        viewBox="-11.5 -9 23 13"
        className="w-full cursor-grab touch-none select-none rounded-xl active:cursor-grabbing"
        style={{ background: COURT.freeZone }}
        onPointerDown={onDown}
        onPointerMove={onMove}
        onPointerUp={() => (drag.current = null)}
        onPointerCancel={() => (drag.current = null)}
        onDoubleClick={() => setView(HOME)}
      >
        <polygon
          points={path([
            [0, 0, 0],
            [18, 0, 0],
            [18, 9, 0],
            [0, 9, 0],
          ])}
          fill={COURT.surface}
        />
        {state.lines.map(([x1, y1, x2, y2], i) => (
          <polyline
            key={i}
            points={path([
              [x1, y1, 0],
              [x2, y2, 0],
            ])}
            stroke="white"
            strokeOpacity={0.8}
            strokeWidth={0.06}
          />
        ))}
        {/* Shadows first, so the flights read above the floor. */}
        {arcs.map((a) => (
          <polyline
            key={`s${a.from}`}
            points={path(a.points.map(([x, y]) => [x, y, 0]))}
            fill="none"
            stroke="black"
            strokeOpacity={0.35}
            strokeWidth={0.08}
          />
        ))}
        {players.map((p) => {
          const [x, y] = at([p.at[0], p.at[1], 0]);
          return (
            <circle
              key={p.key}
              cx={x}
              cy={y}
              r={0.22}
              fill="none"
              stroke="white"
              strokeOpacity={0.8}
              strokeWidth={0.07}
            >
              {p.title && <title>{p.title}</title>}
            </circle>
          );
        })}
        {[y0, y1].map((y) => (
          <polyline
            key={y}
            points={path([
              [9, y, 0],
              [9, y, h],
            ])}
            stroke="white"
            strokeWidth={0.08}
          />
        ))}
        <polygon
          points={path(net)}
          fill="white"
          fillOpacity={0.12}
          stroke="white"
          strokeOpacity={0.7}
          strokeWidth={0.05}
        />
        {arcs.flatMap((a) =>
          arcParts(a).map((part, i) => (
            <polyline
              key={`${a.from}-${i}`}
              points={path(part.points)}
              fill="none"
              stroke={actionColor(a.label)}
              strokeWidth={0.1}
              strokeDasharray={part.dashed ? '0.3 0.2' : undefined}
            />
          )),
        )}
        {lift && (
          <polyline
            points={path(lift)}
            stroke="white"
            strokeWidth={0.05}
            strokeDasharray="0.2 0.15"
          />
        )}
        {ball && (
          <>
            <circle
              cx={at([ball[0], ball[1], 0])[0]}
              cy={at([ball[0], ball[1], 0])[1]}
              r={0.15}
              fill="black"
              fillOpacity={0.5}
            />
            <circle
              cx={at(ball)[0]}
              cy={at(ball)[1]}
              r={0.3}
              fill={COURT.ball}
              stroke="black"
              strokeWidth={0.06}
            />
          </>
        )}
      </svg>
      <p className="text-[10px] text-text-muted">
        Drag to turn · double-click to face the court again
        {ball ? ` · ball ${ball[2].toFixed(1)} m up` : ''}
      </p>
    </div>
  );
}
