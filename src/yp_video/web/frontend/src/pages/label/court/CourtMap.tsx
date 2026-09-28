/** Top-down court: metres on the floor, the near sideline at the bottom so it
 *  reads the way the camera sees it. */

import type { Arc, CourtState, Point, Point3 } from './geometry';

/** Free zone drawn around the court, metres. */
const MARGIN = 3;

export interface MapDot {
  key: string;
  at: Point;
  color: string;
  /** Live players are rings; events are filled. */
  kind: 'player' | 'event';
  title?: string;
  /** The event picked in the positions table. */
  selected?: boolean;
}

export function CourtMap({
  state,
  dots,
  arcs = [],
  ball = null,
}: {
  state: CourtState;
  dots: MapDot[];
  /** The current rally's flights, drawn as their floor shadows. */
  arcs?: Arc[];
  ball?: Point3 | null;
}) {
  const { length, width } = state.court;
  const y = (v: number) => width - v;
  return (
    <svg
      viewBox={`${-MARGIN} ${-MARGIN} ${length + 2 * MARGIN} ${width + 2 * MARGIN}`}
      className="w-full rounded-xl bg-emerald-950/60"
    >
      <rect x={0} y={0} width={length} height={width} className="fill-sky-900/50" />
      {state.lines.map(([x1, y1, x2, y2], i) => (
        <line
          key={i}
          x1={x1}
          y1={y(y1)}
          x2={x2}
          y2={y(y2)}
          stroke="white"
          strokeOpacity={0.8}
          strokeWidth={0.06}
        />
      ))}
      {arcs.map((a) => (
        <polyline
          key={`${a.from}-${a.to}`}
          points={a.points.map((p) => `${p[0]},${y(p[1])}`).join(' ')}
          fill="none"
          stroke="#f97316"
          strokeWidth={0.07}
          strokeOpacity={0.7}
        />
      ))}
      {dots.map((d) =>
        d.kind === 'player' ? (
          <circle
            key={d.key}
            cx={d.at[0]}
            cy={y(d.at[1])}
            r={0.32}
            fill="none"
            stroke={d.color}
            strokeWidth={0.1}
          >
            {d.title && <title>{d.title}</title>}
          </circle>
        ) : (
          <circle
            key={d.key}
            cx={d.at[0]}
            cy={y(d.at[1])}
            r={d.selected ? 0.45 : 0.18}
            fill={d.color}
            fillOpacity={0.85}
            stroke={d.selected ? 'white' : 'none'}
            strokeWidth={0.12}
          >
            {d.title && <title>{d.title}</title>}
          </circle>
        ),
      )}
      {/* The ball last, over the dots: at a touch it sits right above the
          actor's feet, and a dot drawn after it would hide it. */}
      {ball && (
        <g>
          <circle
            cx={ball[0]}
            cy={y(ball[1])}
            r={0.4}
            fill="#facc15"
            stroke="black"
            strokeWidth={0.08}
          />
          <text x={ball[0] + 0.6} y={y(ball[1]) + 0.25} fontSize={0.7} fill="#facc15">
            {ball[2].toFixed(1)} m
          </text>
        </g>
      )}
    </svg>
  );
}
