/** The play seen from the near sideline: distance along the court against
 *  height, so a flight's apex and its clearance over the net read directly. */

import { actionColor } from '@/lib/actionColors';
import type { Arc, Point3 } from './geometry';

const TOP = 6;

export function SideView({
  length,
  netHeight,
  arcs,
  ball,
}: {
  length: number;
  netHeight: number;
  arcs: Arc[];
  ball: Point3 | null;
}) {
  const y = (z: number) => TOP - z;
  return (
    <svg
      viewBox={`-2 -0.5 ${length + 4} ${TOP + 1.5}`}
      className="w-full rounded-xl bg-surface-200/40"
    >
      <line
        x1={-2}
        y1={y(0)}
        x2={length + 2}
        y2={y(0)}
        stroke="currentColor"
        strokeOpacity={0.4}
        strokeWidth={0.05}
      />
      <line x1={0} y1={y(0)} x2={length} y2={y(0)} stroke="#38bdf8" strokeWidth={0.12} />
      <line
        x1={length / 2}
        y1={y(0)}
        x2={length / 2}
        y2={y(netHeight)}
        stroke="white"
        strokeWidth={0.08}
      />
      {[1, 2, 3, 4, 5].map((z) => (
        <text key={z} x={-1.9} y={y(z) + 0.15} fontSize={0.4} fill="currentColor" opacity={0.5}>
          {z}m
        </text>
      ))}
      {arcs.map((a) => (
        <polyline
          key={`${a.from}-${a.to}`}
          points={a.points.map((p) => `${p[0]},${y(p[2])}`).join(' ')}
          fill="none"
          stroke={actionColor(a.label)}
          strokeWidth={0.07}
          strokeOpacity={0.8}
        />
      ))}
      {ball && (
        <circle
          cx={ball[0]}
          cy={y(ball[2])}
          r={0.3}
          fill="#facc15"
          stroke="black"
          strokeWidth={0.06}
        />
      )}
    </svg>
  );
}
