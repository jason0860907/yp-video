/** The play seen from the near sideline: distance along the court against
 *  height, so a flight's apex and its clearance over the net read directly —
 *  and the flight at the playhead's numbers above it. */

import { actionColor, COURT } from '@/lib/actionColors';
import { flightStats, shownFlight, type Arc, type Point3 } from './geometry';

const TOP = 6;

export function SideView({
  length,
  netHeight,
  arcs,
  ball,
  time,
}: {
  length: number;
  netHeight: number;
  /** The current rally's flights; empty between rallies. */
  arcs: Arc[];
  ball: Point3 | null;
  time: number;
}) {
  const y = (z: number) => TOP - z;
  const flight = shownFlight(arcs, time);
  const stats = flight && flightStats(flight, length / 2, netHeight);
  return (
    <div className="space-y-2">
      <div className="flex min-h-[2.25rem] flex-wrap items-baseline gap-x-4 gap-y-1 text-[11px] text-text-muted">
        {flight && stats ? (
          <>
            <span style={{ color: actionColor(flight.label) }} className="font-semibold">
              {flight.label}
            </span>
            <span>{flight.source === 'ball' ? 'fitted to ball' : 'from touches'}</span>
            <Stat label="speed" value={`${stats.speed.toFixed(0)} km/h`} />
            <Stat label="contact" value={`${stats.contact.toFixed(1)} m`} />
            <Stat label="peak" value={`${stats.peak.toFixed(1)} m`} />
            {stats.clearance !== null && <Stat label="over net" value={`${stats.clearance.toFixed(2)} m`} />}
          </>
        ) : (
          <span>No flight at the playhead</span>
        )}
      </div>
      <svg viewBox={`-2 -0.5 ${length + 4} ${TOP + 1.5}`} className="w-full rounded-xl bg-surface-200/40">
        <line
          x1={-2}
          y1={y(0)}
          x2={length + 2}
          y2={y(0)}
          stroke="currentColor"
          strokeOpacity={0.4}
          strokeWidth={0.05}
        />
        <line x1={0} y1={y(0)} x2={length} y2={y(0)} stroke={COURT.surface} strokeWidth={0.12} />
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
            strokeWidth={a === flight ? 0.12 : 0.07}
            strokeOpacity={0.8}
            strokeDasharray={a.source === 'ball' ? undefined : '0.25 0.18'}
          />
        ))}
        {ball && (
          <circle
            cx={ball[0]}
            cy={y(ball[2])}
            r={0.3}
            fill={COURT.ball}
            stroke="black"
            strokeWidth={0.06}
          />
        )}
      </svg>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <span>
      {label} <span className="font-mono text-text-primary">{value}</span>
    </span>
  );
}
