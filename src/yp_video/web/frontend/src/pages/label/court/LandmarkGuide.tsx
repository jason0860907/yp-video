/** Where each landmark is on the court: a top-down sketch stood upright to
 *  fit the side column — the video's left end line at the top, its near
 *  sideline (the camera side) on the left. Clicking a landmark arms it, like
 *  the list below. */

import type { CourtState } from './geometry';

export function LandmarkGuide({
  state,
  armed,
  onArm,
}: {
  state: CourtState;
  armed: string | null;
  onArm: (name: string) => void;
}) {
  const { length, width } = state.court;
  // Court (x along, y across) → sketch: across runs left→right, along runs down.
  const at = (x: number, y: number) => ({ cx: y, cy: x });
  const label = { fontSize: 0.75, fill: 'currentColor' };
  const right = width + 0.6;
  return (
    <svg
      viewBox={`-2 -1.8 ${width + 8} ${length + 3.6}`}
      className="mx-auto mb-3 w-3/4 text-text-muted"
    >
      <rect x={0} y={0} width={width} height={length} className="fill-sky-900/40" />
      {state.lines.map(([x1, y1, x2, y2], i) => (
        <line
          key={i}
          x1={y1}
          y1={x1}
          x2={y2}
          y2={x2}
          stroke="white"
          strokeOpacity={0.6}
          strokeWidth={0.08}
        />
      ))}
      <line
        x1={-0.5}
        y1={length / 2}
        x2={width + 0.5}
        y2={length / 2}
        stroke="white"
        strokeWidth={0.2}
        strokeDasharray="0.3 0.2"
      />
      <text x={width / 2} y={-0.6} textAnchor="middle" {...label}>
        left end
      </text>
      <text x={width / 2} y={length + 1.2} textAnchor="middle" {...label}>
        right end
      </text>
      <text x={right + 0.5} y={length / 2 + 0.25} {...label}>
        net
      </text>
      <text x={right} y={length / 2 - 3 + 0.25} {...label}>
        attack
      </text>
      <text x={right} y={length / 2 + 3 + 0.25} {...label}>
        attack
      </text>
      <text transform={`translate(-1.2 ${length / 2}) rotate(-90)`} textAnchor="middle" {...label}>
        near · camera
      </text>
      {/* Net tops sit on the posts' line, just outside each sideline. */}
      {Object.entries(state.net_landmarks).map(([name, [lx, ly]]) => {
        const isArmed = name === armed;
        const isMarked = name in state.points;
        const side = ly === 0 ? -0.7 : 0.7;
        const s = isArmed ? 1 : 0.7;
        return (
          <rect
            key={name}
            x={ly + side - s / 2}
            y={lx - s / 2}
            width={s}
            height={s}
            onClick={() => onArm(name)}
            className="cursor-pointer"
            fill={isArmed ? '#3b82f6' : isMarked ? '#facc15' : 'transparent'}
            stroke={isArmed || isMarked ? 'white' : '#94a3b8'}
            strokeWidth={0.1}
          >
            <title>{name}</title>
          </rect>
        );
      })}
      {Object.entries(state.landmarks).map(([name, [lx, ly]]) => {
        const isArmed = name === armed;
        const isMarked = name in state.points;
        return (
          <circle
            key={name}
            {...at(lx, ly)}
            r={isArmed ? 0.55 : 0.35}
            onClick={() => onArm(name)}
            className="cursor-pointer"
            fill={isArmed ? '#3b82f6' : isMarked ? '#facc15' : 'transparent'}
            stroke={isArmed || isMarked ? 'white' : '#94a3b8'}
            strokeWidth={0.1}
          >
            <title>{name}</title>
          </circle>
        );
      })}
    </svg>
  );
}
