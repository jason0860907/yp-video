/** Everything drawn over the video, in frame coordinates: the court and net
 *  as the calibration projects them, the reconstructed flights and ball, the
 *  current action's feet and ball, guide lines and the user's marks.
 *
 *  Rendered inside the frame element, so percentages are frame fractions;
 *  `margin` widens the drawing past the frame edge for off-screen marks. */

import { useMemo } from 'react';
import type { PointerEvent as ReactPointerEvent } from 'react';
import { actionColor } from '@/lib/actionColors';
import {
  LANDMARK_LABELS,
  apply,
  extendToFrame,
  project3,
  type Arc,
  type CourtPositions,
  type CourtState,
  type Point,
  type Point3,
} from './geometry';
import { LAYERS, type Layers } from './layers';
import type { Guides } from './useGuides';

export function LayerToggles({
  layers,
  onChange,
}: {
  layers: Layers;
  onChange: (next: Layers) => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-text-muted">
      {LAYERS.map(([key, label]) => (
        <label key={key} className="inline-flex items-center gap-1.5">
          <input
            type="checkbox"
            checked={layers[key]}
            onChange={(e) => onChange({ ...layers, [key]: e.target.checked })}
          />
          {label}
        </label>
      ))}
    </div>
  );
}

type Line = readonly [Point, Point];
const pair = (a: Point | null, b: Point | null): Line[] => (a && b ? [[a, b]] : []);

function Lines({ lines, ...stroke }: { lines: Line[] } & React.SVGProps<SVGLineElement>) {
  return (
    <>
      {lines.map(([a, b], i) => (
        <line
          key={i}
          x1={a[0]}
          y1={a[1]}
          x2={b[0]}
          y2={b[1]}
          vectorEffect="non-scaling-stroke"
          {...stroke}
        />
      ))}
    </>
  );
}

/** Where a marker sits: centred on `at`, drawn at `scale` × its size. */
const pin = (at: Point, scale: number): React.CSSProperties => ({
  left: `${at[0] * 100}%`,
  top: `${at[1] * 100}%`,
  transform: `translate(-50%, -50%) scale(${scale})`,
});

function Dot({
  at,
  scale,
  className,
  style,
  title,
}: {
  at: Point;
  scale: number;
  className: string;
  style?: React.CSSProperties;
  title: string;
}) {
  return (
    <span
      className={`pointer-events-none absolute ${className}`}
      style={{ ...pin(at, scale), ...style }}
      title={title}
    />
  );
}

export function VideoOverlay({
  state,
  margin,
  layers,
  arcs,
  ball,
  current,
  guides,
  editingGuides,
  pointScale,
  onDragMark,
}: {
  state: CourtState;
  margin: number;
  layers: Layers;
  arcs: Arc[];
  ball: Point3 | null;
  current: CourtPositions['events'][number] | null;
  guides: Guides;
  /** Line tool active: guide ends get drag handles. */
  editingGuides: boolean;
  /** Every marker's size, × its default. */
  pointScale: number;
  onDragMark: (e: ReactPointerEvent, name: string) => void;
}) {
  const { fit, camera } = state;
  const toImage = (p: Point3) => (camera ? project3(camera.projection, p) : null);

  const courtLines = useMemo(
    () =>
      fit
        ? state.lines.flatMap(([x1, y1, x2, y2]) =>
            pair(apply(fit.court_to_image, [x1, y1]), apply(fit.court_to_image, [x2, y2])),
          )
        : [],
    [fit, state.lines],
  );

  const netLines = useMemo(() => {
    const [near, far] = Object.values(state.net_landmarks);
    if (!camera || !near || !far) return [];
    const h = state.net_height_m;
    const at = (p: Point3) => project3(camera.projection, p);
    return [
      ...pair(at([near[0], near[1], h]), at([far[0], far[1], h])),
      ...pair(at([near[0], near[1], 0]), at([near[0], near[1], h])),
      ...pair(at([far[0], far[1], 0]), at([far[0], far[1], h])),
    ];
  }, [camera, state.net_landmarks, state.net_height_m]);

  const guideLines = [...guides.guides, ...(guides.draft ? [guides.draft] : [])].flatMap((g) => {
    const full = extendToFrame(g, margin);
    return full ? [full] : [];
  });

  // The line the height was read off: straight up from the feet to the ball.
  const lift =
    current?.ball_3d != null && current.court_xy
      ? pair(toImage([current.court_xy[0], current.court_xy[1], 0]), toImage(current.ball_3d))
      : [];
  const ballAt = ball ? toImage(ball) : null;

  return (
    <>
      <svg
        className="pointer-events-none absolute"
        style={{
          left: `${-margin * 100}%`,
          top: `${-margin * 100}%`,
          width: `${(1 + 2 * margin) * 100}%`,
          height: `${(1 + 2 * margin) * 100}%`,
        }}
        viewBox={`${-margin} ${-margin} ${1 + 2 * margin} ${1 + 2 * margin}`}
        preserveAspectRatio="none"
      >
        {layers.court && (
          <Lines lines={courtLines} stroke="#facc15" strokeOpacity={0.85} strokeWidth={1.5} />
        )}
        {layers.net && (
          <Lines lines={netLines} stroke="#f472b6" strokeWidth={1.5} strokeDasharray="4 3" />
        )}
        {layers.path &&
          arcs.map((a) => (
            <polyline
              key={`${a.from}-${a.to}`}
              points={a.points
                .flatMap((p) => {
                  const at = toImage(p);
                  return at ? [`${at[0]},${at[1]}`] : [];
                })
                .join(' ')}
              fill="none"
              stroke={actionColor(a.label)}
              strokeWidth={2}
              strokeOpacity={0.85}
              vectorEffect="non-scaling-stroke"
            />
          ))}
        <Lines lines={guideLines} stroke="#22d3ee" strokeWidth={1} />
        {layers.action && (
          <Lines lines={lift} stroke="white" strokeWidth={1.5} strokeDasharray="3 3" />
        )}
      </svg>

      {layers.action && current?.foot_image && (
        <Dot
          at={current.foot_image}
          scale={pointScale}
          className="h-3 w-3 rounded-full bg-fuchsia-500 ring-2 ring-white"
          title="Actor's feet"
        />
      )}
      {layers.action && current?.ball_image && (
        <Dot
          at={current.ball_image}
          scale={pointScale}
          className="h-5 w-5 rounded-full border-2"
          style={{ borderColor: actionColor(current.label) }}
          title={`${current.label ?? ''} ball`}
        />
      )}
      {layers.path && ballAt && (
        <Dot
          at={ballAt}
          scale={pointScale}
          className="h-4 w-4 rounded-full border-2 border-yellow-300"
          title="Reconstructed ball"
        />
      )}
      {guides.crossings.map((c, i) => (
        <Dot
          key={i}
          at={c}
          scale={pointScale}
          className="h-3 w-3 rounded-full border border-cyan-300"
          title="Guide crossing"
        />
      ))}
      {editingGuides &&
        guides.guides.flatMap((g, i) =>
          ([0, 1] as const).map((end) => (
            <button
              key={`${i}-${end}`}
              type="button"
              onPointerDown={(e) => guides.dragEnd(e, i, end)}
              onDoubleClick={() => guides.remove(i)}
              className="absolute z-20 h-3 w-3 cursor-move touch-none border border-white bg-cyan-400"
              style={pin(g[end], pointScale)}
              title="Drag to adjust · double-click to delete this line"
            />
          )),
        )}
      {layers.marks &&
        Object.entries(state.points).map(([name, [x, y]]) => (
          <button
            key={name}
            type="button"
            onPointerDown={(e) => onDragMark(e, name)}
            className="absolute z-20 h-4 w-4 cursor-grab touch-none rounded-full border-2 border-white bg-yellow-400/80 active:cursor-grabbing"
            style={pin([x, y], pointScale)}
            title={LANDMARK_LABELS[name] ?? name}
          />
        ))}
    </>
  );
}
