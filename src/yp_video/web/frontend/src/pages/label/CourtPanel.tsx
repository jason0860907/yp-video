/** Court Label panel: mark where the court's line intersections sit in the
 *  frame; the server solves the floor homography from them.
 *
 *  The projected court lines are drawn back over the video, so a bad mark
 *  shows as lines that miss the paint. The map beside it places every tracked
 *  player's feet at the playhead and each action's actor at its contact — the
 *  calibration's first consumers, and the check that it means something.
 *
 *  No dirty guard: every mark is saved the moment it lands.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import type { PointerEvent as ReactPointerEvent } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { API, apiFetch, apiUrl, errMsg } from '@/lib/api';
import { actionColor } from '@/lib/actionColors';
import { cn } from '@/lib/cn';
import { hasRealTime, seekWhenSeekable, usePlayheadHandover } from '@/lib/playheadHandover';
import { fieldCls } from '@/components/form/Field';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { toast } from '@/components/feedback/toast';
import { useVideoLabelingData } from '@/components/labeling/useVideoLabelingData';
import { CourtMap, type MapDot } from './court/CourtMap';
import { LandmarkGuide } from './court/LandmarkGuide';
import { SideView } from './court/SideView';
import {
  LANDMARK_LABELS,
  apply,
  ballAt,
  boxesAt,
  extendToFrame,
  footOf,
  intersect,
  project3,
  rallyArcs,
  type CourtPositions,
  type CourtState,
  type Point,
  type Point3,
  type Segment,
} from './court/geometry';
import type { ModeDescriptor, PlaybackClock } from './mode';

export const COURT_MODE: ModeDescriptor = {
  key: 'court',
  label: 'Court',
  listKey: 'court-videos',
  statusOptions: [
    { value: 'all', label: 'All' },
    { value: 'unlabeled', label: 'Unlabeled' },
    { value: 'in-progress', label: 'In-Progress' },
    { value: 'done', label: 'Done' },
  ],
  status: (row) => row.court?.status ?? 'unlabeled',
  matches: (row, status) => status === 'all' || (row.court?.status ?? 'unlabeled') === status,
  available: (row) => Boolean(row.action),
  hint: () => '需要一支 cut 影片',
  doneApi: (video) => API.court.done(video),
};

/** How far past the frame edge the canvas reaches when "Outside frame" is
 *  on, as a fraction of the frame — room for corners the camera cut off. */
const OUTSIDE = 0.25;
/** How long after a touch it still counts as "the current action" — the
 *  server's longest flight (web/court_positions.py MAX_FLIGHT_S). */
const MAX_FLIGHT_S = 2.5;
/** How close (fraction of frame width) a mark must land to snap onto a
 *  guide crossing. */
const SNAP = 0.015;
const NET_HEIGHTS = [
  { value: 2.43, label: 'Men · 2.43 m' },
  { value: 2.24, label: 'Women · 2.24 m' },
];
const round4 = (v: number) => Math.round(v * 1e4) / 1e4;

export function CourtPanel({ video, clock }: { video: string; clock?: PlaybackClock }) {
  const qc = useQueryClient();
  const query = useQuery({
    queryKey: ['court', video],
    queryFn: () => apiFetch<CourtState>(API.court.video(video)),
  });
  const state = query.data;
  const { meta, tracksQuery } = useVideoLabelingData(video);

  const videoRef = useRef<HTMLVideoElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [aspect, setAspect] = useState(16 / 9);
  const [frameSize, setFrameSize] = useState<[number, number] | null>(null);
  const [frame, setFrame] = useState(0);
  const [armed, setArmed] = useState<string | null>(null);
  // The map's action dots: the one at the playhead, or every action.
  const [eventScope, setEventScope] = useState<'current' | 'all'>('current');
  // Guide lines are a marking aid, not a label: they live only here.
  const [tool, setTool] = useState<'mark' | 'line'>('mark');
  const [guides, setGuides] = useState<Segment[]>([]);
  const [draft, setDraft] = useState<Segment | null>(null);
  const [outside, setOutside] = useState(false);
  const [layers, setLayers] = useState({
    court: true,
    net: true,
    path: true,
    action: true,
    marks: true,
  });
  const takeHandover = usePlayheadHandover(clock ? () => clock.read(video) : undefined, video);

  const fps = meta.fps;

  // Space = play/pause, ←/→ = one frame (Shift: ten) — the same keys as the
  // other Label panels. Text fields keep their keys; the video's own space
  // handling is pre-empted so a focused player does not toggle twice, and a
  // focused button is blurred so space does not also press it.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = videoRef.current;
      const target = e.target as HTMLElement | null;
      const tag = target?.tagName;
      if (!el || tag === 'TEXTAREA' || tag === 'SELECT') return;
      if (tag === 'INPUT' && (target as HTMLInputElement).type !== 'checkbox') return;
      if (e.key === ' ') {
        e.preventDefault();
        if (tag === 'BUTTON' || tag === 'INPUT') target?.blur();
        if (el.paused) void el.play();
        else el.pause();
      } else if ((e.key === 'ArrowLeft' || e.key === 'ArrowRight') && fps) {
        e.preventDefault();
        el.pause();
        const step = (e.key === 'ArrowLeft' ? -1 : 1) * (e.shiftKey ? 10 : 1);
        const f = Math.max(0, Math.floor(el.currentTime * fps) + step);
        // Park mid-frame so floor(t·fps) lands back on f.
        el.currentTime = (f + 0.5) / fps;
      }
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [fps]);

  // Playhead → frame, every painted frame while playing.
  useEffect(() => {
    const el = videoRef.current;
    if (!el || !fps) return;
    let raf = 0;
    const tick = () => {
      setFrame(Math.round(el.currentTime * fps));
      if (!el.paused) raf = requestAnimationFrame(tick);
    };
    const start = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(tick);
    };
    el.addEventListener('play', start);
    el.addEventListener('seeked', start);
    return () => {
      cancelAnimationFrame(raf);
      el.removeEventListener('play', start);
      el.removeEventListener('seeked', start);
    };
  }, [fps, video]);

  const save = async (points: CourtState['points'], netHeight = state?.net_height_m) => {
    if (!state) return;
    // Optimistic: the mark moves now; the fit follows the response.
    qc.setQueryData<CourtState>(['court', video], {
      ...state,
      points,
      net_height_m: netHeight ?? state.net_height_m,
    });
    try {
      const next = await apiFetch<CourtState>(API.court.video(video), {
        method: 'PUT',
        body: { version: 1, points, net_height_m: netHeight, frame_size: frameSize },
      });
      qc.setQueryData(['court', video], next);
      void qc.invalidateQueries({ queryKey: ['court-videos'] });
      void qc.invalidateQueries({ queryKey: ['label-stats'] });
    } catch (e) {
      toast.error(`Save failed: ${errMsg(e)}`);
      void query.refetch();
    }
  };

  const clientToPoint = (cx: number, cy: number): Point | null => {
    const r = wrapRef.current?.getBoundingClientRect();
    if (!r || !r.width || !r.height) return null;
    const clamp = (v: number) => Math.min(1 + margin, Math.max(-margin, v));
    return [round4(clamp((cx - r.left) / r.width)), round4(clamp((cy - r.top) / r.height))];
  };

  // Every pair of guides' crossing — the points a mark snaps onto.
  const crossings = useMemo(
    () =>
      guides.flatMap((a, i) =>
        guides.slice(i + 1).flatMap((b) => {
          const p = intersect(a, b, OUTSIDE);
          return p ? [p] : [];
        }),
      ),
    [guides],
  );
  // Marks already outside the frame keep the canvas open.
  const hasOutsideMark = Object.values(state?.points ?? {}).some(
    ([x, y]) => x < 0 || x > 1 || y < 0 || y > 1,
  );
  const margin = outside || hasOutsideMark ? OUTSIDE : 0;
  const snap = (p: Point): Point => {
    let best: Point | null = null;
    let bestD = SNAP;
    for (const c of crossings) {
      // Distance in frame-width units, so the radius is round on screen.
      const d = Math.hypot(c[0] - p[0], (c[1] - p[1]) / aspect);
      if (d < bestD) [best, bestD] = [c, d];
    }
    return best ? [round4(best[0]), round4(best[1])] : p;
  };

  const place = (e: ReactPointerEvent) => {
    if (!state || !armed) return;
    const raw = clientToPoint(e.clientX, e.clientY);
    if (!raw) return;
    const p = snap(raw);
    const next = { ...state.points, [armed]: p };
    // Walk on to the next unmarked landmark, so marking reads as one pass.
    const order = [...Object.keys(state.landmarks), ...Object.keys(state.net_landmarks)];
    const after = order.slice(order.indexOf(armed) + 1).concat(order);
    setArmed(after.find((n) => !(n in next)) ?? null);
    void save(next);
  };

  const drag = (e: ReactPointerEvent, name: string) => {
    if (!state) return;
    e.preventDefault();
    e.stopPropagation();
    const target = e.currentTarget as HTMLElement;
    target.setPointerCapture(e.pointerId);
    let last: Point | null = null;
    const move = (ev: PointerEvent) => {
      last = clientToPoint(ev.clientX, ev.clientY);
      if (last)
        qc.setQueryData<CourtState>(['court', video], (s) =>
          s ? { ...s, points: { ...s.points, [name]: last! } } : s,
        );
    };
    const up = () => {
      target.removeEventListener('pointermove', move);
      target.removeEventListener('pointerup', up);
      if (last) void save({ ...state.points, [name]: snap(last) });
    };
    target.addEventListener('pointermove', move);
    target.addEventListener('pointerup', up);
  };

  const drawGuide = (e: ReactPointerEvent) => {
    const start = clientToPoint(e.clientX, e.clientY);
    if (!start) return;
    const target = e.currentTarget as HTMLElement;
    target.setPointerCapture(e.pointerId);
    let seg: Segment = [start, start];
    setDraft(seg);
    const move = (ev: PointerEvent) => {
      const p = clientToPoint(ev.clientX, ev.clientY);
      if (p) setDraft((seg = [start, p]));
    };
    const up = () => {
      target.removeEventListener('pointermove', move);
      target.removeEventListener('pointerup', up);
      setDraft(null);
      if (Math.hypot(seg[1][0] - seg[0][0], seg[1][1] - seg[0][1]) > 0.01)
        setGuides((g) => [...g, seg]);
    };
    target.addEventListener('pointermove', move);
    target.addEventListener('pointerup', up);
  };

  // Nudge one end of a drawn guide; the crossings follow live.
  const dragGuideEnd = (e: ReactPointerEvent, index: number, end: 0 | 1) => {
    e.preventDefault();
    e.stopPropagation();
    const target = e.currentTarget as HTMLElement;
    target.setPointerCapture(e.pointerId);
    const move = (ev: PointerEvent) => {
      const p = clientToPoint(ev.clientX, ev.clientY);
      if (!p) return;
      setGuides((gs) =>
        gs.map((g, i) => (i === index ? ((end === 0 ? [p, g[1]] : [g[0], p]) as Segment) : g)),
      );
    };
    const up = () => {
      target.removeEventListener('pointermove', move);
      target.removeEventListener('pointerup', up);
    };
    target.addEventListener('pointermove', move);
    target.addEventListener('pointerup', up);
  };

  const remove = (name: string) => {
    if (!state) return;
    const { [name]: _, ...rest } = state.points;
    void save(rest);
  };

  const fit = state?.fit ?? null;

  // Actor positions are computed server-side (the export); keyed on the fit
  // so a moved mark refetches them.
  const positionsQuery = useQuery({
    queryKey: ['court-positions', video, fit?.image_to_court],
    queryFn: () => apiFetch<CourtPositions>(API.court.positions(video)),
    enabled: Boolean(fit),
    retry: false,
  });
  const positions = useMemo(() => positionsQuery.data?.events ?? [], [positionsQuery.data]);

  // The rally under the playhead: its reconstructed flights, and the ball
  // on them right now — drawn over the video, where a wrong one shows.
  const t = fps ? frame / fps : 0;
  // The action at the playhead: the latest touch not after it, while its
  // flight could still be in the air.
  const current = useMemo(() => {
    const now = t + 0.5 / (fps || 30);
    let hit: (typeof positions)[number] | null = null;
    for (const p of positions) if (p.time <= now) hit = p;
    return hit && t - hit.time <= MAX_FLIGHT_S ? hit : null;
  }, [positions, t, fps]);
  const arcs = useMemo(
    () => rallyArcs(positionsQuery.data?.arcs ?? [], t),
    [positionsQuery.data, t],
  );
  const ball = ballAt(arcs, t);
  const camera = state?.camera ?? null;
  const toImage = (p: Point3) => (camera ? project3(camera.projection, p) : null);
  const netLines = useMemo(() => {
    if (!camera || !state) return [];
    const h = state.net_height_m;
    const [near, far] = Object.values(state.net_landmarks);
    if (!near || !far) return [];
    const segments: [Point3, Point3][] = [
      [
        [near[0], near[1], h],
        [far[0], far[1], h],
      ],
      [
        [near[0], near[1], 0],
        [near[0], near[1], h],
      ],
      [
        [far[0], far[1], 0],
        [far[0], far[1], h],
      ],
    ];
    return segments.flatMap(([a, b]) => {
      const pa = project3(camera.projection, a);
      const pb = project3(camera.projection, b);
      return pa && pb ? [[pa, pb] as const] : [];
    });
  }, [camera, state]);

  // Court lines projected into the frame (normalized coords).
  const overlayLines = useMemo(() => {
    if (!fit || !state) return [];
    return state.lines.flatMap(([x1, y1, x2, y2]) => {
      const a = apply(fit.court_to_image, [x1, y1]);
      const b = apply(fit.court_to_image, [x2, y2]);
      return a && b ? [[a, b] as const] : [];
    });
  }, [fit, state]);

  const dots = useMemo<MapDot[]>(() => {
    if (!fit || !frameSize) return [];
    const toCourt = (box: [number, number, number, number]) =>
      apply(fit.image_to_court, footOf(box, frameSize));
    const players = boxesAt(tracksQuery.data?.tracklets ?? [], frame).flatMap(({ key, box }) => {
      const at = toCourt(box);
      return at
        ? [
            {
              key: `p${key}`,
              at,
              color: '#ffffff',
              kind: 'player' as const,
              title: `track ${key} · (${at[0].toFixed(2)}, ${at[1].toFixed(2)}) m`,
            },
          ]
        : [];
    });
    const events = (eventScope === 'all' ? positions : current ? [current] : []).map((p) => ({
      key: `e${p.id}`,
      at: p.court_xy,
      color: actionColor(p.label),
      kind: 'event' as const,
      title: `${p.label ?? ''} · frame ${p.frame} · (${p.court_xy[0].toFixed(2)}, ${p.court_xy[1].toFixed(2)}) m`,
      selected: p.id === current?.id,
    }));
    return [...events, ...players];
  }, [fit, frameSize, tracksQuery.data, frame, positions, eventScope, current]);
  const selected = current;

  if (query.isPending)
    return (
      <Card>
        <p className="py-12 text-center text-xs text-text-muted">Loading…</p>
      </Card>
    );
  if (!state)
    return (
      <Card>
        <p className="text-xs text-red-400">{errMsg(query.error)}</p>
      </Card>
    );

  const marked = Object.keys(state.points).length;

  return (
    <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_22rem]">
      <Card className="space-y-3">
        <div className="overflow-hidden rounded-2xl bg-surface-200 ring-1 ring-white/[0.06]">
          {/* The stage is the frame plus the optional outside margin; marks
              and guides are in frame coordinates, so they may sit past 0..1. */}
          <div
            className="relative mx-auto"
            style={{
              aspectRatio: `${aspect}`,
              maxWidth: `calc(var(--video-max-h, 60vh) * ${aspect})`,
            }}
          >
            <div
              ref={wrapRef}
              className="absolute bg-black"
              style={{
                left: `${(margin / (1 + 2 * margin)) * 100}%`,
                top: `${(margin / (1 + 2 * margin)) * 100}%`,
                width: `${(1 / (1 + 2 * margin)) * 100}%`,
                height: `${(1 / (1 + 2 * margin)) * 100}%`,
              }}
            >
              <video
                ref={videoRef}
                src={apiUrl(API.actionAnnotate.video(video))}
                className="block h-full w-full bg-black object-contain"
                controls
                playsInline
                preload="metadata"
                onLoadedMetadata={(e) => {
                  const el = e.currentTarget;
                  if (el.videoWidth && el.videoHeight) {
                    setAspect(el.videoWidth / el.videoHeight);
                    setFrameSize([el.videoWidth, el.videoHeight]);
                  }
                  const t = takeHandover();
                  if (t != null) seekWhenSeekable(el, t);
                }}
                onTimeUpdate={(e) => {
                  if (hasRealTime(e.currentTarget))
                    clock?.write(video, e.currentTarget.currentTime);
                }}
              />
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
                {layers.court &&
                  overlayLines.map(([a, b], i) => (
                    <line
                      key={i}
                      x1={a[0]}
                      y1={a[1]}
                      x2={b[0]}
                      y2={b[1]}
                      stroke="#facc15"
                      strokeOpacity={0.85}
                      strokeWidth={1.5}
                      vectorEffect="non-scaling-stroke"
                    />
                  ))}
                {layers.net &&
                  netLines.map(([a, b], i) => (
                    <line
                      key={`net${i}`}
                      x1={a[0]}
                      y1={a[1]}
                      x2={b[0]}
                      y2={b[1]}
                      stroke="#f472b6"
                      strokeWidth={1.5}
                      strokeDasharray="4 3"
                      vectorEffect="non-scaling-stroke"
                    />
                  ))}
                {layers.path &&
                  arcs.map((a) => {
                    const pts = a.points.flatMap((p) => {
                      const at = toImage(p);
                      return at ? [at] : [];
                    });
                    return (
                      <polyline
                        key={`${a.from}-${a.to}`}
                        points={pts.map((p) => `${p[0]},${p[1]}`).join(' ')}
                        fill="none"
                        stroke="#f97316"
                        strokeWidth={2}
                        strokeOpacity={0.85}
                        vectorEffect="non-scaling-stroke"
                      />
                    );
                  })}
                {[...guides, ...(draft ? [draft] : [])].flatMap((g, i) => {
                  const full = extendToFrame(g, margin);
                  return full
                    ? [
                        <line
                          key={i}
                          x1={full[0][0]}
                          y1={full[0][1]}
                          x2={full[1][0]}
                          y2={full[1][1]}
                          stroke="#22d3ee"
                          strokeWidth={1}
                          vectorEffect="non-scaling-stroke"
                        />,
                      ]
                    : [];
                })}
                {layers.action &&
                  selected?.ball_3d &&
                  (() => {
                    // The line the height was read off: straight up from the
                    // feet to the ball.
                    const [x, y] = selected.court_xy;
                    const a = toImage([x, y, 0]);
                    const b = toImage(selected.ball_3d);
                    return a && b ? (
                      <line
                        x1={a[0]}
                        y1={a[1]}
                        x2={b[0]}
                        y2={b[1]}
                        stroke="white"
                        strokeWidth={1.5}
                        strokeDasharray="3 3"
                        vectorEffect="non-scaling-stroke"
                      />
                    ) : null;
                  })()}
              </svg>
              {layers.action && selected?.foot_image && (
                <span
                  className="pointer-events-none absolute -ml-1.5 -mt-1.5 h-3 w-3 rounded-full bg-fuchsia-500 ring-2 ring-white"
                  style={{
                    left: `${selected.foot_image[0] * 100}%`,
                    top: `${selected.foot_image[1] * 100}%`,
                  }}
                  title="Actor's feet"
                />
              )}
              {layers.action && selected?.ball_image && (
                <span
                  className="pointer-events-none absolute -ml-2.5 -mt-2.5 h-5 w-5 rounded-full border-2"
                  style={{
                    left: `${selected.ball_image[0] * 100}%`,
                    top: `${selected.ball_image[1] * 100}%`,
                    borderColor: actionColor(selected.label),
                  }}
                  title={`${selected.label ?? ''} ball`}
                />
              )}
              {layers.path &&
                (() => {
                  const at = ball ? toImage(ball) : null;
                  return at ? (
                    <span
                      className="pointer-events-none absolute -ml-2 -mt-2 h-4 w-4 rounded-full border-2 border-yellow-300"
                      style={{ left: `${at[0] * 100}%`, top: `${at[1] * 100}%` }}
                      title="Reconstructed ball"
                    />
                  ) : null;
                })()}
              {crossings.map(([x, y], i) => (
                <span
                  key={i}
                  className="pointer-events-none absolute -ml-1.5 -mt-1.5 h-3 w-3 rounded-full border border-cyan-300"
                  style={{ left: `${x * 100}%`, top: `${y * 100}%` }}
                />
              ))}
              {tool === 'line' &&
                guides.flatMap((g, i) =>
                  ([0, 1] as const).map((end) => (
                    <button
                      key={`${i}-${end}`}
                      type="button"
                      onPointerDown={(e) => dragGuideEnd(e, i, end)}
                      onDoubleClick={() => setGuides((gs) => gs.filter((_, j) => j !== i))}
                      className="absolute z-20 -ml-1.5 -mt-1.5 h-3 w-3 cursor-move touch-none border border-white bg-cyan-400"
                      style={{ left: `${g[end][0] * 100}%`, top: `${g[end][1] * 100}%` }}
                      title="Drag to adjust · double-click to delete this line"
                    />
                  )),
                )}
              {layers.marks &&
                Object.entries(state.points).map(([name, [x, y]]) => (
                  <button
                    key={name}
                    type="button"
                    onPointerDown={(e) => drag(e, name)}
                    className="absolute z-20 -ml-2 -mt-2 h-4 w-4 cursor-grab touch-none rounded-full border-2 border-white bg-yellow-400/80 active:cursor-grabbing"
                    style={{ left: `${x * 100}%`, top: `${y * 100}%` }}
                    title={LANDMARK_LABELS[name] ?? name}
                  />
                ))}
            </div>
            {/* Catch the pointer only while a tool needs it, so the native
                controls keep working the rest of the time. */}
            {tool === 'line' ? (
              <div className="absolute inset-0 z-10 cursor-crosshair" onPointerDown={drawGuide} />
            ) : (
              armed && (
                <div className="absolute inset-0 z-10 cursor-crosshair" onPointerDown={place} />
              )
            )}
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Button
            size="sm"
            intent={tool === 'mark' ? 'primary' : 'default'}
            onClick={() => setTool('mark')}
          >
            Mark points
          </Button>
          <Button
            size="sm"
            intent={tool === 'line' ? 'primary' : 'default'}
            onClick={() => setTool('line')}
          >
            Draw lines
          </Button>
          <Button
            size="sm"
            onClick={() => setGuides((g) => g.slice(0, -1))}
            disabled={!guides.length}
          >
            Undo line
          </Button>
          <Button size="sm" onClick={() => setGuides([])} disabled={!guides.length}>
            Clear lines
          </Button>
          <Button
            size="sm"
            intent={margin ? 'primary' : 'default'}
            onClick={() => setOutside((v) => !v)}
            disabled={hasOutsideMark}
            title={
              hasOutsideMark
                ? 'Some marks sit outside the frame'
                : 'Room around the frame for off-screen corners'
            }
          >
            Outside frame
          </Button>
        </div>
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-text-muted">
          {(
            [
              ['court', 'Court lines'],
              ['net', 'Net'],
              ['path', 'Ball path'],
              ['action', 'Action point'],
              ['marks', 'Marks'],
            ] as const
          ).map(([key, label]) => (
            <label key={key} className="inline-flex items-center gap-1.5">
              <input
                type="checkbox"
                checked={layers[key]}
                onChange={(e) => setLayers((l) => ({ ...l, [key]: e.target.checked }))}
              />
              {label}
            </label>
          ))}
        </div>
      </Card>

      <div className="space-y-5">
        <Card>
          <SectionLabel>Landmarks · {marked} marked</SectionLabel>
          <LandmarkGuide
            state={state}
            armed={armed}
            onArm={(name) => {
              setArmed(armed === name ? null : name);
              setTool('mark');
            }}
          />
          <div className="grid grid-cols-2 gap-1.5">
            {[...Object.keys(state.landmarks), ...Object.keys(state.net_landmarks)].map((name) => {
              const isMarked = name in state.points;
              return (
                <div key={name} className="flex items-center gap-1">
                  <button
                    type="button"
                    onClick={() => {
                      setArmed(armed === name ? null : name);
                      setTool('mark');
                    }}
                    className={cn(
                      'min-w-0 flex-1 truncate rounded-md border px-2 py-1 text-left text-[11px] transition-colors',
                      armed === name
                        ? 'border-primary text-text-primary'
                        : isMarked
                          ? 'border-yellow-400/40 text-text-secondary'
                          : 'border-border text-text-muted hover:text-text-primary',
                    )}
                  >
                    {LANDMARK_LABELS[name] ?? name}
                  </button>
                  {isMarked && (
                    <button
                      type="button"
                      onClick={() => remove(name)}
                      className="text-[11px] text-text-muted hover:text-red-400"
                      title="Remove mark"
                    >
                      ✕
                    </button>
                  )}
                </div>
              );
            })}
          </div>
          <label className="mt-3 flex items-center justify-between gap-2 text-[11px] text-text-muted">
            Net height
            <select
              value={state.net_height_m}
              onChange={(e) => void save(state.points, Number(e.target.value))}
              className={cn(fieldCls, 'h-7 py-0 text-[11px]')}
            >
              {NET_HEIGHTS.map((h) => (
                <option key={h.value} value={h.value}>
                  {h.label}
                </option>
              ))}
            </select>
          </label>
          <p className={cn('mt-3 text-[11px]', fit ? 'text-text-secondary' : 'text-amber-400')}>
            {fit ? `Fit RMSE ${fit.rmse_m.toFixed(2)} m` : state.fit_error}
          </p>
          <p className={cn('mt-1 text-[11px]', camera ? 'text-text-secondary' : 'text-amber-400')}>
            {camera
              ? `Camera at (${camera.center.map((v) => v.toFixed(1)).join(', ')}) m · reprojection ${(camera.rmse * (frameSize?.[1] ?? 1080)).toFixed(1)} px${camera.off_floor ? '' : ' · mark the net tops to check height'}`
              : state.camera_error}
          </p>
        </Card>

        {fit && (
          <Card>
            <div className="mb-2.5 flex items-center justify-between">
              <SectionLabel className="mb-0">Court</SectionLabel>
              <div className="inline-flex items-center gap-2 text-[11px]">
                {current && (
                  <span style={{ color: actionColor(current.label) }}>
                    {current.label}
                    {ball ? ` · ball ${ball[2].toFixed(1)} m` : ''}
                  </span>
                )}
                <div className="inline-flex overflow-hidden rounded-md border border-border">
                  {(['current', 'all'] as const).map((scope) => (
                    <button
                      key={scope}
                      type="button"
                      onClick={() => setEventScope(scope)}
                      className={cn(
                        'px-2 py-0.5',
                        eventScope === scope
                          ? 'bg-primary/20 text-text-primary'
                          : 'text-text-muted',
                      )}
                    >
                      {scope === 'current' ? 'Current' : 'All'}
                    </button>
                  ))}
                </div>
              </div>
            </div>
            <CourtMap state={state} dots={dots} arcs={arcs} ball={ball} />
          </Card>
        )}

        {camera && arcs.length > 0 && (
          <Card>
            <SectionLabel>Side view · rally {arcs[0]!.rally_id}</SectionLabel>
            <SideView
              length={state.court.length}
              netHeight={state.net_height_m}
              arcs={arcs}
              ball={ball}
            />
          </Card>
        )}

        {fit && (
          <Card>
            <div className="mb-2.5 flex items-center justify-between">
              <SectionLabel className="mb-0">Actor positions · {positions.length}</SectionLabel>
              {positions.length > 0 && (
                <a
                  href={apiUrl(API.court.positions(video))}
                  download={`${video.replace(/\.[^.]+$/, '')}_court_positions.json`}
                  className="text-[11px] text-primary hover:underline"
                >
                  Download JSON
                </a>
              )}
            </div>
            {positionsQuery.isError ? (
              <p className="text-[11px] text-amber-400">{errMsg(positionsQuery.error)}</p>
            ) : (
              <div className="max-h-72 overflow-y-auto">
                <table className="w-full text-[11px] tabular-nums">
                  <thead className="sticky top-0 bg-surface-100 text-text-muted">
                    <tr>
                      <th className="py-1 text-left font-normal">Time</th>
                      <th className="text-left font-normal">Action</th>
                      <th className="text-right font-normal">x (m)</th>
                      <th className="text-right font-normal">y (m)</th>
                      <th className="text-right font-normal" title="Ball height at the touch">
                        z (m)
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {positions.map((p) => (
                      <tr
                        key={p.id}
                        onClick={() => {
                          const el = videoRef.current;
                          if (el && meta.fps) el.currentTime = p.frame / meta.fps;
                        }}
                        className={cn(
                          'cursor-pointer hover:bg-white/5',
                          p.id === current?.id && 'bg-white/10',
                          !p.in_court && 'text-text-muted',
                        )}
                      >
                        <td className="py-0.5 font-mono">{p.time.toFixed(1)}</td>
                        <td style={{ color: actionColor(p.label) }}>{p.label}</td>
                        <td className="text-right font-mono">{p.court_xy[0].toFixed(2)}</td>
                        <td className="text-right font-mono">{p.court_xy[1].toFixed(2)}</td>
                        <td className="text-right font-mono">
                          {p.ball_3d ? p.ball_3d[2].toFixed(2) : '–'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        )}
      </div>
    </div>
  );
}
