/** Court Label panel: mark where the court's line intersections (and the
 *  net tops) sit in the frame; the server solves the floor homography and the
 *  camera from them, and reconstructs the play in court metres — drawn back
 *  over the zoomable video (court/useZoom) so a mark's error shows. Looking
 *  at the play itself is the Analysis page's job.
 *
 *  This file composes: data, saving marks, the playhead. Drawing lives in
 *  court/VideoOverlay, guide lines in court/useGuides, the landmark list in
 *  court/LandmarksCard.
 *
 *  No dirty guard: every mark is saved the moment it lands.
 */

import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import type { PointerEvent as ReactPointerEvent } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { API, apiFetch, apiUrl, errMsg } from '@/lib/api';
import { firstRallyStart, hasRealTime, seekWhenSeekable, usePlayheadHandover } from '@/lib/playheadHandover';
import { ActionTimeline } from '@/components/editor/ActionTimeline';
import { ACTION_COLORS } from '@/lib/actionColors';
import { useActionWaveform } from '@/lib/useActionWaveform';
import { Button } from '@/components/ui/Button';
import { Segmented } from '@/components/ui/Segmented';
import { Card } from '@/components/ui/Card';
import { cn } from '@/lib/cn';
import { toast } from '@/components/feedback/toast';
import { useVideoKeys } from '@/components/labeling/useVideoKeys';
import { useFrameClock } from '@/components/labeling/useFrameClock';
import { FrameStats, KeyHints, PlayerTransport } from '@/components/labeling/PlayerTransport';
import { useVideoLabelingData } from '@/components/labeling/useVideoLabelingData';
import { LandmarksCard } from './court/LandmarksCard';
import { Loupe } from './court/Loupe';
import { useGuides } from './court/useGuides';
import { trackPointer } from './court/pointer';
import type { Layers } from './court/layers';
import { LayerToggles, MIN_POINT_SCALE, VideoOverlay } from './court/VideoOverlay';
import { MAX_ZOOM, useZoom } from './court/useZoom';
import { usePlays } from './court/usePlays';
import { ballAt, type CourtState, type Point, type Segment } from './court/geometry';
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
  hint: () => 'Needs a cut video',
  doneApi: (video) => API.court.done(video),
};

const round4 = (v: number) => Math.round(v * 1e4) / 1e4;

const POINT_SCALE_KEY = 'court.pointScale';

/** A number remembered in this browser — a viewer's preference, so a
 *  blocked or empty store just means the default. */
function useStoredNumber(key: string, fallback: number) {
  const [value, setValue] = useState(() => {
    try {
      const stored = Number(localStorage.getItem(key));
      return stored > 0 ? stored : fallback;
    } catch {
      return fallback;
    }
  });
  const set = (next: number) => {
    setValue(next);
    try {
      localStorage.setItem(key, String(next));
    } catch {
      // Not remembered; still applied.
    }
  };
  return [value, set] as const;
}

export function CourtPanel({ video, clock }: { video: string; clock?: PlaybackClock }) {
  const qc = useQueryClient();
  const query = useQuery({
    queryKey: ['court', video],
    queryFn: () => apiFetch<CourtState>(API.court.video(video)),
  });
  const state = query.data;
  const { meta } = useVideoLabelingData(video);
  const fps = meta.fps;

  const wrapRef = useRef<HTMLDivElement>(null);
  const [aspect, setAspect] = useState(16 / 9);
  const [frameSize, setFrameSize] = useState<[number, number] | null>(null);
  const [armed, setArmed] = useState<string | null>(null);
  // The loupe: where the pointer is over the stage, and which side of it.
  const [pointer, setPointer] = useState<{ at: Point; onRight: boolean } | null>(null);
  const [draggingMark, setDraggingMark] = useState(false);
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const [tool, setTool] = useState<'mark' | 'line'>('mark');
  // How far past the frame the stage reaches, as a fraction of it; null
  // until the slider moves, opening just wide enough for the marks.
  const [outside, setOutside] = useState<number | null>(null);
  const [layers, setLayers] = useState<Layers>({
    court: true,
    net: true,
    path: true,
    action: true,
    marks: true,
  });
  const { zoom, zoomTo, pan, attach: attachViewport } = useZoom();
  const [pointScale, setPointScale] = useStoredNumber(POINT_SCALE_KEY, MIN_POINT_SCALE);
  const takeHandover = usePlayheadHandover(clock ? () => clock.read(video) : undefined, video);

  const [duration, setDuration] = useState(0);
  const { waveform, loadWaveform } = useActionWaveform();
  const numFrames = fps ? Math.round(duration * fps) : 0;
  const { videoRef, bindVideo, frame, playing, seekFrame, step, togglePlay } = useFrameClock({
    fps: fps ?? 0,
    numFrames,
  });
  useVideoKeys(togglePlay, step);
  // Esc puts the armed landmark down, so a stray click marks nothing.
  useEffect(() => {
    if (!armed) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setArmed(null);
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [armed]);

  const reach = state?.outside_frame ?? 0;
  // The farthest a mark or a guide end sits past the frame: the stage never
  // shrinks past it, so nothing marked slips out of view.
  const farthest = Math.max(
    0,
    ...[...Object.values(state?.points ?? {}), ...(state?.guides ?? []).flat()].map(([x, y]) =>
      Math.max(-x, x - 1, -y, y - 1),
    ),
  );
  const margin = Math.min(reach, Math.max(farthest, outside ?? Math.ceil(farthest / 0.05) * 0.05));

  const clientToPoint = (cx: number, cy: number): Point | null => {
    const r = wrapRef.current?.getBoundingClientRect();
    if (!r || !r.width || !r.height) return null;
    const clamp = (v: number) => Math.min(1 + margin, Math.max(-margin, v));
    return [round4(clamp((cx - r.left) / r.width)), round4(clamp((cy - r.top) / r.height))];
  };
  const save = async (
    points: CourtState['points'],
    { netHeight, guides: lines }: { netHeight?: number; guides?: Segment[] } = {},
  ) => {
    if (!state) return;
    if (!frameSize) {
      toast.error('The video has not loaded yet');
      return;
    }
    const net_height_m = netHeight ?? state.net_height_m;
    const guides = lines ?? state.guides;
    // Optimistic: the mark moves now; the fit follows the response.
    qc.setQueryData<CourtState>(['court', video], { ...state, points, net_height_m, guides });
    try {
      const next = await apiFetch<CourtState>(API.court.video(video), {
        method: 'PUT',
        body: { points, net_height_m, frame_size: frameSize, guides },
      });
      qc.setQueryData(['court', video], next);
      void qc.invalidateQueries({ queryKey: ['court-videos'] });
      void qc.invalidateQueries({ queryKey: ['label-stats'] });
    } catch (e) {
      toast.error(`Save failed: ${errMsg(e)}`);
      void query.refetch();
    }
  };

  // Guide lines are saved with the marks: drawn live, written once they land.
  const guides = useGuides(clientToPoint, reach, aspect, state?.guides ?? [], (next, final) => {
    if (!state) return;
    if (final) void save(state.points, { guides: next });
    else qc.setQueryData<CourtState>(['court', video], (s) => (s ? { ...s, guides: next } : s));
  });

  // A calibration not yet started opens with guides the server found along
  // the paint (court/paint.py); they are saved with the first mark.
  const untouched = (s: CourtState) => s.guides.length === 0 && Object.keys(s.points).length === 0;
  const found = useQuery({
    queryKey: ['court-guides', video],
    queryFn: () => apiFetch<{ guides: Segment[] }>(API.court.guides(video)),
    enabled: state != null && untouched(state),
    staleTime: Infinity,
    retry: false,
  });
  useEffect(() => {
    const lines = found.data?.guides;
    if (!lines?.length) return;
    qc.setQueryData<CourtState>(['court', video], (s) => (s && untouched(s) ? { ...s, guides: lines } : s));
  }, [found.data, qc, video]);

  // The loupe follows the pointer over the stage while a tool places
  // something: marking, drawing guides, or dragging a mark.
  const loupeOn = tool === 'line' || armed != null || draggingMark;
  useEffect(() => {
    if (!loupeOn) return;
    const onMove = (e: PointerEvent) => {
      const r = viewportRef.current?.getBoundingClientRect();
      const inside = r && e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom;
      const at = inside ? clientToPoint(e.clientX, e.clientY) : null;
      setPointer(at && r ? { at, onRight: e.clientX > r.left + r.width / 2 } : null);
    };
    window.addEventListener('pointermove', onMove);
    return () => window.removeEventListener('pointermove', onMove);
    // clientToPoint reads the frame's live rect and the margin each call.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loupeOn, margin]);

  const place = (e: ReactPointerEvent) => {
    if (!state || !armed) return;
    const raw = clientToPoint(e.clientX, e.clientY);
    if (!raw) return;
    const next = { ...state.points, [armed]: guides.snap(raw) };
    // Walk on to the next unmarked landmark, so marking reads as one pass.
    const order = [...Object.keys(state.landmarks), ...Object.keys(state.net_landmarks)];
    const after = order.slice(order.indexOf(armed) + 1).concat(order);
    setArmed(after.find((n) => !(n in next)) ?? null);
    void save(next);
  };

  const dragMark = (e: ReactPointerEvent, name: string) => {
    if (!state) return;
    e.preventDefault();
    e.stopPropagation();
    let last: Point | null = null;
    setDraggingMark(true);
    trackPointer(
      e,
      (ev) => {
        const p = clientToPoint(ev.clientX, ev.clientY);
        if (!p) return;
        last = p;
        qc.setQueryData<CourtState>(['court', video], (s) =>
          s ? { ...s, points: { ...s.points, [name]: p } } : s,
        );
      },
      () => {
        setDraggingMark(false);
        if (last) void save({ ...state.points, [name]: guides.snap(last) });
      },
    );
  };

  const remove = (name: string) => {
    if (!state) return;
    const { [name]: _, ...rest } = state.points;
    void save(rest);
  };

  const arm = (name: string) => {
    setArmed(armed === name ? null : name);
    setTool('mark');
  };

  const fit = state?.fit ?? null;
  const t = fps ? frame / fps : 0;
  const { positions, timelineEvents, current, rallyFlights } = usePlays(video, state, t, fps);
  const ball = ballAt(rallyFlights, t);

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

  return (
    <div className="flex flex-col gap-5 xl:flex-row">
      {/* Player */}
      <div className="min-w-0 flex-1 space-y-4">
        <Card>
          <div className="mb-3 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <h3 className="font-heading text-sm font-semibold text-text-primary">
              Court Calibration
            </h3>
            <div className="flex flex-wrap items-center gap-2">
              <Segmented
                options={[
                  { key: 'mark', label: 'Mark points' },
                  { key: 'line', label: 'Draw lines' },
                ]}
                value={tool}
                onChange={setTool}
              />
              <Button size="sm" onClick={guides.undo} disabled={!guides.guides.length}>
                Undo line
              </Button>
              <Button size="sm" onClick={guides.clear} disabled={!guides.guides.length}>
                Clear lines
              </Button>
              {found.isFetching && <span className="text-[11px] text-text-muted">Finding court lines…</span>}
              {(found.isError || found.data?.guides.length === 0) && (
                <span className="text-[11px] text-text-muted" title={found.isError ? errMsg(found.error) : undefined}>
                  No lines found
                </span>
              )}
              <label
                className="inline-flex items-center gap-2 text-[11px] text-text-muted"
                title={
                  farthest > 0
                    ? 'Room around the frame for off-screen corners — it stays wide enough for the marks out there'
                    : 'Room around the frame for off-screen corners'
                }
              >
                Outside frame
                <input
                  type="range"
                  min={0}
                  max={reach}
                  step={0.01}
                  value={margin}
                  onChange={(e) => setOutside(Number(e.target.value))}
                  className="w-24"
                />
                <span className="w-8 tabular-nums text-text-secondary">{Math.round(margin * 100)}%</span>
              </label>
            </div>
          </div>

          <div className="overflow-hidden rounded-2xl bg-surface-200 ring-1 ring-white/[0.06]">
            {/* The stage is the frame plus the optional outside margin; marks
                and guides are in frame coordinates, so they may sit past 0..1. */}
            <div
              ref={(el) => {
                viewportRef.current = el;
                attachViewport(el);
              }}
              className="relative mx-auto overflow-hidden"
              style={{
                aspectRatio: `${aspect}`,
                maxWidth: `calc(var(--video-max-h, 45vh) * ${aspect})`,
              }}
              // Middle or right drag pans anywhere; a plain drag pans the
              // zoomed video wherever no tool has the pointer.
              onPointerDownCapture={(e) => {
                if (e.button === 1 || e.button === 2) pan(e);
                else if (e.button === 0 && zoom > 1 && e.target === videoRef.current) pan(e);
              }}
              onContextMenu={(e) => e.preventDefault()}
            >
              <div
                className="absolute left-0 top-0"
                style={{ width: `${zoom * 100}%`, height: `${zoom * 100}%` }}
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
                    ref={bindVideo}
                    src={apiUrl(API.actionAnnotate.video(video))}
                    className={cn(
                      'block h-full w-full bg-black object-contain',
                      zoom > 1 && 'cursor-grab',
                    )}
                    playsInline
                    preload="metadata"
                    onLoadedMetadata={(e) => {
                      const el = e.currentTarget;
                      if (el.videoWidth && el.videoHeight) {
                        setAspect(el.videoWidth / el.videoHeight);
                        setFrameSize([el.videoWidth, el.videoHeight]);
                      }
                      setDuration(el.duration);
                      loadWaveform(video, el.duration);
                      const at = takeHandover(firstRallyStart(meta.rallies ?? []));
                      if (at != null) seekWhenSeekable(el, at);
                    }}
                    onTimeUpdate={(e) => {
                      if (hasRealTime(e.currentTarget))
                        clock?.write(video, e.currentTarget.currentTime);
                    }}
                  />
                  <VideoOverlay
                    state={state}
                    margin={margin}
                    layers={layers}
                    arcs={rallyFlights}
                    ball={ball}
                    current={current}
                    pointScale={pointScale}
                    edit={{ guides, editingGuides: tool === 'line', onDragMark: dragMark }}
                  />
                </div>
                {/* Catch the pointer only while a tool needs it, so a drag pans
                  the zoomed video the rest of the time. */}
                {tool === 'line' ? (
                  <div
                    className="absolute inset-0 z-10 cursor-crosshair"
                    onPointerDown={guides.draw}
                  />
                ) : (
                  armed && (
                    <div className="absolute inset-0 z-10 cursor-crosshair" onPointerDown={place} />
                  )
                )}
              </div>
              {loupeOn && pointer && (
                <div className={cn('absolute top-3 z-20', pointer.onRight ? 'left-3' : 'right-3')}>
                  <Loupe
                    video={videoRef}
                    frame={wrapRef}
                    point={pointer.at}
                    guides={[...guides.guides, ...(guides.draft ? [guides.draft] : [])]}
                    crossings={guides.crossings}
                    marks={Object.values(state.points)}
                    margin={margin}
                  />
                </div>
              )}
            </div>
          </div>

          {/* The Action Label timeline: rally bands, the touches placed on the
              court, the waveform. */}
          <div className="mt-3">
            <ActionTimeline
              duration={duration}
              fps={fps ?? 0}
              numFrames={numFrames}
              frame={frame}
              rallies={meta.rallies ?? []}
              events={timelineEvents}
              selectedRallyId="all"
              selectedId={current?.id ?? null}
              playing={playing}
              waveform={waveform}
              colors={ACTION_COLORS}
              onSeekFrame={seekFrame}
              onJumpEvent={(id) => {
                const p = positions.find((x) => x.id === id);
                if (p) seekFrame(p.frame);
              }}
            />
          </div>
          <PlayerTransport
            frame={frame}
            fps={fps ?? 0}
            playing={playing}
            onTogglePlay={togglePlay}
            onStep={step}
          >
            <span className="inline-flex items-center gap-1">
              <Button
                size="sm"
                onClick={() => zoomTo(zoom / 1.5)}
                disabled={zoom <= 1}
                title="Zoom out"
              >
                −
              </Button>
              <Button
                size="sm"
                onClick={() => zoomTo(1)}
                disabled={zoom <= 1}
                title="Wheel over the video zooms; drag (or right-drag while marking) pans. Click to reset."
              >
                {zoom.toFixed(1)}×
              </Button>
              <Button
                size="sm"
                onClick={() => zoomTo(zoom * 1.5)}
                disabled={zoom >= MAX_ZOOM}
                title="Zoom in"
              >
                +
              </Button>
            </span>
            <label
              className="inline-flex items-center gap-1.5 text-[11px] text-text-muted"
              title="Size of every point drawn on the video"
            >
              Point size
              <input
                type="range"
                min={MIN_POINT_SCALE}
                max={2.5}
                step={0.1}
                value={pointScale}
                onChange={(e) => setPointScale(Number(e.target.value))}
                className="w-20"
              />
            </label>
          </PlayerTransport>
          <div className="mt-2 flex flex-wrap items-center justify-between gap-2">
            <LayerToggles layers={layers} onChange={setLayers} />
            <FrameStats fps={fps ?? 0} numFrames={numFrames} />
          </div>
        </Card>
        <KeyHints
          keys={[
            ['Space', 'play'],
            ['← →', 'frame (⇧ ×10)'],
            ['Esc', 'stop marking'],
            ['Wheel', 'zoom'],
            ['Drag', 'pan (right-drag while marking)'],
          ]}
        />
      </div>

      {/* Landmarks */}
      <div className="min-w-0 xl:w-[420px] xl:flex-shrink-0">
        <Card className="space-y-4">
          <LandmarksCard
            state={state}
            armed={armed}
            onArm={arm}
            onRemove={remove}
            onNetHeight={(h) => void save(state.points, { netHeight: h })}
            frameHeight={frameSize?.[1] ?? null}
          />
          {fit && (
            <Link
              to={`/analysis?video=${encodeURIComponent(video)}`}
              className="block text-[11px] text-primary hover:underline"
            >
              See the play in Analysis →
            </Link>
          )}
        </Card>
      </div>
    </div>
  );
}
