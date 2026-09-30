/** Court Label panel: mark where the court's line intersections (and the
 *  net tops) sit in the frame; the server solves the floor homography and the
 *  camera from them, and reconstructs the play in court metres.
 *
 *  This file composes: data, saving marks, the playhead and the action under
 *  it. Drawing lives in court/VideoOverlay, guide lines in court/useGuides,
 *  the map / side view / positions list in their own components.
 *
 *  No dirty guard: every mark is saved the moment it lands.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import type { PointerEvent as ReactPointerEvent } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { API, apiFetch, apiUrl, errMsg } from '@/lib/api';
import { hasRealTime, seekWhenSeekable, usePlayheadHandover } from '@/lib/playheadHandover';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { toast } from '@/components/feedback/toast';
import { stepVideo, useVideoKeys } from '@/components/labeling/useVideoKeys';
import { useVideoLabelingData } from '@/components/labeling/useVideoLabelingData';
import type { MapDot } from './court/CourtMap';
import { MapCard } from './court/MapCard';
import { LandmarksCard } from './court/LandmarksCard';
import { PositionsList } from './court/PositionsList';
import { SideView } from './court/SideView';
import { useGuides } from './court/useGuides';
import { trackPointer } from './court/pointer';
import type { Layers } from './court/layers';
import { LayerToggles, VideoOverlay } from './court/VideoOverlay';
import {
  apply,
  ballAt,
  boxesAt,
  footOf,
  rallyArcs,
  type CourtPositions,
  type CourtState,
  type Point,
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

const round4 = (v: number) => Math.round(v * 1e4) / 1e4;

export function CourtPanel({ video, clock }: { video: string; clock?: PlaybackClock }) {
  const qc = useQueryClient();
  const query = useQuery({
    queryKey: ['court', video],
    queryFn: () => apiFetch<CourtState>(API.court.video(video)),
  });
  const state = query.data;
  const { meta, tracksQuery } = useVideoLabelingData(video);
  const fps = meta.fps;

  const videoRef = useRef<HTMLVideoElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [aspect, setAspect] = useState(16 / 9);
  const [frameSize, setFrameSize] = useState<[number, number] | null>(null);
  const [frame, setFrame] = useState(0);
  const [armed, setArmed] = useState<string | null>(null);
  const [tool, setTool] = useState<'mark' | 'line'>('mark');
  const [outside, setOutside] = useState(false);
  const [layers, setLayers] = useState<Layers>({
    court: true,
    net: true,
    path: true,
    action: true,
    marks: true,
  });
  const takeHandover = usePlayheadHandover(clock ? () => clock.read(video) : undefined, video);

  useVideoKeys(
    () => {
      const el = videoRef.current;
      if (!el) return;
      if (el.paused) void el.play();
      else el.pause();
    },
    (n) => {
      if (videoRef.current && fps) stepVideo(videoRef.current, fps, n);
    },
  );

  // Playhead → frame, every painted frame while playing. Bound through the
  // <video>'s own onPlay/onSeeked: the element mounts only once the
  // calibration has loaded, after any mount-time effect has already run.
  const raf = useRef(0);
  const followPlayhead = () => {
    const el = videoRef.current;
    if (!el || !fps) return;
    cancelAnimationFrame(raf.current);
    const tick = () => {
      setFrame(Math.round(el.currentTime * fps));
      if (!el.paused) raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
  };
  useEffect(() => () => cancelAnimationFrame(raf.current), []);

  // Marks already outside the frame keep the canvas open.
  const reach = state?.outside_frame ?? 0;
  const hasOutsideMark = Object.values(state?.points ?? {}).some(
    ([x, y]) => x < 0 || x > 1 || y < 0 || y > 1,
  );
  const margin = outside || hasOutsideMark ? reach : 0;

  const clientToPoint = (cx: number, cy: number): Point | null => {
    const r = wrapRef.current?.getBoundingClientRect();
    if (!r || !r.width || !r.height) return null;
    const clamp = (v: number) => Math.min(1 + margin, Math.max(-margin, v));
    return [round4(clamp((cx - r.left) / r.width)), round4(clamp((cy - r.top) / r.height))];
  };
  const guides = useGuides(clientToPoint, reach, aspect);

  const save = async (points: CourtState['points'], netHeight?: number) => {
    if (!state) return;
    if (!frameSize) {
      toast.error('The video has not loaded yet');
      return;
    }
    const net_height_m = netHeight ?? state.net_height_m;
    // Optimistic: the mark moves now; the fit follows the response.
    qc.setQueryData<CourtState>(['court', video], { ...state, points, net_height_m });
    try {
      const next = await apiFetch<CourtState>(API.court.video(video), {
        method: 'PUT',
        body: { points, net_height_m, frame_size: frameSize },
      });
      qc.setQueryData(['court', video], next);
      void qc.invalidateQueries({ queryKey: ['court-videos'] });
      void qc.invalidateQueries({ queryKey: ['label-stats'] });
    } catch (e) {
      toast.error(`Save failed: ${errMsg(e)}`);
      void query.refetch();
    }
  };

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
  const camera = state?.camera ?? null;

  // Actor positions are computed server-side (the export); keyed on the fit
  // so a moved mark refetches them.
  const positionsQuery = useQuery({
    queryKey: ['court-positions', video, fit?.image_to_court],
    queryFn: () => apiFetch<CourtPositions>(API.court.positions(video)),
    enabled: Boolean(fit),
    retry: false,
  });
  const positions = useMemo(() => positionsQuery.data?.events ?? [], [positionsQuery.data]);

  const t = fps ? frame / fps : 0;
  const maxFlight = state?.max_flight_s ?? 0;
  // The action at the playhead: the latest touch not after it, while its
  // flight could still be in the air.
  const current = useMemo(() => {
    const now = t + 0.5 / (fps || 30);
    let hit: (typeof positions)[number] | null = null;
    for (const p of positions) if (p.time <= now) hit = p;
    return hit && t - hit.time <= maxFlight ? hit : null;
  }, [positions, t, fps, maxFlight]);
  // The rally under the playhead: its reconstructed flights and the ball on
  // them right now.
  const arcs = useMemo(
    () => rallyArcs(positionsQuery.data?.arcs ?? [], t),
    [positionsQuery.data, t],
  );
  const ball = ballAt(arcs, t);

  // Every tracked player's feet at the playhead, on the court.
  const players = useMemo<MapDot[]>(() => {
    if (!fit || !frameSize) return [];
    return boxesAt(tracksQuery.data?.tracklets ?? [], frame).flatMap(({ key, box }) => {
      const at = apply(fit.image_to_court, footOf(box, frameSize));
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
  }, [fit, frameSize, tracksQuery.data, frame]);

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
    <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_420px]">
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
                  const at = takeHandover();
                  if (at != null) seekWhenSeekable(el, at);
                }}
                onPlay={followPlayhead}
                onSeeked={followPlayhead}
                onTimeUpdate={(e) => {
                  if (hasRealTime(e.currentTarget))
                    clock?.write(video, e.currentTarget.currentTime);
                }}
              />
              <VideoOverlay
                state={state}
                margin={margin}
                layers={layers}
                arcs={arcs}
                ball={ball}
                current={current}
                guides={guides}
                editingGuides={tool === 'line'}
                onDragMark={dragMark}
              />
            </div>
            {/* Catch the pointer only while a tool needs it, so the native
                controls keep working the rest of the time. */}
            {tool === 'line' ? (
              <div className="absolute inset-0 z-10 cursor-crosshair" onPointerDown={guides.draw} />
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
          <Button size="sm" onClick={guides.undo} disabled={!guides.guides.length}>
            Undo line
          </Button>
          <Button size="sm" onClick={guides.clear} disabled={!guides.guides.length}>
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
        <LayerToggles layers={layers} onChange={setLayers} />
      </Card>

      <div className="space-y-5">
        <Card>
          <LandmarksCard
            state={state}
            armed={armed}
            onArm={arm}
            onRemove={remove}
            onNetHeight={(h) => void save(state.points, h)}
            frameHeight={frameSize?.[1] ?? null}
          />
        </Card>

        {fit && (
          <Card>
            <MapCard
              state={state}
              positions={positions}
              current={current}
              players={players}
              arcs={arcs}
              ball={ball}
            />
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
            <PositionsList
              video={video}
              positions={positions}
              rallies={meta.rallies ?? []}
              error={positionsQuery.error}
              time={t}
              currentId={current?.id ?? null}
              onSeek={(seconds) => {
                const el = videoRef.current;
                if (el) el.currentTime = seconds;
              }}
            />
          </Card>
        )}
      </div>
    </div>
  );
}
