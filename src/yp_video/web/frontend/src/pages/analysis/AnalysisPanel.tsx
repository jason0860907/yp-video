/** One calibrated video's play, laid out like Action Label: the video with
 *  the reconstruction drawn over it and the action timeline on the left,
 *  under it the rally top-down and in 3D — always in view — and on the
 *  right the touches rally by rally (Positions) or the ball-path statistics
 *  (Flights). Read-only: calibrating is Label → Court. */

import { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { API, apiFetch, apiUrl, errMsg } from '@/lib/api';
import { ACTION_COLORS } from '@/lib/actionColors';
import { useActionWaveform } from '@/lib/useActionWaveform';
import { firstRallyStart, seekWhenSeekable } from '@/lib/playheadHandover';
import { ActionTimeline } from '@/components/editor/ActionTimeline';
import { Card } from '@/components/ui/Card';
import { Tabs } from '@/components/ui/Tabs';
import { cn } from '@/lib/cn';
import { buildTrackBoxes, nearestFrame } from '@/components/labeling/masks';
import { useFrameClock } from '@/components/labeling/useFrameClock';
import { useVideoKeys } from '@/components/labeling/useVideoKeys';
import { FrameStats, KeyHints, PlayerTransport } from '@/components/labeling/PlayerTransport';
import { useVideoLabelingData } from '@/components/labeling/useVideoLabelingData';
import {
  apply,
  ballAt,
  cameraView,
  footOf,
  type Arc,
  type CourtState,
} from '@/pages/label/court/geometry';
import type { Layers } from '@/pages/label/court/layers';
import { usePlays } from '@/pages/label/court/usePlays';
import { LayerToggles, VideoOverlay } from '@/pages/label/court/VideoOverlay';
import type { MapDot } from './CourtMap';
import { Court3D } from './Court3D';
import { FlightsCard } from './FlightsCard';
import { MapCard } from './MapCard';
import { PositionsList } from './PositionsList';

type Tab = 'positions' | 'flights';

export function AnalysisPanel({ video }: { video: string }) {
  const query = useQuery({
    queryKey: ['court', video],
    queryFn: () => apiFetch<CourtState>(API.court.video(video)),
  });
  const state = query.data;
  const fit = state?.fit ?? null;
  const { meta, tracksQuery } = useVideoLabelingData(video);
  const fps = meta.fps;

  const [aspect, setAspect] = useState(16 / 9);
  const [frameSize, setFrameSize] = useState<[number, number] | null>(null);
  const [duration, setDuration] = useState(0);
  const [tab, setTab] = useState<Tab>('positions');
  const [layers, setLayers] = useState<Layers>({
    court: true,
    net: true,
    path: true,
    action: true,
    marks: false,
  });
  const { waveform, loadWaveform } = useActionWaveform();
  const numFrames = fps ? Math.round(duration * fps) : 0;
  const { bindVideo, frame, playing, seekFrame, step, togglePlay } = useFrameClock({
    fps: fps ?? 0,
    numFrames,
  });
  useVideoKeys(togglePlay, step);
  const seekSeconds = (seconds: number) => {
    if (fps) seekFrame(Math.floor(seconds * fps));
  };

  const t = fps ? frame / fps : 0;
  const {
    query: positionsQuery,
    positions,
    timelineEvents,
    current,
    rallyFlights,
  } = usePlays(video, state, t, fps);
  const view = useMemo(() => (fit ? cameraView(fit.court_to_image) : null), [fit]);
  // The flight picked in Flights replaces the rally's on the video.
  const [flightFocus, setFlightFocus] = useState<Arc | null>(null);
  const arcs = tab === 'flights' ? (flightFocus ? [flightFocus] : []) : rallyFlights;
  const ball = ballAt(arcs, t);

  // Every tracked player's feet at the playhead, on the court.
  const trackBoxes = useMemo(
    () => buildTrackBoxes(tracksQuery.data?.tracklets ?? []),
    [tracksQuery.data],
  );
  const players = useMemo<MapDot[]>(() => {
    if (!fit || !frameSize) return [];
    return (nearestFrame(trackBoxes, frame) ?? []).flatMap(({ key, box }) => {
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
  }, [fit, frameSize, trackBoxes, frame]);

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
  if (!fit || !view)
    return (
      <Card>
        <p className="py-12 text-center text-xs text-text-muted">
          The court isn&apos;t calibrated far enough to place the play — mark at least four floor
          points in{' '}
          <Link
            className="text-primary hover:underline"
            to={`/label?mode=court&video=${encodeURIComponent(video)}`}
          >
            Label → Court
          </Link>
          .
        </p>
      </Card>
    );

  return (
    <div className="flex flex-col gap-5 xl:flex-row">
      <div className="min-w-0 flex-1 space-y-4">
        <Card>
          <div className="overflow-hidden rounded-2xl bg-surface-200 ring-1 ring-white/[0.06]">
            <div
              className="relative mx-auto bg-black"
              style={{
                aspectRatio: `${aspect}`,
                maxWidth: `calc(var(--video-max-h, 45vh) * ${aspect})`,
              }}
            >
              <video
                ref={bindVideo}
                src={apiUrl(API.actionAnnotate.video(video))}
                className="block h-full w-full bg-black object-contain"
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
                  const start = firstRallyStart(meta.rallies ?? []);
                  if (start != null) seekWhenSeekable(el, start);
                }}
              />
              <VideoOverlay
                state={state}
                margin={0}
                layers={layers}
                arcs={arcs}
                ball={ball}
                current={current}
                pointScale={1}
              />
            </div>
          </div>
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
          />
          <div className="mt-2 flex flex-wrap items-center justify-between gap-2">
            <LayerToggles layers={layers} onChange={setLayers} without={['marks']} />
            <FrameStats fps={fps ?? 0} numFrames={numFrames} />
          </div>
        </Card>

        {/* The rally under the playhead, top-down and in 3D. */}
        <Card>
          <div className="grid gap-4 lg:grid-cols-2">
            <div>
              <MapCard
                state={state}
                positions={positions}
                current={current}
                players={players}
                arcs={arcs}
                ball={ball}
              />
            </div>
            <div>
              <div className="mb-2.5 text-[11px] font-medium uppercase tracking-wide text-text-muted">
                3D{arcs.length > 0 ? ` · rally ${arcs[0]!.rally_id}` : ''}
              </div>
              <Court3D state={state} arcs={arcs} ball={ball} current={current} players={players} />
            </div>
          </div>
        </Card>

        <KeyHints
          keys={[
            ['Space', 'play'],
            ['← →', 'frame (⇧ ×10)'],
            ['[ ]', 'previous / next flight (Flights)'],
          ]}
        />
      </div>

      {/* Flights takes the room the video can spare. */}
      <div
        className={cn(
          'min-w-0 xl:flex-shrink-0',
          tab === 'flights' ? 'xl:w-[560px]' : 'xl:w-[460px]',
        )}
      >
        <Card className="space-y-4">
          <Tabs
            tabs={[
              { key: 'positions', label: 'Positions' },
              { key: 'flights', label: 'Flights' },
            ]}
            active={tab}
            onChange={setTab}
          />
          {tab === 'positions' && (
            <PositionsList
              video={video}
              positions={positions}
              rallies={meta.rallies ?? []}
              error={positionsQuery.error}
              time={t}
              currentId={current?.id ?? null}
              onSeek={seekSeconds}
            />
          )}
          {tab === 'flights' && (
            <FlightsCard
              view={view}
              positions={positionsQuery.data}
              netHeight={state.net_height_m}
              time={t}
              onFocus={setFlightFocus}
              onSeek={seekSeconds}
            />
          )}
        </Card>
      </div>
    </div>
  );
}
