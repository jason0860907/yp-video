/** The match's source video, played the Label way: the frame clock stops at
 *  the selected clip's end, the Action timeline shows the rallies and their
 *  touches, and the transport and keys are Label's. One long-lived <video>:
 *  selecting a clip seeks it, so the source is fetched once. */

import { forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState } from 'react';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { toast } from '@/components/feedback/toast';
import { ActionTimeline, type TimelineEvent } from '@/components/editor/ActionTimeline';
import { FrameStats, KeyHints, PlayerTransport } from '@/components/labeling/PlayerTransport';
import { useFrameClock } from '@/components/labeling/useFrameClock';
import { useVideoKeys } from '@/components/labeling/useVideoKeys';
import { ACTION_COLORS } from '@/lib/actionColors';
import { formatActionTime } from '@/lib/actionEditorModel';
import { errMsg } from '@/lib/api';
import { EMPTY_WAVEFORM } from '@/lib/useActionWaveform';
import { useVideoRecovery } from '@/lib/useVideoRecovery';
import type { Clip, VideoMeta } from './types';

export interface ClipPlayerHandle {
  seek: (time: number) => void;
}

export const ClipPlayer = forwardRef<
  ClipPlayerHandle,
  {
    title: string;
    src: string;
    meta: VideoMeta;
    clip: Clip | null;
    /** The match's rally clips: the timeline's bands and touch marks. */
    rallies: Clip[];
    /** A clip finished while auto-next is on. */
    onEnded: () => void;
  }
>(function ClipPlayer({ title, src, meta, clip, rallies, onEnded }, ref) {
  const { fps, num_frames: numFrames } = meta;
  const span = useMemo(() => (clip ? { start: clip.start, end: clip.end } : null), [clip]);
  const { videoRef, bindVideo, frame, playing, seekFrame, step, togglePlay } = useFrameClock({
    fps,
    numFrames,
    rally: span,
  });
  useVideoKeys(togglePlay, step);
  useVideoRecovery(videoRef, {
    src: () => src,
    onRecover: () => toast.info('Video stream dropped — reloaded automatically'),
    onGiveUp: () => toast.error('Video still stuck after reloading; auto-retry stopped'),
  });
  const [aspect, setAspect] = useState(16 / 9);
  const [autoNext, setAutoNext] = useState(false);

  const toFrame = useCallback((t: number) => Math.floor(t * fps), [fps]);
  const playFrom = useCallback(
    (f: number) => {
      seekFrame(f);
      void videoRef.current?.play().catch((e) => toast.error(`Play failed: ${errMsg(e)}`));
    },
    [seekFrame, videoRef],
  );
  useImperativeHandle(ref, () => ({ seek: (t) => seekFrame(toFrame(t)) }), [seekFrame, toFrame]);

  // react-query's structural sharing keeps `clip` identical across refetches
  // that change nothing, so this fires on a real selection or reframing only.
  useEffect(() => {
    const el = videoRef.current;
    if (clip && el && el.readyState >= HTMLMediaElement.HAVE_METADATA) playFrom(toFrame(clip.start));
  }, [clip, playFrom, toFrame, videoRef]);

  // The clock stops on the clip's last frame; auto-next picks up from there.
  const endFrame = clip ? Math.max(0, Math.ceil(clip.end * fps) - 1) : -1;
  const wasPlaying = useRef(false);
  useEffect(() => {
    if (wasPlaying.current && !playing && autoNext && frame >= endFrame && endFrame >= 0) onEnded();
    wasPlaying.current = playing;
  }, [playing, frame, endFrame, autoNext, onEnded]);

  const timeline = useMemo(() => {
    const bands = rallies.map((r) => ({ rally_id: r.rally_index ?? 0, start: r.start, end: r.end }));
    const events: TimelineEvent[] = rallies.flatMap((r) =>
      r.touches.map((t) => ({
        id: t.event_id,
        frame: toFrame(t.time),
        visible: true,
        rally_id: r.rally_index,
        label: t.kind,
      })),
    );
    return { bands, events };
  }, [rallies, toFrame]);

  return (
    <div className="min-w-0 flex-1 space-y-4">
      <Card
        label={title}
        right={
          clip && (
            <span className="font-mono text-[11px] tabular-nums text-text-muted">
              {clip.kind !== 'rally' && <span className="capitalize">{clip.kind} · </span>}
              {clip.rally_index != null && `Rally ${clip.rally_index} · `}
              {formatActionTime(clip.start)} →{' '}
              {formatActionTime(clip.end)}
            </span>
          )
        }
      >
        <div className="overflow-hidden rounded-2xl bg-black ring-1 ring-white/[0.06]">
          <div
            className="relative mx-auto"
            style={{ aspectRatio: `${aspect}`, maxWidth: `calc(var(--video-max-h, 45vh) * ${aspect})` }}
          >
            <video
              ref={bindVideo}
              src={src}
              className="block h-full w-full bg-black object-contain"
              playsInline
              preload="metadata"
              onLoadedMetadata={(e) => {
                const el = e.currentTarget;
                if (el.videoWidth && el.videoHeight) setAspect(el.videoWidth / el.videoHeight);
                if (clip) seekFrame(toFrame(clip.start));
              }}
            />
          </div>
        </div>
        <div className="mt-3">
          <ActionTimeline
            duration={meta.duration}
            fps={fps}
            numFrames={numFrames}
            frame={frame}
            rallies={timeline.bands}
            events={timeline.events}
            selectedRallyId={clip?.rally_index ?? 'all'}
            selectedId={null}
            playing={playing}
            waveform={EMPTY_WAVEFORM}
            colors={ACTION_COLORS}
            bandLabel={(r) => `#${r.rally_id}`}
            onSeekFrame={seekFrame}
            onJumpEvent={(id) => {
              const e = timeline.events.find((x) => x.id === id);
              if (e) seekFrame(e.frame);
            }}
          />
        </div>
        <PlayerTransport frame={frame} fps={fps} playing={playing} onTogglePlay={togglePlay} onStep={step}>
          <Button size="sm" disabled={!clip} onClick={() => clip && playFrom(toFrame(clip.start))}>
            Replay clip
          </Button>
          <Button
            size="sm"
            intent={autoNext ? 'primary' : 'default'}
            onClick={() => setAutoNext((v) => !v)}
            title="At a clip's end, play the next one in the list"
          >
            {autoNext ? 'Auto-next on' : 'Auto-next off'}
          </Button>
        </PlayerTransport>
        <div className="mt-2">
          <FrameStats fps={fps} numFrames={numFrames} />
        </div>
      </Card>
      <KeyHints
        keys={[
          ['Space', 'play'],
          ['← →', 'frame'],
          ['Shift ← →', '10 frames'],
        ]}
      />
    </div>
  );
});
