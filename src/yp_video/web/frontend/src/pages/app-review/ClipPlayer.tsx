/** The match's source video under the App's detail chrome: the frame clock
 *  stops at the selected clip's end, the App's TouchScrubber runs over the
 *  clip's window (a rally's attack bands under it) and `children` — the
 *  clip's marks — follow. Frame stepping and keys stay Label's, for review.
 *  One long-lived <video>: selecting a clip seeks it, so the source is
 *  fetched once. */

import { forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState, type ReactNode } from 'react';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { toast } from '@/components/feedback/toast';
import { FrameStats, KeyHints, PlayerTransport } from '@/components/labeling/PlayerTransport';
import { useFrameClock } from '@/components/labeling/useFrameClock';
import { useVideoKeys } from '@/components/labeling/useVideoKeys';
import { errMsg } from '@/lib/api';
import { useVideoRecovery } from '@/lib/useVideoRecovery';
import { AttackBands } from './AttackBands';
import { TouchScrubber } from './TouchScrubber';
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
    /** What the card header says the clip is. */
    caption: string;
    /** A clip finished while auto-next is on. */
    onEnded: () => void;
    children?: ReactNode;
  }
>(function ClipPlayer({ title, src, meta, clip, caption, onEnded, children }, ref) {
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

  return (
    <div className="min-w-0 flex-1 space-y-4">
      <Card
        label={title}
        right={clip && <span className="text-[11px] tabular-nums text-text-muted">{caption}</span>}
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
        {clip && (
          <div className="mt-3">
            <TouchScrubber
              touches={clip.touches.filter((t) => t.kind !== 'score')}
              start={clip.start}
              end={clip.end}
              time={frame / fps}
              onSeek={(t) => seekFrame(toFrame(t))}
            >
              {clip.attacks && (
                <AttackBands
                  attacks={clip.attacks}
                  start={clip.start}
                  end={clip.end}
                  time={frame / fps}
                  onSeek={(t) => seekFrame(toFrame(t))}
                />
              )}
            </TouchScrubber>
          </div>
        )}
        <PlayerTransport frame={frame} fps={fps} playing={playing} onTogglePlay={togglePlay} onStep={step}>
          <Button size="sm" disabled={!clip} onClick={() => clip && playFrom(toFrame(clip.start))}>
            重播
          </Button>
          <Button
            size="sm"
            intent={autoNext ? 'primary' : 'default'}
            onClick={() => setAutoNext((v) => !v)}
            title="片段播完接著播清單的下一個"
          >
            連續播放{autoNext ? '：開' : '：關'}
          </Button>
        </PlayerTransport>
        <div className="mt-2">
          <FrameStats fps={fps} numFrames={numFrames} />
        </div>
        {children}
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
