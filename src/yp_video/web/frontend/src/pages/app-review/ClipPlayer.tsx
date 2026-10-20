/** The match's source video under the App's detail chrome: the frame clock
 *  stops at the selected clip's end, the App's 觸球點時間軸 panel runs over
 *  the clip's window (a rally's attack bands under it), the control bar
 *  steps through the feed, and `children` — the clip's marks — follow.
 *  Frame stepping and keys stay Label's, for review. One long-lived
 *  <video>: selecting a clip seeks it, so the source is fetched once. */

import { forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState, type ReactNode } from 'react';
import { Button } from '@/components/ui/Button';
import { toast } from '@/components/feedback/toast';
import { KeyHints } from '@/components/labeling/PlayerTransport';
import { useFrameClock } from '@/components/labeling/useFrameClock';
import { useVideoKeys } from '@/components/labeling/useVideoKeys';
import { errMsg } from '@/lib/api';
import { formatActionTime } from '@/lib/actionEditorModel';
import { useVideoRecovery } from '@/lib/useVideoRecovery';
import { AttackBands } from './AttackBands';
import { TouchScrubber } from './TouchScrubber';
import type { Clip, VideoMeta } from './types';

export interface ClipPlayerHandle {
  seek: (time: number) => void;
}

/** Where the clip sits in the feed, for the control bar's ‹ 12 / 42 ›. */
export interface FeedPosition {
  at: number;
  count: number;
  go: (delta: number) => void;
}

export const ClipPlayer = forwardRef<
  ClipPlayerHandle,
  {
    src: string;
    meta: VideoMeta;
    clip: Clip | null;
    nav?: FeedPosition;
    /** A clip finished while auto-next is on. */
    onEnded: () => void;
    children?: ReactNode;
  }
>(function ClipPlayer({ src, meta, clip, nav, onEnded, children }, ref) {
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
    <div className="min-w-0 flex-1 space-y-3">
      <div className="overflow-hidden rounded-2xl bg-black ring-1 ring-white/[0.06]">
        <div
          className="relative mx-auto"
          style={{ aspectRatio: `${aspect}`, maxWidth: `calc(var(--video-max-h, 62vh) * ${aspect})` }}
        >
          <video
            ref={bindVideo}
            src={src}
            className="block h-full w-full cursor-pointer bg-black object-contain"
            playsInline
            preload="metadata"
            onClick={togglePlay}
            onLoadedMetadata={(e) => {
              const el = e.currentTarget;
              if (el.videoWidth && el.videoHeight) setAspect(el.videoWidth / el.videoHeight);
              if (clip) seekFrame(toFrame(clip.start));
            }}
          />
        </div>
      </div>

      {clip && (
        <section className="rounded-xl bg-surface-100 px-4 pb-2 pt-3">
          <div className="flex items-center justify-between text-[11px]">
            <span className="font-semibold text-text-secondary">
              <span className="mr-1" style={{ color: '#E8B23A' }}>✦</span>觸球點時間軸
            </span>
            <span className="font-mono tabular-nums text-text-muted">{(clip.end - clip.start).toFixed(1)}s</span>
          </div>
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
        </section>
      )}

      <div className="flex flex-wrap items-center gap-2 rounded-xl bg-surface-100 p-2">
        {nav && (
          <div className="flex items-center gap-1">
            <Button size="sm" disabled={nav.at <= 0} onClick={() => nav.go(-1)} aria-label="上一個">
              ‹
            </Button>
            <span className="min-w-[4.5rem] text-center font-mono text-xs tabular-nums text-text-secondary">
              {nav.at + 1} / {nav.count}
            </span>
            <Button size="sm" disabled={nav.at >= nav.count - 1} onClick={() => nav.go(1)} aria-label="下一個">
              ›
            </Button>
          </div>
        )}
        <Button size="sm" disabled={!clip} onClick={() => clip && playFrom(toFrame(clip.start))}>
          ↺ 重播
        </Button>
        <Button size="sm" intent={autoNext ? 'primary' : 'default'} onClick={() => setAutoNext((v) => !v)} title="片段播完接著播清單的下一個">
          ▷▷ 連續播放
        </Button>
        <span className="flex-1" />
        <span
          className="font-mono text-xs tabular-nums text-text-secondary"
          title={fps && numFrames ? `${fps.toFixed(3)} fps · ${numFrames} frames` : undefined}
        >
          {formatActionTime(frame / (fps || 30))} · f{frame}
        </span>
        <Button size="sm" onClick={() => step(-1)} aria-label="上一格">
          ◂
        </Button>
        <Button size="sm" className="w-16" onClick={togglePlay}>
          {playing ? 'Pause' : 'Play'}
        </Button>
        <Button size="sm" onClick={() => step(1)} aria-label="下一格">
          ▸
        </Button>
      </div>

      {children}

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
