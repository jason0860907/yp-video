import { useCallback, useEffect, useRef, useState } from 'react';
import { toast } from '@/components/feedback/toast';
import { errMsg } from '@/lib/api';
import { clamp } from '@/lib/actionEditorModel';

/** A stretch of the video in seconds — the selected rally. */
export interface ClockSpan {
  start: number;
  end: number;
}

/** The Label players' frame clock: which frame is on screen, and the moves
 *  that keep that answer exact.
 *
 *  One convention: the frame under a time is floor(t·fps). Seeks park
 *  mid-frame ((f + 0.5) / fps); a presented frame's mediaTime is its start,
 *  so it is read half a frame on — never on a boundary a rounding error can
 *  tip either way.
 *
 *  Driven by requestVideoFrameCallback, re-armed per presented frame, with a
 *  slow poll behind it. seekFrame parks on a frame and holds it until play:
 *  the decoder may present a neighbour, but the frame asked for is the one
 *  labelled. With a `rally`, playback crossing its end from inside stops on
 *  its last frame, and play from there replays it.
 *
 *  Bind the <video> with `bindVideo` (it may mount after the hook); read it
 *  through `videoRef`. Any other seek (writing currentTime) releases the
 *  parked frame. */
export function useFrameClock({
  fps,
  numFrames,
  rally = null,
}: {
  fps: number;
  numFrames: number;
  rally?: ClockSpan | null;
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [video, setVideo] = useState<HTMLVideoElement | null>(null);
  const bindVideo = useCallback((el: HTMLVideoElement | null) => {
    videoRef.current = el;
    setVideo(el);
  }, []);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);

  // Latest inputs, read inside the frame callback without re-arming it.
  const opts = useRef({ fps: fps || 30, numFrames, rally });
  useEffect(() => {
    opts.current = { fps: fps || 30, numFrames, rally };
  });
  /** The frame parked on by seekFrame, held until play. */
  const locked = useRef<number | null>(null);
  /** A time inside the frame on screen; null → the element's currentTime. */
  const at = useRef<number | null>(null);
  /** Bumped by every seek of ours, so a callback armed before it is dropped. */
  const gen = useRef(0);
  const prev = useRef(0);

  const currentFrame = useCallback(() => {
    const { fps, numFrames } = opts.current;
    const last = Math.max(0, numFrames - 1);
    if (locked.current !== null) return clamp(locked.current, 0, last);
    const t = at.current ?? videoRef.current?.currentTime ?? 0;
    return clamp(Math.floor(t * fps), 0, last);
  }, []);

  /** The rally's first and last frame. */
  const rallyFrames = useCallback((): [number, number] | null => {
    const { fps, rally } = opts.current;
    return rally
      ? [Math.round(rally.start * fps), Math.max(0, Math.ceil(rally.end * fps) - 1)]
      : null;
  }, []);

  const seekFrame = useCallback((f: number) => {
    const el = videoRef.current;
    if (!el) return;
    const { fps, numFrames } = opts.current;
    const target = clamp(f, 0, Math.max(0, numFrames - 1));
    const t = (target + 0.5) / fps;
    locked.current = target;
    at.current = t;
    gen.current += 1;
    el.currentTime = el.duration > 0 ? Math.min(t, el.duration) : t;
    setFrame(target);
  }, []);

  const refresh = useCallback(() => {
    const f = currentFrame();
    setFrame(f);
    const before = prev.current;
    prev.current = f;
    // Stop at the rally's end only when crossing it from inside: a playhead
    // parked beyond the rally must never trip this.
    const el = videoRef.current;
    const span = rallyFrames();
    if (!el || el.paused || !span) return;
    if (before >= span[0] && before < span[1] && f >= span[1]) {
      el.pause();
      seekFrame(span[1]);
    }
  }, [currentFrame, rallyFrames, seekFrame]);

  const step = useCallback(
    (frames: number) => {
      videoRef.current?.pause();
      seekFrame((locked.current ?? currentFrame()) + frames);
    },
    [currentFrame, seekFrame],
  );

  const togglePlay = useCallback(() => {
    const el = videoRef.current;
    if (!el?.src) return;
    if (!el.paused) {
      el.pause();
      return;
    }
    // Parked at the rally's end the clock would stop again on the very next
    // frame, so play there means "replay the rally".
    const span = rallyFrames();
    if (span && currentFrame() >= span[1]) seekFrame(span[0]);
    locked.current = null;
    void el.play().catch((e) => toast.error(`Play failed: ${errMsg(e)}`));
  }, [currentFrame, rallyFrames, seekFrame]);

  useEffect(() => {
    if (!video) return;
    let alive = true;
    const arm = () => {
      if (!video.requestVideoFrameCallback) return;
      const armedAt = gen.current;
      video.requestVideoFrameCallback((_now, meta) => {
        if (!alive) return;
        if (armedAt === gen.current) {
          if (!video.paused) locked.current = null;
          if (locked.current === null) at.current = meta.mediaTime + 0.5 / opts.current.fps;
          refresh();
        }
        arm();
      });
    };
    arm();
    const poll = setInterval(refresh, 120);
    const onPlay = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    // A seek not made through seekFrame (a scrubber, a handover) moves the
    // playhead off any parked frame: read the frame from where it went.
    const onSeeking = () => {
      if (at.current !== null && Math.abs(video.currentTime - at.current) < 1e-6) return;
      locked.current = null;
      at.current = null;
      refresh();
    };
    // A new source (a load, a recovery reload) starts the clock over.
    const onEmptied = () => {
      locked.current = null;
      at.current = null;
      gen.current += 1;
      prev.current = 0;
      setFrame(0);
    };
    video.addEventListener('play', onPlay);
    video.addEventListener('pause', onPause);
    video.addEventListener('ended', onPause);
    video.addEventListener('seeking', onSeeking);
    video.addEventListener('emptied', onEmptied);
    return () => {
      alive = false;
      clearInterval(poll);
      video.removeEventListener('play', onPlay);
      video.removeEventListener('pause', onPause);
      video.removeEventListener('ended', onPause);
      video.removeEventListener('seeking', onSeeking);
      video.removeEventListener('emptied', onEmptied);
    };
  }, [video, refresh]);

  return { videoRef, bindVideo, frame, playing, currentFrame, seekFrame, step, togglePlay };
}
