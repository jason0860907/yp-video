import { useEffect, useRef, type RefObject } from 'react';

// Videos stream via expiring presigned URLs, and the <video> element fails
// silently when one dies: a range request that 403s or hangs leaves it stuck
// in `seeking` (or starved mid-playback) forever, with no user-visible error.
// A 1s watchdog spots a hard error or a stopped clock, reloads the src — the
// API hands out a fresh URL — and seeks back to where the user was.
//
// Each attempt logs a `[video-recovery]` media-state snapshot to the console:
// when a stall survives reloading, that snapshot is what tells apart a dead
// presigned URL (error/403), a hung connection (networkState loading, nothing
// buffered), and a broken spot in the file itself (stalls at the same time
// with data buffered right up to it).

const STUCK_TICKS = 5; // seconds of stuck seek / starved playback before reloading
const COOLDOWN_MS = 8000; // min gap between recovery attempts
const MAX_ATTEMPTS = 3; // reloads per stall spot before giving up there
const SAME_SPOT_S = 1; // stalls within this window count as the same spot

export interface VideoRecoveryOptions {
  /** Current media URL; return '' when nothing is loaded. */
  src: () => string;
  onRecover?: () => void;
  /** Reloading didn't help MAX_ATTEMPTS times at the same spot; auto-retry
   *  stops there until the stream works elsewhere. Tell the user the truth. */
  onGiveUp?: () => void;
}

function snapshot(el: HTMLVideoElement) {
  return {
    error: el.error ? `${el.error.code} ${el.error.message}` : null,
    readyState: el.readyState,
    networkState: el.networkState,
    seeking: el.seeking,
    paused: el.paused,
    currentTime: el.currentTime,
    buffered: Array.from({ length: el.buffered.length }, (_, i) => `${el.buffered.start(i).toFixed(1)}–${el.buffered.end(i).toFixed(1)}s`).join(' '),
  };
}

export function useVideoRecovery(videoRef: RefObject<HTMLVideoElement>, opts: VideoRecoveryOptions) {
  const optsRef = useRef(opts);
  optsRef.current = opts;

  useEffect(() => {
    const el = videoRef.current;
    if (!el) return;

    let lastAttempt = -Infinity;
    let lastStallT = Infinity;
    let attempts = 0;
    let stuckTicks = 0;
    let lastTime = -1;

    const recover = (reason: string) => {
      const { src, onRecover, onGiveUp } = optsRef.current;
      const url = src();
      const now = performance.now();
      if (!url || now - lastAttempt < COOLDOWN_MS) return;
      // The retry budget is per stall spot. Reloading can't fix a stall that
      // keeps re-forming at the same position (broken file region, dead
      // backend), so that trips the give-up; a stall somewhere new means the
      // stream worked in between and gets a fresh budget. This also survives
      // the recovery's own currentTime churn (0 → seek-back), which defeats
      // any "the clock moved" health check.
      const t = el.currentTime;
      attempts = Math.abs(t - lastStallT) < SAME_SPOT_S ? attempts + 1 : 1;
      lastStallT = t;
      if (attempts > MAX_ATTEMPTS) return; // gave up here; a stall elsewhere re-arms
      console.warn(`[video-recovery] ${reason} at ${t.toFixed(1)}s, attempt ${attempts}/${MAX_ATTEMPTS}`, snapshot(el));
      lastAttempt = now;
      const wasPlaying = !el.paused && !el.ended;
      el.src = url;
      const assigned = el.src; // browser-normalized; identifies this load
      el.load();
      el.addEventListener(
        'loadedmetadata',
        () => {
          if (el.src !== assigned) return; // another load superseded this recovery
          el.currentTime = t;
          if (wasPlaying) void el.play().catch(() => {});
        },
        { once: true },
      );
      if (attempts === MAX_ATTEMPTS) {
        console.error('[video-recovery] giving up — reloading does not clear this stall', snapshot(el));
        onGiveUp?.();
      } else {
        onRecover?.();
      }
    };

    const watchdog = setInterval(() => {
      if (!el.src) return;
      if (el.error) {
        recover(`media error (${el.error.code} ${el.error.message})`);
        return;
      }
      // A seek whose range request died never clears `seeking`; a starved
      // stream keeps "playing" with a stopped clock and nothing buffered ahead.
      const starved = !el.paused && !el.ended && el.currentTime === lastTime && el.readyState < el.HAVE_FUTURE_DATA;
      stuckTicks = el.seeking || starved ? stuckTicks + 1 : 0;
      lastTime = el.currentTime;
      if (stuckTicks >= STUCK_TICKS) {
        stuckTicks = 0;
        recover(el.seeking ? `seek stuck ${STUCK_TICKS}s` : `playback starved ${STUCK_TICKS}s`);
      }
    }, 1000);

    return () => clearInterval(watchdog);
  }, [videoRef]);
}
