import { useEffect, useRef, type RefObject } from 'react';

// Videos stream via expiring presigned URLs, and the <video> element fails
// silently when one dies: a range request that 403s or hangs leaves it stuck
// in `seeking` (or starved mid-playback) forever, with no user-visible error.
// A 1s watchdog spots a hard error or a stopped clock, reloads the src — the
// API hands out a fresh URL — and seeks back to where the user was.

const STUCK_TICKS = 5; // seconds of stuck seek / starved playback before reloading
const COOLDOWN_MS = 8000; // min gap between recovery attempts
const MAX_ATTEMPTS = 3; // consecutive failed loads before giving up until the next successful one

export interface VideoRecoveryOptions {
  /** Current media URL; return '' when nothing is loaded. */
  src: () => string;
  onRecover?: () => void;
}

export function useVideoRecovery(videoRef: RefObject<HTMLVideoElement>, opts: VideoRecoveryOptions) {
  const optsRef = useRef(opts);
  optsRef.current = opts;

  useEffect(() => {
    const el = videoRef.current;
    if (!el) return;

    let lastAttempt = -Infinity;
    let attempts = 0;
    let stuckTicks = 0;
    let lastTime = -1;

    const recover = () => {
      const { src, onRecover } = optsRef.current;
      const url = src();
      const now = performance.now();
      if (!url || attempts >= MAX_ATTEMPTS || now - lastAttempt < COOLDOWN_MS) return;
      lastAttempt = now;
      attempts += 1;
      const t = el.currentTime;
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
      onRecover?.();
    };

    // Any successful load — recovery or a normal video switch — resets the budget.
    const onLoaded = () => {
      attempts = 0;
    };
    el.addEventListener('loadedmetadata', onLoaded);

    const watchdog = setInterval(() => {
      if (!el.src) return;
      if (el.error) {
        recover();
        return;
      }
      // A seek whose range request died never clears `seeking`; a starved
      // stream keeps "playing" with a stopped clock and nothing buffered ahead.
      const starved = !el.paused && !el.ended && el.currentTime === lastTime && el.readyState < el.HAVE_FUTURE_DATA;
      stuckTicks = el.seeking || starved ? stuckTicks + 1 : 0;
      lastTime = el.currentTime;
      if (stuckTicks >= STUCK_TICKS) {
        stuckTicks = 0;
        recover();
      }
    }, 1000);

    return () => {
      el.removeEventListener('loadedmetadata', onLoaded);
      clearInterval(watchdog);
    };
  }, [videoRef]);
}
