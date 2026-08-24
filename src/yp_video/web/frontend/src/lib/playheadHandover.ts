import { useCallback, useMemo, useRef } from 'react';

/**
 * Carrying the playhead from one Label panel to the next.
 *
 * Two things make this harder than assigning `currentTime`, and both bit us:
 *
 * 1. A panel resets its own <video> while loading (removeAttribute('src') →
 *    load() → src = … → load()). That fires a `timeupdate` at 0, and a panel
 *    that writes the shared clock on every tick overwrites the position it was
 *    just handed. So the handover value is taken ONCE, during the first render
 *    for a video, before the element can write anything.
 *
 * 2. Videos stream from a 302 to a presigned R2 URL. At `loadedmetadata` the
 *    element's `seekable` range can still be empty, and the HTML seek
 *    algorithm then returns silently — no error, no seek. Assigning
 *    `currentTime` there looks correct and does nothing.
 *
 * Zero is a real position, so every check here is `!= null`, never truthiness.
 */

/** Seek now if the element can, otherwise as soon as it can. */
export function seekWhenSeekable(el: HTMLVideoElement, t: number): void {
  const apply = () => {
    // duration is NaN until metadata lands; fall back to the raw target.
    el.currentTime = Math.min(t, el.duration || t);
  };
  if (el.seekable.length > 0) {
    apply();
    return;
  }
  el.addEventListener('canplay', apply, { once: true });
}

/**
 * Take the incoming playhead for *videoKey* before this panel's own player can
 * touch the clock. Returns a getter that yields the value once and then null,
 * so a later reload does not yank the user back to where they arrived.
 */
export function usePlayheadHandover(
  read: (() => number | null) | undefined,
  videoKey: string,
): () => number | null {
  // Sampled during render, which is early enough to beat the element's first
  // write — an effect would already be too late. useMemo, not a ref, because
  // reading a ref during render is exactly the hazard the lint rule guards.
  //
  // `read` is deliberately not a dependency: it is an inline arrow at every
  // call site, so tracking its identity would resample on every render and
  // undo the point. One sample per video is the contract.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const handoff = useMemo(() => read?.() ?? null, [videoKey]);
  const consumedFor = useRef<string | null>(null);

  return useCallback(() => {
    // Once only: a later reload of the same video must not yank the user back
    // to where they arrived.
    if (consumedFor.current === videoKey) return null;
    consumedFor.current = videoKey;
    return handoff;
  }, [handoff, videoKey]);
}

/** True when the element is loaded enough that its time means something.
 *  Guards the clock against the 0 that a src reset reports. */
export const hasRealTime = (el: HTMLVideoElement): boolean => el.readyState > 0;
