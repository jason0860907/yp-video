/** Zoom and pan of the video stage, for marking far corners precisely.
 *
 *  Zooming grows the stage inside a clipped viewport and pans by scrolling
 *  it, rather than a CSS transform: every consumer keeps measuring the frame
 *  with getBoundingClientRect, markers keep their on-screen size, and the
 *  overlay's non-scaling strokes stay thin. The wheel zooms about the
 *  pointer; a drag pans. */

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import type { PointerEvent as ReactPointerEvent } from 'react';
import { trackPointer } from './pointer';

export const MAX_ZOOM = 8;

export function useZoom() {
  const [viewport, setViewport] = useState<HTMLDivElement | null>(null);
  const [zoom, setZoom] = useState(1);
  const zoomRef = useRef(1);
  // The viewport point that must stay put across a zoom step, applied once
  // the grown stage has laid out.
  const anchor = useRef<{ x: number; y: number; from: number } | null>(null);

  const zoomTo = useCallback(
    (next: number, at?: { x: number; y: number }) => {
      if (!viewport) return;
      const z = Math.min(MAX_ZOOM, Math.max(1, next));
      if (z === zoomRef.current) return;
      anchor.current = {
        ...(at ?? { x: viewport.clientWidth / 2, y: viewport.clientHeight / 2 }),
        // Wheel steps can outrun a render: scale from the last laid-out zoom.
        from: anchor.current?.from ?? zoomRef.current,
      };
      zoomRef.current = z;
      setZoom(z);
    },
    [viewport],
  );

  useLayoutEffect(() => {
    const a = anchor.current;
    if (!viewport || !a) return;
    anchor.current = null;
    const k = zoom / a.from;
    viewport.scrollTo({
      left: (viewport.scrollLeft + a.x) * k - a.x,
      top: (viewport.scrollTop + a.y) * k - a.y,
    });
  }, [viewport, zoom]);

  // Non-passive, so the wheel zooms the video instead of scrolling the page.
  useEffect(() => {
    if (!viewport) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = viewport.getBoundingClientRect();
      zoomTo(zoomRef.current * Math.exp(-e.deltaY * 0.002), {
        x: e.clientX - r.left,
        y: e.clientY - r.top,
      });
    };
    viewport.addEventListener('wheel', onWheel, { passive: false });
    return () => viewport.removeEventListener('wheel', onWheel);
  }, [viewport, zoomTo]);

  const pan = (e: ReactPointerEvent) => {
    if (!viewport) return;
    e.preventDefault();
    e.stopPropagation();
    const start = {
      x: e.clientX,
      y: e.clientY,
      left: viewport.scrollLeft,
      top: viewport.scrollTop,
    };
    trackPointer(e, (ev) => {
      viewport.scrollTo({
        left: start.left - (ev.clientX - start.x),
        top: start.top - (ev.clientY - start.y),
      });
    });
  };

  return { zoom, zoomTo, pan, attach: setViewport };
}
