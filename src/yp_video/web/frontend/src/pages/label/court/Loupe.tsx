/** A magnified view around the pointer while marking or drawing guides —
 *  the App's calibration loupe: a circle in the top corner away from the
 *  pointer, the frame four times larger than shown, a crosshair on the
 *  exact spot, and the guide lines, their crossings and the marks drawn
 *  over it, so a corner can be placed to the pixel. */

import { useEffect, useRef, type RefObject } from 'react';
import { extendToFrame, type Point, type Segment } from './geometry';

const SIZE = 168;
/** Times the frame as shown on screen. */
const MAGNIFICATION = 4;

export function Loupe({
  video,
  frame,
  point,
  guides,
  crossings,
  marks,
  margin,
}: {
  video: RefObject<HTMLVideoElement | null>;
  /** The frame's element: its on-screen size sets the magnification. */
  frame: RefObject<HTMLElement | null>;
  /** The pointer, in normalized frame coordinates. */
  point: Point;
  guides: Segment[];
  crossings: Point[];
  marks: Point[];
  margin: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Redrawn every animation frame while shown, so a playing video and a
  // guide being dragged stay live.
  useEffect(() => {
    let raf = 0;
    const draw = () => {
      raf = requestAnimationFrame(draw);
      const canvas = canvasRef.current;
      const ctx = canvas?.getContext('2d');
      const rect = frame.current?.getBoundingClientRect();
      const source = video.current;
      if (!canvas || !ctx || !rect?.width || !rect.height) return;
      const dpr = window.devicePixelRatio || 1;
      if (canvas.width !== SIZE * dpr) {
        canvas.width = SIZE * dpr;
        canvas.height = SIZE * dpr;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, SIZE, SIZE);

      // Screen pixels per frame unit, magnified; the pointer at the centre.
      const sx = rect.width * MAGNIFICATION;
      const sy = rect.height * MAGNIFICATION;
      const at = ([x, y]: Point): [number, number] => [
        SIZE / 2 + (x - point[0]) * sx,
        SIZE / 2 + (y - point[1]) * sy,
      ];

      if (source?.videoWidth) {
        const [left, top] = at([0, 0]);
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(source, left, top, sx, sy);
      }

      ctx.lineWidth = 1.5;
      ctx.strokeStyle = '#22d3ee';
      for (const g of guides) {
        const full = extendToFrame(g, margin);
        if (!full) continue;
        const [a, b] = [at(full[0]), at(full[1])];
        ctx.beginPath();
        ctx.moveTo(...a);
        ctx.lineTo(...b);
        ctx.stroke();
      }
      ctx.strokeStyle = '#f472b6';
      ctx.lineWidth = 2;
      for (const c of crossings) {
        const [x, y] = at(c);
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, 2 * Math.PI);
        ctx.stroke();
      }
      ctx.strokeStyle = '#facc15';
      for (const m of marks) {
        const [x, y] = at(m);
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, 2 * Math.PI);
        ctx.stroke();
      }

      // The crosshair, open at the centre so the spot itself shows.
      const c = SIZE / 2;
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (const [dx, dy] of [
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1],
      ] as const) {
        ctx.moveTo(c + dx * 5, c + dy * 5);
        ctx.lineTo(c + dx * 16, c + dy * 16);
      }
      ctx.stroke();
    };
    draw();
    return () => cancelAnimationFrame(raf);
  }, [video, frame, point, guides, crossings, marks, margin]);

  return (
    <canvas
      ref={canvasRef}
      className="pointer-events-none rounded-full shadow-lg ring-2 ring-white"
      style={{ width: SIZE, height: SIZE }}
    />
  );
}
