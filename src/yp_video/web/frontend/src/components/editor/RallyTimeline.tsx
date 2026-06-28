import { useEffect, useMemo, useRef, useState, type RefObject } from 'react';
import { cn } from '@/lib/cn';
import type { EditorAnnotation } from './AnnotationEditor';

interface RallyTimelineProps {
  videoRef: RefObject<HTMLVideoElement>;
  annotations: EditorAnnotation[];
  duration: number;
  markStart: number | null;
  onSeek: (t: number) => void;
}

// Window length shown across the timeline. secondsPerView = 0 means "All"
// (the whole video). Terse range-selector tokens, consistent style.
const ZOOMS: Array<{ label: string; spv: number }> = [
  { label: 'All', spv: 0 },
  { label: '10m', spv: 600 },
  { label: '5m', spv: 300 },
  { label: '3m', spv: 180 },
];

/** Zoomable, horizontally-scrollable rally timeline. Segments are labelled
 *  R1, R2 …; the windowed zoom levels scroll to follow the playhead so a long
 *  match isn't squashed into one screen. */
export function RallyTimeline({ videoRef, annotations, duration, markStart, onSeek }: RallyTimelineProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [viewW, setViewW] = useState(0);
  // Default to a 3-minute window once the video is longer than that.
  const [spv, setSpv] = useState(0);
  const userScrolledRef = useRef(false);

  useEffect(() => {
    // Pick a sensible default zoom the first time we learn the duration.
    if (duration > 0) setSpv((cur) => (cur === 0 && duration > 200 ? 180 : cur));
  }, [duration]);

  // Track the scroll-container width.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setViewW(el.clientWidth));
    ro.observe(el);
    setViewW(el.clientWidth);
    return () => ro.disconnect();
  }, []);

  const pxPerSec = useMemo(() => {
    if (!duration || !viewW) return 0;
    return spv === 0 ? viewW / duration : viewW / spv;
  }, [duration, viewW, spv]);
  const cssWidth = duration && pxPerSec ? duration * pxPerSec : viewW;

  // Refs the draw loop reads so it never closes over stale state.
  const annRef = useRef(annotations);
  annRef.current = annotations;
  const markRef = useRef(markStart);
  markRef.current = markStart;
  const ppsRef = useRef(pxPerSec);
  ppsRef.current = pxPerSec;
  const spvRef = useRef(spv);
  spvRef.current = spv;

  useEffect(() => {
    let raf = 0;
    const draw = () => {
      raf = requestAnimationFrame(draw);
      const canvas = canvasRef.current;
      const scroll = scrollRef.current;
      const el = videoRef.current;
      const pps = ppsRef.current;
      if (!canvas || !scroll || !pps) return;
      const dpr = window.devicePixelRatio || 1;
      const cssW = canvas.clientWidth;
      const cssH = canvas.clientHeight;
      if (canvas.width !== cssW * dpr || canvas.height !== cssH * dpr) {
        canvas.width = cssW * dpr;
        canvas.height = cssH * dpr;
      }
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = 'rgba(255,255,255,0.02)';
      ctx.fillRect(0, 0, w, h);

      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.font = `${11 * dpr}px 'IBM Plex Mono', ui-monospace, monospace`;
      annRef.current.forEach((a, i) => {
        const x1 = a.start * pps * dpr;
        const x2 = a.end * pps * dpr;
        const grad = ctx.createLinearGradient(x1, 0, x1, h);
        grad.addColorStop(0, 'rgba(52,199,89,0.55)');
        grad.addColorStop(1, 'rgba(52,199,89,0.28)');
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.roundRect(x1, 2 * dpr, Math.max(1, x2 - x1), h - 4 * dpr, 3 * dpr);
        ctx.fill();
        if (x2 - x1 > 16 * dpr) {
          ctx.fillStyle = 'rgba(255,255,255,0.9)';
          ctx.fillText(`R${i + 1}`, (x1 + x2) / 2, h / 2);
        }
      });

      if (el && !isNaN(el.currentTime)) {
        const px = el.currentTime * pps * dpr;
        ctx.fillStyle = '#FFFFFF';
        ctx.shadowColor = 'rgba(255,255,255,0.55)';
        ctx.shadowBlur = 6 * dpr;
        ctx.fillRect(px - dpr, 0, 2 * dpr, h);
        ctx.shadowBlur = 0;

        // Follow the playhead: when windowed, keep it on screen.
        if (spvRef.current !== 0 && !el.paused) {
          const pxCss = el.currentTime * pps;
          const left = scroll.scrollLeft;
          const vw = scroll.clientWidth;
          if (pxCss < left + 8 || pxCss > left + vw - 8) {
            scroll.scrollLeft = Math.max(0, pxCss - vw / 2);
          }
        }
      }

      if (markRef.current != null) {
        const mx = markRef.current * pps * dpr;
        ctx.fillStyle = '#E8B23A';
        ctx.shadowColor = 'rgba(232,178,58,0.6)';
        ctx.shadowBlur = 6 * dpr;
        ctx.fillRect(mx - dpr, 0, 2 * dpr, h);
        ctx.shadowBlur = 0;
      }
    };
    draw();
    return () => cancelAnimationFrame(raf);
  }, [videoRef]);

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-end gap-1">
        {ZOOMS.map((z) => (
          <button
            key={z.label}
            type="button"
            onClick={() => {
              userScrolledRef.current = false;
              setSpv(z.spv);
            }}
            className={cn(
              'rounded-md px-2 py-0.5 font-mono text-[10px] transition-colors',
              spv === z.spv ? 'bg-primary text-on-primary' : 'bg-surface-50 text-text-muted hover:text-text-secondary',
            )}
          >
            {z.label}
          </button>
        ))}
      </div>
      <div
        ref={scrollRef}
        className="overflow-x-auto overflow-y-hidden rounded-2xl ring-1 ring-white/[0.06]"
        style={{ background: 'linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01))' }}
        onScroll={() => (userScrolledRef.current = true)}
      >
        <canvas
          ref={canvasRef}
          className="block h-12 cursor-pointer"
          style={{ width: cssWidth }}
          title="Click to seek"
          onClick={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            if (duration && rect.width) onSeek(((e.clientX - rect.left) / rect.width) * duration);
          }}
        />
      </div>
    </div>
  );
}
