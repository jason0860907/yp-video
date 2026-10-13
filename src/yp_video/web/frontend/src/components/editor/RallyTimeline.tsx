import { useEffect, useMemo, useRef, useState, type RefObject } from 'react';
import { cn } from '@/lib/cn';
import { useTheme } from '@/lib/theme';
import type { EditorAnnotation } from './AnnotationEditor';
import { DEFAULT_SPV, ZOOMS } from './timelineZoom';

// "45 95 63" → "45,95,63" for rgba(); falls back to court green.
const rgbList = (v: string, fallback: string) => (v.trim() ? v.trim().split(/\s+/).join(',') : fallback);

interface RallyTimelineProps {
  videoRef: RefObject<HTMLVideoElement>;
  annotations: EditorAnnotation[];
  duration: number;
  markStart: number | null;
  onSeek: (t: number) => void;
}


/** Zoomable, horizontally-scrollable rally timeline. Segments are labelled
 *  R1, R2 …; the windowed zoom levels scroll to follow the playhead so a long
 *  match isn't squashed into one screen. */
export function RallyTimeline({ videoRef, annotations, duration, markStart, onSeek }: RallyTimelineProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [viewW, setViewW] = useState(0);
  // Open at the default window once the video is longer than that.
  const [spv, setSpv] = useState(0);
  const lastTimeRef = useRef<number | null>(null);

  useEffect(() => {
    // Pick a sensible default zoom the first time we learn the duration.
    if (duration > 0) setSpv((cur) => (cur === 0 && duration > DEFAULT_SPV ? DEFAULT_SPV : cur));
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

  // Brand colors read from CSS vars so the timeline follows theme + palette.
  const { theme, palette } = useTheme();
  const brandRef = useRef({ primary: '45,95,63', accent: '232,178,58', fg: '#ffffff' });
  useEffect(() => {
    const cs = getComputedStyle(document.documentElement);
    brandRef.current = {
      primary: rgbList(cs.getPropertyValue('--primary'), '45,95,63'),
      accent: rgbList(cs.getPropertyValue('--accent'), '232,178,58'),
      fg: cs.getPropertyValue('--text-primary').trim() || '#ffffff',
    };
  }, [theme, palette]);

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
      // The canvas covers only the visible window and draws the timeline
      // shifted by the scroll offset: a canvas as wide as the whole timeline
      // (20k px for a 15-minute match at 30 s) exceeds the GPU texture limit
      // and renders blank.
      const off = scroll.scrollLeft;
      const cssW = canvas.clientWidth;
      const cssH = canvas.clientHeight;
      if (canvas.width !== cssW * dpr || canvas.height !== cssH * dpr) {
        canvas.width = cssW * dpr;
        canvas.height = cssH * dpr;
      }
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      const brand = brandRef.current;
      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = 'rgba(255,255,255,0.02)';
      ctx.fillRect(0, 0, w, h);

      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.font = `${11 * dpr}px 'IBM Plex Mono', ui-monospace, monospace`;
      annRef.current.forEach((a, i) => {
        const x1 = (a.start * pps - off) * dpr;
        const x2 = (a.end * pps - off) * dpr;
        if (x2 < 0 || x1 > w) return;
        const grad = ctx.createLinearGradient(x1, 0, x1, h);
        grad.addColorStop(0, `rgba(${brand.primary},0.6)`);
        grad.addColorStop(1, `rgba(${brand.primary},0.3)`);
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.roundRect(x1, 2 * dpr, Math.max(1, x2 - x1), h - 4 * dpr, 3 * dpr);
        ctx.fill();
        if (x2 - x1 > 16 * dpr) {
          ctx.fillStyle = brand.fg;
          ctx.fillText(`R${i + 1}`, (x1 + x2) / 2, h / 2);
        }
      });

      if (el && !isNaN(el.currentTime)) {
        const px = (el.currentTime * pps - off) * dpr;
        ctx.fillStyle = brand.fg;
        ctx.shadowColor = brand.fg;
        ctx.shadowBlur = 6 * dpr;
        ctx.fillRect(px - dpr, 0, 2 * dpr, h);
        ctx.shadowBlur = 0;

        // Follow the playhead: when windowed, keep it on screen — during
        // playback and after a seek (opening a video lands on its first rally,
        // clicking a row jumps). A playhead that has not moved leaves a manual
        // scroll alone.
        const moved = el.currentTime !== lastTimeRef.current;
        lastTimeRef.current = el.currentTime;
        if (spvRef.current !== 0 && moved) {
          const pxCss = el.currentTime * pps;
          const left = scroll.scrollLeft;
          const vw = scroll.clientWidth;
          if (pxCss < left + 8 || pxCss > left + vw - 8) {
            scroll.scrollLeft = Math.max(0, pxCss - vw / 2);
          }
        }
      }

      if (markRef.current != null) {
        const mx = (markRef.current * pps - off) * dpr;
        ctx.fillStyle = `rgb(${brand.accent})`;
        ctx.shadowColor = `rgba(${brand.accent},0.6)`;
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
              // A new scale moves the playhead's pixel: re-centre on it.
              lastTimeRef.current = null;
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
      >
        {/* The spacer gives the scroll its full width; the canvas stays
            pinned over the visible window. */}
        <div style={{ width: cssWidth }}>
          <canvas
            ref={canvasRef}
            className="sticky left-0 block h-12 cursor-pointer"
            style={{ width: viewW }}
            title="Click to seek"
            onClick={(e) => {
              const scroll = scrollRef.current;
              if (!duration || !pxPerSec || !scroll) return;
              const x = e.clientX - e.currentTarget.getBoundingClientRect().left + scroll.scrollLeft;
              onSeek(Math.min(duration, x / pxPerSec));
            }}
          />
        </div>
      </div>
    </div>
  );
}
