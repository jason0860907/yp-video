import { useCallback, useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react';
import { cn } from '@/lib/cn';
import type { ActionEvent, ActionRally, WaveformData } from '@/types/api';

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const fmt = (s: number) => {
  if (!Number.isFinite(s)) return '0:00';
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, '0')}`;
};

// Window length shown across the timeline. spv = 0 means "All" (whole video).
const ZOOMS: Array<{ label: string; spv: number }> = [
  { label: 'All', spv: 0 },
  { label: '10m', spv: 600 },
  { label: '5m', spv: 300 },
  { label: '3m', spv: 180 },
  { label: '30s', spv: 30 },
  { label: '10s', spv: 10 },
];

// Waveform amplitude shaping (ported from the legacy editor).
const WAVE_SCALE_PERCENTILE = 0.98;
const WAVE_SCALE_HEADROOM = 1.35;
const WAVE_MIN_SCALE = 0.02;
const WAVE_VERTICAL_FILL = 0.4;
const WAVE_PEAK_GAIN = 0.55;
const WAVE_RMS_GAIN = 1.15;
const MAX_CANVAS_PX = 32000;

function waveScaleAmp(values: number[]): number {
  const active = values.filter((v) => v > 0.001).sort((a, b) => a - b);
  if (!active.length) return 0.03;
  const idx = clamp(Math.floor((active.length - 1) * WAVE_SCALE_PERCENTILE), 0, active.length - 1);
  return Math.max(WAVE_MIN_SCALE, active[idx]! * WAVE_SCALE_HEADROOM);
}

interface ActionTimelineProps {
  duration: number;
  fps: number;
  numFrames: number;
  frame: number;
  rallies: ActionRally[];
  events: ActionEvent[];
  selectedRallyId: number | 'all';
  selectedIdx: number;
  waveform: WaveformData;
  colors: Record<string, string>;
  onSeekFrame: (f: number) => void;
  onJumpEvent: (idx: number) => void;
}

/** Zoomable, horizontally-scrollable Action Label timeline. A rally-band lane
 *  (labelled R1, R2 …) sits above an audio waveform lane; both share one scroll
 *  container so they stay aligned, with a single playhead through both. Windowed
 *  zoom levels scroll to follow the playhead. */
export function ActionTimeline({
  duration,
  fps,
  numFrames,
  frame,
  rallies,
  events,
  selectedRallyId,
  selectedIdx,
  waveform,
  colors,
  onSeekFrame,
  onJumpEvent,
}: ActionTimelineProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [viewW, setViewW] = useState(0);
  const [spv, setSpv] = useState(0);

  const ready = duration > 0 && numFrames > 0;
  const time = fps ? frame / fps : 0;

  // Default to a 3-minute window the first time we meet a long video.
  useEffect(() => {
    if (duration > 0) setSpv((cur) => (cur === 0 && duration > 200 ? 180 : cur));
  }, [duration]);

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
  const xOf = (t: number) => t * pxPerSec;

  // Follow the playhead when zoomed in: keep it on screen.
  useEffect(() => {
    if (spv === 0 || !pxPerSec) return;
    const scroll = scrollRef.current;
    if (!scroll) return;
    const x = time * pxPerSec;
    const left = scroll.scrollLeft;
    const vw = scroll.clientWidth;
    if (x < left + 8 || x > left + vw - 8) scroll.scrollLeft = Math.max(0, x - vw / 2);
  }, [time, pxPerSec, spv]);

  const drawWave = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const cssW = Math.max(1, Math.floor(canvas.clientWidth));
    const cssH = Math.max(1, Math.floor(canvas.clientHeight));
    const dpr = Math.min(window.devicePixelRatio || 1, MAX_CANVAS_PX / cssW);
    const pw = Math.max(1, Math.floor(cssW * dpr));
    const ph = Math.max(1, Math.floor(cssH * dpr));
    if (canvas.width !== pw || canvas.height !== ph) {
      canvas.width = pw;
      canvas.height = ph;
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    const center = cssH / 2;

    const bg = ctx.createLinearGradient(0, 0, 0, cssH);
    bg.addColorStop(0, 'rgba(56,189,248,0.12)');
    bg.addColorStop(1, 'rgba(249,115,22,0.06)');
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, cssW, cssH);
    ctx.strokeStyle = 'rgba(125,125,135,0.20)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, center);
    ctx.lineTo(cssW, center);
    ctx.stroke();

    if (!waveform.hasAudio || !waveform.peaks.length) return;
    const peaks = waveform.peaks;
    const rms = waveform.rms.length === peaks.length ? waveform.rms : peaks;
    const scaleAmp = waveScaleAmp(rms);
    const halfHeight = Math.max(8, cssH - 8) * WAVE_VERTICAL_FILL;

    const valueAtPixel = (values: number[], x: number) => {
      if (!values.length) return 0;
      if (values.length < cssW * 1.5) {
        const pos = (x / Math.max(1, cssW - 1)) * Math.max(0, values.length - 1);
        const left = Math.floor(pos);
        const right = Math.min(values.length - 1, left + 1);
        const mix = pos - left;
        return (values[left] || 0) * (1 - mix) + (values[right] || 0) * mix;
      }
      const from = Math.floor((x / cssW) * values.length);
      const to = Math.max(from + 1, Math.ceil(((x + 1) / cssW) * values.length));
      let value = 0;
      for (let i = from; i < to; i += 1) value = Math.max(value, values[i] || 0);
      return value;
    };
    const compressedAmp = (value: number, mult = 1) => Math.sqrt(clamp((value * mult) / scaleAmp, 0, 1));
    const fillEnvelope = (values: number[], gain: number, style: string | CanvasGradient) => {
      ctx.fillStyle = style;
      ctx.beginPath();
      ctx.moveTo(0, center);
      for (let x = 0; x < cssW; x += 1) ctx.lineTo(x, center - compressedAmp(valueAtPixel(values, x), gain) * halfHeight);
      for (let x = cssW - 1; x >= 0; x -= 1) ctx.lineTo(x, center + compressedAmp(valueAtPixel(values, x), gain) * halfHeight);
      ctx.closePath();
      ctx.fill();
    };

    fillEnvelope(peaks, WAVE_PEAK_GAIN, 'rgba(56,189,248,0.12)');
    const rmsGradient = ctx.createLinearGradient(0, 0, cssW, 0);
    rmsGradient.addColorStop(0, 'rgba(56,189,248,0.72)');
    rmsGradient.addColorStop(0.58, 'rgba(129,140,248,0.76)');
    rmsGradient.addColorStop(1, 'rgba(249,115,22,0.68)');
    fillEnvelope(rms, WAVE_RMS_GAIN, rmsGradient);
  }, [waveform, cssWidth]);

  useEffect(() => {
    drawWave();
  }, [drawWave]);

  const seek = (e: ReactPointerEvent) => {
    if ((e.target as HTMLElement).closest('[data-marker]')) return;
    if (!ready || !pxPerSec) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const t = (e.clientX - rect.left) / pxPerSec;
    onSeekFrame(clamp(Math.round(t * fps), 0, numFrames - 1));
  };

  const status = waveform.loading
    ? 'Loading audio…'
    : waveform.error
      ? 'Audio unavailable'
      : !waveform.hasAudio || !waveform.peaks.length
        ? 'No audio'
        : 'Audio';

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between gap-2">
        <span className="font-mono text-[11px] tabular-nums text-text-muted">
          {ready ? `${rallies.length} rally · ${spv === 0 ? `0:00–${fmt(duration)}` : `${spv >= 60 ? `${spv / 60}m` : `${spv}s`} window`}` : ''}
        </span>
        <div className="flex items-center gap-1">
          {ZOOMS.map((z) => (
            <button
              key={z.label}
              type="button"
              onClick={() => setSpv(z.spv)}
              className={cn(
                'rounded-md px-2 py-0.5 font-mono text-[10px] transition-colors',
                spv === z.spv ? 'bg-primary text-on-primary' : 'bg-surface-50 text-text-muted hover:text-text-secondary',
              )}
            >
              {z.label}
            </button>
          ))}
        </div>
      </div>
      <div ref={scrollRef} className="overflow-x-auto overflow-y-hidden rounded-lg border border-border bg-surface-200/30">
        <div className="relative cursor-pointer select-none" style={{ width: cssWidth }} onPointerDown={seek}>
          {/* Rally lane */}
          <div className="relative h-8 border-b border-border">
            {rallies.map((rally, i) => {
              const left = xOf(rally.start);
              const width = Math.max(2, xOf(rally.end) - left);
              const active = rally.rally_id === selectedRallyId;
              return (
                <div
                  key={rally.rally_id}
                  className={cn('absolute inset-y-1 flex items-center justify-center overflow-hidden rounded-sm border-x text-[10px] font-mono', active ? 'bg-primary/[0.22] text-primary-text' : 'bg-primary/[0.09] text-text-muted')}
                  style={{ left, width, borderColor: active ? 'rgb(var(--primary) / 0.55)' : 'rgb(var(--primary) / 0.2)' }}
                  title={`R${i + 1} ${fmt(rally.start)}–${fmt(rally.end)}`}
                >
                  {width > 16 && `R${i + 1}`}
                </div>
              );
            })}
            {events
              .map((e, idx) => ({ e, idx }))
              .filter(({ e }) => selectedRallyId === 'all' || e.rally_id === selectedRallyId)
              .map(({ e, idx }) => {
                const color = colors[e.label] || '#8E8E93';
                const active = idx === selectedIdx;
                return (
                  <button
                    key={e.id}
                    data-marker
                    type="button"
                    onClick={(ev) => {
                      ev.stopPropagation();
                      onJumpEvent(idx);
                    }}
                    title={`${e.label} · frame ${e.frame}`}
                    className={cn('absolute top-1/2 -translate-y-1/2 rounded-full border border-black/50 transition-transform', active ? '-ml-[7px] h-5 w-3.5 scale-105' : '-ml-px h-4 w-1.5 hover:scale-125')}
                    style={{ left: xOf(fps ? e.frame / fps : 0), background: e.visible ? color : 'transparent', borderColor: e.visible ? 'rgba(0,0,0,0.5)' : color }}
                  />
                );
              })}
          </div>
          {/* Waveform lane */}
          <div className="relative h-16">
            <canvas ref={canvasRef} className="block h-full" style={{ width: cssWidth }} />
            <span className="pointer-events-none absolute left-2 top-1.5 font-mono text-[10px] uppercase tracking-wide text-text-muted">{status}</span>
          </div>
          {/* Playhead through both lanes */}
          {ready && <div className="pointer-events-none absolute inset-y-0 w-0.5 -translate-x-1/2 bg-ink/80" style={{ left: xOf(time) }} />}
        </div>
      </div>
    </div>
  );
}
