import { useMemo, useState } from 'react';
import { cn } from '@/lib/cn';
import { Card } from '@/components/ui/Card';
import { SectionLabel } from '@/components/ui/SectionLabel';
import type { PerfData } from '@/types/api';

const TIOU_COLORS: Record<string, string> = {
  '0.30': '#22d3ee',
  '0.40': '#34d399',
  '0.50': '#facc15',
  '0.60': '#fb923c',
  '0.70': '#f87171',
};

interface SeriesPoint {
  epoch: number;
  values: Array<number | null>;
}

function pointsForSource(data: PerfData, src: string, tiouKeys: string[]): SeriesPoint[] {
  const entries = data.entries ?? [];
  if (src === 'general') {
    return entries.filter((e) => e.tiou).map((e) => ({ epoch: e.epoch, values: tiouKeys.map((k) => e.tiou?.[k]?.mAP ?? null) }));
  }
  return entries
    .filter((e) => e.per_source?.[src]?.tiou_mAP)
    .map((e) => ({ epoch: e.epoch, values: e.per_source![src]!.tiou_mAP! }));
}

/** TAD validation mAP per epoch, one line per tIoU threshold. */
export function PerformanceChart({ data }: { data: PerfData }) {
  const sources = useMemo(() => {
    const set = new Set<string>();
    for (const e of data.entries ?? []) for (const s of Object.keys(e.per_source ?? {})) set.add(s);
    return [...set].sort();
  }, [data]);
  const [source, setSource] = useState('general');
  const activeSource = source !== 'general' && !sources.includes(source) ? 'general' : source;

  const sample = (data.entries ?? []).find((e) => e.tiou && Object.keys(e.tiou).length > 0);
  if (!sample) return null;
  const tiouKeys = Object.keys(sample.tiou!).sort();
  const series = pointsForSource(data, activeSource, tiouKeys);
  if (!series.length) return null;

  const epochs = series.map((p) => p.epoch);
  let maxVal = 0;
  for (const p of series) for (const v of p.values) if (typeof v === 'number' && v > maxVal) maxVal = v;
  maxVal = Math.min(Math.ceil(maxVal * 10) / 10 + 0.1, 1) || 1;

  const W = 700;
  const H = 260;
  const pad = { t: 20, r: 20, b: 40, l: 50 };
  const cw = W - pad.l - pad.r;
  const ch = H - pad.t - pad.b;
  const xMin = Math.min(...epochs);
  const xMax = Math.max(...epochs);
  const xRange = xMax - xMin || 1;
  const x = (ep: number) => pad.l + ((ep - xMin) / xRange) * cw;
  const y = (val: number) => pad.t + (1 - val / maxVal) * ch;

  const best = tiouKeys.map((tiou, i) => {
    let b = 0;
    let ep = 0;
    for (const p of series) {
      const v = p.values[i];
      if (typeof v === 'number' && v > b) {
        b = v;
        ep = p.epoch;
      }
    }
    return { tiou, color: TIOU_COLORS[tiou] || '#888', best: b, epoch: ep };
  });

  let bestOverall = 0;
  let bestOverallEp = 0;
  for (const p of series) {
    const valid = p.values.filter((v): v is number => typeof v === 'number');
    if (!valid.length) continue;
    const mean = valid.reduce((a, b2) => a + b2, 0) / valid.length;
    if (mean > bestOverall) {
      bestOverall = mean;
      bestOverallEp = p.epoch;
    }
  }
  const latest = (data.entries ?? [])[(data.entries ?? []).length - 1]?.per_source ?? {};
  const srcMeta = activeSource !== 'general' ? latest[activeSource] : null;
  const subtitle = [
    data.name || '',
    activeSource === 'general' ? 'aggregated mAP' : `${activeSource} · ${srcMeta?.n_videos ?? '?'} val videos`,
    `best mAP ${(bestOverall * 100).toFixed(1)}% @ ep${bestOverallEp}`,
  ]
    .filter(Boolean)
    .join(' · ');

  const ySteps = 5;
  const xStep = Math.max(1, Math.floor(epochs.length / 6));

  return (
    <Card>
      <SectionLabel>Performance · {subtitle}</SectionLabel>
      <div className="mb-3 flex flex-wrap gap-2">
        {['general', ...sources].map((s) => {
          const active = s === activeSource;
          const m = s !== 'general' ? latest[s] : null;
          return (
            <button
              key={s}
              type="button"
              onClick={() => setSource(s)}
              className={cn(
                'rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors',
                active ? 'border-primary/25 bg-primary/15 text-primary-light' : 'border-border bg-surface-50 text-text-secondary hover:text-text-primary',
              )}
            >
              {s === 'general' ? 'General' : s.toUpperCase()}
              {m && typeof m.mAP === 'number' && <span className="ml-1 text-text-muted">{(m.mAP * 100).toFixed(1)}</span>}
            </button>
          );
        })}
      </div>
      <div className="grid items-center gap-8 lg:grid-cols-[3fr_1fr]">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxHeight: 280 }}>
          {Array.from({ length: ySteps + 1 }, (_, i) => {
            const val = (maxVal / ySteps) * i;
            const yy = y(val);
            return (
              <g key={i}>
                <line x1={pad.l} x2={pad.l + cw} y1={yy} y2={yy} stroke="currentColor" className="text-text-muted" strokeOpacity={0.12} />
                <text x={pad.l - 8} y={yy + 3} textAnchor="end" className="fill-text-muted" fontSize={10}>
                  {(val * 100).toFixed(0)}%
                </text>
              </g>
            );
          })}
          {epochs.map((ep, i) =>
            i % xStep === 0 ? (
              <text key={ep} x={x(ep)} y={H - 8} textAnchor="middle" className="fill-text-muted" fontSize={10}>
                {ep}
              </text>
            ) : null,
          )}
          {tiouKeys.map((tiou, i) => {
            const color = TIOU_COLORS[tiou] || '#888';
            const pts = series
              .filter((p) => typeof p.values[i] === 'number')
              .map((p) => ({ x: x(p.epoch), y: y(p.values[i] as number), epoch: p.epoch, mAP: p.values[i] as number }));
            if (!pts.length) return null;
            const d = pts.map((p, j) => `${j === 0 ? 'M' : 'L'}${p.x},${p.y}`).join(' ');
            return (
              <g key={tiou}>
                <path d={d} fill="none" stroke={color} strokeWidth={2} opacity={0.85} />
                {pts.map((p) => (
                  <circle key={p.epoch} cx={p.x} cy={p.y} r={3} fill={color} opacity={0.9}>
                    <title>{`Epoch ${p.epoch} | tIoU ${tiou} | mAP ${(p.mAP * 100).toFixed(1)}%`}</title>
                  </circle>
                ))}
              </g>
            );
          })}
        </svg>
        <div className="space-y-2.5">
          {best.map((b) => (
            <div key={b.tiou} className="flex items-center gap-2 whitespace-nowrap">
              <span className="h-2 w-2 flex-shrink-0 rounded-full" style={{ background: b.color }} />
              <span className="text-[11px] text-text-muted">tIoU={parseFloat(b.tiou)}</span>
              <span className="text-xs font-medium tabular-nums text-text-primary">{(b.best * 100).toFixed(1)}%</span>
              <span className="text-[10px] text-text-muted">ep{b.epoch}</span>
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
}
