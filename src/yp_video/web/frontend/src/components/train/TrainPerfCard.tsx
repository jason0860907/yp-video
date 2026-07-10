import { useMemo } from 'react';
import { cn } from '@/lib/cn';
import { Card } from '@/components/ui/Card';
import { SectionLabel } from '@/components/ui/SectionLabel';
import type { ActionPerfData, ActionPerfEntry, ActionVideoMap } from '@/types/api';

// Three related metrics, not arbitrary categories: harmonic is the headline
// (the checkpoint criterion), temporal/spatial are the two components it is the
// harmonic mean of. Rally (segment) runs degenerate — spatial is all zero and
// temporal equals harmonic — so those series are dropped and the single
// remaining line is just called "mAP".
//
// Colors are theme tokens (applied as text-* + currentColor) so the charts
// follow the switchable brand palette instead of clashing with it.
const METRICS = [
  { key: 'harmonic', label: 'Harmonic', colorClass: 'text-primary-light' },
  { key: 'temporal', label: 'Temporal', colorClass: 'text-accent-light' },
  { key: 'spatial', label: 'Spatial', colorClass: 'text-text-secondary' },
] as const;

type MetricKey = (typeof METRICS)[number]['key'];

const FIELD: Record<MetricKey, keyof ActionPerfEntry> = {
  harmonic: 'val_mAP',
  temporal: 'val_mAP_temporal',
  spatial: 'val_mAP_spatial',
};

interface Point {
  ep: number;
  v: number;
}

interface Series {
  key: MetricKey;
  label: string;
  colorClass: string;
  pts: Point[];
  best: Point;
}

function toSeries(m: (typeof METRICS)[number], pts: Point[]): Series {
  const best = pts.reduce((a, b) => (b.v > a.v ? b : a), pts[0] ?? { ep: 0, v: 0 });
  return { key: m.key, label: m.label, colorClass: m.colorClass, pts, best };
}

function buildSeries(entries: ActionPerfEntry[]): Series[] {
  const all = METRICS.map((m) =>
    toSeries(
      m,
      entries
        .map((e) => ({ ep: e.epoch, v: e[FIELD[m.key]] as number | undefined }))
        .filter((p): p is Point => typeof p.v === 'number'),
    ),
  );
  let series = all.filter((s) => s.pts.some((p) => p.v > 0));
  const harmonic = series.find((s) => s.key === 'harmonic');
  const temporal = series.find((s) => s.key === 'temporal');
  if (
    harmonic &&
    temporal &&
    harmonic.pts.length === temporal.pts.length &&
    harmonic.pts.every((p, i) => p.ep === temporal.pts[i]!.ep && p.v === temporal.pts[i]!.v)
  ) {
    series = series.filter((s) => s.key !== 'temporal');
    harmonic.label = 'mAP';
  }
  return series;
}

/** Shorten a long broadcast filename to something a bar row can show.
 *  Prefer the segment naming the teams ("A vs. B") over the date/time head. */
function shortLabel(video: string): string {
  const setSuffix = video.match(/_set\d+$/)?.[0] ?? '';
  const stem = video.replace(/_set\d+$/, '');
  const parts = stem.split(/[｜|]/).map((s) => s.trim()).filter(Boolean);
  const head = parts.find((p) => /vs/i.test(p)) ?? parts[0] ?? stem;
  return (head.length > 42 ? head.slice(0, 41) + '…' : head) + setSuffix;
}

export function TrainPerfCard({ data, onSelectRun }: { data: ActionPerfData; onSelectRun: (run: string) => void }) {
  const entries = useMemo(() => (data.entries ?? []).filter((e) => typeof e.val_mAP === 'number'), [data.entries]);
  const series = useMemo(() => buildSeries(entries), [entries]);

  // Per-video comes from the best epoch (fall back to the latest epoch that has it).
  const perVideo = useMemo<ActionVideoMap[]>(() => {
    const withPv = entries.filter((e) => Array.isArray(e.val_per_video) && e.val_per_video.length);
    const bestEp = data.best?.epoch;
    const pick = withPv.find((e) => e.epoch === bestEp) ?? withPv[withPv.length - 1];
    if (!pick) return [];
    return [...(pick.val_per_video ?? [])].sort((a, b) => b.harmonic - a.harmonic);
  }, [entries, data.best?.epoch]);

  // Per-class (per-action) temporal mAP at the best epoch, strongest first.
  const perClass = useMemo<Array<{ label: string; mAP: number }>>(() => {
    const withPc = entries.filter((e) => e.per_class && Object.keys(e.per_class).length);
    const bestEp = data.best?.epoch;
    const pick = withPc.find((e) => e.epoch === bestEp) ?? withPc[withPc.length - 1];
    if (!pick?.per_class) return [];
    return Object.entries(pick.per_class)
      .map(([label, mAP]) => ({ label, mAP }))
      .sort((a, b) => b.mAP - a.mAP);
  }, [entries, data.best?.epoch]);

  if (!entries.length || !series.length) return null;

  const runs = data.runs ?? [];
  const hasDetail = perVideo.length > 0 || perClass.length > 0;
  const subtitle = [
    data.run,
    data.best && typeof data.best.value === 'number'
      ? `best mAP ${(data.best.value * 100).toFixed(1)}%${data.best.epoch != null ? ` @ ep${data.best.epoch}` : ''}`
      : null,
  ]
    .filter(Boolean)
    .join(' · ');

  return (
    <Card>
      <SectionLabel>Validation performance{subtitle ? ` · ${subtitle}` : ''}</SectionLabel>
      {runs.length > 1 && (
        <div className="mb-3 flex flex-wrap gap-2">
          {runs.map((r) => {
            const active = r === data.run;
            return (
              <button
                key={r}
                type="button"
                onClick={() => onSelectRun(r)}
                className={cn(
                  'rounded-lg border px-3 py-1.5 font-mono text-xs font-medium transition-colors',
                  active ? 'border-primary/25 bg-primary/15 text-primary-light' : 'border-border bg-surface-50 text-text-secondary hover:text-text-primary',
                )}
              >
                {r}
              </button>
            );
          })}
        </div>
      )}
      <div className="grid items-center gap-8 lg:grid-cols-[minmax(0,3fr)_minmax(0,1fr)]">
        <EpochChart series={series} entries={entries} bestEpoch={data.best?.epoch} />
        <div className="space-y-2.5">
          {series.map((s) => (
            <div key={s.key} className="flex items-center gap-2 whitespace-nowrap">
              <span className={cn('h-2 w-2 flex-shrink-0 rounded-full bg-current', s.colorClass)} />
              <span className="text-[11px] text-text-muted">{s.label}</span>
              <span className="text-xs font-medium tabular-nums text-text-primary">{(s.best.v * 100).toFixed(1)}%</span>
              <span className="text-[10px] text-text-muted">ep{s.best.ep}</span>
            </div>
          ))}
        </div>
      </div>
      {/* A single class (rally runs) carries no comparison — the headline line already shows it. */}
      {perClass.length > 1 && <PerClassChart rows={perClass} bestEpoch={data.best?.epoch} />}
      {perVideo.length > 0 && <PerVideoChart rows={perVideo} bestEpoch={data.best?.epoch} />}
      {!hasDetail && (
        <p className="mt-4 text-[11px] text-text-muted">
          Per-action and per-video mAP appear here for runs trained after this feature landed (older runs don't record it).
        </p>
      )}
    </Card>
  );
}

// Matches components/job/ProgressBar.tsx so metric bars read as the same
// element family as every other progress bar in the app.
function Bar({ frac, colorClass }: { frac: number; colorClass: string }) {
  return (
    <div className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-ink/[0.06]">
      <div className={cn('absolute inset-y-0 left-0 rounded-full bg-current opacity-85', colorClass)} style={{ width: `${Math.max(frac * 100, 1.5)}%` }} />
    </div>
  );
}

function PerClassChart({ rows, bestEpoch }: { rows: Array<{ label: string; mAP: number }>; bestEpoch?: number }) {
  const max = Math.max(0.001, ...rows.map((r) => r.mAP));
  return (
    <div className="mt-4">
      <div className="mb-1.5 text-xs font-semibold text-text-primary">
        Per-action temporal mAP{typeof bestEpoch === 'number' ? ` · best epoch ${bestEpoch}` : ''}
      </div>
      <div className="space-y-1.5">
        {rows.map((r) => {
          const pct = r.mAP * 100;
          return (
            <div key={r.label} className="flex items-center gap-2" title={`${r.label} · temporal mAP ${pct.toFixed(1)}%`}>
              <span className="w-72 flex-shrink-0 truncate font-mono text-[11px] capitalize text-text-secondary">{r.label}</span>
              <Bar frac={r.mAP / max} colorClass="text-accent-light" />
              <span className="w-12 flex-shrink-0 text-right font-mono text-[11px] tabular-nums text-text-primary">{pct.toFixed(1)}%</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function EpochChart({ series, entries, bestEpoch }: { series: Series[]; entries: ActionPerfEntry[]; bestEpoch?: number }) {
  const lrByEpoch = new Map(entries.map((e) => [e.epoch, typeof e.lr === 'number' ? e.lr : null]));
  const W = 720;
  const H = 260;
  const pad = { t: 16, r: 16, b: 34, l: 40 };
  const cw = W - pad.l - pad.r;
  const ch = H - pad.t - pad.b;
  const eps = [...new Set(series.flatMap((s) => s.pts.map((p) => p.ep)))].sort((a, b) => a - b);
  const xMin = Math.min(...eps);
  const xMax = Math.max(...eps);
  const xRange = xMax - xMin || 1;
  let maxVal = Math.max(...series.map((s) => s.best.v));
  maxVal = Math.min(Math.ceil(maxVal * 10) / 10 + 0.1, 1) || 1;
  const x = (ep: number) => pad.l + ((ep - xMin) / xRange) * cw;
  const y = (v: number) => pad.t + (1 - v / maxVal) * ch;
  const ySteps = 5;
  const xStep = Math.max(1, Math.floor(eps.length / 8));

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxHeight: 280 }} preserveAspectRatio="xMinYMid meet">
      {Array.from({ length: ySteps + 1 }, (_, i) => {
        const val = (maxVal / ySteps) * i;
        const yy = y(val);
        return (
          <g key={i}>
            <line x1={pad.l} x2={pad.l + cw} y1={yy} y2={yy} stroke="currentColor" className="text-text-muted" strokeOpacity={0.12} />
            <text x={pad.l - 6} y={yy + 3} textAnchor="end" className="fill-text-muted" fontSize={10}>{(val * 100).toFixed(0)}%</text>
          </g>
        );
      })}
      {eps.map((ep, i) => (i % xStep === 0 ? (
        <text key={ep} x={x(ep)} y={H - 8} textAnchor="middle" className="fill-text-muted" fontSize={10}>{ep}</text>
      ) : null))}
      {typeof bestEpoch === 'number' && bestEpoch >= xMin && bestEpoch <= xMax && (
        <line x1={x(bestEpoch)} x2={x(bestEpoch)} y1={pad.t} y2={pad.t + ch} stroke="currentColor" className="text-text-muted" strokeOpacity={0.35} strokeDasharray="3 3" />
      )}
      {series.map((s) => {
        const pts = s.pts.map((p) => ({ ...p, X: x(p.ep), Y: y(p.v) }));
        const d = pts.map((p, j) => `${j === 0 ? 'M' : 'L'}${p.X},${p.Y}`).join(' ');
        return (
          <g key={s.key} className={s.colorClass}>
            <path d={d} fill="none" stroke="currentColor" strokeWidth={2} opacity={0.9} />
            {pts.map((p) => (
              <circle key={p.ep} cx={p.X} cy={p.Y} r={2.5} fill="currentColor" opacity={0.9}>
                <title>{`Epoch ${p.ep} · ${s.label} ${(p.v * 100).toFixed(1)}%${lrByEpoch.get(p.ep) != null ? ` · lr ${lrByEpoch.get(p.ep)!.toExponential(2)}` : ''}`}</title>
              </circle>
            ))}
          </g>
        );
      })}
    </svg>
  );
}

function PerVideoChart({ rows, bestEpoch }: { rows: ActionVideoMap[]; bestEpoch?: number }) {
  const max = Math.max(0.001, ...rows.map((r) => r.harmonic));
  return (
    <div className="mt-4">
      <div className="mb-1.5 text-xs font-semibold text-text-primary">
        Per-video harmonic mAP{typeof bestEpoch === 'number' ? ` · best epoch ${bestEpoch}` : ''}
      </div>
      <div className="space-y-1.5">
        {rows.map((r) => {
          const pct = r.harmonic * 100;
          return (
            <div key={r.video} className="flex items-center gap-2" title={`${r.video}\nharmonic ${pct.toFixed(1)}% · temporal ${(r.temporal * 100).toFixed(1)}% · spatial ${(r.spatial * 100).toFixed(1)}% · ${r.events} events`}>
              <span className="w-72 flex-shrink-0 truncate font-mono text-[10.5px] text-text-secondary">{shortLabel(r.video)}</span>
              <Bar frac={r.harmonic / max} colorClass="text-primary-light" />
              <span className="w-12 flex-shrink-0 text-right font-mono text-[11px] tabular-nums text-text-primary">{pct.toFixed(1)}%</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
