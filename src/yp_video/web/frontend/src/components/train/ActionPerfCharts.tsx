import { useMemo } from 'react';
import { Card } from '@/components/ui/Card';
import { SectionLabel } from '@/components/ui/SectionLabel';
import type { ActionPerfData, ActionVideoMap } from '@/types/api';

// Three related metrics, not arbitrary categories: harmonic is the headline
// (the checkpoint criterion), temporal/spatial are the two components it is the
// harmonic mean of. Legend + hover carry identity so colour is never alone.
const METRICS = [
  { key: 'harmonic', label: 'Harmonic', color: '#facc15' },
  { key: 'temporal', label: 'Temporal', color: '#22d3ee' },
  { key: 'spatial', label: 'Spatial', color: '#34d399' },
] as const;

type MetricKey = (typeof METRICS)[number]['key'];

const FIELD: Record<MetricKey, keyof import('@/types/api').ActionPerfEntry> = {
  harmonic: 'val_mAP',
  temporal: 'val_mAP_temporal',
  spatial: 'val_mAP_spatial',
};

/** Shorten a long broadcast filename to something a bar row can show. */
function shortLabel(video: string): string {
  const setSuffix = video.match(/_set\d+$/)?.[0] ?? '';
  const stem = video.replace(/_set\d+$/, '');
  const head = ((stem.split('｜')[0] ?? stem).split('|')[0] ?? stem).trim();
  return (head.length > 26 ? head.slice(0, 25) + '…' : head) + setSuffix;
}

export function ActionPerfCharts({ data }: { data: ActionPerfData }) {
  const entries = useMemo(() => (data.entries ?? []).filter((e) => typeof e.val_mAP === 'number'), [data.entries]);

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

  if (!entries.length) return null;

  const hasDetail = perVideo.length > 0 || perClass.length > 0;

  return (
    <Card>
      <SectionLabel>Validation performance{data.run ? ` · ${data.run}` : ''}</SectionLabel>
      <EpochChart entries={entries} bestEpoch={data.best?.epoch} />
      {perClass.length > 0 && <PerClassChart rows={perClass} bestEpoch={data.best?.epoch} />}
      {perVideo.length > 0 && <PerVideoChart rows={perVideo} bestEpoch={data.best?.epoch} />}
      {!hasDetail && (
        <p className="mt-4 text-[11px] text-text-muted">
          Per-action and per-video mAP appear here for runs trained after this feature landed (older runs don't record it).
        </p>
      )}
    </Card>
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
          const w = (r.mAP / max) * 100;
          return (
            <div key={r.label} className="flex items-center gap-2" title={`${r.label} · temporal mAP ${pct.toFixed(1)}%`}>
              <span className="w-24 flex-shrink-0 truncate font-mono text-[11px] capitalize text-text-secondary">{r.label}</span>
              <div className="relative h-4 flex-1 overflow-hidden rounded bg-surface-100">
                <div className="absolute inset-y-0 left-0 rounded" style={{ width: `${Math.max(w, 1.5)}%`, background: '#22d3ee', opacity: 0.85 }} />
              </div>
              <span className="w-12 flex-shrink-0 text-right font-mono text-[11px] tabular-nums text-text-primary">{pct.toFixed(1)}%</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function EpochChart({ entries, bestEpoch }: { entries: ActionPerfData['entries']; bestEpoch?: number }) {
  const W = 720;
  const H = 260;
  const pad = { t: 16, r: 16, b: 34, l: 40 };
  const cw = W - pad.l - pad.r;
  const ch = H - pad.t - pad.b;
  const eps = entries.map((e) => e.epoch);
  const lrByEpoch = new Map(entries.map((e) => [e.epoch, typeof e.lr === 'number' ? e.lr : null]));
  const xMin = Math.min(...eps);
  const xMax = Math.max(...eps);
  const xRange = xMax - xMin || 1;
  let maxVal = 0;
  for (const e of entries) for (const m of METRICS) { const v = e[FIELD[m.key]] as number | undefined; if (typeof v === 'number' && v > maxVal) maxVal = v; }
  maxVal = Math.min(Math.ceil(maxVal * 10) / 10 + 0.1, 1) || 1;
  const x = (ep: number) => pad.l + ((ep - xMin) / xRange) * cw;
  const y = (v: number) => pad.t + (1 - v / maxVal) * ch;
  const ySteps = 5;
  const xStep = Math.max(1, Math.floor(eps.length / 8));

  return (
    <>
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1">
        {METRICS.map((m) => {
          const last = entries[entries.length - 1]?.[FIELD[m.key]] as number | undefined;
          return (
            <span key={m.key} className="inline-flex items-center gap-1.5 text-[11px] text-text-secondary">
              <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: m.color }} />
              {m.label}
              {typeof last === 'number' && <span className="font-mono tabular-nums text-text-muted">{(last * 100).toFixed(1)}%</span>}
            </span>
          );
        })}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="mt-1 w-full" style={{ maxHeight: 300 }}>
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
        {METRICS.map((m) => {
          const pts = entries
            .map((e) => ({ ep: e.epoch, v: e[FIELD[m.key]] as number | undefined }))
            .filter((p): p is { ep: number; v: number } => typeof p.v === 'number')
            .map((p) => ({ ...p, X: x(p.ep), Y: y(p.v) }));
          if (!pts.length) return null;
          const d = pts.map((p, j) => `${j === 0 ? 'M' : 'L'}${p.X},${p.Y}`).join(' ');
          return (
            <g key={m.key}>
              <path d={d} fill="none" stroke={m.color} strokeWidth={2} opacity={0.9} />
              {pts.map((p) => (
                <circle key={p.ep} cx={p.X} cy={p.Y} r={2.5} fill={m.color} opacity={0.9}>
                  <title>{`Epoch ${p.ep} · ${m.label} ${(p.v * 100).toFixed(1)}%${lrByEpoch.get(p.ep) != null ? ` · lr ${lrByEpoch.get(p.ep)!.toExponential(2)}` : ''}`}</title>
                </circle>
              ))}
            </g>
          );
        })}
      </svg>
    </>
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
          const w = (r.harmonic / max) * 100;
          return (
            <div key={r.video} className="flex items-center gap-2" title={`${r.video}\nharmonic ${pct.toFixed(1)}% · temporal ${(r.temporal * 100).toFixed(1)}% · spatial ${(r.spatial * 100).toFixed(1)}% · ${r.events} events`}>
              <span className="w-40 flex-shrink-0 truncate font-mono text-[10.5px] text-text-secondary">{shortLabel(r.video)}</span>
              <div className="relative h-4 flex-1 overflow-hidden rounded bg-surface-100">
                <div className="absolute inset-y-0 left-0 rounded" style={{ width: `${Math.max(w, 1.5)}%`, background: '#facc15', opacity: 0.85 }} />
              </div>
              <span className="w-12 flex-shrink-0 text-right font-mono text-[11px] tabular-nums text-text-primary">{pct.toFixed(1)}%</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
