import { useMemo } from 'react';
import { cn } from '@/lib/cn';
import { Card } from '@/components/ui/Card';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { METRIC_LABELS, TASK_LABELS, TASK_ORDER } from '@/components/train/metricLabels';
import { TaskBreakdownPanel } from '@/components/train/TaskBreakdown';
import { TaskMetricsTable } from '@/components/train/TaskMetricsTable';
import type { TrainEpochRecord, TrainPerfData } from '@/types/api';

// Colors are theme tokens (applied as text-* + currentColor) so the charts
// follow the switchable brand palette instead of clashing with it. The
// overview and every per-task panel draw from the same list, in the same
// order, so a task's primary metric keeps its color from chart to chart.
const SERIES_COLORS = [
  'text-primary-light',
  'text-accent-light',
  'text-text-secondary',
  'text-amber-400',
  'text-rose-400',
] as const;

interface Point {
  ep: number;
  v: number;
}

interface Series {
  key: string;
  label: string;
  colorClass: string;
  pts: Point[];
  best: Point;
}

function toSeries(
  m: { key: string; label: string; colorClass: string },
  pts: Point[],
): Series {
  const best = pts.reduce((a, b) => (b.v > a.v ? b : a), pts[0] ?? { ep: 0, v: 0 });
  return { key: m.key, label: m.label, colorClass: m.colorClass, pts, best };
}

/** Per-epoch series for one head: every scalar validation metric it reports,
 *  primary metric first so it takes the headline color. */
function buildTaskSeries(entries: TrainEpochRecord[], task: string): Series[] {
  const snapshot = [...entries]
    .reverse()
    .map((e) => e.tasks[task])
    .find(Boolean);
  if (!snapshot) return [];

  const metrics = Object.keys(snapshot.validation.metrics).sort((a, b) => {
    if (a === snapshot.primary_metric) return -1;
    if (b === snapshot.primary_metric) return 1;
    return a.localeCompare(b);
  });

  return metrics
    .map((metric, index) =>
      toSeries(
        {
          key: metric,
          label: METRIC_LABELS[metric] ?? metric,
          colorClass: SERIES_COLORS[index % SERIES_COLORS.length]!,
        },
        entries
          .map((e) => ({ ep: e.epoch, v: e.tasks[task]?.validation.metrics[metric] }))
          .filter((p): p is Point => typeof p.v === 'number' && Number.isFinite(p.v)),
      ),
    )
    .filter((s) => s.pts.length > 0);
}

/** One series per task — its primary metric per epoch. The cross-task
 *  overview: five heads, five lines, same 0-1 axis. */
function buildOverviewSeries(entries: TrainEpochRecord[]): Series[] {
  const lastTasks = entries[entries.length - 1]?.tasks ?? {};
  const names = TASK_ORDER.filter((t) => t in lastTasks);
  return names
    .map((task, index) => {
      const primary = lastTasks[task]!.primary_metric;
      return toSeries(
        {
          key: task,
          label: TASK_LABELS[task] ?? task,
          colorClass: SERIES_COLORS[index % SERIES_COLORS.length]!,
        },
        entries
          .map((e) => ({ ep: e.epoch, v: e.tasks[task]?.validation.metrics[primary] }))
          .filter((pnt): pnt is Point => typeof pnt.v === 'number' && Number.isFinite(pnt.v)),
      );
    })
    .filter((s) => s.pts.length > 0);
}

export function TrainPerfCard({
  data,
  onSelectRun,
}: {
  data: TrainPerfData;
  onSelectRun: (run: string) => void;
}) {
  const entries = data.entries;
  const overview = useMemo(() => buildOverviewSeries(entries), [entries]);
  const taskCharts = useMemo(() => {
    const names = TASK_ORDER.filter((t) => t in (entries[entries.length - 1]?.tasks ?? {}));
    return (
      names
        .filter((t) => t !== 'location') // location's spatial line rides with action below
        .map((task) => {
          let taskSeries = buildTaskSeries(entries, task);
          if (task === 'action') {
            // Spatial mAP is recorded under the location head but is the other
            // half of action's harmonic — chart them together.
            const spatial = buildTaskSeries(entries, 'location').map((sp, i) => ({
              ...sp,
              colorClass: SERIES_COLORS[(taskSeries.length + i) % SERIES_COLORS.length]!,
            }));
            taskSeries = [...taskSeries, ...spatial];
          }
          return { task, series: taskSeries };
        })
        // A one-line panel duplicates its overview line; only multi-metric
        // panels add information.
        .filter((c) => c.series.length > 1)
    );
  }, [entries]);
  const latestEntry = entries[entries.length - 1];
  const bestEntry = entries.find((entry) => entry.epoch === data.best?.epoch) ?? latestEntry;

  if (!latestEntry || !overview.length) return null;

  const charts: Array<{ title: string; series: Series[]; bestEpoch?: number }> = [
    { title: 'Primary metric by task', series: overview, bestEpoch: data.best?.epoch },
    ...taskCharts.map(({ task, series }) => ({
      title: `${TASK_LABELS[task] ?? task} · validation`,
      series,
      bestEpoch: series[0]!.best.ep,
    })),
  ];

  const runs = data.runs ?? [];
  const bestValue = data.best?.value;
  const subtitle = [
    data.run,
    typeof bestValue === 'number'
      ? `best ${latestEntry.selection.metric.endsWith('loss') ? bestValue.toFixed(4) : `${(bestValue * 100).toFixed(1)}%`}${
          data.best?.epoch != null ? ` @ ep${data.best.epoch}` : ''
        }`
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
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        {charts.map((chart) => (
          <EpochChartPanel
            key={chart.title}
            title={chart.title}
            series={chart.series}
            entries={entries}
            bestEpoch={chart.bestEpoch}
            className={charts.length === 1 ? 'xl:col-span-2' : undefined}
          />
        ))}
      </div>
      <TaskMetricsTable
        latest={latestEntry.tasks}
        best={bestEntry?.tasks}
        title="Task validation metrics"
      />
      <TaskBreakdownPanel
        best={bestEntry ? { epoch: bestEntry.epoch, tasks: bestEntry.tasks } : undefined}
        latest={{ epoch: latestEntry.epoch, tasks: latestEntry.tasks }}
      />
    </Card>
  );
}

/** One epoch chart with its title and best-value legend — the single visual
 *  style every per-task panel in the card shares. */
function EpochChartPanel({
  title,
  series,
  entries,
  bestEpoch,
  className,
}: {
  title: string;
  series: Series[];
  entries: TrainEpochRecord[];
  bestEpoch?: number;
  className?: string;
}) {
  return (
    <div className={cn('min-w-0', className)}>
      <div className="mb-1.5 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <span className="text-xs font-semibold text-text-primary">{title}</span>
        <div className="flex flex-wrap gap-x-3 gap-y-1">
          {series.map((s) => (
            <span key={s.key} className="flex items-center gap-1.5 whitespace-nowrap">
              <span className={cn('h-2 w-2 flex-shrink-0 rounded-full bg-current', s.colorClass)} />
              <span className="text-[11px] text-text-muted">{s.label}</span>
              <span className="text-xs font-medium tabular-nums text-text-primary">{(s.best.v * 100).toFixed(1)}%</span>
              <span className="text-[10px] text-text-muted">ep{s.best.ep}</span>
            </span>
          ))}
        </div>
      </div>
      <EpochChart series={series} entries={entries} bestEpoch={bestEpoch} />
    </div>
  );
}

function EpochChart({ series, entries, bestEpoch }: { series: Series[]; entries: TrainEpochRecord[]; bestEpoch?: number }) {
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
