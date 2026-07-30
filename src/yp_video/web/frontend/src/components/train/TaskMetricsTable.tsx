import { cn } from '@/lib/cn';
import type {
  ActionPerfEntry,
  TaskMetricPhase,
  TaskMetrics,
} from '@/types/api';

const TASK_LABELS: Record<string, string> = {
  action: 'Action',
  location: 'Location',
  actor: 'Actor',
  rally: 'Rally',
};

const METRIC_LABELS: Record<string, string> = {
  harmonic_mAP: 'Harmonic mAP',
  temporal_mAP: 'Temporal mAP',
  spatial_mAP: 'Spatial mAP',
  overall_top1: 'Overall Top-1',
  player_top1: 'Player Top-1',
  occluded_recall: 'Occluded recall',
  untracked_recall: 'Untracked recall',
  loss: 'Loss',
};

const SERIES_COLORS = [
  'text-primary-light',
  'text-accent-light',
  'text-text-secondary',
  'text-amber-400',
] as const;

function primaryValue(
  phase: TaskMetricPhase,
  metric: string,
): number | null {
  if (metric === 'loss') return phase.loss;
  const value = phase.metrics[metric];
  return typeof value === 'number' ? value : null;
}

function formatMetric(value: number | null, metric: string): string {
  if (value == null || !Number.isFinite(value)) return '—';
  return metric === 'loss'
    ? value.toFixed(4)
    : `${(value * 100).toFixed(2)}%`;
}

function denominator(phase: TaskMetricPhase, metric: string): number | null {
  const preferred =
    metric === 'player_top1'
      ? phase.counts.player_events
      : phase.counts.events ?? phase.counts.samples;
  return typeof preferred === 'number' && Number.isFinite(preferred)
    ? preferred
    : null;
}

/** Task-agnostic view over the common SPOT task metrics contract. */
export function TaskMetricsTable({
  latest,
  best,
  title = 'Task validation',
}: {
  latest?: TaskMetrics;
  best?: TaskMetrics;
  title?: string;
}) {
  const names = Object.keys(latest ?? {});
  if (!names.length) return null;

  return (
    <div className="mt-3 overflow-hidden rounded-lg border border-border">
      <div className="border-b border-border bg-surface-50 px-3 py-2 text-[10px] font-semibold uppercase tracking-widest text-text-muted">
        {title}
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[580px] text-[11px]">
          <thead className="bg-surface-50 text-text-muted">
            <tr>
              <th className="px-3 py-1.5 text-left font-normal">Task</th>
              <th className="px-3 py-1.5 text-left font-normal">Primary metric</th>
              <th className="px-3 py-1.5 text-right font-normal">Latest</th>
              <th className="px-3 py-1.5 text-right font-normal">Best checkpoint</th>
              <th className="px-3 py-1.5 text-right font-normal">Train loss</th>
              <th className="px-3 py-1.5 text-right font-normal">Val loss</th>
              <th className="px-3 py-1.5 text-right font-normal">Validation N</th>
            </tr>
          </thead>
          <tbody>
            {names.map((name) => {
              const task = latest![name]!;
              const bestTask = best?.[name];
              const metric = task.primary_metric;
              return (
                <tr key={name} className="border-t border-border/60">
                  <td className="px-3 py-2 font-medium text-text-primary">
                    {TASK_LABELS[name] ?? name}
                  </td>
                  <td className="px-3 py-2 text-text-muted">
                    {METRIC_LABELS[metric] ?? metric}
                  </td>
                  <td className="px-3 py-2 text-right font-mono tabular-nums text-text-primary">
                    {formatMetric(primaryValue(task.validation, metric), metric)}
                  </td>
                  <td className="px-3 py-2 text-right font-mono tabular-nums text-text-secondary">
                    {bestTask
                      ? formatMetric(
                          primaryValue(bestTask.validation, bestTask.primary_metric),
                          bestTask.primary_metric,
                        )
                      : '—'}
                  </td>
                  <td className="px-3 py-2 text-right font-mono tabular-nums text-text-secondary">
                    {task.train.loss == null ? '—' : task.train.loss.toFixed(4)}
                  </td>
                  <td className="px-3 py-2 text-right font-mono tabular-nums text-text-secondary">
                    {task.validation.loss == null
                      ? '—'
                      : task.validation.loss.toFixed(4)}
                  </td>
                  <td className="px-3 py-2 text-right font-mono tabular-nums text-text-muted">
                    {denominator(task.validation, metric)?.toLocaleString() ?? '—'}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

interface MetricPoint {
  epoch: number;
  value: number;
}

interface MetricSeries {
  metric: string;
  label: string;
  colorClass: string;
  points: MetricPoint[];
}

function metricValue(
  entry: ActionPerfEntry,
  task: string,
  metric: string,
): number | null {
  const phase = entry.tasks?.[task]?.validation;
  if (!phase) return null;
  if (metric === 'loss') return phase.loss;
  const value = phase.metrics[metric];
  return typeof value === 'number' ? value : null;
}

function taskSeries(
  entries: ActionPerfEntry[],
  task: string,
): MetricSeries[] {
  const snapshot = [...entries]
    .reverse()
    .map((entry) => entry.tasks?.[task])
    .find(Boolean);
  if (!snapshot) return [];

  const scalarMetrics =
    snapshot.primary_metric === 'loss'
      ? ['loss']
      : Object.entries(snapshot.validation.metrics)
          .filter(([, value]) => typeof value === 'number')
          .map(([metric]) => metric)
          .sort((a, b) => {
            if (a === snapshot.primary_metric) return -1;
            if (b === snapshot.primary_metric) return 1;
            return a.localeCompare(b);
          });

  return scalarMetrics.map((metric, index) => ({
    metric,
    label: METRIC_LABELS[metric] ?? metric,
    colorClass: SERIES_COLORS[index % SERIES_COLORS.length]!,
    points: entries
      .map((entry) => ({
        epoch: entry.epoch,
        value: metricValue(entry, task, metric),
      }))
      .filter(
        (point): point is MetricPoint =>
          typeof point.value === 'number' && Number.isFinite(point.value),
      ),
  }));
}

/** Per-epoch charts over the same common task contract as the summary table. */
export function TaskMetricHistory({
  entries,
  bestEpoch,
  excludeTasks = [],
}: {
  entries: ActionPerfEntry[];
  bestEpoch?: number;
  excludeTasks?: string[];
}) {
  const excluded = new Set(excludeTasks);
  const tasks = [
    ...new Set(
      entries.flatMap((entry) => Object.keys(entry.tasks ?? {})),
    ),
  ].filter((task) => !excluded.has(task));

  const charts = tasks
    .map((task) => ({ task, series: taskSeries(entries, task) }))
    .filter((chart) => chart.series.some((series) => series.points.length));
  if (!charts.length) return null;

  return (
    <div className="mt-4 space-y-4 border-t border-border pt-4">
      <div className="text-xs font-semibold text-text-primary">
        Task metric history
      </div>
      {charts.map(({ task, series }) => (
        <TaskHistoryChart
          key={task}
          task={task}
          series={series}
          bestEpoch={bestEpoch}
        />
      ))}
    </div>
  );
}

function TaskHistoryChart({
  task,
  series,
  bestEpoch,
}: {
  task: string;
  series: MetricSeries[];
  bestEpoch?: number;
}) {
  const points = series.flatMap((item) => item.points);
  const epochs = [...new Set(points.map((point) => point.epoch))].sort(
    (a, b) => a - b,
  );
  const percentScale = series.every((item) => item.metric !== 'loss');
  const maxObserved = Math.max(0, ...points.map((point) => point.value));
  const yMax = percentScale
    ? 1
    : Math.max(0.1, Math.ceil(maxObserved * 10) / 10);
  const width = 720;
  const height = 210;
  const pad = { top: 12, right: 16, bottom: 28, left: 42 };
  const chartWidth = width - pad.left - pad.right;
  const chartHeight = height - pad.top - pad.bottom;
  const xMin = epochs[0] ?? 0;
  const xMax = epochs[epochs.length - 1] ?? xMin;
  const xRange = xMax - xMin || 1;
  const x = (epoch: number) =>
    pad.left + ((epoch - xMin) / xRange) * chartWidth;
  const y = (value: number) =>
    pad.top + (1 - value / yMax) * chartHeight;
  const xStep = Math.max(1, Math.floor(epochs.length / 8));

  return (
    <div className="rounded-lg border border-border bg-surface-50 p-3">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <span className="text-xs font-medium text-text-primary">
          {TASK_LABELS[task] ?? task}
        </span>
        <div className="flex flex-wrap gap-3">
          {series.map((item) => {
            const latest = item.points[item.points.length - 1]?.value;
            return (
              <span
                key={item.metric}
                className="flex items-center gap-1.5 text-[10px] text-text-muted"
              >
                <span
                  className={cn(
                    'h-1.5 w-1.5 rounded-full bg-current',
                    item.colorClass,
                  )}
                />
                {item.label}
                {latest != null ? (
                  <span className="font-mono tabular-nums text-text-secondary">
                    {formatMetric(latest, item.metric)}
                  </span>
                ) : null}
              </span>
            );
          })}
        </div>
      </div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full"
        preserveAspectRatio="xMinYMid meet"
      >
        {Array.from({ length: 5 }, (_, index) => {
          const value = (yMax / 4) * index;
          const yy = y(value);
          return (
            <g key={index}>
              <line
                x1={pad.left}
                x2={pad.left + chartWidth}
                y1={yy}
                y2={yy}
                stroke="currentColor"
                className="text-text-muted"
                strokeOpacity={0.12}
              />
              <text
                x={pad.left - 6}
                y={yy + 3}
                textAnchor="end"
                className="fill-text-muted"
                fontSize={10}
              >
                {percentScale
                  ? `${(value * 100).toFixed(0)}%`
                  : value.toFixed(1)}
              </text>
            </g>
          );
        })}
        {epochs.map((epoch, index) =>
          index % xStep === 0 ? (
            <text
              key={epoch}
              x={x(epoch)}
              y={height - 6}
              textAnchor="middle"
              className="fill-text-muted"
              fontSize={10}
            >
              {epoch}
            </text>
          ) : null,
        )}
        {typeof bestEpoch === 'number' &&
        bestEpoch >= xMin &&
        bestEpoch <= xMax ? (
          <line
            x1={x(bestEpoch)}
            x2={x(bestEpoch)}
            y1={pad.top}
            y2={pad.top + chartHeight}
            stroke="currentColor"
            className="text-text-muted"
            strokeOpacity={0.35}
            strokeDasharray="3 3"
          />
        ) : null}
        {series.map((item) => {
          const chartPoints = item.points.map((point) => ({
            ...point,
            px: x(point.epoch),
            py: y(point.value),
          }));
          const path = chartPoints
            .map(
              (point, index) =>
                `${index === 0 ? 'M' : 'L'}${point.px},${point.py}`,
            )
            .join(' ');
          return (
            <g key={item.metric} className={item.colorClass}>
              <path
                d={path}
                fill="none"
                stroke="currentColor"
                strokeWidth={2}
                opacity={0.9}
              />
              {chartPoints.map((point) => (
                <circle
                  key={point.epoch}
                  cx={point.px}
                  cy={point.py}
                  r={2.5}
                  fill="currentColor"
                >
                  <title>
                    {`Epoch ${point.epoch} · ${item.label} ${formatMetric(point.value, item.metric)}`}
                  </title>
                </circle>
              ))}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
