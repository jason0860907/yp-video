import { METRIC_LABELS } from '@/components/train/metricLabels';
import type { TaskMetricPhase, TaskMetrics } from '@/types/api';

const TASK_LABELS: Record<string, string> = {
  action: 'Action',
  location: 'Location',
  actor: 'Actor',
  rally: 'Rally',
};

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
