import { TaskBreakdownPanel } from '@/components/train/TaskBreakdown';
import { TaskMetricsTable } from '@/components/train/TaskMetricsTable';
import type { TrainProgress } from '@/types/api';

const fmtMetric = (v: unknown) => (Number.isFinite(Number(v)) ? Number(v).toFixed(4) : '');

/** Live metric tiles, the task table and each head's breakdown for a running training job. */
export function TrainDetail({
  progress: p,
  epochsFallback,
  mapLabel = 'Last mAP',
}: {
  progress?: TrainProgress;
  epochsFallback: number;
  mapLabel?: string;
}) {
  if (!p) return null;
  const phaseProgress = Number.isFinite(Number(p.phase_progress)) ? `${Math.round(Number(p.phase_progress) * 100)}%` : '';
  const step =
    Number.isFinite(Number(p.step)) && Number.isFinite(Number(p.total)) ? `${p.step}/${p.total}${phaseProgress ? ` (${phaseProgress})` : ''}` : '';
  const latestMap = Number.isFinite(Number(p.latest_val_map)) ? `${(Number(p.latest_val_map) * 100).toFixed(2)}%` : '';
  const bestVal = Number.isFinite(Number(p.best_value))
    ? Number(p.best_value) <= 1
      ? (Number(p.best_value) * 100).toFixed(2) + '%'
      : Number(p.best_value).toFixed(4)
    : '';
  const best = bestVal ? `${bestVal}${p.best_epoch != null ? ` · Epoch ${Number(p.best_epoch) + 1}` : ''}` : '';
  const rows: Array<[string, string]> = [
    ['Epoch', `${p.epoch_display || 1}/${p.epochs || epochsFallback || '?'}`],
    ['Phase', p.phase_label || p.phase || ''],
    ['Step', step],
    ['Cur loss', fmtMetric(p.current_loss)],
    ['Last train', fmtMetric(p.latest_train_loss)],
    ['Last val', fmtMetric(p.latest_val_loss)],
    [mapLabel, latestMap],
    ['Best', best],
  ];
  const visible = rows.filter(([, v]) => v !== '' && v != null);
  if (!visible.length) return null;
  return (
    <>
      <div className="mt-3 grid grid-cols-2 gap-2 md:grid-cols-4">
        {visible.map(([label, value]) => (
          <div key={label} className="min-w-0 rounded-lg border border-border bg-surface-100 px-2.5 py-2">
            <div className="text-[9px] uppercase tracking-wider text-text-muted">{label}</div>
            <div className="mt-0.5 truncate font-mono text-[11px] tabular-nums text-text-secondary" title={value}>
              {value}
            </div>
          </div>
        ))}
      </div>
      <TaskMetricsTable
        latest={p.latest_task_metrics}
        best={p.best_task_metrics}
      />
      <TaskBreakdownPanel
        best={p.best_task_metrics && p.best_epoch != null ? { epoch: p.best_epoch, tasks: p.best_task_metrics } : undefined}
        latest={p.latest_task_metrics && p.epoch != null ? { epoch: p.epoch, tasks: p.latest_task_metrics } : undefined}
      />
    </>
  );
}
