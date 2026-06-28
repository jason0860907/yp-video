import { cn } from '@/lib/cn';
import type { JobItem } from '@/types/api';

const STATUS_CLS: Record<string, string> = {
  completed: 'text-emerald-400 bg-emerald-500/10',
  failed: 'text-red-400 bg-red-500/10',
  running: 'text-primary-light bg-primary/10',
  cancelled: 'text-amber-400 bg-amber-500/10',
  pending: 'text-text-muted bg-ink/5',
};
const BAR_CLS: Record<string, string> = {
  completed: 'bg-emerald-400',
  failed: 'bg-red-400',
  running: 'bg-primary-light',
  cancelled: 'bg-amber-400',
  pending: 'bg-ink/20',
};

const clsFor = (map: Record<string, string>, status: string | undefined) =>
  map[status ?? 'pending'] ?? map.pending!;

interface JobItemsProps {
  items: JobItem[];
  maxVisible?: number;
}

/** Per-video sub-progress for batch jobs (job.params.items). */
export function JobItems({ items, maxVisible = 12 }: JobItemsProps) {
  if (!items.length) return null;

  const counts = {
    completed: items.filter((i) => i.status === 'completed').length,
    failed: items.filter((i) => i.status === 'failed').length,
    running: items.filter((i) => i.status === 'running').length,
    pending: items.filter((i) => i.status === 'pending').length,
    cancelled: items.filter((i) => i.status === 'cancelled').length,
  };
  const visible = items.slice(0, maxVisible);
  const more = items.length - visible.length;

  return (
    <div className="mt-2 space-y-2">
      <div className="flex flex-wrap gap-x-3 gap-y-1 text-[10px] tabular-nums text-text-muted">
        <span>
          {counts.completed}/{items.length} done
        </span>
        {counts.running > 0 && <span>{counts.running} running</span>}
        {counts.pending > 0 && <span>{counts.pending} pending</span>}
        {counts.failed > 0 && <span className="text-red-400">{counts.failed} failed</span>}
        {counts.cancelled > 0 && <span className="text-amber-400">{counts.cancelled} cancelled</span>}
      </div>
      <div className="max-h-56 space-y-1 overflow-y-auto pr-1">
        {visible.map((item, idx) => {
          const pct = Math.round(Math.max(0, Math.min(Number(item.progress ?? 0), 1)) * 100);
          return (
            <div key={item.video ?? idx} className="grid grid-cols-[minmax(0,1fr)_3.25rem] items-center gap-2 py-1">
              <div className="min-w-0 space-y-1">
                <div className="flex min-w-0 items-center gap-2">
                  <span className={cn('shrink-0 rounded px-1.5 py-0.5 text-[9px] uppercase tracking-wide', clsFor(STATUS_CLS, item.status))}>
                    {item.status ?? 'pending'}
                  </span>
                  <span className="min-w-0 truncate text-[10px] text-text-secondary" title={item.video ?? ''}>
                    {item.video ?? ''}
                  </span>
                </div>
                <div className="h-1 overflow-hidden rounded-full bg-ink/[0.06]">
                  <div className={cn('h-full rounded-full transition-all duration-300', clsFor(BAR_CLS, item.status))} style={{ width: `${pct}%` }} />
                </div>
                {item.message && <div className="truncate text-[9px] text-text-muted">{item.message}</div>}
              </div>
              <span className="text-right text-[10px] tabular-nums text-text-muted">{pct}%</span>
            </div>
          );
        })}
        {more > 0 && <div className="text-[10px] text-text-muted">+{more} more video(s)</div>}
      </div>
    </div>
  );
}
