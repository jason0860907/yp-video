import { useState } from 'react';
import { cn } from '@/lib/cn';
import { formatClock, formatDuration } from '@/lib/format';
import { statusTheme } from '@/lib/job';
import type { JobItem } from '@/types/api';

interface JobItemsProps {
  items: JobItem[];
  maxVisible?: number;
}

/** Per-video sub-progress for batch jobs (job.params.items).
 *
 *  Each video is a collapsible row: collapsed shows status + name + percent;
 *  expanded adds the bar, the detailed message, start time and duration.
 *  Running items start expanded so the video currently being processed is
 *  visible without a click; a manual toggle always wins. */
export function JobItems({ items, maxVisible = 12 }: JobItemsProps) {
  const [toggled, setToggled] = useState<Record<string, boolean>>({});
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
      <div className="max-h-72 space-y-0.5 overflow-y-auto pr-1">
        {visible.map((item, idx) => {
          const key = item.video ?? String(idx);
          const pct = Math.round(Math.max(0, Math.min(Number(item.progress ?? 0), 1)) * 100);
          const open = toggled[key] ?? item.status === 'running';
          const elapsed = item.started_at
            ? (item.finished_at ?? Date.now() / 1000) - item.started_at
            : null;
          return (
            <div key={key} className="rounded-lg transition-colors hover:bg-ink/[0.03]">
              <button
                type="button"
                onClick={() => setToggled((t) => ({ ...t, [key]: !open }))}
                className="flex w-full items-center gap-2 px-1 py-1.5 text-left"
              >
                <svg
                  className={cn('h-3 w-3 flex-shrink-0 text-text-muted transition-transform', open && 'rotate-90')}
                  fill="none"
                  stroke="currentColor"
                  strokeWidth={2}
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                </svg>
                <span className={cn('flex-shrink-0 rounded px-1.5 py-0.5 text-[9px] uppercase tracking-wide', statusTheme(item.status).pill)}>
                  {item.status ?? 'pending'}
                </span>
                <span className="min-w-0 flex-1 truncate text-[10.5px] text-text-secondary" title={item.video ?? ''}>
                  {item.video ?? ''}
                </span>
                <span className="flex-shrink-0 text-right text-[10px] tabular-nums text-text-muted">{pct}%</span>
              </button>
              {open && (
                <div className="space-y-1.5 px-1 pb-2 pl-6">
                  <div className="h-1 overflow-hidden rounded-full bg-ink/[0.06]">
                    <div
                      className={cn('h-full rounded-full transition-all duration-300', statusTheme(item.status).bar)}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  {item.message && <div className="break-words text-[10px] text-text-muted">{item.message}</div>}
                  {item.error && <div className="break-words text-[10px] text-red-400/80">{item.error}</div>}
                  <div className="flex flex-wrap gap-x-3 font-mono text-[9.5px] tabular-nums text-text-muted">
                    <span>Started {formatClock(item.started_at)}</span>
                    {elapsed != null && <span>{formatDuration(elapsed)}</span>}
                    {item.finished_at != null && <span>Finished {formatClock(item.finished_at)}</span>}
                  </div>
                </div>
              )}
            </div>
          );
        })}
        {more > 0 && <div className="text-[10px] text-text-muted">+{more} more video(s)</div>}
      </div>
    </div>
  );
}
