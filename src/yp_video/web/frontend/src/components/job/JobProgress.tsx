import { cn } from '@/lib/cn';
import type { Job } from '@/types/api';
import { ProgressBar } from './ProgressBar';

function statusColor(status: Job['status']): string {
  switch (status) {
    case 'running':
      return 'text-primary-light';
    case 'completed':
      return 'text-primary-light';
    case 'failed':
      return 'text-red-400';
    case 'cancelled':
      return 'text-amber-400';
    default:
      return 'text-text-muted';
  }
}

function statusLabel(job: Job): string {
  const pct = Math.round((job.progress ?? 0) * 100);
  if (job.status === 'failed') return 'failed';
  if (job.status === 'cancelled') return 'cancelled';
  if (job.status === 'completed') return job.message?.includes('failed') ? 'partial' : 'done';
  return `${pct}%`;
}

interface JobProgressProps {
  job: Job;
  detail?: string;
  showLogs?: boolean;
  truncateMsg?: boolean;
}

/** Single-job progress block — name + status label + bar + message/error,
 *  with an optional collapsible log tail on failure. */
export function JobProgress({ job, detail = '', showLogs = false, truncateMsg = true }: JobProgressProps) {
  const isRunning = job.status === 'running';
  const isDone = job.status === 'completed';
  const isFailed = job.status === 'failed';
  const showMessage = job.message && (isRunning || isDone || isFailed);
  const trunc = truncateMsg ? 'truncate' : '';
  const hasLogs = Array.isArray(job.logs) && job.logs.length > 0;
  const showLogsBlock = showLogs && hasLogs && (isFailed || (isDone && job.message?.includes('failed')));

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <span className="truncate text-xs font-medium text-text-primary">{job.name}</span>
        <span className={cn('text-[11px] font-medium tabular-nums', statusColor(job.status))}>{statusLabel(job)}</span>
      </div>
      <ProgressBar progress={job.progress} />
      {detail && <div className="text-[11px] tabular-nums text-text-muted">{detail}</div>}
      {showMessage && <p className={cn('text-[10px] text-text-muted', trunc)}>{job.message}</p>}
      {job.error && <p className={cn('text-[10px] text-red-400/80', trunc)}>{job.error}</p>}
      {showLogsBlock && (
        <details className="mt-1">
          <summary className="cursor-pointer text-[10px] text-text-muted hover:text-text-primary">
            Show logs ({job.logs!.length} lines)
          </summary>
          <pre className="mt-1 max-h-64 overflow-y-auto whitespace-pre-wrap break-words rounded-lg border border-ink/5 bg-black/40 p-2 font-mono text-[10px] text-red-300/80">
            {job.logs!.join('\n')}
          </pre>
        </details>
      )}
    </div>
  );
}
