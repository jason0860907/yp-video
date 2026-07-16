import type { QueryClient } from '@tanstack/react-query';
import type { Job, JobStatus } from '@/types/api';

/** A job is settled once it reaches a terminal state. */
export const isTerminal = (status: Job['status']): boolean =>
  status === 'completed' || status === 'failed' || status === 'cancelled';

// Single source of truth for status colors: running/completed = green (running
// pulses), failed = red, cancelled = amber, pending/stopped = muted. Every
// component picks the slot it needs (pill/text/dot/bar) instead of keeping its
// own map.
export const STATUS_THEME: Record<JobStatus, { pill: string; text: string; dot: string; bar: string }> = {
  running: {
    pill: 'bg-primary/20 text-primary-light ring-1 ring-primary/30',
    text: 'text-primary-light',
    dot: 'bg-primary-light',
    bar: 'bg-primary-light',
  },
  completed: {
    pill: 'bg-primary/15 text-primary-light ring-1 ring-primary/25',
    text: 'text-primary-light',
    dot: 'bg-primary-light',
    bar: 'bg-primary-light',
  },
  failed: {
    pill: 'bg-red-500/15 text-red-400 ring-1 ring-red-500/20',
    text: 'text-red-400',
    dot: 'bg-red-400',
    bar: 'bg-red-400',
  },
  cancelled: {
    pill: 'bg-amber-500/15 text-amber-400 ring-1 ring-amber-500/20',
    text: 'text-amber-400',
    dot: 'bg-amber-400',
    bar: 'bg-amber-400',
  },
  pending: {
    pill: 'bg-ink/5 text-text-muted ring-1 ring-ink/10',
    text: 'text-text-muted',
    dot: 'bg-text-muted',
    bar: 'bg-ink/20',
  },
  stopped: {
    pill: 'bg-ink/5 text-text-muted ring-1 ring-ink/10',
    text: 'text-text-muted',
    dot: 'bg-text-muted',
    bar: 'bg-ink/20',
  },
};

/** Theme lookup tolerant of unknown/missing statuses (falls back to pending). */
export const statusTheme = (status: string | undefined) =>
  STATUS_THEME[status as JobStatus] ?? STATUS_THEME.pending;

/** Batch jobs write {params: {total, failed}} on finalize; completed with
 *  failures is a partial success. Structural — never sniff job.message. */
export const isPartialSuccess = (job: Job): boolean =>
  job.status === 'completed' && Number(job.params?.failed ?? 0) > 0;

/** Terminal statuses read as words, in-flight as percent. */
export const statusLabel = (job: Job): string => {
  if (job.status === 'completed') return isPartialSuccess(job) ? 'partial' : 'completed';
  if (job.status === 'failed' || job.status === 'cancelled') return job.status;
  return `${Math.round((job.progress ?? 0) * 100)}%`;
};

// Query keys each job type's side effects make stale — the single registry,
// so every page that watches a job invalidates the same set and new job types
// get wired up here instead of in per-page onSettled handlers. Keyed by the
// backend's create_job() type string.
const STALE_QUERIES: Record<string, string[][]> = {
  vlm_detect: [['system-videos'], ['annotate-results']],
  rally_spot_predict: [['spot-predict-videos'], ['annotate-results']],
  spot_prelabel: [['action-videos']],
  spot_prelabel_batch: [['action-videos']],
  player_reid: [['reid-videos'], ['reid-results'], ['reid-clusters'], ['reid-players'], ['reid-tracks']],
  player_tracking: [['reid-videos'], ['reid-tracks']],
  player_embed: [['reid-videos'], ['reid-clusters'], ['reid-players']],
  rally_spot_train: [['spot-train-status'], ['spot-predict-info']],
  action_train: [['action-train-status'], ['spot-info']],
  download: [['cut-videos']],
  r2_upload: [['upload-status']],
  r2_download: [['upload-status']],
};

// Stale after any job settles: queue views and the dashboard counts.
const ALWAYS_STALE: string[][] = [['jobs-list'], ['jobs-active-count'], ['system-stats']];

/** Invalidate every query a settled job may have touched. Runs on any
 *  terminal status — failed batch jobs can still have partial results. */
export function invalidateJobQueries(qc: QueryClient, job: Job): void {
  for (const key of [...(STALE_QUERIES[job.type ?? ''] ?? []), ...ALWAYS_STALE]) {
    void qc.invalidateQueries({ queryKey: key });
  }
}
