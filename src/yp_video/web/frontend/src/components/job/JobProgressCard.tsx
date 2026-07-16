import { API, apiFetch, ApiError } from '@/lib/api';
import { toast } from '@/components/feedback/toast';
import type { Job } from '@/types/api';
import { JobProgress } from './JobProgress';
import { JobItems } from './JobItems';

const errMsg = (e: unknown) => (e instanceof ApiError ? e.body : e instanceof Error ? e.message : String(e));

/** A job rendered as a bordered card: progress block + batch sub-items.
 *  Running jobs get a Cancel action (the current video still runs to
 *  completion — GPU work is not interruptible). */
export function JobProgressCard({ job, showLogs = false }: { job: Job; showLogs?: boolean }) {
  const cancel = async () => {
    try {
      await apiFetch(API.jobs.cancel(job.id), { method: 'POST' });
      toast.warning('Job cancelled');
    } catch (e) {
      toast.error(`Cancel failed: ${errMsg(e)}`);
    }
  };
  return (
    <div className="rounded-xl border border-border bg-surface-50 p-3.5">
      <div className="flex items-start gap-2">
        <div className="min-w-0 flex-1">
          <JobProgress job={job} showLogs={showLogs} truncateMsg={false} />
        </div>
        {job.status === 'running' && (
          <button
            type="button"
            onClick={() => void cancel()}
            className="shrink-0 rounded-lg border border-border-light px-2 py-1 text-[11px] text-text-muted hover:border-red-500/40 hover:text-red-400"
            title="Cancel this job — the video currently running finishes first"
          >
            Cancel
          </button>
        )}
      </div>
      <JobItems items={job.params?.items ?? []} />
    </div>
  );
}
