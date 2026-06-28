import type { Job } from '@/types/api';
import { JobProgress } from './JobProgress';
import { JobItems } from './JobItems';

/** A job rendered as a bordered card: progress block + batch sub-items. */
export function JobProgressCard({ job, showLogs = false }: { job: Job; showLogs?: boolean }) {
  return (
    <div className="rounded-xl border border-border bg-surface-50 p-3.5">
      <JobProgress job={job} showLogs={showLogs} truncateMsg={false} />
      <JobItems items={job.params?.items ?? []} />
    </div>
  );
}
