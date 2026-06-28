import { API } from '@/lib/api';
import { isTerminal } from '@/lib/job';
import { useSSE } from '@/lib/useSSE';
import type { Job } from '@/types/api';
import { JobProgressCard } from './JobProgressCard';

interface LiveJobProps {
  job: Job;
  /** Called on every SSE frame with the updated job. */
  onUpdate: (job: Job) => void;
  /** Called once when the job reaches a terminal state. */
  onSettled?: (job: Job) => void;
  showLogs?: boolean;
}

/** Renders a job card and keeps it live over SSE until it settles. Multiple
 *  can run concurrently — each subscribes only while its job is active. */
export function LiveJob({ job, onUpdate, onSettled, showLogs = true }: LiveJobProps) {
  useSSE<Job>(isTerminal(job.status) ? null : API.jobs.eventsSSE(job.id), (data) => {
    onUpdate(data);
    if (isTerminal(data.status)) onSettled?.(data);
  });
  return <JobProgressCard job={job} showLogs={showLogs} />;
}
