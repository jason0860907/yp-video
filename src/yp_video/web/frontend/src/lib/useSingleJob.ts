import { useEffect, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { API, apiFetch, errMsg } from '@/lib/api';
import { invalidateJobQueries, isTerminal } from '@/lib/job';
import { useSSE } from '@/lib/useSSE';
import { toast } from '@/components/feedback/toast';
import type { Job } from '@/types/api';

interface UseSingleJobOptions {
  /** The server's currently active job, adopted on first load so a reload
   *  mid-run re-attaches instead of showing an idle page. */
  activeJob?: Job | null;
  /** Toast prefix ("Action training") — or a function when the page runs
   *  more than one job kind through the same slot. */
  label: string | ((job: Job) => string);
  /** Page-specific cleanup on settle, beyond the shared cache invalidation. */
  onSettled?: (job: Job) => void;
}

/** One-at-a-time job pages: the Train pages and Upload all follow a single
 *  job over SSE. Adoption, settle toasts and cache invalidation (through
 *  lib/job's STALE_QUERIES registry) live here once — five pages used to
 *  hand-roll this block each, and none of them invalidated the registry. */
export function useSingleJob({ activeJob, label, onSettled }: UseSingleJobOptions) {
  const [job, setJob] = useState<Job | null>(null);
  const queryClient = useQueryClient();

  useEffect(() => {
    if (activeJob && !job) setJob(activeJob);
  }, [activeJob, job]);

  useSSE<Job>(job && !isTerminal(job.status) ? API.jobs.eventsSSE(job.id) : null, (next) => {
    setJob(next);
    if (!isTerminal(next.status)) return;
    const name = typeof label === 'function' ? label(next) : label;
    if (next.status === 'completed') toast.success(next.message || `${name} complete`);
    if (next.status === 'failed') toast.error(`${name} failed: ${next.error || next.message}`);
    invalidateJobQueries(queryClient, next);
    onSettled?.(next);
  });

  const running = !!job && !isTerminal(job.status);

  const cancel = async () => {
    if (!job) return;
    try {
      await apiFetch(API.jobs.cancel(job.id), { method: 'POST' });
      toast.warning('Job cancelled');
    } catch (e) {
      toast.error(`Cancel failed: ${errMsg(e)}`);
    }
  };

  return { job, setJob, running, cancel };
}
