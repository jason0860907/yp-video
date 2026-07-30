import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { API, apiFetch } from '@/lib/api';
import type { Job } from '@/types/api';

/** Jobs of the given types, newest first: the server list keeps them across
 *  reloads, local upserts (POST responses, SSE updates) override it. */
export function useTypedJobs(types: readonly string[]) {
  const [overrides, setOverrides] = useState<Record<string, Job>>({});
  const jobsQuery = useQuery({
    queryKey: ['jobs-list'],
    queryFn: () => apiFetch<Job[]>(API.jobs.list),
  });

  const upsertJob = (job: Job) =>
    setOverrides((prev) => ({ ...prev, [job.id]: job }));

  const jobs = useMemo(() => {
    const merged = new Map<string, Job>();
    for (const job of jobsQuery.data ?? []) {
      if (types.includes(job.type ?? '')) merged.set(job.id, job);
    }
    for (const job of Object.values(overrides)) merged.set(job.id, job);
    return [...merged.values()].sort(
      (a, b) => (b.created_at ?? 0) - (a.created_at ?? 0),
    );
    // types is a constant per page; spreading keeps the memo key stable.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobsQuery.data, overrides, ...types]);

  return { jobs, upsertJob };
}
