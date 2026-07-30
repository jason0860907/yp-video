import { useState } from 'react';
import { keepPreviousData, useQuery } from '@tanstack/react-query';
import { apiFetch } from '@/lib/api';
import type { ActionPerfData } from '@/types/api';

/** The per-epoch performance card's data: keyed by run, refreshed while a
 *  job runs, and kept mounted while a newly selected run loads — otherwise
 *  the page collapses and the browser jumps back to the top. */
export function useTrainPerformance(queryKey: string, endpoint: string, running: boolean) {
  const [perfRun, setPerfRun] = useState<string>();
  const perfQuery = useQuery({
    queryKey: [queryKey, perfRun],
    queryFn: () =>
      apiFetch<ActionPerfData>(
        perfRun ? `${endpoint}?run=${encodeURIComponent(perfRun)}` : endpoint,
      ),
    refetchInterval: running ? 30_000 : false,
    placeholderData: keepPreviousData,
  });
  return { perf: perfQuery.data, perfRun, setPerfRun };
}
