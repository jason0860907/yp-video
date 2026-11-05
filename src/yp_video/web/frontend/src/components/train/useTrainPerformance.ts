import { useState } from 'react';
import { keepPreviousData, useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api';
import type { TrainPerfData } from '@/types/api';

/** The Fusion Train performance card's data: keyed by run, refreshed while a
 *  job runs, and kept mounted while a newly selected run loads — otherwise
 *  the page collapses and the browser jumps back to the top. */
export function useTrainPerformance(running: boolean) {
  const [perfRun, setPerfRun] = useState<string>();
  const perfQuery = useQuery({
    queryKey: ['fusion-model-performance', perfRun],
    queryFn: () =>
      // The epoch records are yp-spot's, open on this backend.
      api.get('/api/fusion-model/performance', {
        params: { query: { run: perfRun } },
      }) as unknown as Promise<TrainPerfData>,
    refetchInterval: running ? 30_000 : false,
    placeholderData: keepPreviousData,
  });
  return { perf: perfQuery.data, perfRun, setPerfRun };
}
