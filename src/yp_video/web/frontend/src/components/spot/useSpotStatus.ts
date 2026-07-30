import { useQuery } from '@tanstack/react-query';
import { apiFetch, errMsg } from '@/lib/api';
import type { SpotInfo } from '@/types/api';

/** Availability of a yp-spot inference surface (action or rally flavor).
 *  `problem` is null while the status request is still pending — rendering
 *  the pending state as "not ready" flashes a false alarm on load. */
export function useSpotStatus(queryKey: string[], endpoint: string) {
  const spotQuery = useQuery({
    queryKey,
    queryFn: () => apiFetch<SpotInfo>(endpoint),
  });
  const spot = spotQuery.data;
  const checkpoints = spot?.checkpoints ?? [];
  const ready = Boolean(spot?.available && checkpoints.length);
  const problem = spotQuery.isPending
    ? null
    : spotQuery.isError
      ? `status check failed: ${errMsg(spotQuery.error)}`
      : spot?.error ||
        (spot?.available
          ? checkpoints.length
            ? null
            : 'no checkpoint found'
          : `${spot?.spot_dir || 'yp-spot directory'} not ready`);
  return { spot, checkpoints, ready, problem };
}
