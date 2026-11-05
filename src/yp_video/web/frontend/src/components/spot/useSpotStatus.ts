import { useQuery } from '@tanstack/react-query';
import { api, errMsg } from '@/lib/api';

/** Availability of yp-spot inference (the Inference page's fusion packages).
 *  `problem` is null while the status request is still pending — rendering
 *  the pending state as "not ready" flashes a false alarm on load. */
export function useSpotStatus(queryKey: string[]) {
  const spotQuery = useQuery({
    queryKey,
    queryFn: () => api.get('/api/inference/spot'),
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
