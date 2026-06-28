import { QueryClient } from '@tanstack/react-query';

/** Shared TanStack Query client. Server state for an internal tool: modest
 *  caching, retry once, and no window-focus refetch storms. Pages that need
 *  live data drive their own polling interval via `refetchInterval`. */
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5_000,
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});
