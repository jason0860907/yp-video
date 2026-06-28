import { useEffect, useRef } from 'react';
import { SSEClient } from './sse';

/**
 * Subscribe to a server-sent-event stream for the component's lifetime.
 * Connects when `path` is non-null and disconnects on change/unmount. The
 * handler is held in a ref so changing it doesn't tear down the connection.
 *
 * @param path path relative to /api, or null to stay disconnected.
 */
export function useSSE<T = unknown>(path: string | null, onMessage: (data: T) => void): void {
  const handler = useRef(onMessage);
  handler.current = onMessage;

  useEffect(() => {
    if (!path) return;
    const client = new SSEClient<T>(path, { onMessage: (d) => handler.current(d) }).start();
    return () => client.stop();
  }, [path]);
}
