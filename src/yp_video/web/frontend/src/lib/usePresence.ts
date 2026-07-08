import { useEffect, useRef, useState } from 'react';
import { API, apiFetch } from '@/lib/api';

/** Server reply: browsers with the page in the foreground within the last
 *  minute; `active` = those whose user gave any input within IDLE_AFTER_MS. */
export interface Presence {
  online: number;
  active: number;
}

const CLIENT_ID_KEY = 'vq:client-id';
const HEARTBEAT_MS = 30_000;
const IDLE_AFTER_MS = 5 * 60_000;

// Stable per-browser id — two tabs in the same browser count as one person.
function clientId(): string {
  let id = localStorage.getItem(CLIENT_ID_KEY);
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem(CLIENT_ID_KEY, id);
  }
  return id;
}

const INPUT_EVENTS = ['mousemove', 'mousedown', 'keydown', 'scroll', 'touchstart'] as const;

/** Heartbeat + read: reports this browser's presence (and whether the user is
 *  idle) and returns the current counts. Beats pause while the tab is hidden,
 *  so a backgrounded tab drops out of the count after the server TTL. */
export function usePresence(): Presence | null {
  const [presence, setPresence] = useState<Presence | null>(null);
  const lastInputRef = useRef(Date.now());

  useEffect(() => {
    const touch = () => {
      lastInputRef.current = Date.now();
    };
    INPUT_EVENTS.forEach((e) => window.addEventListener(e, touch, { passive: true }));

    let stopped = false;
    const beat = async () => {
      try {
        const res = await apiFetch<Presence>(API.system.presence, {
          method: 'POST',
          body: {
            client_id: clientId(),
            active: Date.now() - lastInputRef.current < IDLE_AFTER_MS,
          },
        });
        if (!stopped) setPresence(res);
      } catch {
        /* presence is best-effort */
      }
    };

    void beat();
    const timer = setInterval(() => {
      if (!document.hidden) void beat();
    }, HEARTBEAT_MS);
    // Coming back to the tab re-enters the count immediately, not on the
    // next 30 s tick.
    const onVisible = () => {
      if (!document.hidden) {
        touch();
        void beat();
      }
    };
    document.addEventListener('visibilitychange', onVisible);

    return () => {
      stopped = true;
      clearInterval(timer);
      INPUT_EVENTS.forEach((e) => window.removeEventListener(e, touch));
      document.removeEventListener('visibilitychange', onVisible);
    };
  }, []);

  return presence;
}
