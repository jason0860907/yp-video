/**
 * SSE client that survives transient disconnects.
 *
 * When the browser suspends EventSource (e.g. a backgrounded tab), the client
 * retries with exponential backoff (1s → 2s → 5s → 10s, capped at 30s) and
 * reconnects immediately when the page becomes visible again. Callers must call
 * `stop()` exactly when they no longer want updates (terminal job state, effect
 * cleanup, etc.) — that is treated as a permanent close.
 *
 * Ported from the legacy vanilla client; behaviour is identical.
 */
import { apiUrl } from './api';

// Registry is message-type-agnostic — it only ever calls kick(), so the
// payload generic is irrelevant here.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const aliveClients = new Set<SSEClient<any>>();
const BACKOFF_STEPS_MS = [1000, 2000, 5000, 10000, 30000];

export interface SSEHandlers<T = unknown> {
  onMessage?: (data: T) => void;
}

export class SSEClient<T = unknown> {
  private readonly url: string;
  private source: EventSource | null = null;
  private alive = false;
  private retry = 0;
  private retryTimer: ReturnType<typeof setTimeout> | null = null;

  /** @param path path relative to /api (e.g. `API.jobs.eventsSSE(id)`). */
  constructor(
    path: string,
    private readonly handlers: SSEHandlers<T> = {},
  ) {
    this.url = apiUrl(path);
  }

  start(): this {
    this.alive = true;
    aliveClients.add(this);
    this.open();
    return this;
  }

  private open(): void {
    if (!this.alive) return;
    this.source?.close();
    this.source = new EventSource(this.url);
    this.source.onmessage = (e) => {
      this.retry = 0; // reset backoff on any successful frame
      try {
        this.handlers.onMessage?.(JSON.parse(e.data) as T);
      } catch {
        /* ignore parse errors */
      }
    };
    this.source.onerror = () => {
      if (!this.alive) return;
      // Stay quiet on transient errors; just schedule a reconnect.
      this.scheduleReconnect();
    };
  }

  private scheduleReconnect(): void {
    if (!this.alive || this.retryTimer) return;
    this.source?.close();
    this.source = null;
    const delay = BACKOFF_STEPS_MS[Math.min(this.retry, BACKOFF_STEPS_MS.length - 1)]!;
    this.retry += 1;
    this.retryTimer = setTimeout(() => {
      this.retryTimer = null;
      this.open();
    }, delay);
  }

  /** Force an immediate reconnect (e.g. page returned to foreground). */
  private kick(): void {
    if (!this.alive) return;
    if (this.retryTimer) {
      clearTimeout(this.retryTimer);
      this.retryTimer = null;
    }
    this.retry = 0;
    if (!this.source || this.source.readyState === EventSource.CLOSED) {
      this.open();
    }
  }

  stop(): void {
    this.alive = false;
    aliveClients.delete(this);
    if (this.retryTimer) {
      clearTimeout(this.retryTimer);
      this.retryTimer = null;
    }
    this.source?.close();
    this.source = null;
  }

  static kickAll(): void {
    aliveClients.forEach((c) => c.kick());
  }
}

if (typeof document !== 'undefined') {
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') SSEClient.kickAll();
  });
}
