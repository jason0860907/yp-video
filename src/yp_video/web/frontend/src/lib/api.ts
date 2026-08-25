/**
 * API endpoint map + fetch helper. Single source of truth so backend route
 * renames don't require grepping. Every backend route is mounted under /api;
 * `apiUrl()` is the one place that prefix lives, so it works for fetch, SSE,
 * and media `src` alike.
 */

export const API_BASE = '/api';

/** Absolute URL for a path relative to /api — use for fetch, EventSource, and
 *  <video>/<a> src/href. */
export function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

type QueryParams = Record<string, string | number | boolean | null | undefined>;

const q = (params: QueryParams): string => {
  const entries = Object.entries(params).filter(([, v]) => v != null && v !== '');
  return entries.length
    ? '?' + entries.map(([k, v]) => `${k}=${encodeURIComponent(String(v))}`).join('&')
    : '';
};

export interface ApiOptions extends Omit<RequestInit, 'body'> {
  /** Plain object — JSON-encoded automatically. */
  body?: unknown;
}

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly body: string,
  ) {
    super(`API ${status}: ${body}`);
    this.name = 'ApiError';
  }
}

/** The user-facing text of any thrown value. Lives next to ApiError because
 *  its whole job is unwrapping one; twenty-one file-local copies preceded it. */
export const errMsg = (e: unknown): string =>
  e instanceof ApiError ? e.body : e instanceof Error ? e.message : String(e);

/** Fetch, reloading the page if the Cloudflare Access session has lapsed.
 *
 *  Access sessions expire while a labeling tab sits open all afternoon. When
 *  that happens the edge answers an XHR with a redirect to the login page on
 *  another origin, which fetch reports as an opaque TypeError — every page
 *  would show a bogus "network error" and no amount of retrying would fix it.
 *  Only a top-level navigation can complete the login flow, so reload and let
 *  Access take over. A 403 from our own origin means the same thing.
 */
async function fetchOrReauth(path: string, init: RequestInit): Promise<Response> {
  let res: Response;
  try {
    res = await fetch(apiUrl(path), { credentials: 'same-origin', ...init });
  } catch {
    window.location.reload();
    throw new ApiError(0, 'Session expired — reloading');
  }
  if (res.status === 403) {
    window.location.reload();
    throw new ApiError(403, 'Session expired — reloading');
  }
  return res;
}

/** Fetch a JSON endpoint relative to /api. Throws {@link ApiError} on non-2xx. */
export async function apiFetch<T = unknown>(path: string, options: ApiOptions = {}): Promise<T> {
  const { body, headers, ...rest } = options;
  const res = await fetchOrReauth(path, {
    headers: { 'Content-Type': 'application/json', ...headers },
    ...rest,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    throw new ApiError(res.status, await res.text());
  }
  return res.json() as Promise<T>;
}

/** POST JSON and return the response body as a Blob (mp4 / zip clip endpoints). */
export async function apiPostBlob(path: string, body: unknown): Promise<Blob> {
  const res = await fetchOrReauth(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new ApiError(res.status, await res.text());
  return res.blob();
}

// ── Endpoint map ──
// Leaves are literal paths or functions returning paths (relative to /api).
// SSE URLs are passed through apiUrl() by callers via the SSEClient.
export const API = {
  jobs: {
    list: '/jobs',
    activeCount: '/jobs/active-count',
    get: (id: string) => `/jobs/${id}`,
    logs: (id: string) => `/jobs/${id}/logs`,
    cancel: (id: string) => `/jobs/${id}/cancel`,
    eventsSSE: (id: string) => `/jobs/${id}/events`,
  },
  system: {
    videos: '/system/videos',
    me: '/system/me',
    presence: '/system/presence',
    vllmStart: '/system/vllm/start',
    vllmStop: '/system/vllm/stop',
    vllmStatus: '/system/vllm/status',
  },
  upload: {
    start: '/upload/start',
    status: '/upload/status',
    categories: '/upload/categories',
    download: '/upload/download',
    deleteLocal: '/upload/delete-local',
    deleteR2: '/upload/delete-r2',
    files: (category: string) => `/upload/files?category=${encodeURIComponent(category)}`,
    r2Files: (category: string) => `/upload/r2-files?category=${encodeURIComponent(category)}`,
  },
  download: {
    start: '/download/start',
    playlist: (url: string) => `/download/playlist?url=${encodeURIComponent(url)}`,
    cancel: (sessionId: string) => `/download/${sessionId}/cancel`,
    progressSSE: (sessionId: string) => `/download/${sessionId}/progress`,
  },
  cut: {
    videos: '/cut/videos',
    export: '/cut/export',
    video: (name: string) => `/cut/video/${encodeURIComponent(name)}`,
  },
  detect: {
    start: '/detect/start',
    convert: '/detect/convert',
  },
  label: {
    stats: '/label/stats',
  },
  annotate: {
    results: '/annotate/results',
    annotations: '/annotate/annotations',
    result: (name: string, params: QueryParams = {}) => `/annotate/results/${encodeURIComponent(name)}${q(params)}`,
    video: (path: string) => `/annotate/video/${encodeURIComponent(path)}`,
    clip: '/annotate/clip',
    clipZip: '/annotate/clip-zip',
    done: (name: string) => `/annotate/done/${encodeURIComponent(name)}`,
  },
  actionAnnotate: {
    labels: '/action-annotate/labels',
    videos: '/action-annotate/videos',
    spot: '/action-annotate/spot',
    prelabelBatch: '/action-annotate/prelabel-batch',
    annotations: '/action-annotate/annotations',
    annotation: (name: string, params: QueryParams = {}) => `/action-annotate/annotations/${encodeURIComponent(name)}${q(params)}`,
    waveform: (name: string) => `/action-annotate/waveform/${encodeURIComponent(name)}`,
    export: '/action-annotate/export',
    video: (name: string) => `/action-annotate/video/${encodeURIComponent(name)}`,
    done: (name: string) => `/action-annotate/done/${encodeURIComponent(name)}`,
  },
  fusionModel: {
    status: '/fusion-model/status',
    train: '/fusion-model/train',
    performance: '/fusion-model/performance',
  },
  spotPredict: {
    videos: '/spot-predict/videos',
    spot: '/spot-predict/spot',
    start: '/spot-predict/start',
  },
  tracklets: {
    run: '/tracklets/run',
    get: (name: string) => `/tracklets/${encodeURIComponent(name)}`,
    masks: (name: string, rally: number) => `/tracklets/masks/${encodeURIComponent(name)}?rally=${rally}`,
  },
  // Player detection — the sparse perception stage. `extraction` is where its
  // records and crops live, shared with association, which writes the pick
  // into the same record.
  extraction: {
    videos: '/extraction/videos',
    detect: '/extraction/detect',
    records: (name: string) => `/extraction/records/${encodeURIComponent(name)}`,
    crop: (name: string, cropFile: string, masked = false) =>
      `/extraction/crop/${encodeURIComponent(name)}/${encodeURIComponent(cropFile)}${masked ? '?masked=1' : ''}`,
  },
  // ReID is grouping the same person, and nothing else — finding people and
  // choosing who acted is `extraction`, who is on court is `tracklets`.
  reid: {
    videos: '/reid/videos',
    options: '/reid/options',
    embed: '/reid/embed',
    clusters: (name: string, threshold: number, model = 'clip-reid') => `/reid/clusters/${encodeURIComponent(name)}?threshold=${threshold}&model=${encodeURIComponent(model)}`,
    players: (name: string, model = 'clip-reid') => `/reid/players/${encodeURIComponent(name)}?model=${encodeURIComponent(model)}`,
    seedCluster: (name: string) => `/reid/seed-cluster/${encodeURIComponent(name)}`,
    done: (name: string) => `/reid/done/${encodeURIComponent(name)}`,
  },
  reidTrain: {
    status: '/reid-train/status',
    // POST starts the dataset-export job; GET /export (exportPlan) only plans.
    export: '/reid-train/export',
    train: '/reid-train/train',
    runs: '/reid-train/runs',
    performance: (model?: string) =>
      `/reid-train/performance${model ? `?model=${encodeURIComponent(model)}` : ''}`,
    exportPlan: (p: { split_mode: string; test_ratio: number; seed: number; masked: boolean }) =>
      `/reid-train/export?split_mode=${p.split_mode}&test_ratio=${p.test_ratio}&seed=${p.seed}&masked=${p.masked}`,
  },
  // Who did what, when. Written by the backend for every state-changing call
  // and every job transition; this is the read side.
  audit: {
    filters: '/audit/filters',
    worklog: (p: { since: string; until: string }) => `/audit/worklog${q(p)}`,
    saves: (id: number) => `/audit/events/${id}/saves`,
    events: (p: {
      actor?: string;
      action?: string;
      target?: string;
      since?: string;
      until?: string;
      before?: number;
      limit?: number;
    }) => `/audit/events${q(p)}`,
  },
  association: {
    videos: '/actor-association/videos',
    fix: (name: string) => `/actor-association/fix/${encodeURIComponent(name)}`,
    confirm: (name: string) => `/actor-association/confirm/${encodeURIComponent(name)}`,
    status: '/actor-association/status',
    performance: '/actor-association/performance',
    trainPerformance: '/actor-association/train-performance',
    train: '/actor-association/train',
    predict: '/actor-association/predict',
    done: (name: string) => `/actor-association/done/${encodeURIComponent(name)}`,
  },
} as const;
