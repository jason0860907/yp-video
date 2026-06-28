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

/** Fetch a JSON endpoint relative to /api. Throws {@link ApiError} on non-2xx. */
export async function apiFetch<T = unknown>(path: string, options: ApiOptions = {}): Promise<T> {
  const { body, headers, ...rest } = options;
  const res = await fetch(apiUrl(path), {
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
  const res = await fetch(apiUrl(path), {
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
    cancel: (id: string) => `/jobs/${id}/cancel`,
    eventsSSE: (id: string) => `/jobs/${id}/events`,
  },
  system: {
    stats: '/system/stats',
    videos: (params: QueryParams = {}) => `/system/videos${q(params)}`,
    vllmStart: '/system/vllm/start',
    vllmStop: '/system/vllm/stop',
    vllmStatus: '/system/vllm/status',
  },
  upload: {
    start: '/upload/start',
    status: '/upload/status',
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
  annotate: {
    results: '/annotate/results',
    annotations: '/annotate/annotations',
    result: (name: string) => `/annotate/results/${encodeURIComponent(name)}`,
    video: (path: string) => `/annotate/video/${encodeURIComponent(path)}`,
    publish: '/annotate/publish',
  },
  actionAnnotate: {
    labels: '/action-annotate/labels',
    videos: '/action-annotate/videos',
    spot: '/action-annotate/spot',
    prelabel: '/action-annotate/prelabel',
    prelabelBatch: '/action-annotate/prelabel-batch',
    annotations: '/action-annotate/annotations',
    annotation: (name: string) => `/action-annotate/annotations/${encodeURIComponent(name)}`,
    waveform: (name: string) => `/action-annotate/waveform/${encodeURIComponent(name)}`,
    export: '/action-annotate/export',
    video: (name: string) => `/action-annotate/video/${encodeURIComponent(name)}`,
  },
  actionTrain: {
    status: '/action-train/status',
    start: '/action-train/start',
  },
  review: {
    results: '/review/results',
    annotations: '/review/annotations',
    result: (name: string, params: QueryParams = {}) =>
      `/review/results/${encodeURIComponent(name)}${q(params)}`,
    video: (path: string) => `/review/video/${encodeURIComponent(path)}`,
    clip: '/review/clip',
    clipZip: '/review/clip-zip',
  },
  predict: {
    videos: '/predict/videos',
    start: '/predict/start',
  },
  train: {
    configDefaults: '/train/config-defaults',
    convertAnnotations: '/train/convert-annotations',
    extractFeatures: '/train/extract-features',
    start: '/train/start',
    status: (params: QueryParams = {}) => `/train/status${q(params)}`,
    performance: (params: QueryParams = {}) => `/train/performance${q(params)}`,
    checkpoints: (params: QueryParams = {}) => `/train/checkpoints${q(params)}`,
  },
} as const;
