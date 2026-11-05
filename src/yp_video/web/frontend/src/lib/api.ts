/**
 * The typed API client (`api`, paths and types from the generated OpenAPI)
 * plus the URL map for what is not a JSON call — media `src`, SSE streams and
 * downloads, which `apiUrl()` prefixes with /api.
 */

import createClient, { type MaybeOptionalInit } from 'openapi-fetch';
import type { PathsWithMethod, RequiredKeysOf } from 'openapi-typescript-helpers';
import type { paths } from '@/types/openapi';

export const API_BASE = '/api';

/** Absolute URL for a path relative to /api — use for fetch, EventSource, and
 *  <video>/<a> src/href. */
export function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

type QueryParams = Record<string, string | number | boolean | null | undefined>;

/** `?a=1&b=x` without the null / empty entries — the backend reads a missing
 *  parameter as its default, an empty one as a value. */
const queryString = (params: Record<string, unknown>): string =>
  Object.entries(params)
    .filter(([, v]) => v != null && v !== '')
    .map(([k, v]) => `${k}=${encodeURIComponent(String(v))}`)
    .join('&');

const q = (params: QueryParams): string => {
  const query = queryString(params);
  return query ? `?${query}` : '';
};

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

/** Fetch, reloading the page if the Cloudflare Access session has lapsed, and
 *  throwing {@link ApiError} on any non-2xx.
 *
 *  Access sessions expire while a labeling tab sits open all afternoon. When
 *  that happens the edge answers an XHR with a redirect to the login page on
 *  another origin. With `redirect: 'manual'` that surfaces as an
 *  `opaqueredirect` response instead of being followed (and failing) — the
 *  one signal that only a top-level navigation can fix, so reload and let
 *  Access take over. A 403 from our own origin means the same thing. No API
 *  route of ours redirects, so nothing else is affected.
 *
 *  Anything else that makes fetch throw — an aborted request, a backend
 *  restart, a tunnel blip, a laptop waking up — is NOT a session problem and
 *  must not reload the page: the 30 s polls (presence, jobs, progress) would
 *  otherwise turn every transient failure into a lost editor.
 */
async function fetchOrReauth(request: Request): Promise<Response> {
  let res: Response;
  try {
    res = await fetch(request);
  } catch (e) {
    if (e instanceof DOMException && e.name === 'AbortError') throw e;
    throw new ApiError(0, `Network error: ${e instanceof Error ? e.message : String(e)}`);
  }
  if (res.type === 'opaqueredirect' || res.status === 403) {
    window.location.reload();
    throw new ApiError(403, 'Session expired — reloading');
  }
  if (!res.ok) throw new ApiError(res.status, await res.text());
  return res;
}

const client = createClient<paths>({
  credentials: 'same-origin',
  redirect: 'manual',
  fetch: fetchOrReauth,
  querySerializer: queryString,
});

type ApiMethod = 'get' | 'post' | 'put' | 'delete';
/** A route's 200 JSON body as generated. openapi-fetch's own `data` type
 *  runs it through `Readable`, which flattens tuples (`[x, y]` → `number[]`);
 *  this schema has no read/write-only markers for that to strip. */
type Body<Op> = Op extends { responses: { 200: { content: { 'application/json': infer R } } } }
  ? R
  : never;
type Args<I> = RequiredKeysOf<I> extends never ? [init?: I] : [init: I];

function call<M extends ApiMethod>(method: M) {
  return async <P extends PathsWithMethod<paths, M>>(
    path: P,
    ...init: Args<MaybeOptionalInit<paths[P], M>>
  ): Promise<Body<paths[P][M]>> => {
    // Non-2xx never returns: fetchOrReauth threw ApiError.
    const { data } = await (
      client.request as (m: M, p: P, i?: unknown) => Promise<{ data?: unknown }>
    )(method, path, init[0]);
    return data as Body<paths[P][M]>;
  };
}

/** The typed client: every call names an OpenAPI path (`/api/...`); its
 *  params, body and response are the generated types of that route. */
export const api = {
  get: call('get'),
  post: call('post'),
  put: call('put'),
  delete: call('delete'),
  /** POST answering a file (the mp4 / zip clip endpoints). */
  postBlob: async <P extends PathsWithMethod<paths, 'post'>>(
    path: P,
    init: MaybeOptionalInit<paths[P], 'post'>,
  ): Promise<Blob> =>
    (
      await (client.POST as (p: P, i: unknown) => Promise<{ data?: unknown }>)(path, {
        ...init,
        parseAs: 'blob',
      })
    ).data as Blob,
};

// ── URL map ──
// What is not a JSON call — media `src`, SSE streams, downloads and the
// keepalive flush on page leave — as paths relative to /api, passed through
// apiUrl() by callers. JSON calls go through `api` by their OpenAPI path.
export const API = {
  appReview: {
    video: (user: string, match: string) =>
      `/app-review/users/${encodeURIComponent(user)}/matches/${encodeURIComponent(match)}/video`,
  },
  jobs: {
    eventsSSE: (id: string) => `/jobs/${id}/events`,
  },
  download: {
    progressSSE: (sessionId: string) => `/download/${sessionId}/progress`,
  },
  cut: {
    video: (name: string) => `/cut/video/${encodeURIComponent(name)}`,
  },
  court: {
    positions: (name: string) => `/court/positions/${encodeURIComponent(name)}`,
  },
  annotate: {
    annotations: '/annotate/annotations',
    video: (name: string) => `/annotate/video/${encodeURIComponent(name)}`,
  },
  actionAnnotate: {
    annotations: '/action-annotate/annotations',
    video: (name: string) => `/action-annotate/video/${encodeURIComponent(name)}`,
  },
  extraction: {
    crop: (name: string, cropFile: string, masked = false) =>
      `/extraction/crop/${encodeURIComponent(name)}/${encodeURIComponent(cropFile)}${masked ? '?masked=1' : ''}`,
  },
  reidTrain: {
    // GET plans the export as a download; POST (api.post) starts the job.
    exportPlan: (p: { split_mode: string; test_ratio: number; seed: number; masked: boolean }) =>
      `/reid-train/export${q(p)}`,
  },
} as const;
