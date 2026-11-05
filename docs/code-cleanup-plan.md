# Code cleanup plan (yp-video)

Source: 2026-11-04 audit against Linus "good taste" and SOLID. Cross-repo order:
P1 bugs (all repos) → P2 backend lanes → P3 yp-spot → P4 VolleyIQ → P5 yp-video.

## P1 — bugs
- [x] `/video/{path}` serves any absolute path and does not reject `..`
      (`web/routers/annotate.py`). Resolve only through known video dirs and drop
      the historical fallback chain.

## P5 — contracts and structure
- [x] Response models on routes; generate frontend types from OpenAPI instead of
      the hand-mirrored `types/api.ts` (already drifted: `JobStatus 'stopped'`).
      Routes return TypedDicts / models; `make api-types` writes
      `frontend/src/types/openapi.ts`, `contract-check` guards it — the
      pre-commit hook runs it on any package module, since response TypedDicts
      live throughout `src/yp_video/`, not only in `contracts/`. Still open
      on purpose: extraction records (next item), the Worker admin passthrough
      (`MatchDetail.match`, library), yp-spot's metrics records and yp-reid's
      progress lines (`job.params.last_eval` / `best`), and the SSE payloads
      OpenAPI cannot describe (`DownloadPage`'s per-video `VideoProgress`).
      `JobSummary.params` types the keys pages read (batch items and
      tallies, SPOT train progress, transfer counters); each job's own
      bookkeeping passes through untyped — a per-kind union would need jobs to
      keep typed state instead of a merged dict.
- [ ] Extraction record as a Pydantic model with association fields nested
      (replaces `_ASSOCIATION_FIELDS` and `.get(...) or default` reads).
- [ ] One sync + one async subprocess line reader with process-group kill
      (`action/predict.py`, `reid/embedder.py`, `web/job_helpers.py`).
- [ ] Shared frame-prefetch iterator for `detect_video` and `tracking.py`;
      split `detect_video` into small helpers.
- [x] Per-task training params: `build_command` loops one `TASK_OPTIONS`
      table (no `mixed` branch) and the request rejects a field its recipe
      does not show.
- [ ] Split `start_training_job`.
- [ ] `masked_input` on the `Embedder` Protocol; drop `getattr` and `-masked` name checks.
- [ ] Remove compat fallbacks: `reid/store.py` mtime fallback, pre-GTA track special case.
- [ ] Shared `useFlushOnLeave` hook for AnnotationEditor / ActionPanel.
- [ ] Split `EventVideoPlayer` (20 props); fix the 14 `exhaustive-deps` disables.
- [ ] Layering: R2 helpers and extraction paths into `core/`; routers build responses.
- [ ] One yt-dlp format helper; drop the redundant download progress channel;
      remove "used to…" history comments.
