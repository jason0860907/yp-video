# Code cleanup plan (yp-video)

Source: 2026-11-04 audit against Linus "good taste" and SOLID. Cross-repo order:
P1 bugs (all repos) → P2 backend lanes → P3 yp-spot → P4 VolleyIQ → P5 yp-video.

## P1 — bugs
- [x] `/video/{path}` serves any absolute path and does not reject `..`
      (`web/routers/annotate.py`). Resolve only through known video dirs and drop
      the historical fallback chain.

## P5 — contracts and structure
- [ ] Response models on routes; generate frontend types from OpenAPI instead of
      the hand-mirrored `types/api.ts` (already drifted: `JobStatus 'stopped'`).
- [ ] Extraction record as a Pydantic model with association fields nested
      (replaces `_ASSOCIATION_FIELDS` and `.get(...) or default` reads).
- [ ] One sync + one async subprocess line reader with process-group kill
      (`action/predict.py`, `reid/embedder.py`, `web/job_helpers.py`).
- [ ] Shared frame-prefetch iterator for `detect_video` and `tracking.py`;
      split `detect_video` into small helpers.
- [ ] Per-task training params as `tasks: dict[TaskName, TaskTrainParams]`;
      `build_command` loops it (no `mixed` branch); split `start_training_job`.
- [ ] `masked_input` on the `Embedder` Protocol; drop `getattr` and `-masked` name checks.
- [ ] Remove compat fallbacks: `reid/store.py` mtime fallback, pre-GTA track special case.
- [ ] Shared `useFlushOnLeave` hook for AnnotationEditor / ActionPanel.
- [ ] Split `EventVideoPlayer` (20 props); fix the 14 `exhaustive-deps` disables.
- [ ] Layering: R2 helpers and extraction paths into `core/`; routers build responses.
- [ ] One yt-dlp format helper; drop the redundant download progress channel;
      remove "used to…" history comments.
