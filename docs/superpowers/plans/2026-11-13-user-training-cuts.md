# User Training Cuts + Four Camera Angles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Copy a VolleyIQ user's match into the training bucket as a `cuts-user-<angle>` cut with the App's corrected analysis as pre-labels, triggered from yp-video App Review or a DEBUG App; replace the App's three camera angles with four.

**Architecture:** yp-video owns the import (customer→pipeline R2 copy, pre-label conversion from the App Review `Bundle`). App Review calls it directly; the DEBUG App writes a `training_requests` row through the Worker, which yp-video polls via the admin API. Angles are a four-value string enum shared by iOS `CameraAngle`, D1 `matches.angle` (validated by the Worker) and the yp-video cut kinds.

**Tech Stack:** Python 3 / FastAPI / boto3 / pytest (yp-video); React + TS + openapi types (yp-video frontend); Cloudflare Worker (Hono, D1, node:test + Miniflare) (upload-service); SwiftUI + SwiftData + Swift Testing (VolleyIQ).

**Spec:** `yp-video/docs/superpowers/specs/2026-11-13-user-training-cuts-design.md`

## Global Constraints

- Angles, in this order: `baseline` 底線, `corner` 角落, `high` 高處, `center` 中線側面.
- Cut kinds `user-baseline|user-corner|user-high|user-center`; R2 category / local dir `cuts-user-<angle>` (flat).
- Old → new: `phoneSideline`→`corner`, `broadcast`→`center`, `fixedVenue`→`high`. Old values are deleted everywhere, no fallback.
- Cut file name `<match_id>.mp4`; pre-labels go to `rally-spot/pre-annotations` and `action/pre-annotations` only — never `annotations`, never Done.
- Re-adding never overwrites an existing pre-annotation or annotation; an angle change only moves the mp4.
- Refuse a match whose `source_video.r2_key` starts with any `cuts-` category prefix (operator-published cut).
- Training copies survive user deletion (nothing in storage purge touches them).
- App version `1.5.0`; `MIN_IOS_VERSION = "1.5.0"`.
- D1 developer flag `users.developer INTEGER NOT NULL DEFAULT 0`; App training endpoints 403 unless 1.
- Tests touching writes use scratch dirs / fakes, never real labels (`VIDEOS_DIR` monkeypatched to tmp).
- Commits: English Conventional Commits, one per task, ending with the session attribution lines.

## Review Focus

1. A match with no analysis result yet → request fails with "This match has no analysis result yet", nothing copied (Task 2 test).
2. A transient failure (admin API unreachable, R2 error) during a poll → request stays `pending` and is retried; only domain errors (`ValueError`, `NotFound`) mark it `failed` (Task 4 test).
3. A human annotation already exists for the stem → re-add leaves it and the pre-annotation byte-identical (Task 2 test).
4. A non-developer session calls `POST /matches/:id/training` → 403, no row (Task 6 test).
5. Upload with an old angle value (`phoneSideline`) → 400 `invalid_angle` (Task 5 test).

---

## yp-video

### Task 1: Four user cut kinds in the registry and every kind picker

**Files:**
- Modify: `src/yp_video/config.py` (CUTS_* dirs, `CUTS_DIRS`, `R2_CATEGORIES`, `CUT_KINDS`, `CutKind` Literal)
- Modify: `src/yp_video/action/training.py:99` (`by_view` built from `CUT_KINDS`)
- Create: `src/yp_video/web/frontend/src/lib/cutKinds.ts`
- Modify: `components/video/VideoMultiSelectList.tsx`, `components/video/KindBadge.tsx`, `lib/useUnionVideos.ts`, `pages/label/LabelPage.tsx`, `pages/analysis/AnalysisPage.tsx`, `pages/CutPage.tsx` (all under `src/yp_video/web/frontend/src`)
- Test: `tests/test_cut_kinds.py`

**Interfaces:**
- Produces: `USER_CUT_KINDS: dict[str, str]` in config (`angle -> kind`, e.g. `"corner" -> "user-corner"`); `CUT_KINDS["user-corner"] == CutKind(VIDEOS_DIR / "cuts-user-corner", "cuts-user-corner")`; `CutKind` Literal = `"broadcast" | "sideline" | "user-baseline" | "user-corner" | "user-high" | "user-center"`.
- Produces (TS): `CUT_KIND_OPTIONS: { kind: CutKind; label: string }[]` in display order Broadcast, Sideline, User · Baseline, User · Corner, User · High, User · Center.

- [ ] **Step 1: Failing test**

```python
def test_user_kinds_registered():
    from yp_video import config
    assert config.USER_CUT_KINDS == {
        "baseline": "user-baseline", "corner": "user-corner",
        "high": "user-high", "center": "user-center",
    }
    for kind in config.USER_CUT_KINDS.values():
        info = config.CUT_KINDS[kind]
        assert info.r2_category == f"cuts-{kind}"
        assert info.local_dir == config.VIDEOS_DIR / f"cuts-{kind}"
        assert info.r2_category in config.R2_CATEGORIES
        assert info.local_dir in config.CUTS_DIRS
        assert config.cut_kind_of(info.local_dir / "x.mp4") == kind

def test_annotation_stats_by_view_has_every_kind(tmp_path, monkeypatch):
    # by_view keys == CUT_KINDS keys (no KeyError for a user-* video)
```

- [ ] **Step 2: Run** `pytest tests/test_cut_kinds.py -v` → FAIL (`USER_CUT_KINDS` missing).
- [ ] **Step 3: Implement.** Derive the four dirs/categories/kinds from one `USER_ANGLES = ("baseline", "corner", "high", "center")` tuple; `CUTS_DIRS` = all `CUT_KINDS` local dirs; `by_view` = `{kind: zeros for kind in CUT_KINDS}`. Training `CameraView` stays `all|broadcast|sideline` (non-goal). Frontend: every hard-coded broadcast/sideline `<option>`/Segmented/count/badge reads `CUT_KIND_OPTIONS`; `KindBadge` shows `cast`/`side`/`user·<angle>`; `useUnionVideos` passes `r.kind` through. Regenerate openapi types (`make openapi` or the repo's existing script) so `CutKind` includes the new values.
- [ ] **Step 4: Run** `pytest tests/test_cut_kinds.py -v && pytest -q` → PASS; `cd src/yp_video/web/frontend && npm run build` → no type errors.
- [ ] **Step 5: Commit** `feat: register four user cut kinds`

### Task 2: Import a user match into training

**Files:**
- Create: `src/yp_video/web/app_review/training.py`
- Test: `tests/test_app_review_training.py`

**Interfaces:**
- Consumes: `sources.library_match`, `sources.match_bundle`, `sources.customer` (CustomerArtifacts), `r2_client`, `config.USER_CUT_KINDS`, `CUT_KINDS`, `projection.clip_key`, `projection.kind`, `web.action_annotations.{pre_annotation_path, annotation_path, normalize_events, persistable_events, PREDICTED_EVENT_FIELDS, write_annotation_atomic}`, `web.fusion_inference.rally_spot_pre_annotation_path`, `core.rallies.number_rallies`, `core.ffmpeg.probe_video_metadata`, `r2_client.sync_to_r2`.
- Produces:
  - `Angle = Literal["baseline", "corner", "high", "center"]`
  - `def prelabels(bundle: Bundle) -> tuple[list[dict], list[dict]]` — `(rally_rows, action_events)`; rally rows `{"start","end","label":"rally","winner"}` from the same corrected rally set `projection.project` uses (library rallies when present, minus `deleted_rally_indices`, winner = override or own); action events = `result.action_events` minus those whose `clip_key` is in the removed actions (non-score) or removed scores (score), each `{"frame","label","xy","visible","score"}` (`score` only if present).
  - `class TrainingOutcome(TypedDict): video: str; kind: str; status: Literal["added", "exists", "moved"]; rallies: int; events: int`
  - `def add_to_training(user: str, match: str, angle: Angle | None) -> TrainingOutcome` — `angle=None` means the row's `angle`.

- [ ] **Step 1: Failing tests** (fake `sources`, `r2_client`, probe; `VIDEOS_DIR`-derived dirs → tmp)

```python
def test_prelabels_apply_user_corrections(): ...
    # bundle: 2 rallies, rally 2 deleted, rally 1 winner overridden "far";
    # events serve/receive/score, receive removed via actions[key=f"{t:.3f}"]
    # -> rallies == [{"start":..,"end":..,"label":"rally","winner":"far"}]
    # -> [e["label"] for e in events] == ["serve", "score"]

def test_add_copies_source_and_writes_prelabels():
    out = add_to_training("u1", "m1", None)    # row angle "corner"
    assert out == {"video": "m1.mp4", "kind": "user-corner", "status": "added", "rallies": 1, "events": 2}
    # copy called customer:<source key> -> pipeline:"cuts-user-corner/m1.mp4"
    # rally-spot/pre-annotations/m1_annotations.jsonl meta source == {"type":"app","user_id":"u1","match_id":"m1","job_id":...}
    # action/pre-annotations/m1_actions.jsonl events carry score; no file under annotations/

def test_add_again_is_noop_and_keeps_labels():   # Review Focus 3
    # pre-existing human annotation + pre-annotation -> bytes unchanged, status "exists", no copy

def test_angle_change_moves_only_the_video():
    # existing cuts-user-corner/m1.mp4, angle "high" -> copy to cuts-user-high/m1.mp4, delete old, status "moved"

def test_refuses_operator_cut():
    # source r2_key "cuts-sideline/x.mp4" -> ValueError("already training data")

def test_no_result_fails_before_copy():           # Review Focus 1
    # row r2_keys.result None -> ValueError, copy never called
```

- [ ] **Step 2: Run** `pytest tests/test_app_review_training.py -v` → FAIL (module missing).
- [ ] **Step 3: Implement** in this order inside `add_to_training`: `library_match` → refuse operator cut → resolve angle/kind (`ValueError` on unknown) → look for `<match>.mp4` in every `cuts-user-*` category (`r2_client.head` / existing listing helpers): same kind → `exists` (return without touching labels), other kind → server-side copy then delete (`moved`) → else `match_bundle` (raises on no result) → copy customer→pipeline with boto3 `copy(CopySource={"Bucket": customer.bucket, "Key": src}, Bucket=r2_client.bucket, Key=dst)` (managed, multipart-safe; no re-encode) → probe fps/num_frames from a presigned URL of the new object → write pre-annotations only if neither the pre-annotation nor the human annotation file exists → `sync_to_r2` both files. Rally meta `{"video": "<match>.mp4", "duration": result.total_duration, "max_rally_id": N, "source": {"type":"app",...}}`; action meta like `prelabel.predictions_to_annotation` with `source={"type":"app",...}`, events through `normalize_events(..., rallies=[])` + `persistable_events(..., PREDICTED_EVENT_FIELDS)`. Add `AppSource` TypedDict to the `source` union in `web/action_annotations.py`.
- [ ] **Step 4: Run** `pytest tests/test_app_review_training.py -v && pytest -q` → PASS.
- [ ] **Step 5: Commit** `feat: import a user match into training as a user cut`

### Task 3: App Review endpoint and button

**Files:**
- Modify: `src/yp_video/web/routers/app_review.py`
- Modify: `src/yp_video/web/frontend/src/pages/app-review/MatchSummary.tsx` (or the match header component that already shows the match; builder picks)
- Test: `tests/test_app_review_router.py` (extend existing if present)

**Interfaces:**
- Consumes: `training.add_to_training`, `TrainingOutcome`, `Angle`.
- Produces: `POST /api/app-review/users/{user}/matches/{match}/training` body `{"angle": Angle | null}` → `TrainingOutcome`; `ValueError`/`NotFound` mapped by the router's existing `checked()`.

- [ ] **Step 1: Failing test** — POST with monkeypatched `add_to_training` returns its outcome; `ValueError("x")` → 400 with detail `x`.
- [ ] **Step 2: Run** → FAIL (404 route).
- [ ] **Step 3: Implement** the route (run in a thread like `decide`); frontend: a 加入訓練資料 button + angle `<select>` (底線/角落/高處/中線側面, default the match's `angle`) beside the match summary; on success show `已加入 user-corner（N 回合、M 動作）` / `已存在` / `已搬到 …`; regenerate openapi types.
- [ ] **Step 4: Run** `pytest -q` and `npm run build` → PASS.
- [ ] **Step 5: Commit** `feat: add App Review button to send a match to training`

### Task 4: Poll the DEBUG App's training requests

**Files:**
- Create: `src/yp_video/web/app_review/training_requests.py`
- Modify: `src/yp_video/web/app_review/sources.py` (add `admin_post(path, json)` beside `admin_get`)
- Modify: `src/yp_video/web/app.py` lifespan (daemon thread)
- Test: `tests/test_training_requests.py`

**Interfaces:**
- Consumes: Task 6 admin API — `GET /admin/training-requests?status=pending` → `[{"id","user_id","match_id","angle","created_at"}]`; `POST /admin/training-requests/{id}` body `{"status": "done"|"failed", "error": str|null}`.
- Produces: `def process_pending() -> int` (requests finished this round); `def poll_forever(interval_s: float = 120.0) -> None`.

- [ ] **Step 1: Failing tests**

```python
def test_done_and_failed_are_reported(): ...
    # two pending; first ok -> POST {"status":"done","error":None}; second ValueError("no result") -> {"status":"failed","error":"no result"}

def test_transient_error_leaves_request_pending():   # Review Focus 2
    # add_to_training raises AdminApiError / BotoCoreError -> no POST, process_pending() == 0
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement**; `poll_forever` logs and swallows errors per round; start it in `lifespan` only when `UPLOAD_SERVICE_URL` and `AUTH_TOKEN` are configured.
- [ ] **Step 4: Run** `pytest -q` → PASS.
- [ ] **Step 5: Commit** `feat: process DEBUG App training requests`

## volleyiq-backend/upload-service

### Task 5: Four angles in D1 and the Worker; app version gate

**Files:**
- Create: `migrations/0042_camera_angles_training_requests.sql`
- Create: `src/angles.ts`
- Modify: `src/library/media.ts:98`, `src/jobs.ts` (`CreateJobBody.angle`, INSERT at :289), `src/appVersion.ts:8`
- Test: `tests/camera-angles.test.mjs`

**Interfaces:**
- Produces: `ANGLES = ["baseline", "corner", "high", "center"] as const`, `type Angle`, `parseAngle(value: unknown): Angle | null` in `src/angles.ts`; 400 body `{"error":"invalid_angle"}`.
- Migration: `UPDATE matches SET angle = CASE angle WHEN 'phoneSideline' THEN 'corner' WHEN 'broadcast' THEN 'center' WHEN 'fixedVenue' THEN 'high' ELSE angle END;` + `ALTER TABLE users ADD COLUMN developer INTEGER NOT NULL DEFAULT 0;` + `CREATE TABLE training_requests (id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE, match_id TEXT NOT NULL, angle TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','done','failed')), error TEXT, created_at INTEGER NOT NULL DEFAULT (unixepoch()), finished_at INTEGER)` + index on `(status, created_at)` and `(user_id, match_id, created_at)`.

- [ ] **Step 1: Failing tests** — migration maps the three old values; `/media/upload-url` with `angle:"phoneSideline"` → 400 `invalid_angle` (Review Focus 5); missing angle → 400; `angle:"high"` stored; `/jobs` YouTube with `angle:"center"` stores `center`; `MIN_IOS_VERSION === "1.5.0"`.
- [ ] **Step 2: Run** `npm test` → FAIL.
- [ ] **Step 3: Implement**; angle is required on both endpoints (no default). Update any test helper that posts an old angle.
- [ ] **Step 4: Run** `npm test` → PASS.
- [ ] **Step 5: Commit** `feat: four camera angles with server validation`

### Task 6: Training request endpoints

**Files:**
- Create: `src/trainingRequests.ts`
- Modify: `src/auth.ts` (select `users.developer`; `AuthenticatedUser.developer: boolean`), `src/worker.ts`, `src/admin/routes.ts`
- Test: `tests/training-requests.test.mjs`

**Interfaces:**
- Produces (App, `appUser`): `POST /matches/:id/training` body `{"angle": Angle}` → 201 `{"id","status":"pending","angle","error":null,"created_at","finished_at":null}`; `GET /matches/:id/training` → latest request for (user, match) or 404. Non-developer → 403 `{"error":"developer_only"}`; bad angle → 400 `invalid_angle`.
- Produces (admin): `GET /admin/training-requests?status=pending` (oldest first); `POST /admin/training-requests/:id` body `{"status":"done"|"failed","error":string|null}` sets `finished_at`.

- [ ] **Step 1: Failing tests** — non-developer 403 and no row (Review Focus 4); developer 201 then GET returns it; admin list shows it; admin POST done → App GET shows `done`; unauthenticated admin → 401.
- [ ] **Step 2: Run** `npm test` → FAIL.
- [ ] **Step 3: Implement** (UUID ids via `crypto.randomUUID()`, UUID_RE on `:id`).
- [ ] **Step 4: Run** `npm test` → PASS.
- [ ] **Step 5: Commit** `feat: training requests from developer accounts`

## VolleyIQ (iOS — uncompiled here; user builds on Mac)

### Task 7: Four camera angles in the model

**Files:**
- Modify: `VolleyIQ/Models/CameraAngle.swift`, `VolleyIQ/Models/Match.swift:246`, `VolleyIQ/Networking/LibraryRemote.swift:33,120` (decode `angle` as `String` into `angleRaw`, so a 1.5.0 App syncing before migration 0042 doesn't fail to decode old values), `VolleyIQ/Models/TeamSides.swift:54`, `VolleyIQ/App/UITestFixture.swift`, `VolleyIQ/Resources/Localizable.xcstrings`, `project.yml` (`MARKETING_VERSION: "1.5.0"`), every test using `.phoneSideline`
- Test: `VolleyIQTests/CameraAngleTests.swift`

**Interfaces:**
- Produces: `enum CameraAngle: String { case baseline, corner, high, center }` (that order), `localizedTitleKey`/`localizedSubtitleKey` returning the Chinese originals (`"底線"`/`"站在一方底線後面，正對整個球場"`, `"角落"`/`"在球場一角斜斜拍過去"`, `"高處"`/`"從看台或二樓往下拍"`, `"中線側面"`/`"在中線旁側拍，兩邊場地都拍得到"`); `symbolName` removed. `Match.angle` getter falls back to `.corner` until the next sync rewrites `angleRaw`.

- [ ] **Step 1: Tests** — `CameraAngle.allCases.map(\.rawValue) == ["baseline","corner","high","center"]`; `TeamSides` default `.left` for `.center`, `.near` for the other three; `Match(angleRaw:"phoneSideline").angle == .corner`.
- [ ] **Step 2: Implement**; delete `angle_phone*`/`angle_fixed*`/`angle_tv*` keys, add the four title + four subtitle keys (zh-Hant source, en translation).
- [ ] **Step 3: Commit** `feat: four camera angles`

### Task 8: Court diagrams on the angle step

**Files:**
- Create: `VolleyIQ/Screens/Upload/CameraAngleDiagram.swift`
- Modify: `VolleyIQ/Screens/Upload/UploadAngleStep.swift`

**Interfaces:**
- Produces: `struct CameraAngleDiagram: View { let angle: CameraAngle; let selected: Bool }` 64×64, drawn with `Canvas`: court rect + net line in `theme.textSecondary`, camera dot + view cone in `theme.primary` (cone ~18% opacity, deeper when `selected`); geometry per the spec's mockup A (baseline: below baseline centre, cone up; corner: outside bottom-left corner, diagonal cone; high: baseline position + raised base + 「高」; center: court rotated landscape, net vertical, camera below centre line, cone over both halves).
- [ ] **Step 1: Implement** and replace the SF Symbol tile in `row(for:)` with the diagram (keep the row's selection outline and check circle).
- [ ] **Step 2: Add a `#Preview` with all four rows.**
- [ ] **Step 3: Commit** `feat: court diagrams for camera angles`

### Task 9: DEBUG 加入訓練資料 sheet

**Files:**
- Create: `VolleyIQ/Networking/TrainingRequest.swift`, `VolleyIQ/Screens/RallyFeed/TrainingRequestSheet.swift` (both `#if DEBUG`)
- Modify: `VolleyIQ/Screens/RallyFeed/RallyFeedView.swift` (DEBUG menu item under 動作門檻…, `.appSheet`)

**Interfaces:**
- Consumes: Task 6 App endpoints via `APIClient.fetch`.
- Produces: `struct TrainingRequest: Decodable { id: String; status: Status; angle: CameraAngle; error: String? }` (`Status: pending|done|failed`); `extension APIClient { func requestTraining(matchID: UUID, angle: CameraAngle) async throws -> TrainingRequest; func trainingRequest(matchID: UUID) async throws -> TrainingRequest? }` (404 → nil).
- [ ] **Step 1: Implement** the sheet: angle picker with the four diagrams (default `match.angle`), 送出, then show 排隊中（yp-video 處理中）/ 已加入 / 失敗：<error>; 403 shows 這個帳號不是開發者. Menu item `Label("加入訓練資料…", systemImage: "tray.and.arrow.down")`, disabled unless `match.status == .done`.
- [ ] **Step 2: Commit** `feat: DEBUG sheet to send a match to training`

## Rollout (user-run steps, reported at the end)

1. Mac: build + run tests; Codemagic → TestFlight → App Store 1.5.0.
2. Only after 1.5.0 is live (the 1.5.0 gate would otherwise lock every user out): `wrangler d1 migrations apply --remote` (0042) + Worker deploy.
3. `UPDATE users SET developer = 1 WHERE id = '<your id>'` (remote D1).
4. Restart yp-app so the poller starts.
