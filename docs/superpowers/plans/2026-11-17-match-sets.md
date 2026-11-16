# Match Sets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Users mark 局開頭 / 局結尾 on rallies; every reader derives each rally's set number or "not counted" from those marks, and not-counted rallies leave the App's lists, scores and stats and the share page.

**Architecture:** The marks ride in the corrections blob as an optional `sets: {starts, ends}` (rally indices). One pure rule, ported three times (Swift, TypeScript, Python), is pinned by one shared JSON fixture. The App keeps every rally in a renamed `allRallies` relationship and serves the counted ones through `rallies`, so the ~58 existing readers change behaviour without edits.

**Tech Stack:** SwiftUI + SwiftData (VolleyIQ), Hono Worker + node:test/Miniflare (volleyiq-backend/upload-service), Python + pytest (yp-video).

**Spec:** `yp-video/docs/superpowers/specs/2026-11-17-match-sets-design.md`

## Global Constraints

- Corrections `schemaVersion` stays `"7.0"`; `sets` is optional and omitted when there are no marks.
- Wire shape: `"sets": {"starts": [Int], "ends": [Int]}`, values are `Rally.index` (same key space as `deleted_rally_indices`, `team_sides.switches`).
- Rally play order is `(start seconds, index)` ascending.
- Marks naming a rally that doesn't exist (deleted) are ignored.
- Training prelabels (`app_review/training.py`) do NOT apply the rule (user decision).
- `feedback.py` needs no change: not-counted rallies are not deleted, so they never become "not a rally" feedback.
- Sample match (`sample.ts`) does not carry `sets`.
- UI copy in Traditional Chinese; xcstrings keys are the Chinese source text; JSON round-trip of `Localizable.xcstrings` with `json.dumps(d, indent=2, ensure_ascii=False) + "\n"` (no key sorting).
- No Swift toolchain on the Linux host: Swift tasks are written test-first but run on the user's Mac; push iOS commits so the Mac can pull.
- No backward compatibility (CLAUDE.md): the model's per-rally `set` stops being read by the App.

## The rule (all three ports implement exactly this)

```
assign(rallies: [(index, start)], starts, ends) -> {index: set number}   # absent = not counted
  order rallies by (start, index); drop marks whose index is not a rally
  if starts is empty: current = 1, count = 1  else: current = none, count = 0
  for rally in order:
      if rally in starts: count += 1; current = count
      if current: result[rally] = current
      if rally in ends and current: current = none
```

## Review Focus

1. A mark on a rally the user later deletes — must be ignored, not shift other sets.
2. A rally both 局開頭 and 局結尾 — a one-rally set; the next rally is not counted until the next start.
3. Two 局結尾 in a row — the second has no effect and the sheet shows it as ineffective.
4. Events (touches, score clips) inside a not-counted rally — must vanish from clips, stats and score, not re-home as "outside every rally".
5. A whole-match share and a single-rally share whose rally became not counted — both drop it.

---

### Task 1: Shared fixture + Python rule

**Files:**
- Create: `VolleyIQ/VolleyIQTests/Fixtures/match_sets.json`
- Create: `yp-video/src/yp_video/web/app_review/sets.py`
- Modify: `yp-video/src/yp_video/web/app_review/models.py` (Corrections)
- Test: `yp-video/tests/test_match_sets_fixture.py`

**Interfaces:**
- Produces: fixture format `{"cases": [{"name": str, "rallies": [{"index": int, "start": float}], "starts": [int], "ends": [int], "expected": {"<index>": int | null}}]}` — every rally appears in `expected`.
- Produces: `sets.assign(rallies: Iterable[tuple[int, float]], starts: Iterable[int], ends: Iterable[int]) -> dict[int, int]`; `models.MatchSets(starts: list[int], ends: list[int])`; `Corrections.sets: MatchSets | None = None`.

- [ ] **Step 1: Write the fixture** with at least these cases: `no marks` (all 1); `spec example` (12 rallies, starts [4,10], ends [8] → 1–3 null, 4–8 → 1, 9 null, 10–12 → 2); `only an end` (end [5] of 8 → 1–5 → 1, 6–8 null); `only starts` (starts [3] → 1–2 null); `two ends in a row` (starts [1], ends [3,4], 6 rallies → 1–3 → 1, rest null); `start and end on one rally` (starts [2,5], ends [2] → 2 → 1, 3–4 null, 5+ → 2); `mark on a missing rally` (starts [99], 4 rallies → all 1); `index order differs from time order` (indices whose starts are not ascending; expected follows time).

- [ ] **Step 2: Write the failing test** — loads the fixture via `Path(__file__).resolve().parents[2] / "VolleyIQ/VolleyIQTests/Fixtures/match_sets.json"` (as `test_app_clip_fixture.py`), and for each case asserts `{str(i): assign(...).get(i) for i in indices} == case["expected"]`; plus `Corrections.model_validate({... , "sets": {"starts": [4], "ends": []}}).sets.starts == [4]` and that a blob without `sets` validates with `sets is None`.

- [ ] **Step 3: Run** `cd yp-video && uv run pytest tests/test_match_sets_fixture.py -q` — Expected: FAIL (ImportError).

- [ ] **Step 4: Implement** `sets.assign` per the rule and the models change.

- [ ] **Step 5: Run** the test — Expected: PASS. Then `uv run pytest -q` — Expected: all pass.

- [ ] **Step 6: Commit** in VolleyIQ (`test: shared match sets fixture`) and yp-video (`feat: match sets rule and corrections field`).

### Task 2: App Review projection applies sets

**Files:**
- Modify: `yp-video/src/yp_video/web/app_review/projection.py` (`project`, the rally list at ~l.403–412)
- Test: `yp-video/tests/test_app_review_sets.py`

**Interfaces:**
- Consumes: `sets.assign`, `Corrections.sets` (Task 1).

- [ ] **Step 1: Failing test** — a Bundle (build like `tests/test_app_clip_fixture.py::fixture_bundle`) with 4 rallies, a spike event inside rally 1 and rally 3, corrections `sets: {starts: [2], ends: [2]}` plus starts adding 4 (`starts [2,4], ends [2]`): `project(bundle)` has rallies 2 and 4 only, their `set` values are 1 and 2, the spike in rally 1 and 3 are absent from the preview's clips, and the score totals are per set 1 and 2. With `corrected=False` all 4 rallies show.
- [ ] **Step 2: Run** — Expected: FAIL.
- [ ] **Step 3: Implement** — after the deleted-rally filter, compute `assign` over the kept rallies when `correction and correction.sets`; keep only assigned rallies and `model_copy(update={"set": n})`; drop events whose time falls inside a not-counted rally's `[start, end]` before clips/scores are built.
- [ ] **Step 4: Run** the test, then `uv run pytest -q` — Expected: all pass.
- [ ] **Step 5: Commit** `feat(app-review): hide rallies outside the user's sets`.

### Task 3: Worker port + share page

**Files:**
- Create: `volleyiq-backend/upload-service/src/matchSets.ts`
- Modify: `volleyiq-backend/upload-service/src/shares.ts` (`selectMatchRallies` and its two callers)
- Test: `volleyiq-backend/upload-service/tests/match-sets.test.mjs`

**Interfaces:**
- Produces: `export function assignSets(rallies: { index: number; start: number }[], sets: { starts: number[]; ends: number[] } | null | undefined): Map<number, number>`.

- [ ] **Step 1: Failing tests** — (a) fixture: read `../../VolleyIQ/VolleyIQTests/Fixtures/match_sets.json` (as `app-events.test.mjs` reads `usage_events.json`), import the compiled module the way other tests import `src` code, assert every case. (b) Miniflare share: a match with 3 rallies and a corrections blob in R2 (`corrections/<user>/<match>.json`, key stored in `matches.corrections_r2_key`) with `sets: {starts: [2], ends: []}`; the public payload of a whole-match share lists rallies 2–3 only, and a share whose items reference rally 1 returns no range for it.
- [ ] **Step 2: Run** `npm test` — Expected: the new tests FAIL.
- [ ] **Step 3: Implement** `assignSets`; make `selectMatchRallies` select `start_seconds`, read the match's corrections blob (`matches.corrections_r2_key`, R2 `VIDEOS`) once, and return only assigned rallies. A missing/unreadable blob means no filtering.
- [ ] **Step 4: Run** `npm run typecheck && npm test` — Expected: clean, all pass.
- [ ] **Step 5: Commit** `feat(upload-service): shares leave out rallies outside the owner's sets`. (Deployable before the App: no `sets` → unchanged.)

### Task 4: Swift rule + fixture test

**Files:**
- Create: `VolleyIQ/VolleyIQ/Models/MatchSets.swift`
- Test: `VolleyIQ/VolleyIQTests/MatchSetsFixtureTests.swift`

**Interfaces:**
- Produces: `struct MatchSets: Equatable { var starts: [Int]; var ends: [Int]; var isEmpty: Bool; static func assign(_ rallies: [(index: Int, start: Double)], starts: [Int], ends: [Int]) -> [Int: Int]; mutating func toggleStart(_ index: Int); mutating func toggleEnd(_ index: Int) }`.

- [ ] **Step 1: Failing test** — loads `match_sets.json` like `ClipWindowFixtureTests` (`Bundle(for:)…url(forResource:"match_sets", withExtension:"json")`) and asserts each case; plus toggle tests (toggling twice removes; arrays stay sorted).
- [ ] **Step 2: Implement** per the rule.
- [ ] **Step 3: Commit + push** `feat: match sets rule` (tests run on the Mac).

### Task 5: Swift storage, corrections, counted rallies

**Files:**
- Modify: `VolleyIQ/Models/Match.swift` (relationship l.200–201; new stored marks beside `teamSidesSwitches`)
- Modify: `VolleyIQ/Models/Rally.swift` (`setNumber` becomes computed; drop it from `init`)
- Modify: `VolleyIQ/Models/MatchCorrections.swift` (wire field, build, apply; `modelRallyIndices`/`currentIndices`/winner-override loops use `allRallies`)
- Modify: `VolleyIQ/Models/Match+ScoreThreshold.swift` (`shownActions`, `shippedActions` drop events inside not-counted rallies)
- Modify: `VolleyIQ/Models/Tag.swift` (+ `countedRallies`), and every `tag.rallies` reader (AppSidebar, TagRalliesView, Favorites*)
- Modify: writers of the relationship — `Networking/LibraryRemote.swift` (l.150–175: stop assigning `setNumber`, append to `allRallies`), `Networking/ProcessingJob.swift` (l.437–457 same), `Networking/LibrarySyncService.swift`, `Networking/RallyDeleter.swift`, `Networking/SampleMatchService.swift`, `App/UITestFixture.swift`
- Test: `VolleyIQ/VolleyIQTests/MatchSetsTests.swift`

**Interfaces:**
- Produces: `Match.allRallies: [Rally]` (`@Relationship(deleteRule: .cascade, inverse: \Rally.match, originalName: "rallies")`); `Match.rallies: [Rally]` (computed: counted, unordered like today); `Match.sets: MatchSets` (get/set over stored `setStarts: [Int]`, `setEnds: [Int]`); `Match.setNumbers: [Int: Int]`; `Rally.setNumber: Int` (`match?.setNumbers[index] ?? 1`); `Rally.isCounted: Bool`; `MatchCorrections.sets: SetsCorrection?` (`struct SetsCorrection: Codable { let starts: [Int]; let ends: [Int] }`, decoded with `decodeIfPresent`).

- [ ] **Step 1: Failing tests** — in-memory ModelContainer (as `TeamSidesTests`): 12 rallies, `match.sets = MatchSets(starts: [4,10], ends: [8])` → `match.rallies.count == 8`, `allRallies.count == 12`, `setNumber` of #10 is 2; corrections round-trip carries `sets` and an empty `MatchSets` encodes no `sets` key; `deleted_rally_indices` stays empty (not-counted ≠ deleted); a spike inside rally #2 is not in `match.actionClips()`; `CourtScoreTimeline(rallies: match.rallies).totals.map(\.setNumber) == [1, 2]`.
- [ ] **Step 2: Implement.** Every write to the relationship and every place that must see all rallies (sync, corrections build/apply, deleter, sets sheet) uses `allRallies`; everything else keeps `rallies`. Grep `\.rallies\b` and decide each of the ~58 sites; list the ones moved to `allRallies` in the commit message.
- [ ] **Step 3: Commit + push** `feat: rallies outside the user's sets stop counting`.

### Task 6: 設定局 sheet

**Files:**
- Create: `VolleyIQ/Screens/Analysis/MatchSetsSheet.swift`
- Modify: `VolleyIQ/Screens/Analysis/MatchAnalysisView.swift` (button beside `teamsButton`, `.appSheet`, `.task(id:)` also keyed on `match.sets`)
- Modify: `VolleyIQ/Resources/Localizable.xcstrings`

**Interfaces:**
- Consumes: `Match.sets`, `Match.allRallies`, `MatchSets.assign`, `commitCorrections(context:account:)`.

- [ ] **Step 1: Build the sheet** modelled on `TeamSidesSheet`: draft `MatchSets`; one row per rally in play order over `allRallies` with `Rally #n · time`, the pause before it (warning colour at ≥ 60 s), and the derived state from `MatchSets.assign` on the draft (`第 N 局` / `不算`); two trailing toggles 「局開頭」「局結尾」; an end on a not-counted rally shows dimmed with 「沒有作用」. Cancel / Done (`save`), and 「清除」 (destructive) when marks exist. Thin-line list rows, no tinted cards.
- [ ] **Step 2: Entry** — 「設定局」 button (label 「設定局：N 局」 once marked) beside 「設定我方」, only when `match.canEdit`.
- [ ] **Step 3: xcstrings** — add zh-Hant + en for: 設定局, 設定局：%lld 局, 局開頭, 局結尾, 第 %lld 局, 不算, 沒有作用, 清除, and the footer 「局開頭之前與局結尾之後的 Rally 不會算進比賽」.
- [ ] **Step 4: Commit + push** `feat: 設定局 sheet`.

### Task 7: Version gate

**Files:**
- Modify: `VolleyIQ/project.yml` (`MARKETING_VERSION: "1.6.0"`)
- Modify: `volleyiq-backend/upload-service/src/appVersion.ts` (`MIN_IOS_VERSION = "1.6.0"`) + its test expectations

- [ ] **Step 1:** bump both; `npm test` green.
- [ ] **Step 2: Commit** separately; the Worker change deploys only after 1.6.0 is on the App Store (an older App re-saving corrections would drop `sets`).

## Rollout (user-run)

1. Tasks 1–3: push, deploy the Worker (Task 3 only — not Task 7), restart yp-app.
2. Mac: build + run `MatchSetsFixtureTests`, `MatchSetsTests`, full suite; try the sheet. Then tag v1.6.0 → TestFlight → App Store.
3. After 1.6.0 is live: deploy the Worker with Task 7.
