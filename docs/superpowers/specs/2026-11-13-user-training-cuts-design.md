# 用戶影片加入訓練資料（cuts-user-*）＋ 四種拍攝角度

日期：2026-11-13　狀態：設計，待使用者核准

## 為什麼

使用者（開發者）在手機 App 看到用戶上傳的比賽，覺得適合拿來訓練，但目前沒有路徑把
用戶影片變成可標註的訓練資料。現有 App Review 只能把用戶的修正**套回已經在訓練集裡的
影片**（operator-published cuts，見 `docs/app-review.md`），無法把一部新的用戶比賽加進來。

同時 App 上傳時的「拍攝角度」只有手機／固定機位／電視三個圖示，不足以描述真正的機位，
也無法作為訓練資料分類的依據。

## 目標

1. 一顆「加入訓練資料」按鈕，把一場用戶比賽的原片複製進訓練 bucket 的
   `cuts-user-<角度>/`，之後在 yp-video 的標註佇列出現，像 sideline / broadcast 一樣可以標
   Rally、Action（及其後的流程）。
2. 按鈕在兩個地方：
   - yp-video 的 App Review 頁（看某用戶某場比賽時）——即時執行。
   - 開發者版 App（`#if DEBUG`）的比賽頁「⋯」選單（和「動作門檻…」同一處）——送出請求，
     由 yp-video 取走執行（非即時，見下）。
3. 加入時把該比賽的**線上分析結果＋用戶的修正**帶過來當預標（rally 與 action），
   使用者從修正開始標。
4. App 上傳的拍攝角度改成四種，**取代**現有三種，所有用戶都從這四個選。
5. 上傳畫面每個角度左邊畫俯視球場示意圖（已選定樣式 A，見下）。

## 非目標

- 用戶刪除比賽或帳號時，**不刪**訓練資料裡的複本與標註（使用者決定保留）。隱私權政策已涵蓋
  「經授權的團隊成員…用於改進我們的分析模型」。
- 不改訓練／評估 recipe：user-* kinds 先只進標註佇列，是否、如何進入訓練是之後另案。
- 後端分析（worker）不讀角度，不需改動。
- 不做相機標定、不從畫面自動判斷角度。

## 四種角度

| 順序 | 中文 | iOS `CameraAngle` / D1 `matches.angle` | yp-video cut kind | R2 分類／本機資料夾 |
|---|---|---|---|---|
| 1 | 底線 | `baseline` | `user-baseline` | `cuts-user-baseline` |
| 2 | 角落 | `corner` | `user-corner` | `cuts-user-corner` |
| 3 | 高處 | `high` | `user-high` | `cuts-user-high` |
| 4 | 中線側面 | `center` | `user-center` | `cuts-user-center` |

- 底線：站在一方底線後方，正對整個球場。
- 角落：現在 sideline 最常見，在某個角落斜拍。
- 高處：從看台／二樓往下拍。
- 中線側面：在中線旁側拍、兩個半場都入鏡，類似轉播但鏡頭不動。

R2 分類用平的 `cuts-user-<角度>`，與現有 `cuts-sideline` / `cuts-broadcast` 同一形狀
（使用者決定）：`config.py` 的 `CUT_KINDS` / `R2_CATEGORIES` / `CutKind` 每個角度加一行即可。

### 既有資料一次轉換（不保留舊值、不做 fallback）

| 舊值 | 新值 |
|---|---|
| `phoneSideline` | `corner` |
| `broadcast` | `center` |
| `fixedVenue` | `high` |

（三項皆使用者決定。）

轉換範圍：
- D1 `matches.angle`：migration 一次改完。
- upload-service：所有寫死舊值的地方（如 `library/media.ts` 預設 `"phoneSideline"`、
  `jobs.ts` 的 INSERT）改成新值；新增**伺服器端驗證**，只接受四個新值，其他回 400。
- App 最低版本閘門（`MIN_IOS_VERSION`，現為 1.4.3）升到帶新角度的那一版，舊版 App
  不會再寫入舊值。代價：所有用戶要更新才能上傳（我的判斷）。
  部署順序：iOS 上架 → migration + Worker（含驗證與閘門）一起上。
- iOS SwiftData 既有 `Match.angle`：`CameraAngle` 換 case 後舊 raw value 解不開，需一次性處理
  （SwiftData 勿用 enum predicate，見 iOS 慣例）；從 library 同步回來的值已是新值。
- fixture／測試裡的舊值一併改。轉換完刪掉舊 case 與舊字串。

### 「我方」預設邊（`TeamSides`）

目前 `angle == .broadcast ? .left : .near`。改成：`center` → `.left`，其他三種 → `.near`
（我的判斷）。

## 上傳畫面（iOS `UploadAngleStep`）

已選定樣式 A（mockup：`volleyiq/.superpowers/brainstorm/*/content/angle-diagram.html`）：

- 四列，依上表順序。每列左邊一張俯視球場小圖（約 64×64）：球場長方形＋網線，藍點是攝影機，
  淺藍扇形是拍攝範圍。
  - 底線：攝影機在一端底線後方正中，扇形朝向整個球場。
  - 角落：攝影機在球場一角外側，斜向扇形。
  - 高處：攝影機在底線後方，旁邊標「高」（平面圖表達不了高度，靠標字）。
  - 中線側面：球場橫放、網線直立，攝影機在中線外側，扇形涵蓋兩個半場。
- 右側標題＋一行說明（文案用上表說明，可潤飾；中文文案不用句號）。
- 選取樣式沿用現有 row（選取時外框 primary 色）。小圖用 SwiftUI `Canvas`/`Path` 繪製，隨 theme 變色。
- 舊的 SF Symbol（`iphone.gen3`/`video.fill`/`tv`）與 `angle_phone*`/`angle_fixed*`/`angle_tv*`
  字串刪除，新增四組 xcstrings key（中文原文即 key 的慣例照舊）。

## 加入訓練資料

### 哪些比賽可以加入（我的判斷）

- 可以：原片在 customer bucket 的比賽（用戶上傳、YouTube 匯入）。
- 拒絕：operator-published cut（原片本來就在 pipeline bucket、已是訓練資料），回報原因。
- 分享匯入（唯讀引用）的比賽：以原擁有者的那場為準（match id、角度、修正都取原擁有者的）。

### 共同的匯入動作（yp-video 執行）

輸入：user id + match id + 角度（預設該比賽的 `angle`，按的人可以改）。

1. 原片從 customer bucket 複製到 pipeline bucket：`cuts-user-<角度>/<match_id>.mp4`。
   用 match id 當檔名：冪等、可回查來源。原片不重新編碼。
2. 預標：用 App Review 既有的 `sources.match_bundle` / `projection.project`
   （分析結果＋用戶修正後的投影）轉成 yp-video 的 rally 預標與 action 預標，寫進
   **`rally-spot/pre-annotations` 與 `action/pre-annotations`**（與其他預標同處），同步 R2。
   不寫進人工 `annotations` 資料夾、不標 Done，避免模型輸出被當成人工標註拿去訓練。
   action 事件保留 score。檔名照既有預標命名慣例。
3. 依「所有 cut 只在 R2」原則，本機不留 cut 副本（讀幀流程照既有 resolve_cut 下載再刪）。

重複與改角度（我的判斷）：
- 同一場再按一次 → 已存在就什麼都不做、回報「已加入」，**不重建、不覆蓋**預標
  （可能已經有人在標）。
- 改角度 → 只搬檔案（刪舊 key、寫新 key，預標／標註檔跟著改名），不重新產生預標。

### yp-video 端

- `config.py`：四個 kind 加進 `CUT_KINDS`、`CutKind`、`R2_CATEGORIES`；`cut_kind_of()` 能辨識。
- 標註佇列、kind 下拉、worklists、label_sources 等凡是列舉 kind 的地方都認得新 kinds。
- App Review 頁（`pages/app-review`）比賽詳細區加「加入訓練資料」按鈕＋角度選擇；
  新端點 `POST /app-review/users/{user}/matches/{match}/training`，即時執行匯入動作。
- **請求取件**：yp-video 定期（數分鐘一次）呼叫 Worker admin API 取出待處理的「加入訓練請求」，
  逐筆執行匯入動作，回寫結果（完成／失敗＋原因）。yp-video 沒開時請求就排著。

### 後端（upload-service）

- D1 新 migration：
  - `users` 加開發者標記（例如 `developer INTEGER NOT NULL DEFAULT 0`），只開使用者自己的帳號；
    由使用者手動 `UPDATE`（wrangler remote 由使用者執行）。
  - `training_requests` 表：match、要求的角度、提出者、狀態（pending／done／failed）、錯誤、時間。
  - `matches.angle` 轉換（見上）。
- App 端點（需登入且 `developer = 1`，否則 403）：建立請求、查詢某場比賽的最新請求狀態。
- admin 端點（`AUTH_TOKEN`）：列出 pending、回寫結果——供 yp-video 取件。
- `/quota` 或既有帳號資訊回傳 developer 旗標，讓 DEBUG App 決定是否顯示選單項（顯示與否只是
  介面；真正的保護在伺服器）。

### 開發者 App 端（DEBUG）

- `RallyFeedView` 的「⋯」選單 `#if DEBUG` 區塊新增「加入訓練資料…」（帳號是 developer 才顯示），
  開小 sheet：角度選擇（預設 `match.angle`）＋送出，顯示請求狀態（排隊中／已加入／失敗原因）。

## 限制與慣例

- 不保留向後相容（CLAUDE.md）：舊角度值直接轉換後刪除。
- D1 remote migration／Worker deploy 由使用者執行。
- iOS 在這台 Linux 無法編譯，先 push 再由使用者在 Mac build。
- 寫入類端點的測試一律用 scratch 資料，不碰真實標註。
- commit 英文 Conventional Commits、依主題拆、測完才 push（iOS 例外）。

## 交給 builder 的部分

- 小圖的精確幾何、顏色、選取時的變化。
- 端點錯誤訊息、前端按鈕的確切位置與樣式、取件間隔。
- 預標轉換細節（rally 邊界、action 欄位對應），以現有 projection 為準。
- 測試範圍與 fixture。

## 我做的判斷（請使用者檢查）

1. 英文識別字 `baseline / corner / high / center`。
2. 檔名用 match id；重複加入什麼都不做、不覆蓋預標；改角度只搬檔案。
3. 「我方」預設：`center` 左邊，其他近方。
4. 預標放 `pre-annotations`，不進人工標註。
5. 伺服器只收四個新角度值＋App 最低版本閘門升級（所有用戶需更新才能上傳）。
6. operator-published cut 拒絕加入；分享匯入以原擁有者為準。
7. 開發者標記放 D1 `users`，由使用者手動開給自己。
8. 原片不重新編碼。
