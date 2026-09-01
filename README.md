# yp-video

排球影片分析 pipeline，整合影片下載、剪輯、VLM 偵測、Rally / Action 標註、SPOT 模型訓練與推論，全部透過統一的 Web Dashboard 操作。

## 功能

- **Download** - 批次下載 YouTube 播放清單影片
- **Cut** - 將完整比賽影片切分為個別 set
- **Detect** - 使用 Qwen3-VL 模型偵測 rally 片段（VLM + 投票平滑）
- **Annotate** - 檢視偵測結果並人工校正 rally 標註
- **Action Annotate** - 逐 frame 動作事件標註（serve / receive / set / spike / block / score）
- **Train** - 用校正後的標註訓練 SPOT 模型（rally 分段與 action 事件各一套流程）
- **Predict** - 用訓練好的 SPOT checkpoint 對影片做推論
- **Jobs** - 監控背景任務、控制 vLLM 伺服器
- **Audit** - 誰在什麼時候做了什麼：每個會改變狀態的操作與背景工作都留下紀錄

SPOT 模型本體住在獨立的 `~/yp-spot` repo（自己的 venv），yp-video 透過 subprocess + JSON 檔案跨進程呼叫它——這裡只負責組指令、解析 checkpoint、轉換輸出格式。

## 安裝

```bash
# 使用 uv 安裝依賴
uv sync

# 所有設定都在 workspace 根目錄的單一 .env（R2 金鑰、服務 token、
# Cloudflare Access、稽核資料庫、vLLM 參數全都在裡面）
cp ../.env.example ../.env    # 填好之後 ../scripts/sync-env.sh 會散佈給後端兩個元件

# 稽核資料庫（本機 Postgres，只綁 loopback）
make db-up
```

需要系統安裝 `ffmpeg` 和 `ffprobe`，以及 `docker compose`。

## 存取與稽核

這個 dashboard 只從 `https://label.volley-iq.com` 進入，由 **Cloudflare Access**
擋在 named tunnel 前面。yp-app 只綁 `127.0.0.1`，cloudflared 是它唯一的客戶端：
沒有 LAN 直連、沒有本機後門，因為任何繞過登入的路徑都會在稽核紀錄裡變成一筆
沒有名字的操作。

每個非 GET 的 `/api` 請求，以及每次背景工作的狀態轉換，都會寫進 Postgres 的
`audit_events`：時間、執行者（Access 驗出來的 email）、動作、對象、關鍵數字與
結果。請求內容本身不會存。Rally / Action 編輯器的自動存檔會摺疊成一列（`repeats`），
否則一次標註就會把整張表淹掉。

一次性設定（在 Cloudflare Zero Trust 後台）：

1. **Networks → Tunnels** 建立 named tunnel `yp-video`，Public Hostname 設成
   `label.volley-iq.com` → `http://localhost:8080`。
2. **Access → Applications** 新增 self-hosted application，網域填同一個 hostname，
   policy 允許標註人員的 email。
3. 把該 application 的 **Application Audience (AUD) Tag** 與 team 網域填進 `~/volleyiq/.env`
   的 `CF_ACCESS_AUD` / `CF_ACCESS_TEAM_DOMAIN`。

前端開發（`npm run dev`）也需要一張真的 Access token，因為後端不接受無身分請求：

```bash
cloudflared access login https://label.volley-iq.com
export YP_ACCESS_TOKEN=$(cloudflared access token -app=https://label.volley-iq.com)
```

ReID 的 appearance embedder（CLIP-ReIdent 系譜）訓練/推論住在 sibling package `../yp-reid/`（獨立 venv，subprocess 邊界 + contract 握手，比照 yp-spot）。細節見 [docs/third_party.md](docs/third_party.md)。

## 使用方式

### Web Dashboard（主要入口）

```bash
uv run yp-app
```

開啟瀏覽器至 http://localhost:8080，即可操作所有功能。

### CLI 工具（按 Pipeline 順序）

```bash
# 1. 下載 YouTube 影片
uv run yp-download "https://youtube.com/watch?v=xxx"
uv run yp-download "https://youtube.com/watch?v=xxx" -q 720

# 2. VLM 偵測（需先啟動 vLLM 伺服器）
./start_vllm_server.sh
uv run yp-vlm-segment --video ../videos/cuts-broadcast/set1.mp4

# 3. VLM 片段偵測 → Rally 標註合併
uv run yp-vlm-to-rally
# 讀取 videos/rally/seg-annotations/ → 輸出至 videos/rally/pre-annotations/

# 4. 人工校正標註、SPOT 訓練與推論 → 使用 Web Dashboard
```

### VLM 偵測參數

```bash
uv run yp-vlm-segment --video path/to/video.mp4 \
    --server http://localhost:8000 \
    --clip-duration 6.0 \
    --slide-interval 3.0 \
    --batch-size 32
```

### TPVL 影片重命名（選用）

```bash
uv run python -m yp_video.youtube.rename_tpvl --dry-run
```

## 工作流程

```
Download → Cut → Detect → VLM→Rally → Annotate → Train → Predict
   │        │       │         │          │          │        │
   │        │       │         │          │          │        └─ SPOT 推論（rally / action）
   │        │       │         │          │          └─ 訓練 SPOT 模型（yp-spot）
   │        │       │         │          └─ 人工校正 → ground truth
   │        │       │         └─ 片段偵測合併為 rally 標註
   │        │       └─ VLM 偵測（Qwen3-VL）
   │        └─ 切分為個別 set
   └─ 下載 YouTube 影片
```

全部步驟都可在 Web Dashboard (`yp-app`) 中完成。

## 專案結構

```
yp-video/
├── src/yp_video/               # 主要程式碼
│   ├── config.py               # 集中管理路徑與設定
│   ├── core/                   # 核心邏輯（無 Web 依賴）
│   │   ├── ffmpeg.py           # FFmpeg 影片處理
│   │   ├── vlm_segment.py      # VLM 排球偵測
│   │   ├── vlm_to_rally.py     # VLM 片段 → rally 標註合併
│   │   ├── jsonl.py            # JSONL 讀寫
│   │   └── sampling.py         # 影片取樣工具
│   ├── action/                 # SPOT 流程編排：frame 快取、預標、推論輸出轉換
│   ├── contracts/              # 跨 repo 資料格式（yp-video ↔ yp-spot / yp-reid / selfhost-worker）
│   ├── person/                 # 感知基元：人物偵測與 instance mask
│   ├── tracklets/              # 追蹤：誰在場上、何時（rally 內 track）
│   ├── extraction/             # 屋頂層：偵測 → 選 actor → 裁切 → embedding 的編排
│   ├── actor/                  # 誰做了這個動作（association 規則與學習策略）
│   ├── reid/                   # 這個人是誰（embedding、聚類、身分標註）
│   ├── youtube/                # CLI 工具
│   │   ├── download.py         # YouTube 下載
│   │   └── rename_tpvl.py      # TPVL 重命名
│   └── web/                    # Web Dashboard
│       ├── app.py              # FastAPI 應用
│       ├── access.py           # Cloudflare Access 驗證（誰在操作）
│       ├── audit.py            # 稽核軌跡：middleware + 寫入佇列
│       ├── db.py               # 稽核用 Postgres 連線池與 migration
│       │                       # （設定一律讀 workspace 根目錄的 ../.env）
│       ├── jobs.py             # 背景任務管理
│       ├── vllm_manager.py     # vLLM 生命週期管理
│       ├── routers/            # API 路由
│       └── frontend/           # React SPA（build 到 frontend/dist）
├── migrations/                 # 稽核資料庫 schema（NNNN_*.sql）
├── docker-compose.yml          # 本機 Postgres
├── prompts/                    # VLM Prompt 模板
├── start_vllm_server.sh        # vLLM 啟動腳本
└── pyproject.toml
```

> **`contracts/` 是跨 repo 公開 API**：`yp_video/contracts/__init__.py` 的
> re-export 除了 yp-spot / yp-reid 之外，也被 `volleyiq-backend/selfhost-worker`
> 以 `from yp_video.contracts import ...` 直接使用。就算在 yp-video 內部
> 看起來沒有人引用，也不可當 dead code 清除；`contracts/*.schema.json`
> 由 `make contract` 產生，是 iOS app 與 yp-spot 消費的 source of truth。

## 資料目錄

以 `src/yp_video/config.py` 為準（`YP_VIDEOS_DIR` 可覆寫，預設 workspace 的 `../videos/`）。
規則：屬於某個模型家族的東西——人工標註與機器產物——都住在該家族的目錄下；
各家族的 `annotations/` 子目錄是手工、不可重建的部分，其餘皆為衍生資料。

```
videos/
├── raw-videos/              # 下載的完整比賽影片
├── cuts-broadcast/          # 剪輯後的 set 影片（轉播視角）
├── cuts-sideline/           # 剪輯後的 set 影片（場邊視角）
├── rally/
│   ├── seg-annotations/     # VLM 逐片段偵測結果（自動）
│   └── pre-annotations/     # VLM 合併後的 rally 預標註（自動）
├── rally-spot/
│   ├── annotations/         # 人工校正後的 rally ground truth（含 winner）
│   └── pre-annotations/     # rally SPOT 推論結果
├── action/
│   ├── annotations/         # 人工校正後的 action ground truth
│   ├── pre-annotations/     # action SPOT 推論/預標結果
│   ├── frames/              # SPOT 訓練用 frame 快取
│   ├── audio/               # SPOT late-fusion 音訊特徵
│   └── waveforms/           # 標註輔助波形
├── spot/checkpoints/        # 所有 SPOT/association checkpoint package（依 manifest 分辨）
├── tracks/                  # 追蹤輸出：誰在場上（衍生）
├── extraction/              # 偵測記錄與 actor crop（records/、crops/、crops-masked/；衍生）
├── association/annotations/ # 人工 actor verdict（<stem>_actors.json）
├── reid/
│   ├── annotations/         # 人工球員身分（<stem>_players.json）
│   ├── checkpoints/         # yp-reid checkpoint package
│   ├── datasets/            # 匯出的 ReID 訓練資料集（衍生）
│   └── embeddings/          # crop embedding（衍生）
├── action-val-set.txt       # action recipe 的人工驗證集清單
└── label-done.jsonl         # 每支影片各標註模式的 Done 旗標
```

## CLI 指令一覽

| 指令 | Pipeline 順序 | 說明 |
|------|:---:|------|
| `yp-app` | — | 啟動 Web Dashboard（port 8080，只綁 loopback） |
| `yp-download` | 1 | 下載 YouTube 影片 |
| `yp-vlm-segment` | 2 | VLM 排球偵測 |
| `yp-vlm-to-rally` | 3 | VLM 片段偵測 → Rally 標註合併 |

SPOT 訓練與推論沒有獨立 CLI，統一走 Web Dashboard 的 Train / Predict 頁面。
