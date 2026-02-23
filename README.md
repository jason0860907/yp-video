# yp-video

排球影片分析工具，用於建立 temporal segmentation 訓練資料集。整合 YouTube 下載、影片剪輯、排球活動偵測與標註功能。

## 功能

- **YouTube 下載** - 下載 YouTube 影片（CLI 或 Web UI 批次下載）
- **TPVL 重命名** - 批次將 TPVL 影片從長標題重命名為簡潔格式
- **影片剪輯** - Web UI 介面，將完整比賽影片切分為個別 set
- **排球偵測** - 使用 Qwen3-VL 模型分析 set 影片中的排球活動（支援並行處理）
- **VLM→Rally 轉換** - 將逐片段偵測結果合併為 rally 標註
- **Rally 標註** - 檢視自動標註並人工校正，產生 ground truth 訓練資料
- **TAD 推論** - 使用 Temporal Action Detection 模型預測 rally 片段

## 安裝

```bash
# 使用 uv 安裝依賴
uv sync
```

需要系統安裝 `ffmpeg` 和 `ffprobe`。

## 使用方式

### 1. YouTube 下載

#### CLI 單一影片下載

```bash
# 下載影片（預設輸出至 ~/videos）
uv run yp-download "https://youtube.com/watch?v=xxx"

# 指定畫質
uv run yp-download "https://youtube.com/watch?v=xxx" -q 720

# 下載純音訊 (MP3)
uv run yp-download "https://youtube.com/watch?v=xxx" --audio-only

# 指定輸出目錄
uv run yp-download "https://youtube.com/watch?v=xxx" -o ~/my-videos

# 列出可用格式
uv run yp-download "https://youtube.com/watch?v=xxx" --list
```

#### Web UI 批次下載（播放清單）

```bash
uv run yp-downloader
```

開啟瀏覽器至 http://localhost:8001

功能：
- 貼上 YouTube 播放清單網址
- 勾選要下載的影片
- 批次下載並顯示進度

### 2. TPVL 影片重命名

將 TPVL 影片從長標題重命名為簡潔格式：

```
原始：【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL 2025-26 例行賽 G96 5/17 18:30 台中連莊 vs 桃園雲豹飛將.mp4
目標：2025-05-17_G96_台中連莊_vs_桃園雲豹飛將.mp4
```

```bash
# 預覽變更（不實際執行）
uv run python -m youtube.rename_tpvl --dry-run

# 執行重命名（會詢問確認）
uv run python -m youtube.rename_tpvl

# 直接執行不詢問
uv run python -m youtube.rename_tpvl --yes

# 指定目錄
uv run python -m youtube.rename_tpvl -d ~/my-videos --dry-run
```

### 3. 影片剪輯 (Video Cutter)

將完整比賽影片切分為個別 set，方便後續分析。

```bash
uv run yp-cutter
```

開啟瀏覽器至 http://localhost:8002

功能：
- 從 `~/videos` 載入完整比賽影片
- 播放並標記每個 set 的起始/結束時間
- 批次匯出 set 影片至 `~/videos/cuts/`

快捷鍵：
- `←` / `→` - 快轉 5 秒

### 4. 排球活動偵測

對切分好的 set 影片進行 VLM 分析，初步偵測 rally 片段。

首先啟動 vLLM 伺服器：

```bash
# 使用預設模型 (Qwen3-VL-8B) 在 port 8000
./start_vllm_server.sh

# 或指定模型和 port
./start_vllm_server.sh Qwen/Qwen3-VL-4B-Instruct 8001
```

執行偵測：

```bash
# 基本使用
uv run python vlm_segment.py --video path/to/video.mp4

# 指定伺服器與輸出檔案
uv run python vlm_segment.py --video path/to/video.mp4 \
    --server http://localhost:8000 \
    --output results.json

# 調整分析參數與並行數量
uv run python vlm_segment.py --video path/to/video.mp4 \
    --clip-duration 6.0 \
    --slide-interval 3.0 \
    --batch-size 8
```

參數說明：
- `--video, -v` - 影片檔案路徑
- `--server, -s` - vLLM 伺服器 URL（預設：http://localhost:8000）
- `--model, -m` - 模型名稱（預設：Qwen/Qwen3-VL-8B-Instruct）
- `--clip-duration, -d` - 每個片段長度（秒，預設：6.0）
- `--slide-interval, -i` - 滑動視窗間隔（秒，預設：3.0）
- `--batch-size, -b` - 並行處理的片段數量（預設：32）
- `--output, -o` - 輸出 JSON 檔案路徑

### 5. VLM→Rally 轉換

將 VLM 逐片段偵測結果合併為 rally 標註：

```bash
uv run python -m tad.vlm_to_rally
# 讀取 ~/videos/seg-annotations/ → 輸出至 ~/videos/rally-pre-annotations/
```

#### Rally 自動合併邏輯

自動將連續的 clips 合併成 rally 片段：

```
輸入 clips:  [gameplay] [gameplay] [non-gameplay] [gameplay] [gameplay]
                 ↓          ↓            ↓             ↓          ↓
合併後:      [─── rally 1 ───]       分隔       [──── rally 2 ────]
```

判斷規則：
- **Gameplay** = `in_rally: true` **且** `shot_type: full_court`
- **Non-gameplay** = 其他情況（無排球活動、特寫畫面等）

合併規則：
1. 連續的 gameplay clips 合併為同一個 rally
2. 相鄰 gameplay clips 間隔 ≤ 2 秒也會合併
3. 遇到 non-gameplay 時結束當前 rally

### 6. Rally 標註器

人工校正自動產生的 rally 標註，產生 ground truth 訓練資料。

```bash
uv run yp-annotator
```

開啟瀏覽器至 http://localhost:8003

功能：
- 載入 `~/videos/rally-pre-annotations/`（自動）和 `~/videos/rally-annotations/`（人工校正）的檔案
- 已校正的版本優先顯示
- 播放影片並檢視各片段的偵測結果
- 微調 rally 邊界，標註 rally（keep）或非 rally（skip）
- 儲存校正結果至 `~/videos/rally-annotations/`

## 工作流程範例

完整的 **下載 → 剪輯 → 偵測 → 轉換 → 標註** 流程：

```bash
# 1. 下載 YouTube 比賽影片
uv run yp-download "https://youtube.com/watch?v=xxx"
# 或使用 Web UI 批次下載：uv run yp-downloader

# 2. 重命名 TPVL 影片（可選）
uv run python -m youtube.rename_tpvl

# 3. 切分比賽影片為個別 set
uv run yp-cutter
# 輸出至 ~/videos/cuts/

# 4. 啟動 vLLM 伺服器（另開 terminal）
./start_vllm_server.sh

# 5. 對 set 影片進行 VLM 排球活動偵測
uv run python vlm_segment.py --video ~/videos/cuts/set1.mp4
# 輸出至 ~/videos/seg-annotations/

# 6. 將 VLM 片段偵測合併為 rally 標註
uv run python -m tad.vlm_to_rally
# 讀取 ~/videos/seg-annotations/ → ~/videos/rally-pre-annotations/

# 7. 人工校正標註，產生 ground truth
uv run yp-annotator
# 讀取 rally-pre-annotations + rally-annotations，存入 rally-annotations
```

產生的標註資料可用於訓練 temporal segmentation 模型。

### ~/videos 目錄結構

```
~/videos/
├── cuts/                    # 剪輯後的 set 影片
├── seg-annotations/         # VLM 逐片段偵測結果（自動）
├── rally-pre-annotations/   # 自動合併的 rally 標註（自動）
├── rally-annotations/       # 人工校正後的 ground truth（人工）
└── tad-predictions/         # TAD 模型預測結果（自動）
```

## 專案結構

```
yp-video/
├── pyproject.toml            # 專案設定與依賴
├── vlm_segment.py      # VLM 排球偵測主程式
├── start_vllm_server.sh  # vLLM 伺服器啟動腳本
├── utils/                    # 共用工具
│   └── ffmpeg.py             # FFmpeg 操作函式
├── youtube/                  # YouTube 相關功能
│   ├── download.py           # YouTube 下載器（CLI）
│   ├── rename_tpvl.py        # TPVL 影片重命名
│   ├── downloader/           # 批次下載器（Web UI）
│   │   ├── main.py           # FastAPI 伺服器
│   │   └── static/           # Web UI
│   └── cutter/               # 影片剪輯器
│       ├── main.py           # FastAPI 伺服器
│       └── static/           # Web UI
├── annotator/                # Rally 標註器
│   ├── main.py               # FastAPI 伺服器
│   └── static/               # Web UI
└── tad/                      # Temporal Action Detection
    ├── vlm_to_rally.py       # VLM→rally 轉換
    ├── convert_annotations.py # 標註格式轉換
    ├── infer.py              # TAD 推論
    └── output_converter.py   # MambaTAD 輸出轉換
```

## CLI 指令

安裝後可使用以下指令：

| 指令 | 說明 |
|------|------|
| `yp-download` | 下載 YouTube 影片（CLI） |
| `yp-downloader` | 啟動批次下載伺服器（Web UI，port 8001） |
| `python -m youtube.rename_tpvl` | 批次重命名 TPVL 影片 |
| `yp-cutter` | 啟動影片剪輯伺服器（Web UI，port 8002） |
| `yp-annotator` | 啟動 Rally 標註伺服器（Web UI，port 8003） |
