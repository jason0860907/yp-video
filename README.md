# yp-video

排球影片分析工具，整合 YouTube 下載、影片剪輯與排球活動偵測功能。

## 功能

- **YouTube 下載** - 下載 YouTube 影片（支援多種畫質與純音訊）
- **影片剪輯** - Web UI 介面，可視覺化標記並匯出影片片段
- **排球偵測** - 使用 Qwen3-VL 模型分析影片中的排球活動（支援並行處理）
- **Rally 標註** - 檢視偵測結果並標註 rally 片段

## 安裝

```bash
# 使用 uv 安裝依賴
uv sync
```

需要系統安裝 `ffmpeg` 和 `ffprobe`。

## 使用方式

### 1. YouTube 下載

```bash
# 下載影片（預設輸出至 ~/videos）
python -m youtube.download "https://youtube.com/watch?v=xxx"

# 指定畫質
python -m youtube.download "https://youtube.com/watch?v=xxx" -q 720

# 下載純音訊 (MP3)
python -m youtube.download "https://youtube.com/watch?v=xxx" --audio-only

# 指定輸出目錄
python -m youtube.download "https://youtube.com/watch?v=xxx" -o ~/my-videos

# 列出可用格式
python -m youtube.download "https://youtube.com/watch?v=xxx" --list
```

### 2. 影片剪輯 (Video Cutter)

啟動 Web 伺服器：

```bash
uvicorn youtube.cutter.main:app --port 8001
```

開啟瀏覽器至 http://localhost:8001

功能：
- 從 `~/videos` 載入影片
- 播放並標記起始/結束時間點
- 批次匯出多個片段至 `~/videos/cuts/`

快捷鍵：
- `←` / `→` - 快轉 5 秒

### 3. 排球活動偵測

首先啟動 vLLM 伺服器：

```bash
# 使用預設模型 (Qwen3-VL-8B) 在 port 8000
./start_qwen3_vl_server.sh

# 或指定模型和 port
./start_qwen3_vl_server.sh Qwen/Qwen3-VL-4B-Instruct 8001
```

執行偵測：

```bash
# 基本使用
python detect_volleyball.py --video path/to/video.mp4

# 指定伺服器與輸出檔案
python detect_volleyball.py --video path/to/video.mp4 \
    --server http://localhost:8000 \
    --output results.json

# 調整分析參數與並行數量
python detect_volleyball.py --video path/to/video.mp4 \
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

### 4. Rally 標註器

檢視偵測結果並標註 rally 片段：

```bash
# 啟動標註伺服器
yp-annotator

# 或使用 uvicorn
uvicorn annotator.main:app --port 8002
```

開啟瀏覽器至 http://localhost:8002

功能：
- 載入偵測結果 JSON 檔案
- 播放影片並檢視各片段的偵測結果
- 標註 rally（keep）或非 rally（skip）
- 儲存標註結果

#### Rally 自動合併邏輯

載入偵測結果時，會自動將連續的 clips 合併成 rally 片段：

```
輸入 clips:  [gameplay] [gameplay] [non-gameplay] [gameplay] [gameplay]
                 ↓          ↓            ↓             ↓          ↓
合併後:      [─── rally 1 ───]       分隔       [──── rally 2 ────]
```

判斷規則：
- **Gameplay** = `has_volleyball: true` **且** `shot_type: full_court`
- **Non-gameplay** = 其他情況（無排球活動、特寫畫面等）

合併規則：
1. 連續的 gameplay clips 合併為同一個 rally
2. 相鄰 gameplay clips 間隔 ≤ 2 秒也會合併
3. 遇到 non-gameplay 時結束當前 rally

## 工作流程範例

完整的 **下載 → 分析 → 標註 → 剪輯** 流程：

```bash
# 1. 下載 YouTube 影片
yp-download "https://youtube.com/watch?v=xxx"

# 2. 啟動 vLLM 伺服器（另開 terminal）
./start_qwen3_vl_server.sh

# 3. 分析排球活動
python detect_volleyball.py --video ~/videos/影片名稱.mp4 --output results.json

# 4. 檢視結果並標註（可選）
yp-annotator

# 5. 根據分析結果剪輯精彩片段
yp-cutter
```

## 專案結構

```
yp-video/
├── pyproject.toml            # 專案設定與依賴
├── detect_volleyball.py      # 排球偵測主程式
├── start_qwen3_vl_server.sh  # vLLM 伺服器啟動腳本
├── utils/                    # 共用工具
│   └── ffmpeg.py             # FFmpeg 操作函式
├── youtube/                  # YouTube 相關功能
│   ├── download.py           # YouTube 下載器
│   └── cutter/               # 影片剪輯器
│       ├── main.py           # FastAPI 伺服器
│       └── static/           # Web UI
├── annotator/                # Rally 標註器
│   ├── main.py               # FastAPI 伺服器
│   └── static/               # Web UI
└── InternVideo/              # InternVideo 模型（子模組）
```

## CLI 指令

安裝後可使用以下指令：

| 指令 | 說明 |
|------|------|
| `yp-download` | 下載 YouTube 影片 |
| `yp-cutter` | 啟動影片剪輯伺服器 |
| `yp-annotator` | 啟動 Rally 標註伺服器 |

## 依賴

主要依賴：
- `yt-dlp` - YouTube 下載
- `fastapi` + `uvicorn` - Web 伺服器
- `torch` + `transformers` - 模型推論
- `aiohttp` - 並行 API 請求
- `tqdm` - 進度條顯示
- `ffmpeg` (系統) - 影片處理
