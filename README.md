# yp-video

排球影片分析 pipeline，整合影片下載、剪輯、VLM 偵測、Rally 標註、TAD 模型訓練與推論，全部透過統一的 Web Dashboard 操作。

## 功能

- **Download** - 批次下載 YouTube 播放清單影片
- **Cut** - 將完整比賽影片切分為個別 set
- **Detect** - 使用 Qwen3-VL 模型偵測 rally 片段（VLM + 投票平滑）
- **Annotate** - 檢視偵測結果並人工校正，產生 ground truth
- **Train** - 標註轉換、R3D-18 特徵提取、ActionFormer TAD 模型訓練
- **Predict** - 使用訓練好的 TAD 模型預測 rally 片段
- **Jobs** - 監控背景任務、控制 vLLM 伺服器

## 安裝

```bash
# 使用 uv 安裝依賴
uv sync
```

需要系統安裝 `ffmpeg` 和 `ffprobe`。

## 使用方式

### Web Dashboard（主要入口）

```bash
uv run yp-app
```

開啟瀏覽器至 http://localhost:8080，即可操作所有功能。

### CLI 工具

```bash
# 下載 YouTube 影片
uv run yp-download "https://youtube.com/watch?v=xxx"
uv run yp-download "https://youtube.com/watch?v=xxx" -q 720
uv run yp-download "https://youtube.com/watch?v=xxx" --audio-only

# VLM 偵測（需先啟動 vLLM 伺服器）
./start_vllm_server.sh
uv run yp-vlm-segment --video ~/videos/cuts/set1.mp4

# 批次偵測多場比賽
./rally.sh G1 G2 G3

# TAD 相關
uv run yp-tad-extract    # 提取 R3D-18 特徵
uv run yp-tad-train      # 訓練 TAD 模型
uv run yp-tad-infer      # TAD 推論

# TPVL 影片重命名
uv run python -m yp_video.youtube.rename_tpvl --dry-run
```

### VLM 偵測參數

```bash
uv run yp-vlm-segment --video path/to/video.mp4 \
    --server http://localhost:8000 \
    --clip-duration 6.0 \
    --slide-interval 3.0 \
    --batch-size 32
```

## 工作流程

```
Download → Cut → Detect → Annotate → Train → Predict
   │        │       │         │         │        │
   │        │       │         │         │        └─ TAD 模型預測 rally
   │        │       │         │         └─ 轉換標註 + 提取特徵 + 訓練
   │        │       │         └─ 人工校正 → ground truth
   │        │       └─ VLM 偵測 + 投票平滑 → pre-annotations
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
│   │   └── vlm_segment.py      # VLM 排球偵測
│   ├── tad/                    # Temporal Action Detection
│   │   ├── train.py            # 模型訓練
│   │   ├── infer.py            # 模型推論
│   │   ├── extract_features.py # R3D-18 特徵提取
│   │   ├── convert_annotations.py  # 標註格式轉換
│   │   ├── vlm_to_rally.py     # VLM→Rally 合併
│   │   ├── output_converter.py # TAD 輸出轉換
│   │   ├── configs/            # MMEngine 設定檔
│   │   ├── data/               # 特徵與標註資料
│   │   └── checkpoints/        # 模型權重
│   ├── youtube/                # CLI 工具
│   │   ├── download.py         # YouTube 下載
│   │   └── rename_tpvl.py      # TPVL 重命名
│   └── web/                    # Web Dashboard
│       ├── app.py              # FastAPI 應用
│       ├── jobs.py             # 背景任務管理
│       ├── vllm_manager.py     # vLLM 生命週期管理
│       ├── routers/            # API 路由
│       └── static/             # 前端 SPA
├── OpenTAD/                    # TAD 框架（外部依賴）
├── prompts/                    # VLM Prompt 模板
├── vllm.env                    # vLLM 伺服器設定
├── rally.sh                    # 批次偵測腳本
├── start_vllm_server.sh        # vLLM 啟動腳本
└── pyproject.toml
```

## 資料目錄

```
~/videos/
├── cuts/                    # 剪輯後的 set 影片
├── seg-annotations/         # VLM 逐片段偵測結果（自動）
├── rally-pre-annotations/   # 投票平滑後的 rally 標註（自動）
├── rally-annotations/       # 人工校正後的 ground truth
└── tad-predictions/         # TAD 模型預測結果
```

## CLI 指令一覽

| 指令 | 說明 |
|------|------|
| `yp-app` | 啟動 Web Dashboard（port 8080） |
| `yp-download` | 下載 YouTube 影片（CLI） |
| `yp-vlm-segment` | VLM 排球偵測（CLI） |
| `yp-tad-extract` | 提取 R3D-18 影片特徵 |
| `yp-tad-train` | 訓練 TAD 模型 |
| `yp-tad-infer` | TAD 模型推論 |
