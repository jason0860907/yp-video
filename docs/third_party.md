# third_party 外部模型 checkout

ReID pipeline 的三個外部研究 repo 住在 `<volleyiq>/third_party/`(yp-video repo 之外,不進版控)。
它們全部遵循同一套掛載模式:

1. **路徑**由 `src/yp_video/config.py` 定義,環境變數可覆蓋(`KPR_DIR` / `SAM3D_DIR` / `CLIP_REIDENT_DIR`),預設是 `<repo 上兩層>/third_party/<名稱>`
2. **匯入**是執行時 lazy 地把 checkout 塞進 `sys.path`(依賴裝在 yp-video 自己的 venv,checkout 本身不 pip install)
3. **註冊**只在權重檔存在時發生 —— 缺權重就自動從 registry 退場,`/reid/options` 不會列出它,其餘功能不受影響

換機器重建時,照下面各節的步驟 clone + 下載權重即可,程式碼零修改。

## 總覽

| checkout | 來源 | 角色 | 權重來源 | 膠水程式碼 |
|---|---|---|---|---|
| `kpr/` | [VlSomers/keypoint_promptable_reidentification](https://github.com/VlSomers/keypoint_promptable_reidentification)(ECCV'24) | `kpr` embedder | 手動下載(repo README 的連結) | `reid/embedder.py::KprEmbedder` |
| `sam-3d-body/` | [facebookresearch/sam-3d-body](https://github.com/facebookresearch/sam-3d-body)(Meta) | `sam-3d-body` keypoint source | Hugging Face(gated) | `reid/sam3d.py` |
| `clip_reident/` | [KonradHabel/clip_reid](https://github.com/KonradHabel/clip_reid)(MMSports'22) | `clip-reident` embedder | Google Drive | `reid/clip_reident.py` |

## KPR — Keypoint Promptable ReID

本人骨架當正向 prompt、crop 裡其他人的骨架當負向,embedding 鎖定目標球員 —— 針對攔網/救球堆疊的遮擋場景。用 SOLIDER backbone(跨 domain 遮擋基準較強)。

```bash
git clone https://github.com/VlSomers/keypoint_promptable_reidentification third_party/kpr
# 權重:到 repo README 的 model zoo 下載 SOLIDER / Occluded-PoseTrack 版,放到:
#   third_party/kpr/pretrained_models/kpr_occ_pt_SOLIDER_81.24_90.59_42326409.pth.tar
```

設定檔 `configs/kpr/solider/kpr_occ_posetrack_test.yaml` 由 `reid/embedder.py` 的 `KPR_CONFIG` 指定;輸入尺寸/normalization 是 extractor 的建構參數,不從 cfg 讀 —— 兩處都跟著 config 走,升級時別讓它們分岔。

## SAM 3D Body(Meta)

可提示的單張人體網格重建模型,不是偵測器 —— 框永遠來自 RF-DETR,它只負責把每個候選演員的骨架重估一次(MHR-70 → OpenPose → COCO-17),比 RF-DETR 內建 keypoint head 準但慢(top-down,逐人 forward,只跑接觸點附近的高信心框)。

```bash
git clone https://github.com/facebookresearch/sam-3d-body third_party/sam-3d-body
# 權重 gated,先在 HF 網頁申請存取,然後:
hf download facebook/sam-3d-body-dinov3 \
    --local-dir third_party/sam-3d-body/checkpoints/sam-3d-body-dinov3
```

存在檢查看兩個檔案:`checkpoints/sam-3d-body-dinov3/model.ckpt` 和 `…/assets/mhr_model.pt`。

## CLIP-ReIdent(MMSports'22)

CLIP 的語言-影像對比訓練改寫成影像-影像 InfoNCE,在 MMSports 2022 籃球轉播球員資料上微調(該挑戰賽冠軍)—— 三個 embedder 中訓練 domain 最接近排球轉播的一個。OpenCLIP ViT-L/14 去投影層,輸出 1024 維。

```bash
git clone https://github.com/KonradHabel/clip_reid third_party/clip_reident
cd third_party/clip_reident
gdown 1Gm5J19okhLdnZTQLUsjfYoI0rwrLQ09i -O model/checkpoints.zip
unzip model/checkpoints.zip -d model/ && mv model/model/* model/ && rmdir model/model
rm model/checkpoints.zip   # 3.6G,解完即可刪
```

使用的 checkpoint:`model/ViT-L-14_openai/all_data_seed_1/weights_e4.pth`。

**QuickGELU 陷阱**:checkpoint 訓練年代的 open_clip 對 `('ViT-L-14', 'openai')` 隱式使用 QuickGELU;新版 open_clip 必須明確用 `ViT-L-14-quickgelu` 才對得上,否則權重照樣載入成功但 activation 錯了、精度默默劣化。`reid/clip_reident.py` 已編碼此 workaround。

## pip 層級的外部模型(不需 checkout)

- **RF-DETR**(`rfdetr` 套件):人物偵測 + 預設 keypoint source + rally tracking 的密集偵測;權重自動下載到 `~/.roboflow/`
- **CLIP-ReID**(預設 embedder):HF `occurra/person_vit_clip_reid` 的 ONNX,onnxruntime CPU 推論,首次使用自動下載
- **ByteTrack**:來自 `supervision` 套件

## 升級注意事項

- `supervision` 0.28 起 `ByteTrack` 標為 deprecated、**0.30 移除**(遷去獨立的 `trackers` 套件)—— 升級前先遷移 `reid/tracking.py`
- `rfdetr` 的 `optimize_for_inference()` 會改變 `predict()` 回傳型別(丟失 keypoint 包裝)且 batch 維度烙死在 traced graph 裡;`>1.8.3` 另有分數尺度正規化,升級要重新校準 `reid/detector.py` 的 0.1 / 0.5 門檻
- 任何外部模型升級後,拿已標注影片重跑一次對應的驗證(embedder 看 labeled-pair 距離分佈、tracking 看 tracklet 數量與同軌一致性)再信任結果
