# 外部模型

Appearance embedder 的訓練與推論住在 sibling package
`<volleyiq>/yp-reid/`（獨立 venv，subprocess 邊界 + contract 握手，比照
yp-spot）。見下方〈ReID 模型（yp-reid）〉。

## 總覽

| 元件 | 來源 | 角色 | 權重來源 | 膠水程式碼 |
|---|---|---|---|---|
| `yp-reid/`（sibling package） | CLIP-ReIdent 系譜（[KonradHabel/clip_reid](https://github.com/KonradHabel/clip_reid)，MMSports'22），已 vendor 重寫 | `clip-reident` embedder + 訓練 | checkpoint package（`videos/reid/checkpoints/`） | `reid/checkpoints.py` + `reid/embedder.py::SubprocessEmbedder` |

## ReID 模型（yp-reid）

CLIP 的語言-影像對比訓練改寫成影像-影像 InfoNCE（Habel et al., MMSports'22，
籃球轉播球員 ReID 挑戰賽冠軍）。OpenCLIP ViT-L/14 去投影層，輸出 1024 維。
模型定義、訓練 loop、embedding CLI 全部重寫在 `yp-reid/` 裡（不再依賴
`third_party/clip_reident` 的程式碼）；yp-video 透過 contract
（`yp_video/contracts/reid.py` ⇄ `yp_reid/contract.py`）跨 subprocess 使用。

```bash
cd yp-reid && uv sync   # 自己的 venv：torch / open-clip-torch / opencv
```

權重以 **checkpoint package** 形式住在 `videos/reid/checkpoints/<run>/`
（manifest.json 完整描述 architecture 與 preprocessing，載入端不看 CLI 參數）。
自己訓練的 run 由 `yp_reid.train --export-dir` 直接產出；論文釋出的權重用一次性
import 包裝：

```bash
# 論文權重（Google Drive 釋出）若尚未下載：
git clone https://github.com/KonradHabel/clip_reid third_party/clip_reident
cd third_party/clip_reident
gdown 1Gm5J19okhLdnZTQLUsjfYoI0rwrLQ09i -O model/checkpoints.zip   # 需要 gdown（pipx 或 uvx 皆可）
unzip model/checkpoints.zip -d model/ && mv model/model/* model/ && rmdir model/model
rm model/checkpoints.zip   # 3.6G，解完即可刪

# 包成 checkpoint package（一次性）：
cd ../../yp-reid && uv run python -m yp_reid.import_weights \
    --weights ../third_party/clip_reident/model/ViT-L-14_openai/all_data_seed_1/weights_e4.pth \
    --arch ViT-L-14-quickgelu --remove-proj \
    --out ../videos/reid/checkpoints/clip-reident-paper \
    --note "Habel et al. MMSports'22, all_data_seed_1 epoch 4"
```

包裝完成後 `third_party/clip_reident` 只剩參考價值（訓練 recipe 的出處），
runtime 不再讀它。

**QuickGELU 陷阱**：checkpoint 訓練年代的 open_clip 對 `('ViT-L-14', 'openai')`
隱式使用 QuickGELU；新版 open_clip 必須明確用 `ViT-L-14-quickgelu` 才對得上，
否則權重照樣載入成功但 activation 錯了、精度默默劣化。上面的 `--arch` 與
package manifest 已編碼此 workaround，之後的載入端只讀 manifest。

`clip-reident` embedder 綁定 `reid/checkpoints.py::default_checkpoint()`
（依 manifest 記錄的 best metric 排序，訓練出的 run 勝過無指標的 imported
package）；哪組權重產生了哪個矩陣，看 extraction 時 `embedder.weights_id()`
的記錄。

## pip 層級的外部模型（不需 checkout）

- **RF-DETR Seg Medium**（`rfdetr` 套件）：action frame 的人物框、crop masking 與 rally tracking 的密集偵測；權重自動下載到 `~/.roboflow/`
- **CLIP-ReID**（`clip-reid` embedder）：HF `occurra/person_vit_clip_reid` 的 ONNX，onnxruntime CPU 推論，首次使用自動下載
- **ByteTrack**：來自 `supervision` 套件

## 升級注意事項

- `supervision` 0.28 起 `ByteTrack` 標為 deprecated、**0.30 移除**（遷去獨立的 `trackers` 套件）—— 升級前先遷移 `reid/tracking.py`
- `rfdetr` 的 `optimize_for_inference()` 會把 batch 維度烙死在 traced graph 裡；`>1.8.3` 另有分數尺度正規化，升級要重新校準 `person/detector.py` 的 0.1 與 Association 的 0.5 門檻
- 任何外部模型升級後，拿已標注影片重跑一次對應的驗證（embedder 看 labeled-pair 距離分佈、tracking 看 tracklet 數量與同軌一致性）再信任結果
- yp-reid 的 contract 變更要同步 bump `yp_video/contracts/reid.py` 與 `yp_reid/contract.py` 的版本（握手會擋不一致，見兩檔案的 docstring）
