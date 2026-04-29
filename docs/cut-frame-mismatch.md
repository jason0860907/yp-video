# Cut Frame Count Mismatch

## TL;DR

`~/videos/cuts/` 裡 **302 / 400 (75.5%)** 個 mp4 的 container header 宣稱的 frame 數比實際解碼出來多。最嚴重的多算 411 frames(~6.9s)。decord 在這些檔案上做 random-access(`get_batch(indices)`)時會 seek 到不存在的 frame,觸發 `DECORD_EOF_RETRY_MAX` exception,導致 V-JEPA feature extraction 卡住。

## 已驗證會 crash 的檔案

兩支實際看到 decord 報錯的影片:

| 檔名 | Header `nb_frames` | 實際 decoded | 差距 |
|---|---:|---:|---:|
| `2025-11-01_G16_桃園雲豹飛將_vs_台鋼天鷹_set4.mp4` | 59,834 | 59,636 | −198 frames (~6.6s) |
| `2025-10-25_G12_台鋼天鷹_vs_臺中連莊_set5.mp4` | 30,641 | 30,604 | −37 frames (~1.2s) |

這兩支也都在 mAP < 0.5 的失敗預測清單裡 — 受影響的 feature 被截斷,training 跟 inference 結果都不可靠。

## 全清單

掃描方式:用 `nb_frames / fps` 跟 `format.duration` 比對,差距 > 30 frames(1 sec @30fps)就列入。完整清單存在 [`cut-frame-mismatch-list.txt`](cut-frame-mismatch-list.txt)。

分佈:

| Δframes | 檔數 |
|---|---:|
| ≥ 300 | 12 |
| 200–300 | 67 |
| 100–200 | 97 |
| 60–100 | 55 |
| 30–60 | 71 |
| **合計** | **302** |

100% 都是「header > 實際」,沒有反向案例 → 系統性 bug。

## 為什麼會發生

源頭在 `src/yp_video/core/ffmpeg.py:73-113` 的 `export_segment()`,被 `src/yp_video/web/routers/cut.py:85` 用 `copy=True` 呼叫:

```python
cmd = [
    "ffmpeg", "-y",
    "-ss", str(start),       # ← input-side seek
    "-i", str(source),
    "-t", str(end - start),
    "-c:v", "copy", "-c:a", "copy",  # ← stream copy
    "-movflags", "+faststart",
    str(output),
]
```

`-ss` 在 `-i` 之前 + `-c copy` 是 ffmpeg 的經典陷阱:

1. **input seek** 會跳到 `start` 之前最近的 keyframe(不會 frame-accurate)。
2. **`-c copy`** 不重編碼,只能在 packet 邊界切。最後一個 packet 的時間落點跟 `-t` 給的 duration 不會精確對齊。
3. 寫出去的 mp4 容器,header 裡的 `nb_frames` / `duration` 是用「請求值」計算的,跟 packet 流的實際長度對不上。

decord 用 random-access(`vr[i]` / `vr.get_batch(indices)`)的時候,先看 `len(vr) = header.nb_frames`,然後計算 clip 起點。當索引超過實際存在的 frame 數,decord 去那個位置 seek 失敗,EOF retry 一直累加到 `DECORD_EOF_RETRY_MAX=10240` 用完就丟例外。

ffmpeg sequential decode(例如 `ffmpeg -i FILE -f null -`)沒事,因為它讀完 packet 就停,不在乎 header 騙人。

## 怎麼修

### 修源頭(避免之後再發生)

`core/ffmpeg.py` 裡 `export_segment` 有 `copy: bool = False` 參數,**default 已經是 re-encode**。問題在 `web/routers/cut.py:85` 顯式傳了 `copy=True`,等於繞過了預設值。

選項:

- **A. 直接拿掉 `copy=True`**(最乾淨):走 libx264 re-encode 路徑,frame-accurate,header 正確。代價是 cut 時間變長(取決於 GPU/CPU)。
- **B. 保留 `copy=True` 但加 `-fflags +genpts`**:強制 ffmpeg 重算 PTS,通常能修 metadata。仍可能因 keyframe 邊界產生小偏差。
- **C. 改用 output-side `-ss`**(`-i` 在 `-ss` 前):frame-accurate but 慢(要從頭 decode 找到 start)。跟 re-encode 的成本差不多,沒明顯好處。

建議 A:可靠且不挑檔。

### 修現有的 cut(不重切)

對既有 mp4 強制 remux + 重算 PTS:

```bash
ffmpeg -i in.mp4 -fflags +genpts -c copy -movflags +faststart out.mp4
```

不一定每支都救得回來。最保險是 re-encode:

```bash
ffmpeg -i in.mp4 -c:v libx264 -preset fast -crf 18 -c:a copy out.mp4
```

### 修現有 cut 的批次腳本範例

```bash
# 只處理 cut-frame-mismatch-list.txt 裡列出的檔案
while IFS= read -r line; do
  # 從檔尾的「filename」欄位取出檔名
  fn=$(echo "$line" | awk '{$1=$2=$3=$4=$5=""; sub(/^ +/,""); print}')
  [ -z "$fn" ] && continue
  src="$HOME/videos/cuts/$fn"
  tmp="$src.fixed.mp4"
  [ -f "$src" ] || continue
  ffmpeg -y -i "$src" -c:v libx264 -preset fast -crf 18 -c:a copy "$tmp" \
    && mv "$tmp" "$src"
done < <(awk '/^ *[+-][0-9]+/' docs/cut-frame-mismatch-list.txt)
```

⚠️ 重切前提:對應的 V-JEPA feature 跟 prediction 都需要重新生成,因為 frame-accurate 結果會稍微改變 timestamps 跟 GT 對齊。

### Decoder-side workaround(不修檔,只繞過)

`extract_features.py` 裡 clip 起點 clamp 到 `min(actual_frames, header_nb_frames - 1)`,或者 try/except 接住 decord exception 跳過該 clip。對 V-JEPA feature 而言會少抽片尾幾個 clip,影響輕微。

如果只想讓 extraction 跑完,可以:

```bash
export DECORD_EOF_RETRY_MAX=20480
```

某些檔案還是會 fail,但能撐過大多數。

## 影響範圍評估

- **Feature extraction**:已知會在最嚴重的幾支 crash;其餘可能抽到的 feature 在尾段是錯的(decord 重複返回最後一個 keyframe 的解碼結果)。
- **TAD training / inference**:用了壞 feature 的 video,model 在這些片段學到的是 garbage,跟 mAP < 0.5 的失敗清單高度相關。
- **VLM detect / annotate**:這兩個流程主要走 ffmpeg 抽 clip(不靠 decord random-access),受影響較小。
- **Web 播放**:HTML5 video element 走 sequential decode,不受影響。

## 建議行動順序

1. **優先**:修 `web/routers/cut.py:85` 拿掉 `copy=True`,避免新切的 cut 也壞。
2. 跑 batch re-encode 修現有 302 支 mp4(備份原檔)。
3. 重抽 V-JEPA feature(`tad-features/vjepa-l/`)。
4. 重跑 TAD training。
5. 重跑 inference 出新的 `tad-predictions/`。
