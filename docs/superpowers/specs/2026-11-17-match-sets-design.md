# 局的開頭與結尾（Match sets）— Design

2026-11-17

## Why

切出來的 rally 常混進熱身、局間與賽後的練球，使用者只能一個一個刪。
使用者要能在一場比賽的 rally 上標「局開頭」「局結尾」，局外的 rally
自動不算，局數也跟著出來（現在每個 rally 的 set 都是 1——線上 7,857
個全部是 1，模型不判斷局）。

## Goals

- 每個 rally 可標「局開頭」、「局結尾」兩種記號，兩者都選填。
- 由記號推出每個 rally 屬於第幾局，或「不算」。
- 不算的 rally 從列表、比分、統計、分享頁與 App Review 中拿掉（訓練預標照常保留，見下）。
- 局數取代模型給的 set number，比分與得分走勢自然按局分開。

## Non-goals

- 不偵測局（不做自動判斷換局）；只提示停頓長度幫使用者找。
- 不用影片時間標記；單位就是 rally。
- 不碰「設定我方」的換邊（team_sides）——兩者獨立。

## Anti-goals

- 不算的 rally 不可變成刪除：拿掉記號必須恢復，且與使用者手動刪除的
  rally 分得開。
- 規則在 App、分享頁、App Review 算出不同結果。

## The rule（使用者確認）

Rally 依播放順序（開始時間）排列。

1. 「局開頭」：從這個 rally 開始是新的一局。局號依局開頭在影片中的先後
   自動編 1、2、3…，無上限，使用者不選第幾局。
2. 有任何局開頭時，第一個局開頭之前的 rally 不算（熱身）。一個局開頭都
   沒有時，第一局從第一個 rally 開始。
3. 「局結尾」：這一局打到這個 rally 為止（含）。之後到下一個局開頭之前的
   rally 不算。
4. 沒標結尾的局算到下一個局開頭的前一個 rally；沒有下一局就到最後一個
   rally。
5. 結尾標在已不算的 rally 上（例如連續兩個結尾）沒有作用；標記頁要看得
   出它沒作用。
6. 同一個 rally 可同時是局開頭與局結尾（一球的局，邊界情況，照規則算）。
7. 一個記號都沒有 = 全部是第一局（與今天相同）。

例（12 個 rally）：#1–3 熱身不算；#4 局開頭；#8 局結尾；#9 不算；#10 局開頭
→ 第一局 #4–#8，第二局 #10–#12。

## Approaches

- **A. 存在使用者修正（corrections blob）— 選用。** 加選填欄位，與 team_sides、
  deleted_rally_indices 同一套機制、同一次同步；各讀者自己套規則。代價：規則
  在 Swift / TypeScript / Python 各一份，用共用 fixture 鎖住一致。
- B. 存進 D1 每個 rally（set_number + excluded）由新 API 寫入。讀者不用算，但
  要 migration、新 API、上線順序；重新分析換掉 rally 時要搬記號；「刪除」在
  修正、「排除」在 D1，同類修改分兩處。使用者選 A。

## Design

### Wire format

Corrections blob（`corrections/<user>/<match>.json`，App 以 snake_case 編碼）
加選填欄位：

```json
"sets": { "starts": [4, 10], "ends": [8] }
```

值是 rally index（`Rally.index`，與 `deleted_rally_indices`、
`team_sides.switches`、`rally_winner_overrides` 同一個鍵空間）。沒有任何記號時
省略整個欄位。schema 版本維持 "7.0"——只加選填欄位（見記憶 corrections 版本閘：
升版會讓既有 blob 全失效）。

### The rule as one function, three ports

一個純函式：輸入依播放順序的 rally index 序列與 starts / ends，輸出每個 index
的局號或「不算」。

- Swift（VolleyIQ，Models 下新檔）
- TypeScript（upload-service，shareAnalysis 旁）
- Python（yp-video `web/app_review/`）

共用 fixture `VolleyIQ/VolleyIQTests/Fixtures/match_sets.json`：一組案例（上面
七條規則各至少一例，含無記號、只有結尾、連續結尾、同 rally 開頭兼結尾、記號
指向已刪除 rally）。iOS 測試、upload-service 測試、yp-video 測試都讀同一檔
（沿用 `clip_windows.json` 的做法：yp-video 以相對路徑讀 VolleyIQ 那份）。

記號指向不存在／已刪除的 rally：忽略該記號（等於沒標）。

### App (VolleyIQ)

- Match 存記號（與 teamSides 一樣的 SwiftData 欄位），經 corrections 上傳／套用。
- 新的「設定局」sheet：與 TeamSidesSheet 同型——依播放順序列出全部 rally（含
  不算的），每列顯示與前一球的停頓、目前算出的局號或「不算」，可切換局開頭／
  局結尾記號。沒作用的結尾要看得出來。Done 才存並同步。入口放在分析頁「設定
  我方」旁。
- 套用結果：每個 rally 得到局號（取代模型的 setNumber 給比分、走勢、排序用）
  與是否不算。不算的 rally 不出現在 rally 列表／feed、比分、統計、精選、匯出，
  也不出現在使用者自己的播放清單、標籤、收藏裡（我的決定：一律隱藏，好在一層
  統一過濾）；只在「設定局」sheet 看得到。
- 局號與「不算」在記號或 rally 改變時重算；模型的原始 set 不覆寫進同步資料。

### Share page (upload-service)

shareAnalysis／shares 讀 corrections 時套用同一規則：不算的 rally 不進統計，也不
出現在分享頁的片段清單——擁有者先前明確挑進單球／合輯分享的 rally 被標成不算後
同樣隱藏（我的決定）。分享頁這次不加按局的比分或走勢 UI（我的決定）：只少掉練球；
按局的比分只在 App（CourtScoreTimeline 已按 set 分組）。局號以整場 rally 計算。

範例比賽（sample.ts 的 trimCorrections）不帶 `sets`，維持現狀（我的決定）。

### App Review + training (yp-video)

- `projection.py` 套用規則：不算的 rally 不出現，局號照算。
- 模型回饋（`feedback.py`）不把不算的 rally 列為「這不是 rally」——它們不是誤偵測
  （我的決定）。
- `training.prelabels` **照常**把不算的 rally 與其動作寫成預標（使用者決定）：練球
  也是真的在打球，對 rally／動作模型是正確樣本；分辨比賽與練球是使用者標局的事。
  因此 training.py 不需套用規則。預標只反映送出那一刻，送出後改記號不回寫（現行
  寫一次即不重寫，我的決定）。

### Rollout

舊版 App 讀到 blob 會忽略 `sets`，但它存修正時會寫回沒有 `sets` 的 blob，把記號
清掉。所以：Worker 的 MIN_IOS_VERSION 升到帶這功能的版本，並照慣例先上架 App、
再 deploy Worker（分享頁的 TypeScript 可以先上，沒有 `sets` 時行為不變）；
yp-video 改完重啟 yp-app。

## Left to the builder

- Swift 端在哪一層過濾（集中在 Match 的一個 accessor，或在 Rally 上存衍生欄位），
  以不漏掉任何 rally 消費者為準。
- 「設定局」sheet 的版面細節（照 TeamSidesSheet 與 ios-fewer-tinted-blocks 的
  細線清單風格）。
- 重新分析換掉 rally 後記號的處理，比照 team_sides 現行做法。
