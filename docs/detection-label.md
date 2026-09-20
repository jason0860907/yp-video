# Detection 人物框標註

入口：[Label → Detection](https://label.volley-iq.com/label?mode=detection)。選一支 cut 影片後，可標註任意影格，不限動作發生時刻。

## 標註規則與操作

- **所有可辨識人物**都標，包含場邊人員；遮擋時只框可見部分，不推測被遮住的身體。
- 上一幀／下一幀、影格編號跳轉、每 5 秒抽樣、動作影格選單和已標註影格選單共用同一編輯器。影格編號從 0 開始。
- 拖曳新增人物框，選取模式可移動與拖曳角點縮放；支援刪除選取框及復原。
- 預設顯示 native JPEG 快取，勾選「原始解析度」會從原影片精確解碼該幀，適合遠處小人物。R2 影片可能載入較慢。
- 既有框自動顯示，不需先匯入。Source 可選 Fusion、Tracking、Action detections；顯示各自的資料，不會在空白影格偷偷切換來源。預設 Min score = 0 保留所有既有框，可自行篩選。
- 已儲存人工標註（包括確認無人）優先顯示；Reset to source 才會取代現有人工框，取代前有確認提示。來源框仍是待確認預標註。
- 沒有跨分頁傳入的播放位置時，直接開到第一個有框的影格；Prev boxes / Next boxes 跳至所選來源有框的前後影格。Fusion 鄰近抽樣會顯示實際來源影格，Tracking 與 Action detections 僅顯示當幀資料。
- 沿用 Action 分頁的左側畫面／控制列與右側人物清單配置；人物清單可選取對應框，標註規則收在 Labeling guide。
- 「儲存草稿」不代表已檢查完整畫面。「確認全部人物並儲存」才是人工真值。
- 「確認此幀無人」有獨立確認提示；空白且未標註的畫面絕不自動視為無人。
- 切換影片、Label 分頁或影格會提示未儲存變更；關閉／重整瀏覽器有離開提示。
- 同一影格有 revision 衝突時拒絕覆寫，需重新載入。儲存失敗保留目前編輯內容。

影片列表的 In-Progress 表示已有人工工作，不代表整支影片已逐幀檢查。頁面另列已確認與草稿的影格數，不使用整部影片的 Done 判定。

## 儲存與訓練

人工資料：`videos/person/annotations/<stem>_persons.json`，R2 category 為 `person/annotations`。每幀保存 normalized xyxy、draft/reviewed 和 revision；`reviewed + boxes: []` 是明確的無人真值。預測檔 `tracks/<stem>_persons.npz` 不會被修改。

Fusion 訓練建立 person sidecar 時，對**本次訓練選入的影片**：

1. 既有對齊的 tracking 輸出繼續提供 pseudo labels。
2. 已確認人工影格完整取代同幀 pseudo labels，包含確認無人和回合外影格。
3. 已儲存草稿的影格從 supervision 排除。
4. 無 tracking 的影片也能輸出已確認人工影格。人工標註影格數與 training frame cache 不一致會中止匯出，不冒險套到錯誤畫面。

未操作過的其他影格仍遵循既有 pseudo-label 訓練流程；不能把它們解讀為人工確認。訓練來源仍由既有 Rally/Fusion 資料選取決定，因此只有 Detection 標註、未被訓練選入的影片不會自動加入。既有訓練 run 的快照和已部署模型不會因標註而改變；修正供下一次建立訓練資料使用。

## 驗證

`tests/test_detection_label.py` 覆蓋草稿／確認無人／未標註的區別、revision 衝突、框驗證、無 tracks 的人工匯出、影格數不一致、native PTS 時基與 API 預測隔離。瀏覽器驗證畫框、移動、縮放、確認、儲存重開、離開提示和已標註影格跳轉。
