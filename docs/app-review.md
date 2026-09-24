# App 結果預覽與修正審核

入口：側邊欄 **Video → App Review**（`/app-review`）。

這個頁面用與 iOS 相同的片段規則檢查分析結果，再將使用者回饋交由人工審核。預覽不啟動推論，也不修改 App 雲端資料。

## 預覽結果

可從四個入口開始：

- **Pipeline 結果**：選擇現有影片，讀取目前 Rally／Action 標註或預測與人工 ReID 名稱。
- **雲端修正**：列出 customer bucket 的 corrections，選擇比賽和明確的 analysis job，再保存來源快照。使用工作區既有 `R2_BUCKET_CUSTOMER` 與 R2 憑證；原始影片透過短效簽名 URL 播放。
- **匯入 JSON**：載入 analysis result、corrections 6.0，以及選用的 identify v4、Library Rally 快照。
- **審核紀錄**：重新開啟已保存的來源快照、審核意見及匯入結果。

預覽提供 Rally／Action／Score、球員和動作篩選、In Rally／全部、指定回合的 Action、動作時間定位與連續播放。Action 可切換「組織進攻／單一動作／整個 Rally」。Score 顯示篩選後的失分原因次數，以及不受片段篩選影響的完整場邊累計；場邊累計不推測換場後的球隊身分。

匯入 App 資料後，可切換「套用 App 修正」，對照原始分析與修正後的片段。播放器使用來源影片的絕對秒數；選擇 pipeline 影片播放時，需自行確認它與 customer 影片是同一段原片、沒有裁切位移。

### 片段規則

排球規則集中在 [`yp_video/action/rules.py`](../src/yp_video/action/rules.py)，結果以 `attacks`／`rally_outcomes` 隨 analysis result 送出，iOS 與這個頁面都直接讀取，不各自推導。本機 pipeline 來源會先經同一模組補上這兩個欄位。頁面只負責依規則輸出決定取景範圍：

- 組織進攻：每次扣球一組，向前找接→舉，不越過前一次扣球，也不把發球或攔網當成組織。
- 回合歸屬使用 `start ≤ time ≤ end`，不加入邊界容差。
- 一個 Rally 只產生一個 Score，以最後一聲哨音（`score_event_index`，相鄰重複哨音只取最後一聲）為結尾；沒有哨音則使用回合結束。
- Score 的歸屬球員取自 `deciding_event_indices`：哨音前最後一次組織進攻的扣球，沒有扣球才取最後一次實際觸球。只有發球或接發的回合標為 `serve_point`。
- 人工單次觸球指派優先於整組人物配對；指派存成 null 表示使用者標記「無人」，會清空該次觸球的球員。整組配對只在辨識 result ID 相符時套用。
- 使用者裁切覆蓋取景模式；隱藏片段不刪除原始動作時間軸。

### Library Rally 快照

`corrections` **不包含 Rally 自由裁切後的範圍**。原始分析結果也沒有 App 的 Rally UUID。因此，要重現這部分 App 狀態，需要提供影片庫 `/sync` 回傳的 `rallies` 中，**這一場比賽的完整陣列**（不是增量更新片段）：

```json
[
  {
    "id": "11111111-1111-4111-8111-111111111111",
    "match_id": "與 analysis result 相同的 match_id",
    "index": 1,
    "set": 1,
    "start": 10.0,
    "end": 24.0,
    "score": 90,
    "winner": "left",
    "updated_at": 1789862400,
    "deleted_at": null
  }
]
```

未提供時，頁面使用模型的 Rally 界線，明確列出無法還原的部分；不會猜測 `rally-UUID` 型 Score 修正的對象。雲端入口目前讀取 R2 artifacts，不讀取影片庫資料庫；要補入 Library Rally 快照，可用 JSON 匯入入口載入同一份分析、修正與快照。

corrections 6.0 本身沒有 analysis job ID。選擇要比對的 job 是審核者的明確決定；同一 match ID 或相近時間，不等於已證明同一版本。找不到唯一片段的修正會保留並標示，不套用猜測。

## 審核與標註寫入

每項修正可選「保留為已審核回饋」或「不採用」，並記錄觀察。這兩種操作只改變審核紀錄。

下列項目另有明確的標註寫入操作：

| 回饋 | 審核者確認的事實 | 寫入 |
|---|---|---|
| 隱藏 Action | 影片中確實沒有這個動作，不是單純不想看 | 移除唯一匹配的 Action 人工標註 |
| 刪除 Rally | 原片此範圍不是一個回合 | 移除唯一匹配的 Rally 人工標註 |
| Library Rally 範圍修改 | 修改後符合完整回合起訖，不只是剪輯偏好 | 更新既有 Rally 起訖，保留 ID、勝方與其他標註 |

寫入前必須選擇對應的 pipeline 影片、核對影片來源，並填寫視覺依據。只修改已存在的人工標註；若只有模型預測，先到 **Label** 完整審核並儲存，不因一項回饋就將整份預測列為人工標註。

來源核對還包括影片時長、Action frame/time、唯一相同 frame/label，或唯一相同 Rally 起訖。目標檔案的 SHA-256 必須與載入版本相同；標註已被修改時，需重新載入。更新 Rally 不得與其他回合重疊。

寫入後該模式的 Done 會清除，標註仍可在原本 Label 編輯器檢查，並透過既有流程同步至 pipeline R2。已套用的項目不能重複匯入；之後的調整在 Label 完成。

人物指派會展開成逐次觸球回饋，供審核與定位。**人物名稱不直接寫入 ReID 訓練標註**：App 指派的是執行動作的人，pipeline 的 ReID 標註描述的是裁圖中的人，兩者必須先在 Association／ReID 確認。頁面提供對應編輯入口。得失分、失分原因、收藏、取景裁切與辨識門檻亦保留為回饋，不冒充動作接觸點或模型勝方標籤。

## 保存與驗證

- 來源快照與逐項審核紀錄保存在 `videos/app-feedback/<sha256>.json`。相同內容重複匯入會開啟同一筆紀錄。
- 審核紀錄目前保存在這個 yp-video 工作目錄；不隨 App 帳號同步。資料備份需包含此目錄。
- 紀錄包含來源 match/job、審核者、時間、意見，以及實際標註寫入前後的 SHA-256。
- 先保存寫入意圖，再以原子替換更新標註。若中斷，可使用「恢復中斷的標註匯入」；恢復前會重新核對檔案版本，不將舊操作套到新標註。
- 舊 corrections schema、不相符的 match、重複／模糊對應不做相容轉換。

測試位於 [`tests/test_app_review.py`](../tests/test_app_review.py)，涵蓋規則輸出的取景、辨識版本、Library UUID／裁切、明確標註寫入、版本衝突、錯誤來源、中斷恢復與 HTTP 匯入流程。
