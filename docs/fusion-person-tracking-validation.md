# Fusion 人物框追蹤的驗證邊界

記錄日期：2026-09-16。以下描述尚在工作區中的推論流程變更；本文件提交
不代表相關程式已發布，也不代表正式 App 已切換模型。

## 候選流程

一次 SPOT Fusion pass 產生回合、動作及人物框，保存人物框後，交由 ByteTrack
形成回合內 tracklets，事件候選使用同一份框，最後由獨立球員 clip classifier
選擇 actor。這條完整 Inference 路徑不另跑 RF-DETR，也沒有 instance masks。
RF-DETR 的獨立工具仍作離線標註用途。

這是對推論資料來源的實質更動，應與 RF-GSM backbone 訓練實驗分開評估。
缺少 person head 或人物框輸出不完整時直接報錯，不回退至另一個 detector。

## 已有驗證與未證實的效果

目前測試涵蓋：

- 人物框保留原始 frame 編號、stride、全片覆蓋與 checkpoint 身分。
- 拒絕不完整框檔與超出影片範圍的事件。
- 每個回合重設 ByteTrack，保留原始時間索引，移除舊 instance masks。
- tracking 與事件候選消費同一份人物框；重建候選時保留既有選擇。
- 在保留既有時間標註時仍產生全片人物框，並重用有效結果。

相關測試位於 `tests/test_fusion_tracking.py` 與 `tests/test_fusion_inference.py`。
流程測試能證明接線與資料處理行為，不能證明真實影片的模型品質。

| 尚未證實的項目 | 所需比較 |
| --- | --- |
| 人物偵測品質 | 同一批人工標註影片上的 precision／recall／AP，特別是小人物、遮擋、快速移動 |
| 追蹤品質 | ID 切換、軌跡碎裂與有效球員覆蓋；使用相同回合標註比較 |
| Actor 品質 | 固定 clip classifier，比較候選召回及最終 Actor 準確度 |
| 聯合產品品質 | 預測事件的時間、位置與執行者是否一起正確 |
| 實際效率 | 相同硬體與影片上的完整流程耗時、顯存與磁碟使用，包含快取冷／熱狀態 |

減少一次 detector 呼叫不等於已證明整體加速；全片推論與額外框資料的
處理成本也需要計入。無 instance masks 的點選與標註體驗也需實際檢查。

## 整合與提交條件

正式替代之前，應在代表性且未用於調參的影片上對照目前 RF-DETR 流程。
固定 SPOT、clip classifier、回合來源及評分條件，記錄上述品質與成本，
根據實際產品需求決定可接受的差異；尚無足夠證據宣稱等效或更好。

相關程式宜作一個獨立功能 commit，連同流程測試與說明一起提交。
它依賴 `yp-spot/yp_spot/inference.py` 輸出的 `stride`、`num_frames`，
兩個 repo 的對應 commit 應在整合紀錄中互相註記。正式部署前另行驗證。

正式 worker 的 release 隔離屬部署維護，不是模型效果實驗，應獨立提交。
其他 repo 的環境範例檔刪除不屬於這項功能。
