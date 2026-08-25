# 新增功能準則

這份文件定義 yp-video 新功能的產品、UI、資料與模型整合規則。新增頁面前，先找最接近的既有流程作為模板；目前 Train 以 Fusion Train（所有 SPOT recipe 的單一入口）為準，Predict 以 Action Predict / Rally SPOT Predict 為準。

## 1. 先定義功能邊界

- 一個頁面只處理一個清楚的工作目標。若同頁有 Train / Predict 等模式，目前模式必須同時控制統計、設定、資料清單與 jobs，不顯示另一模式的資訊。
- 新功能先寫出輸入、輸出、前置條件、會覆寫的資料、失敗條件，再設計 UI。
- 尚未實作的模式可以登記，但必須標示為 planned / contract needed 並禁止執行；不要放看似可用的空表單。
- Router 只做 request validation 與流程編排。資料選擇、checkpoint 判定、label 轉換等規則應放在 domain/service 層，避免同一規則散落在 UI 和 API。

## 2. Dashboard 的固定骨架

### Train 頁

依序使用：

1. `PageHeader`：只放頁級說明與少量頁級操作。
2. 四個 `StatTile`：資料量、標註量、validation 或 checkpoint、目前狀態。
3. 主區域：`1.6fr / 1fr`。
   - 左：`Training config`
   - 右：資料集摘要、實際 selection/split 與輸出 heads
4. `Training job`：放在主區域下方，獨占整列。
5. `Validation performance`：有結果才顯示。

### Predict 頁

依序使用：

1. `PageHeader`：相關 Label 入口與主要 Run action。
2. 四個 `StatTile`：Videos、Selected、已有輸出、Running。
3. 主區域：`1fr / 1.6fr`。
   - 左：`Config`
   - 右：`VideoMultiSelectList`
4. Jobs：放在主區域下方，獨占整列。

### UI 元件

- 使用既有的 `Card`、`SectionLabel`、`Field`、`fieldCls`、`Button`、`Badge`、`StatTile`、`VideoMultiSelectList`、`JobProgress` / `LiveJob`。
- 不在頁面內重新宣告 input、button、card 的通用樣式。缺少通用能力時先擴充共用元件。
- 一個概念只使用一種控制方式。固定 enum 用 select；互斥且只有 2–3 個的頁級模式可用 segmented buttons；不要同時用大卡片、tabs、select 表達同一層選擇。
- 預設只顯示完成目前工作所需的欄位。進階選項與條件欄位採 progressive disclosure，例如只有 Random ratio 模式才顯示 ratio 與 seed。
- 警告只用於會改變資料、模型語意或結果可信度的事情；一般說明用 muted helper text。
- 統計必須描述目前畫面真正會執行的資料範圍，不能用全 corpus 數字代替已套用 view/scope/filter 後的數字。

## 3. 訓練資料選擇

每個 trainer 必須有單一、可測試的 eligible predicate，並在 backend status 與 start endpoint 共用同一套規則。

資料選擇順序固定為：

1. 找出具備必要人工標註且有本地影片的樣本。
2. 套用 camera view / dataset scope。
3. 驗證每個 task head 的 supervision。
4. 以整部影片為單位切 train / validation，禁止同一影片跨 split。
5. 將最後使用的 labels、actor candidates、split 與 config 凍結到 run。

多任務訓練必須明確選擇：

- `joint_only`：使用所有 task 標註的交集；缺少任一 supervision 直接拒絕或排除。
- `partial_labels`：使用聯集；缺少標註的 head 必須 mask loss，UI 必須說明 shared backbone 仍會被其他 head 更新。

不可默默把「沒有該 task label」當成 negative label，也不可由檔名存在就推定內容可用。start endpoint 應重新驗證實際 target 數量並 fail loudly。

## 4. Validation 與 Resume

- 手動 validation 是明確的影片清單；random validation 才使用 ratio + seed。
- UI 顯示的 validation 候選必須已套用與 training 相同的 view/scope eligibility。
- validation 不可為空；train 與 validation 都必須至少有一部影片。
- split 必須 deterministic，並在 run 內保存實際 manifest，而不只保存 ratio。
- Resume 使用原 run 凍結的架構、label snapshot、candidate snapshot 與 split；UI 不應讓使用者誤以為這些欄位會生效。
- `Epochs` 在 Resume 時表示新的總 epoch 目標，介面必須明說。

## 5. Checkpoint 與模型能力

- Checkpoint 是否可用由集中式 family/manifest 檢查決定，不能只靠頁面上的名稱 filter。
- Train init、Resume、Predict 是三種不同相容性：
  - init 只需要可載入相容 backbone/head。
  - Resume 需要完整架構、optimizer 與 frozen run data。
  - Predict 需要使用者選擇的 output head 真正存在。
- 專案不承擔一般性的向後相容；但已經作為產品入口公開的舊 checkpoint，必須明確支援、提供 migration，或在 UI 說明拒絕原因。
- 多 head checkpoint 的 best epoch 選擇標準必須顯示。若某 head 尚未參與 best metric，不能讓 UI 暗示它已由 validation 最佳化。

## 6. Predict 的資料安全

- 顯示執行所需的 pipeline prerequisites，並在 backend 再驗證。
- 覆寫人工或既有結果必須有明確的 overwrite 開關；實際會覆寫時再顯示 confirmation。
- 人工 reviewed / final 資料預設不可被 prediction 覆寫。
- Checkpoint、output head、video selection 與 job 統計都必須屬於目前 Predict 模式。

## 7. API、Jobs 與前端接線

新增 Web 功能時逐項確認：

- FastAPI router 與 request/response model
- `web/app.py` router mount
- frontend `lib/api.ts`
- frontend `types/api.ts`
- `App.tsx` route
- sidebar `nav.ts`
- job type、progress key、SSE 更新、cancel、terminal toast
- `lib/job.ts` 完成後應失效的 queries

Status endpoint 是 UI 的 source of truth，應一次回傳：

- dependency availability 與可操作原因
- filtered dataset counts 與 per-video metadata
- compatible checkpoints / resumable runs
- active job

不要讓前端用多個不一致的 endpoint 自行猜測 eligibility 或 checkpoint family。

## 8. 完成條件

至少執行：

```bash
cd src/yp_video/web/frontend
npm run typecheck
npm run build

cd ../../../..
.venv/bin/python -m unittest <相關測試>
git diff --check
```

若 frontend 已配置 linter，`npm run lint` 也必須通過。不可只留下沒有 dependency / config、實際無法執行的 lint script。

測試應涵蓋：

- eligible / excluded dataset cases
- manual 與 random validation
- empty split 與 missing target failure
- fresh / init / resume
- checkpoint family 與 output head
- overwrite protection
- job payload、progress 與完成後 query refresh

最後人工對照最接近的 Action 或 Rally 頁面，確認資訊順序、欄位命名、responsive grid、empty/loading/error/disabled 狀態一致。
