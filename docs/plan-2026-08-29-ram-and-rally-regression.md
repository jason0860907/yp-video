# 計劃：主機 RAM 壓力與 rally 訓練退化（2026-08-29）

## 目標與原則

1. 先恢復 rally 的效果，再處理 winner；兩者不能用同一個實驗互相證明。
2. 保留共用的 native-fps frame cache，不恢復已移除的低 fps cache。`sample_fps`
   只決定訓練時從 native frames 取樣的 stride。
3. 5 fps 是 rally 在 `clip_len=64` 下目前合適的時間尺度，不是未來多任務訓練的限制：
   - action：30 fps × 64 frames，涵蓋約 2.1 秒；
   - rally：5 fps × 64 frames，涵蓋約 12.8 秒；
   - 每個 batch 的 frame 數與 tensor shape 相同，所以不同 FPS 本身不增加 peak VRAM；
   - 未來合併 action / rally / winner 時，共用 backbone 與 optimizer，但用
     task-specific sampler 交替餵 batch，而不是強迫所有 task 共用一個 `sample_fps`。
4. RAM 驗收量整個 trainer process tree / cgroup 的 peak，不只看單一 PID 的瞬間 RSS。
5. 不增加舊 cache、舊 request 或舊訓練路徑的相容層；直接讓現行路徑具備正確語意。

## 背景（已確認與待確認）

### 已確認

1. **8/28 19:51 訓練被 kernel OOM 砍掉**。`RuntimeError: Segment mAP
   (mean over tIoU): 40.87%` 只是子程序死前最後一行 stdout，被
   `spot_training.py:398` 拿來當錯誤訊息。當時 62 GB RAM 只剩約 150 MB：
   trainer 22.8 GB（4.9 GB anon + 17.9 GB shm）、web 12.85 GB，其餘是
   Chrome / claude / rg。
2. **web 程序常駐約 12 GB，啟動後不再成長**。主要來源是
   `tracklets/store.py:_index_cache`：warm-up 走過所有 association worklist，
   `TrackletIndex` 因而長期持有大量 tracks JSONL 的 parsed records 與 frame index。
3. tracks 還會經過 `core/jsonl.py:read_jsonl_cached`。限制 `_index_cache` 並不足以
   保證釋放 records：index 被淘汰後，同一批 parsed objects 仍可能由通用 JSONL
   cache 保留。tracks 必須只有一個有界的 cache owner。
4. trainer 的高 shared-memory 用量與大量 DataLoader workers、prefetch、pinning
   一致；目前 train 16 workers、val 8 workers，`evaluate()` 另建 8 workers。
   但現行 DataLoader 沒有啟用 `persistent_workers`，不能先假定三組 worker 在
   epoch 交界同時常駐，必須分階段量 peak 才能確認。
5. **30 fps 是 rally 退化的首要嫌疑**：
   - 7/7 run：5 fps、rally-only，第一個 epoch 已約 0.90 mAP；
   - 8/27 run：5 fps、rally+winner，第 8 個完成的 epoch（metrics `epoch=7`）
     為 0.884；
   - 8/28 run：30 fps、rally+winner，第 6 個完成的 epoch（metrics `epoch=5`）
     為 0.345；
   - `a1b37aa` 把 rally 改成共用 native-fps cache 後，request 的
     `sample_fps=30` 沒有被 rally recipe 覆蓋；
   - `clip_len=64` 在 30 fps 只涵蓋 2.1 秒，在 5 fps 涵蓋 12.8 秒。
6. decode 改動只走 inference 的 `SlidingClipBatchLoader`；訓練讀 JPEG cache，
   不經過該路徑。CPU eval transform 與 `ClipAugment(is_eval=True)` 的輸出差異也
   只有約 1.2e-6。

### 尚未確認，不能先當結論

1. winner loss 是否確實拖慢 rally：7/7 與 8/27 並非只有 winner 一個變因，
   需要同一版程式的 rally-only / rally+winner 對照。
2. winner top-1 約 0.5 不等於全域亂猜。winner 是 left/right/near/far 四類；
   0.5 可能代表模型只學會 camera view 對應的二類 vocabulary，卻無法分辨同一
   view 裡的兩側。需要 confusion matrix 與 per-view baseline。
3. winner 問題不一定只在 target 建立。現有 augmentation 會做 rotation、affine、
   perspective，但 winner target 目前只處理 horizontal flip；camera-frame side
   可能在其他幾何變換後失去原語意。

### 2026-08-29 執行中觀察

- 新版 web 重啟後實際 RSS 約 846 MiB，有界 tracks cache 已在服務程序生效。
- `rally+winner` 5 fps run 的 epoch 0–2 rally validation mAP 為
  0.381／0.432／0.475；winner validation top-1 約 0.49–0.53，但 class recall
  在 left/right 與 near/far 內來回塌到單側，尚未證明能分辨同一鏡頭的兩邊。
- Trainer 的約 12 GiB `RssShmem` 由 pinned-host `/dev/zero` arenas 構成；它在
  train／val／full-evaluate 不同 batch shape 間累積。下一個 run 已準備移除
  `pin_memory`，並讓 full-evaluate 的 raw uint8 clips 到模型裝置後才 resize／normalize。
- `camera_view=broadcast` 不能視為嚴格的 left/right vocabulary：全 snapshot 的 15 個
  broadcast near/far events 經 frame 抽查，rally 當下實為端線鏡頭，near/far 標籤合理。
  因此 per-view baseline 只作 corpus slice，不作合法類別約束；最終仍需四類 confusion。

## 要做的事

### A. rally 恢復 5 fps 取樣

#### 實作

- `contracts/action.py` 的 `_RALLY_DEFAULTS` 加上 `"sample_fps": 5.0`。
- 保持 native-fps frame cache；不恢復 `rally-spot/frames/fps5` 或 `extract_fps`。
- `FusionTrainPage.tsx:pickRecipe` 已會在切換 recipe 時套 defaults。初始 recipe 是
  `association_action`，不需要額外的 mount effect；避免 status 載入或 refetch 時
  覆蓋使用者已修改的欄位。
- 補測試：
  - status payload 的 rally / rally_winner defaults 都是 5 fps；
  - UI 切換到 rally recipe 後 request 為 5 fps；
  - `sample_stride_for(30, 5) == 6`、`sample_stride_for(60, 5) == 12`；
  - train dataset、val-loss dataset、full evaluate dataset 使用相同 stride。

#### 受控驗證

- 先跑 **rally-only**，不要用尚未解決的 winner 一起驗收。
- 固定與對照 run 相同的：
  - label snapshot / video subset；
  - train/val split 與 split seed；
  - backbone、temporal head、clip_len、batch size、epoch frame budget、augmentation；
  - 程式 commit。
- 只改 `sample_fps`，先做 30 fps 與 5 fps 對照；metrics 一律同時記錄
  zero-based `epoch` 與人類可讀的「第 N 個完成 epoch」，避免 off-by-one。
- 驗收：5 fps 的 rally mAP 曲線明顯恢復到 8/27 同級；以第 8 個完成 epoch
  （metrics `epoch=7`）mAP ≥ 0.85 為最低門檻，並記錄 peak RAM 與 throughput。

#### 2026-08-29 受控結果

- 在相同 label snapshot（725 train／81 val）、split seed 42、training seed 42、
  backbone、GRU、clip 64、batch 8、500,000 frames/epoch 與 augmentation 下，只改
  `sample_fps`，各跑 8 個 rally-only epoch。trainer 在實驗前補齊獨立的 training
  seed，固定 model initialization、sampling 與 augmentation RNG；config 會保存該值。
- mAP（5 fps／30 fps）：epoch 0 `0.4212／0.3045`、epoch 3
  `0.8027／0.4331`、epoch 6 `0.84925／0.4596`、epoch 7
  `0.8337／0.4714`。30 fps 在每個觀測點都大幅落後，證明 64-frame clip 對 rally
  需要 5 fps 提供的約 12.8 秒上下文，而不是 30 fps 的約 2.13 秒。
- 5 fps 明顯恢復到 8/27 同級，但字面上的 epoch 7 `>= 0.85` 門檻仍差
  `0.0163`；最佳 epoch 6 差 `0.00075`。不得把它記為通過，A3 前置門檻須在 A2
  與 joint 對照後重新驗證。
- process-tree peak PSS：5 fps **5.944 GiB**、30 fps **5.869 GiB**，顯示不同
  sampling FPS 不會疊加 peak host RAM；batch/clip shape 相同時 peak VRAM 也相同。
- 5 fps 約 7.2 分鐘/epoch；30 fps 約 20.3 分鐘/epoch。train 都約 4 分鐘，差異
  主要來自 full evaluate 必須遍歷約六倍 sampled timeline；多 FPS 本身不增加單一
  batch 的計算量，但保留完整 30 fps 全片 budget 會增加總工作量。
- 後續 joint gate 追查發現，trainer 雖印出「Linear Warmup + Cosine
  Annealing」，實際卻使用 `ChainedScheduler`，使 warm-up 與 cosine 從第一步
  同時前進。cosine `T_max` 為 6 epochs，因此在 epoch 5 底先降到 0，
  epoch 6--7 又反向上升，最終 LR 回到 `7.5e-5`。這與宣告的訓練語意
  不一致，也可解釋後段 mAP 振盪。
- 已直接改為 `SequentialLR`：2 epochs 完整 warm-up 後，再開始 6-epoch
  cosine，最後一個 optimizer step 準確降到 0，且全程不反彈。新增測試
  覆蓋 phase 切換、最終 LR、zero warm-up 與無 cosine phase 的非法設定；
  yp-spot 全套 37 個測試通過。
- 原 30/5 fps 實驗兩邊共用同一個 scheduler bug，而 5 fps 對 30 fps 差距
  極大，所以 FPS 因果結論仍成立。但 gate 與 rally-only/joint 最終曲線必須
  在修正版上各重跑一次，不放寬 `epoch=7 >= 0.85` 的原定驗收。
- `SequentialLR` 修正版的 rally-only 5 fps 重跑已完成。沿用完全相同的 725/81
  split、seed 42、clip 64、batch 8、500,000 frames/epoch 與 8 epochs，mAP
  epoch 0--7 依序為 `0.4071`、`0.3615`、`0.5450`、`0.7176`、`0.7107`、
  `0.8577`、`0.8632`、`0.86586`。最後 epoch 7 高於 `0.85` 門檻，且最佳
  checkpoint 正是 epoch 7，rally-only gate 正式通過。
- 該 run 正常以 exit code 0 結束，8 個 epoch checkpoint 與 metrics 均完整；
  `optim_007.pt` 內所有 optimizer param group 的實際 LR 都是 `0.0`，證明
  cosine phase 完整走到底且沒有舊版末期反彈。下一個 gate 是用相同修正版與
  實驗控制重跑 rally+winner 5 fps，完成前 A3 仍保持關閉。

### A2. 找出 winner 無法學習的原因

#### 先建立可解讀的量測

1. 輸出 train / val 的：
   - 四類 label count；
   - per-camera-view label count；
   - confusion matrix；
   - global majority、per-view majority baseline；
   - 每支影片實際出現的 side 集合。
2. 確認 rally annotation 的 `winner` 定義始終是 camera-frame court side，而不是
   team identity；檢查換邊、鏡頭切換與錯誤 camera-view metadata。

#### 最小 overfit ladder

依序進行，每一層成功後才加下一層：

1. 挑一支確實含兩個 winner side、label 數足夠且 camera view 固定的影片。
2. 固定少量 clips，關閉所有 augmentation，只訓練 winner head；train top-1
   應快速接近 1.0。
3. 恢復共享 backbone 更新，確認 joint gradient 本身不會阻止 overfit。
4. 只開啟不改變 side 語意的 photometric augmentation。
5. 逐項驗證 crop / zoom / perspective / rotation / affine：
   - horizontal flip 必須交換 left/right，near/far 不動；
   - 無法可靠轉換 camera-frame side 的幾何 augmentation，winner batch 直接不用，
     不建立猜測式 target remapping。
6. 驗證 `WINNER_TAIL_S`：tail 落在 span 尾端、clip 邊界裁切正確、sampling stride
   後仍對齊原生 frame。
7. 最後才比較 per-frame MLP 與 winner GRU；`tests/test_winner_head.py` 的 shape / context
   測試不能取代 overfit 測試。

#### 與 rally 合併驗證

- A2 修好前，不用 `rally_winner` 證明 rally 5 fps 已恢復。
- A2 修好後，在同一版程式做 rally-only 與 rally+winner 的 5 fps 對照，量化
  winner loss 對 rally 收斂速度及最終 mAP 的影響。
- 短期 `rally_winner` 共用 5 fps。不要新增目前 contract 無法獨立訓練／推論的
  winner-only recipe。

#### 2026-08-29 winner 診斷結果

- 原始 30-epoch `rally+winner` 5 fps run 最後的 rally mAP 為 `0.9258`；
  winner validation top-1 為 `0.8600`，明顯高於 global majority baseline
  `0.3721`。left/right recall 為 `0.944/0.948`，near/far 為
  `0.647/0.609`；winner 不是全域亂猜，而是 sideline 的 near/far 較難。
- event-level label 統計為：train left/right/near/far
  `3806/3647/2111/2090`，validation `498/444/165/186`。每支實際含
  winner 標註的影片都記錄了 side 集合；受控 overfit 影片
  `0316小窩季打 9` 為固定 sideline view，並同時含 near/far。
- 同一個初始 checkpoint 與同一組 8 near + 8 far 固定 clips 的
  overfit ladder：
  - 凍結 backbone、只訓練 winner head：best top-1 `0.9676`，最後
    `0.8496`，不穩定且未達接近 1.0 的驗收。
  - 開放共用 backbone 更新：第 30 epoch 達 top-1 `1.0000`，loss
    `0.0189`。
  - 加入不改變 side 語意的 photometric augmentation：第 10 epoch
    達 top-1 `0.9971`，loss `0.0370`。
- `WINNER_TAIL_S` 測試已覆蓋 native fps、sampling stride、span 尾端與
  clip 早於 tail 時不得看到未來標籤。horizontal flip 只交換
  left/right，near/far 保持不變；無法可靠 remap side 的 crop/zoom/
  perspective/rotation/affine 已不用於 winner batch。
- 結論：GRU head、winner loss 與 tail alignment 可學，沒有發現 silent
  target bug。凍結的 pretrained feature 不足以穩定分開 near/far，共用
  backbone 適應是必要的；生產訓練應保留共同更新與 side-safe
  augmentation，不再為此增加其他模型分支。接著用固定 seed 的 8-epoch
  `rally+winner` 5 fps run 與 rally-only 受控結果直接對照。

#### 2026-08-29 rally+winner 受控結果

- 使用與 rally-only 完全相同的 725/81 split、training/split seed 42、
  5 fps、backbone、GRU、clip 64、batch 8、500,000 frames/epoch 與 8-epoch
  budget，只將 tasks 從 `rally` 改為 `rally,winner`。程序正常完成並以
  exit status 0 退出，8 個 checkpoint 與 metrics 都連續落盤。
- rally mAP（rally-only／joint）：epoch 0 `0.4212/0.3797`、epoch 2
  `0.5877/0.6709`、epoch 3 `0.8027/0.8112`、epoch 5 `0.8325/0.8148`、
  epoch 6 `0.84925/0.85021`、epoch 7 `0.8337/0.7943`。winner loss 沒有
  導致持續性 rally 收斂失敗，但 joint 的 epoch 間方差較大；後三輪
  rally mAP 平均為 `0.8198`，低於 rally-only 的 `0.8385`。
- joint 的 best checkpoint 為 epoch 6，rally mAP `0.85021`，略高於同輪
  rally-only `0.84925`；但計劃明定的第 8 個完成 epoch（metrics
  `epoch=7`）只有 `0.79430`，低於 `>=0.85` 門檻 `0.05570`。不得用
  best checkpoint 改寫原定最終 epoch 驗收；A3 gate 依然關閉。
- winner validation top-1 由 epoch 0 `0.5477` 波動到 epoch 7 `0.5902`，
  高於當輪 majority baseline `0.3685`；最終 left/right/near/far recall 為
  `0.611/0.672/0.992/0.044`。這比前期整個類別為 0 的塌縮好，但 8
  epochs 仍未讓 far 穩定收斂。先前 30-epoch joint run 的 winner top-1
  `0.8600` 與四類非零 recall，加上 overfit ladder，證明路徑可學；本結果只證明
  8-epoch 受控 budget 下仍有顯著振盪與未收斂。

#### 2026-08-30 SequentialLR 修正版 rally+winner 結果

- 使用與修正版 rally-only 完全相同的 725/81 split、seed 42、5 fps、模型、
  clip 64、batch 8、500,000 frames/epoch 與 8-epoch budget，只加入 winner task。
  run 正常以 exit code 0 結束，8 個 checkpoint 與 metrics 完整，最終 optimizer
  param groups 的 LR 都是 `0.0`；可排除舊 scheduler 反彈或不完整 run。
- rally mAP（rally-only／joint）epoch 0--7 為：`0.4071/0.3926`、
  `0.3615/0.4228`、`0.5450/0.7537`、`0.7176/0.6601`、`0.7107/0.7285`、
  `0.8577/0.8307`、`0.8632/0.8438`、`0.86586/0.83362`。joint 最後三輪
  平均 `0.83604`，比 rally-only 的 `0.86227` 低 `0.02623`；best epoch 6
  也只有 `0.84378`。因此不論依最終 epoch 或 best checkpoint，都未達
  `>= 0.85`，A3 gate 仍關閉。
- winner validation top-1 在 epoch 6 達最佳 `0.75600`，高於當輪 majority
  baseline `0.40463`；left/right/near/far recall 為
  `0.945/0.757/0.446/0.564`。epoch 7 top-1 回落至 `0.73777`，recall 為
  `0.901/0.819/0.779/0.199`。模型已學到顯著訊號，但 8-epoch budget 下
  near/far 仍互相塌縮且 winner loss 後段反彈；這是 joint 收斂／task
  interference 問題，不是資料路徑、winner head 或 scheduler correctness 問題。

### A3. 未來 action / rally / winner 合併訓練

這是 A、A2 完成後的獨立產品增量，不阻塞本次 regression 修復。

- 一個 checkpoint，共用 visual backbone、optimizer 與 task heads。
- 每個 task 有自己的 dataset sampler，但讀同一份 native frame cache：
  - action batch：預設 30 fps；
  - rally batch：預設 5 fps；
  - winner batch：由 A2 實驗決定，短期先跟 rally 一樣為 5 fps。
- trainer 交替執行各 task batch；每次只保留一個 task batch 的 activations，所以
  peak VRAM 接近最重的單一 batch，而不是各 task 相加。
- 用固定的整體 frames/steps budget 分配 task 比例。若每個 task 都保留完整的舊
  budget，總訓練時間會相加；這是 budget 選擇，不是多 FPS 本身的成本。
- 不把 30 fps × 384 frames 當預設解法：它雖能給 rally 約 12.8 秒視窗，但 CNN
  計算、activation、loader 流量約為 5 fps × 64 frames 的六倍。
- 從最小端到端版本開始：先支援 action/rally 交替 batch，再疊加 winner；不先建
  通用 scheduler、任意 task graph 或其他尚未需要的抽象。

#### 2026-08-30 multi-FPS 實作與前置驗收

- 原先的 A3 gate 因 8-epoch `rally+winner` 未達 `0.85` 而關閉；使用者已明確決定
  不再等待該 gate，直接進行多 FPS 聯合訓練。這不改寫前述受控實驗結果，也不把
  未通過記成通過。
- contract 已直接升為 3.0.0，不保留舊 checkpoint/config fallback。一個 checkpoint
  內有獨立的 action／rally temporal heads、winner head、共同 visual backbone 與
  optimizer；推論多 spotting checkpoint 時必須明確指定 action 或 rally。
- trainer 只建立一個 train `DataLoader` 與一組 worker pool；action、rally、winner
  各自有 dataset/sampler，batch 以固定 round-robin 順序交替，每個 batch 只啟用該
  task 的 head。整體 `epoch_num_frames` 是共享 budget，不會因三個 task 乘三倍。
- action 使用 30 fps，rally 與 winner 各使用 5 fps；三者都從同一份 native-fps
  frame cache 依 stride 讀取。winner stream 會直接以有 winner 標籤的 rally tail
  為 anchor，避免隨機 clip 沒有 winner supervision。
- train/validation split 先以所有 task 的影片 stem 做全域分割，再建立各 task
  dataset，禁止同一影片從一個 task 的 train 洩漏到另一個 task 的 validation。
  validation 逐 task 執行；沒有 location head 的 action 以 temporal mAP 作 primary
  metric，rally 以 segment mAP，checkpoint 以兩個 spotting metric 的平均選擇。
- 真實資料 smoke（`rny008_tv_gsm`、clip 64、batch 8、8 train workers）完整跑過
  action → rally → winner train batch 與三組 validation batch，exit code 0；每個
  stream 都有獨立 loss/count，winner batch 170 個 supervised events。
- 同設定的保守 process-tree peak PSS 為 **6.535 GiB**，低於 8 GiB gate；父程序
  `RssShmem` 約 **149 MiB**，沒有恢復舊 pinned-memory 膨脹。full-model compile 的
  GPU peak 為 **19,034 MiB（18.588 GiB）**，在 24 GiB GPU 內。改成 multi eager
  反而升到 **20,194 MiB**；只 compile backbone 為 **19,030 MiB**，沒有節省且會
  增加額外 compile latency，因此保留最簡單且較快的 full-model compile。
- 完整 map/checkpoint smoke 以真實 cache 跑完 action、rally、winner：action
  temporal mAP `0.5833`、rally segment mAP `0.3333`，macro selection 正確為
  `0.4583`；嚴格載入同一 checkpoint 後，action 推論只回 action scores，rally
  推論同時回 rally 與 winner scores。
- 第一個正式 run `20260830_all_view_act_ral_win_rny008_tv` 在 epoch 0 的完整
  action evaluate 暴露另一個 evaluator RAM 問題：即使不儲存 prediction JSON，
  evaluator 仍對約 1,200 萬 frames 的所有 class scores 呼叫 `tolist()`，並替沒有
  location head 的 action 建立無用的 zero location channel。process-tree PSS 因此
  升到 **10,265,182 KiB（9.79 GiB）**，父程序 anonymous memory 升到
  **9,100,444 KiB**。該 run 在產生錯誤 metric/checkpoint 前安全停止，不列入正式
  結果。
- evaluator 現在於 metric-only 模式只保留通過既有候選 threshold 的 compact
  events；不建立 dense score JSON、argmax records 或 action location accumulator，
  temporal-only action 也不再製造假 spatial channel。候選 threshold 與後續 metric
  原有的 cutoff 相同；tiny regression 的 action `0.5833`、rally `0.3333` 完全不變。
- 修正版正式 run `20260830_all_view_act_ral_win_rny008_tv-2` 的 epoch 0 已完整跑過
  train → val-loss → action full evaluate → rally/winner full evaluate；每秒遞迴量測
  的 process-tree peak PSS 為 **6,236,470 KiB（5.95 GiB）**，低於 8 GiB gate，
  完成 metric 聚合時亦未跳升。GPU peak 為 **19,192 MiB**。共享 500,000-frame
  budget 實際分配為 action `166,912`、rally `166,400`、winner `166,400` frames；
  winner stream 含 `52,729` 個 supervised events。`checkpoint_000.pt`、
  `optim_000.pt`、`checkpoint_best.pt` 與 epoch metric 均已落盤，epoch 1 已開始。
- 正式 run 使用 recipe 的既定 50 epochs、3 warm-up epochs、`3e-5` LR、clip 64、
  batch 8、8 workers、500,000 total frames/epoch、全資料與 seed 42。驗收要求程序
  正常完成、50 筆 metrics/checkpoint 連續、每輪三個 stream 都有 supervision、
  macro-best checkpoint 可同時嚴格載入 action/rally/winner；最終另外對照 action
  temporal mAP、rally segment mAP、winner top-1／majority baseline 與四類 recall，
  不以其中一個 task 的改善掩蓋另一個 task 的退化。

### B. web 記憶體：tracks 只有一個有界 cache owner

#### 實作

- `core/cache.py:StatCache` 加入 byte budget 與 LRU；明確處理：
  - hit 時更新 recency；
  - source stat 改變時扣除舊 weight；
  - concurrent miss；
  - 單一 entry 超過 budget 時只保留該 entry，不無限累積其他 entries。
- 不再讓同一份 tracks parsed records 同時由 `_index_cache` 與
  `read_jsonl_cached` 決定生命週期。
- 在 `tracklets/store.py` 建立 tracks 專用 cache entry，統一持有該影片的
  meta、records 與 `TrackletIndex`；`tracklet_index()` 與需要 raw records 的
  tracklet routes 都從這個入口讀。
- 把目前直接對 tracks 呼叫 `read_jsonl_cached` 或自行建 `TrackletIndex` 的 callers
  改到上述入口；只需要 header 的 caller 使用 `read_jsonl_header`。
- cache weight 先以來源 bytes 為基準並用真實最大／中位 tracks 檔校準；不要直接
  假定 `st_size × 3` 就等於 resident bytes。預算以「常用的一至兩支影片」為目標，
  最終數值由實測 RSS 決定。現有 3.0 GiB／387 份 tracks 的 worklist 實測採
  8 MiB source-byte budget，且必須在建構下一份索引前先淘汰，以免 miss 短暫
  同時持有兩份大型 object graph。
- `_warm_worklists` 保留衍生出的 `set[str]`／小型 worklist cache，不保留 121 支影片
  的 track records/index。

#### 驗收

- cold boot、warm-up 完成後 RSS < 1 GB。
- 依序開啟最大三支 tracks 影片後，RSS 有界；回到第一支時只允許符合 LRU 預期的
  一次重建，不得每個 endpoint 都重 parse。
- `/label` 各列表 endpoint warm 後維持 stat-check 速度。
- 新增 LRU、source invalidation、oversized entry 與 tracks 單一 owner 的測試。

#### 2026-08-29 實測

- 387 份 tracks（來源檔合計 3.0 GiB）、121 筆 association worklist：完整四組
  warm-up 後 RSS 962,228 KiB；同程序第二次 warm-up 0.409 秒。
- 依序讀取最大三份 tracks（24.0／21.3／20.8 MiB）再回到第一份：第一份依 LRU
  預期重建，cache 只保留一份；峰值 RSS 848,336 KiB、結束 RSS 640,028 KiB。
- 完整 worklist warm-up 的解析瞬間高水位約 1.14 GiB，但完成後穩態低於 1 GiB；
  高水位來自單份 JSONL 解析的暫存配置，不是 cache 持續累積。

### C. trainer 的 host RAM

#### 先量測

- 在同一個 epoch 分別記錄 train、val-loss、full evaluate 的：
  - process-tree / cgroup peak memory；
  - parent `RssAnon`、`RssShmem`；
  - worker 數與存活 PID；
  - iteration throughput。
- 先確認 peak 發生在哪一階段，不再以「三組 worker 一定同時常駐」作為前提。

#### 實作

- `--num_workers` 改成實際 train DataLoader worker 數，不再隱含乘 2 或乘 4；同步
  修正 API description、config output 與測試，不保留舊倍率語意。
- 初始值以 8 train workers 測試；val-loss 與 evaluate 最多使用
  `min(num_workers, 4)`。
- 保持 `persistent_workers=False`。
- train 保持 `prefetch_factor=2` 起跑；若仍超標，再用實測比較 1 與 2，不先新增
  額外設定項。
- 三組 DataLoader 都不使用 `pin_memory`。舊 run 的主程序 `smaps` 可見多個
  `/dev/zero (deleted)` shared arena，`RssShmem` 約 12 GiB，來源是 CUDA pinned
  host allocator，不是模型權重或 Python label cache。
- full evaluate 的 worker 只傳 native `uint8` frame；resize、normalize 與轉成
  float tensor 延後到 model device。避免 worker IPC／prefetch 放大已轉換的
  float32 clip，同時讓 train、val-loss、evaluate 共用同一條 device-side prepare
  路徑。

#### 驗收

- 用 8/28 同級的 batch/clip 設定完整跑過 train → val-loss → full evaluate。
- 整個 trainer process tree peak < 8 GB，且不只量 epoch 結束的一個時間點。
- throughput 相較 16 workers 不得出現不成比例的退化；記錄結果後固定最簡配置。

#### 2026-08-29 實測

- 修正版以同一份 725 train／81 val snapshot、rally-only、5 fps、clip 64、batch 8、
  train workers 8、val/eval workers 4，連續跑完 2 個 train → val-loss → full
  evaluate epoch。
- 每 5 秒遞迴加總 trainer 與所有解碼／DataLoader 子程序的 PSS；兩輪全程 peak
  **6.070 GiB**，低於 8 GiB 門檻。第二輪 evaluate 約 5.4--5.6 GiB，沒有逐 epoch
  累積；舊 run 同口徑約 16 GiB。
- train 約 4.0 iteration/s（977 iterations 約 4 分鐘），val-loss 約 11
  iteration/s；2 個完整 epoch 總牆鐘時間約 15 分 23 秒。epoch 0/1 rally mAP
  分別為 0.3567／0.4398；此 run 只驗 RAM，不作為第 8 epoch 的 FPS 受控結論。

### D. 錯誤訊息與 terminal log

- 所有 `rc != 0` 都先報 exit status，再把 `last_line` 當 context，不能反過來讓最後
  一行正常 stdout 冒充錯誤。
- `rc < 0` 顯示 signal 名稱與編號，例如 `killed by SIGKILL (9)`。
- SIGKILL 只能附註「possible host OOM; verify kernel/cgroup logs」，不能只憑 `-9`
  斷言 OOM。
- 在共用的 `stream_subprocess()` 關閉 log 前，把正常退出、非零退出或 signal
  結果寫到 `terminal.log`；SPOT、ReID、association 共用同一語意。
- 測試正常退出、正數 exit code、`-SIGTERM`、`-SIGKILL` 及最後 stdout 看似成功指標
  的情況。

### E. 整理 yp-spot 現有未提交改動

目前改動不能用一個「已驗證正確」的 commit 一次收下。依職責拆開並各自驗證：

1. uint8 loader + GPU `ClipAugment` + 移除 mixup；
2. `torch.compile`；
3. audio mmap；
4. OpenCV grab/retrieve decode + 移除 `decoder_threads`；
5. winner GRU head（實驗性，等 A2 overfit ladder 證明後才視為完成）；
6. worker 數調整屬於 C，不能混進上述 commit。

每一組至少跑直接相關測試；loader/augmentation 與 compile 還要跑一個小型 train +
eval smoke test，decode 要做輸出 frame index / count 對照。移除的 mixup、
`decoder_threads` 路徑直接刪除，不加 compatibility flag 或 fallback。

## 執行順序

1. **E：拆分現有改動**，先取得可比較、可回溯的 commit；winner GRU 保持實驗性。
2. **D：修正退出狀態紀錄**，確保後續長跑失敗時留下可信證據。
3. **B、C：先壓低 web 與 trainer RAM**，避免驗證 run 再次被 host OOM 中止。
4. **A：用 rally-only 受控實驗驗證 5 fps**。
5. **A2：完成 winner 診斷與 overfit ladder**。
6. **rally_winner 5 fps 聯合驗證**。
7. **A3：另以最小端到端增量實作 task-specific sampler 的多任務訓練**。

B、D 與 yp-spot 訓練邏輯彼此獨立，可以平行開發；但任何長時間 GPU 驗證都放在
B、C 的 RAM 驗收通過之後。
