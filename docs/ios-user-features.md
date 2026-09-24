# VolleyIQ iOS App：使用者功能整理

整理日期：2026-09-25。

VolleyIQ 目前的主要用途，是把排球比賽影片整理成可快速回看的回合與動作片段，讓使用者修正、分類、配對球員，再輸出個人或球隊需要的影片。

本文依目前 App 的畫面、操作入口與對應流程整理，描述已接入程式的能力；尚未以這一版在 iPhone／Mac 上逐項實測。需要雲端分析、球員辨識或同步的功能，也取決於服務是否正常運作。

## 1. 使用者從哪裡開始

底部有五個主要入口：

| 入口 | 使用者可以做什麼 |
|---|---|
| 影片 | 管理比賽影片與資料夾、查看處理進度、進入回看片段 |
| 你的收藏 | 依自訂分類整理片段、查看系統挑選的回合、製作合輯 |
| 上傳 | 從照片圖庫或 YouTube 匯入影片，送出分析 |
| 球員配對 | 辨識人物、確認是誰、查看球員片段與觸球次數 |
| 設定 | 登入、播放偏好、語言與外觀、清理快取 |

第一次開啟會先選語言，再顯示入門介紹。上傳前需要完成 Apple 登入，並連接可用的分析服務。

典型使用流程：

1. 匯入一場比賽影片，命名並選擇拍攝角度。
2. 等待分析，查看已產生的回合、動作與回合結束片段。
3. 按球員或動作篩選，回看需要檢討的球。
4. 調整片段範圍、確認人物、修正得分方，並確認或標記失分原因。
5. 加入收藏分類，匯出片段、製作合輯，或分享連結。

## 2. 匯入影片與分析

### 影片來源

| 來源 | 目前可操作的內容 |
|---|---|
| iPhone 照片圖庫 | 一次選取多支影片；逐支命名，或用批次名稱加編號；選擇存放資料夾 |
| YouTube 單支影片 | 貼上網址、自動取得標題、修改名稱、選擇下載畫質 |
| YouTube 播放清單 | 展開影片清單、勾選要匯入的項目、修改各影片名稱，再批次送出 |

YouTube 畫質選項為 360p、480p、720p、1080p，預設 1080p。實際可取得的影片與畫質仍受來源及服務限制。

送出前可選「手機場邊拍攝」「場館固定鏡頭」「電視轉播」三種拍攝角度，也可指定資料夾。從資料夾內開啟上傳時，會預先選取該資料夾。

### 分析期間與重新分析

- 查看處理百分比與目前階段；影片庫另有「處理中」清單。
- 分析會逐步提供結果，不必等整支影片處理完才看到所有已產生的內容；處理中會顯示已找到的回合數，可先點入預覽部分結果。
- 運算主機暫時沒有回應時，畫面會顯示等待主機的提示。
- 可以取消進行中的分析。
- 完成後可從影片選單重新分析 Rally，或只重新分析 Action。

| 重新分析選項 | 對使用者資料的影響 |
|---|---|
| Rally | 重跑整支影片，重新產生回合及 Action／Score；原有回合分類標籤與已看標記會重置 |
| Action | 沿用現有回合，包含手動調整過的範圍，只重跑 Action／Score |

手動修正的得分方、球員與失分原因，只有在新舊結果仍能對應相同時間點時才會保留；不能把重新分析視為所有修正都一定保留的操作。

## 3. 影片庫與資料夾

- 查看比賽影片、分析狀態與基本資訊。
- 重新命名、刪除影片，或移入／移出資料夾。
- 建立資料夾，依比賽、日期或球隊整理影片。
- 批次選取影片後移動或刪除。
- 手動調整資料夾與影片的排列順序。
- 刪除資料夾時，可選擇只刪資料夾，或連同其中的影片一起刪除。
- 為完成分析的影片或資料夾內的完成影片建立分享連結。

## 4. 一場影片的三種回看方式

進入影片後，可以切換 Rally、Action、Score。這三個頁籤觀看的是同一支影片的不同片段。

| 頁籤 | 適合回答的問題 | 目前功能 |
|---|---|---|
| Rally | 「這一球從開始到結束發生了什麼？」 | 回合列表、時間範圍、組織進攻次數與失分原因、排序、播放、裁切、分類與刪除 |
| Action | 「我要看某位球員的接球／舉球／扣球」 | 依觸球動作切出的片段、球員與動作篩選、動作時間軸、人物修正 |
| Score | 「這一球最後怎麼結束？我們為何失分？」 | 回合結束片段、場邊得分累計、組織進攻次數、得分方修正與各側失分原因彙整 |

### 篩選與片段取景

- 依球員篩選；需要先建立名單，並有可對應的球員標記或配對。
- 依發球、接球、舉球、扣球、攔網篩選。
- Action／Score 可選「In Rally」或「全部」，決定是否包含回合外的事件。
- 可從某個 Rally 跳到該回合內的 Action。
- Rally 可依比賽時間順序或畫面稱為「精彩」的分數排序。
- 切換頁籤時，會保留共用的球員／動作篩選條件。

Action 另有三種觀看範圍，不必重新分析就能切換：

| 取景 | 使用情境 |
|---|---|
| 組織進攻 | 看目標動作之前的接、舉、扣組織與後續結果 |
| 單一動作 | 聚焦目標動作到下一個動作之間的片段 |
| 整個 Rally | 回到目標動作所屬的完整回合 |

## 5. 播放、修剪與排除誤判

- 播放／暫停，拖曳進度定位，切換前後片段。
- 全螢幕觀看，並支援橫向播放。
- 開啟或關閉自動接續播放。
- Rally 播放到結尾後記錄為已看。
- Action／Score 顯示動作時間軸，方便定位片段內的觸球。
- Rally 可以拖曳起點與終點、預覽修剪結果，或還原 AI 建議範圍。
- Action／Score 可以調整片段範圍；拖曳端點會吸附到可用的動作邊界，也可還原預設。
- 可以刪除誤判的回合或片段，也有批次刪除入口。

片段修剪影響回看與匯出的範圍。排除 Action／Score 誤判片段不會刪除原始影片；「刪除整支影片」是影片庫中的另一項操作。

## 6. 得失分與失分原因

Score 詳情與 Rally 詳情都可以修正：

- 得分方：預設帶入系統辨識的獲勝側，有誤可直接改選；系統無法判定的回合也能手動指定。
- 失分原因：選定得分方後，為另一側標記原因，例如扣球失誤、被攔網、舉球失誤、接發失誤、嗆司接噴、發球失誤、防守失誤或其他。
- 新增這場比賽專用的自訂失分原因，並可移除自訂原因。
- 失分原因卡片可以收合，收合後仍顯示目前的原因；收合狀態會記住。

只發球就結束的回合，系統會預判為「發球失誤」；發球後接球就結束的回合，預判為「接發失誤」。預判會標示「自動判定」，看完這一球後，可在 Rally 詳情、Score 詳情或全螢幕播放中一鍵確認，也可以改選其他原因；使用者自己標記的原因一律優先。其他回合不會自動判定。

Score 列表可以查看：

- 整場的組織進攻次數。
- 依場側展開失分分析，查看各側失分數量、最常見原因與原因分布；尚未標記的失分會另列為「未標記」，得分方未確認的回合不納入統計。選取球員等篩選條件時，失分分析也會跟著縮小範圍，組織進攻次數則維持整場。

Rally 列表的每個回合也會顯示該回合的組織進攻次數與目前的失分原因，並以不同顏色區分自動判定與人工標記。

這裡有兩種不同的資訊：

| 資訊 | 代表什麼 |
|---|---|
| 場邊比分／獲勝側 | 累計左、右、近、遠側的獲勝回合，採用系統辨識結果或使用者修正後的得分方；無法判定的回合會列為未知 |
| 失分原因 | 使用者標記的原因，加上發球、接發就結束之回合的自動判定；失分分析依這些原因計算 |

**場邊累計目前不會追蹤隊伍換邊，也不是完整的正式賽事計分。** 現行分析輸出尚未自動分局，不能將一整場多局影片的結果當成已正確分局的比分。除了上述發球、接發就結束的回合，失分原因仍需使用者自行判斷與標記。

## 7. 球員名單、辨識與配對

### 建立與管理名單

- 為每場比賽新增球員，填寫背號、姓名與位置。
- 編輯或刪除球員。
- 從其他比賽匯入既有球員名單。
- 在動作詳情直接指定球員，或開啟名單編輯；Rally 詳情與全螢幕播放也能指定目前播放位置附近那次觸球的球員。

### 確認畫面中的人是誰

1. 在完成分析的影片中啟動球員辨識，查看進度；可取消或重新辨識。
2. 系統將可能屬於同一人的出場片段分組，列為待確認項目。
3. 查看人物裁切、展開完整畫面或播放預覽，確認身分。
4. 將整組片段指派給名單中的球員，或逐筆修正。

還可以：

- 調整「配對鬆緊」，讓未完成配對的項目合併得更多或分得更細。
- 將裁判、觀眾等誤抓片段標為「不是球員」，並復原被移除的項目。
- 解除已完成的配對。
- 將某次觸球改回「未指定」；即使它所屬的出場片段已配對到球員，也會維持未指定。
- 修正動作的球員時，選擇只改這一次，或一併修改同一人物出場片段內的多個動作。

目前需要使用者確認人物身分，不能視為已能全自動讀取背號、姓名並保證配對正確。

### 以球員為中心回看

- 球員詳情顯示觸球總數，以及各種動作的次數。
- 點擊「全部」「扣球」「接球」等統計，直接在同一頁篩選下方的觸球片段列表；點片段進入動作詳情播放。
- 在片段上解除配對：觸球來自已配對的出場片段時，會解除整個出場片段；否則只解除這一次觸球。
- 從右上選單輸出這位球員在該場比賽的個人合輯，也可編輯球員資料。
- 把某場比賽的背號連結到同一位跨比賽球員；建立連結後，可以切換「本場／全部比賽」查看該球員的片段。

這些數量取決於已偵測的動作與人物配對／手動修正。目前不是完整的球員能力評分、攻擊成功率或戰術評估報表。

## 8. 收藏分類與系統精選

- 自訂收藏分類名稱，也可使用常見動作或日期的建議名稱。
- 將 Rally、Action 或 Score 片段加入分類，之後從「你的收藏」集中回看。
- 同一分類可以收集不同影片的片段。
- 調整分類與分類內片段的順序，刪除不需要的分類。
- 勾選分類中的片段，製作跨影片合輯。
- 從收藏頂部進入系統精選，每支完成影片最多列出分數最高的四個回合，再進入完整列表。

**「精彩」目前是介面上的名稱。** 現行後端把回合偵測分數換算成 0–100，App 以此排序與挑選精選；它不代表已建立獨立的精彩程度或球技評分模型。

## 9. 匯出與分享

### 輸出影片檔

- 匯出單一 Rally、Action 或 Score 片段。
- 選取多個片段，分別輸出成多支影片，或串成一支合輯。
- 儲存到照片圖庫，或交給 iOS 系統分享面板，傳送至裝置上可用的 App。
- 匯出採用使用者調整後的片段範圍。
- 顯示準備、下載、合成與匯出進度；失敗時可重試。
- 匯出影片帶有 VolleyIQ 浮水印。

### 分享連結

可以針對完成的比賽影片、選取的 Rally，以及包含 Rally 的收藏建立雲端連結。資料夾分享會收集其中完成分析的影片。

| 權限 | 收件者可以做什麼 |
|---|---|
| 僅供觀看 | 預覽分享內容；登入後可匯入自己的獨立副本 |
| 共同編輯 | 登入後加入同一場比賽，透過同步共享編輯結果 |

App 可以開啟分享連結、播放其中的影片／回合，再匯入影片庫或加入共編。

**Action／Score 可匯出影片檔，但目前沒有各自的雲端片段連結。** 混合收藏的雲端分享只包含 Rally；影片合輯則可以包含 Rally、Action 和 Score。

## 10. 帳號、同步與個人偏好

- 使用 Apple 登入及登出。
- 不同帳號有各自的影片庫。
- 登入並連線時，自動同步影片庫、資料夾、回合、收藏及支援的人工修正；前景使用時持續更新，回到 App 時重新啟動同步。
- 設定頁可顯示同步錯誤，供使用者察覺尚未成功送出的編輯。
- 切換繁體中文／英文。
- 選擇淺色、深色或跟隨系統，以及介面配色。
- 設定預設回合排序與自動接續播放。
- 查看影片快取佔用空間並清除快取。
- 查看 App 版本與 build 編號。

本機快取可以減少重複下載，但目前不應宣稱整套產品可離線使用；匯入 YouTube、雲端分析、辨識、分享與同步都需要服務連線。

## 11. 目前仍屬測試設定或尚未具備的能力

| 項目 | 目前狀態 |
|---|---|
| Free／Pro 方案 | 設定頁的開發測試選項，尚未接上正式訂閱或付款；目前用來控制照片圖庫單批可選影片數，Free 2 支、Pro 10 支 |
| 後端服務網址 | 設定頁仍提供的開發選項；一般使用流程依賴已配置的可用服務 |
| 全自動球員身分辨識 | 有人物分組與配對建議，但仍需使用者確認與修正 |
| 全自動失分原因分析 | 只有發球、接發就結束的回合會自動預判，且需使用者確認；其餘原因仍來自人工標記 |
| 正式隊伍比分、自動分局 | 現有場邊累計尚未涵蓋隊伍換邊追蹤與自動分局 |
| 球員技術評分／完整戰術報表 | 目前提供動作片段、觸球數與人工標記彙整，尚無這類完整報表 |
| App 內直接拍攝、檔案 App 匯入 | 目前上傳入口是照片圖庫與 YouTube，未提供這兩種來源入口 |

## 附錄：整理依據

本次檢查的 iOS 原始碼版本為 `VolleyIQ@14c707b`；與分數、分局相關的服務行為另對照 `volleyiq-backend@726ceec`。以下連結供維護者更新文件時查核，正文以使用者操作為主。

| 功能 | 主要依據 |
|---|---|
| 分頁、登入前提、同步啟動 | [RootView.swift](../../VolleyIQ/VolleyIQ/App/RootView.swift)、[AccountLibraryView.swift](../../VolleyIQ/VolleyIQ/App/AccountLibraryView.swift) |
| 匯入、播放清單、畫質與拍攝角度 | [UploadSheet.swift](../../VolleyIQ/VolleyIQ/Screens/Upload/UploadSheet.swift)、[UploadSourceStep.swift](../../VolleyIQ/VolleyIQ/Screens/Upload/UploadSourceStep.swift)、[CameraAngle.swift](../../VolleyIQ/VolleyIQ/Models/CameraAngle.swift) |
| 影片庫、資料夾、處理進度 | [LibraryView.swift](../../VolleyIQ/VolleyIQ/Screens/Library/LibraryView.swift)、[FolderDetailView.swift](../../VolleyIQ/VolleyIQ/Screens/Library/FolderDetailView.swift)、[ProcessingHeader.swift](../../VolleyIQ/VolleyIQ/Screens/RallyFeed/ProcessingHeader.swift) |
| 三種片段與重新分析 | [RallyFeedView.swift](../../VolleyIQ/VolleyIQ/Screens/RallyFeed/RallyFeedView.swift)、[AdvancedFeedOptions.swift](../../VolleyIQ/VolleyIQ/Components/AdvancedFeedOptions.swift)、[ActionEvent.swift](../../VolleyIQ/VolleyIQ/Models/ActionEvent.swift) |
| 播放與修剪 | [RallyDetailView.swift](../../VolleyIQ/VolleyIQ/Screens/RallyDetail/RallyDetailView.swift)、[TrimSheet.swift](../../VolleyIQ/VolleyIQ/Screens/Trim/TrimSheet.swift)、[ActionRangeTrimmer.swift](../../VolleyIQ/VolleyIQ/Screens/Trim/ActionRangeTrimmer.swift) |
| 得失分與比分限制 | [ScoreScopeView.swift](../../VolleyIQ/VolleyIQ/Screens/RallyFeed/ScoreScopeView.swift)、[ScoreDetailView.swift](../../VolleyIQ/VolleyIQ/Screens/ScoreDetail/ScoreDetailView.swift)、[CourtScore.swift](../../VolleyIQ/VolleyIQ/Models/CourtScore.swift)、[RallyWinnerPicker.swift](../../VolleyIQ/VolleyIQ/Components/RallyWinnerPicker.swift) |
| 失分原因與自動判定、組織進攻次數 | [LossReasonEditor.swift](../../VolleyIQ/VolleyIQ/Components/LossReasonEditor.swift)、[LossReason.swift](../../VolleyIQ/VolleyIQ/Models/LossReason.swift)、[Match+Actions.swift](../../VolleyIQ/VolleyIQ/Models/Match+Actions.swift)、[RallyRules.swift](../../VolleyIQ/VolleyIQ/Models/RallyRules.swift) |
| 球員配對與跨比賽回看 | [MatchPairingView.swift](../../VolleyIQ/VolleyIQ/Screens/PlayerPairing/MatchPairingView.swift)、[PlayerDetailView.swift](../../VolleyIQ/VolleyIQ/Screens/PlayerPairing/PlayerDetailView.swift)、[PlayerClipsList.swift](../../VolleyIQ/VolleyIQ/Screens/PlayerPairing/PlayerClipsList.swift) |
| 收藏與精選 | [FavoritesView.swift](../../VolleyIQ/VolleyIQ/Screens/Favorites/FavoritesView.swift)、[HighlightsView.swift](../../VolleyIQ/VolleyIQ/Screens/Highlights/HighlightsView.swift) |
| 匯出、連結與共編 | [ExportSheet.swift](../../VolleyIQ/VolleyIQ/Screens/Share/ExportSheet.swift)、[HighlightReelSheet.swift](../../VolleyIQ/VolleyIQ/Screens/Share/HighlightReelSheet.swift)、[CloudShareSheet.swift](../../VolleyIQ/VolleyIQ/Screens/Share/CloudShareSheet.swift)、[SharedCloudShareView.swift](../../VolleyIQ/VolleyIQ/Screens/Share/SharedCloudShareView.swift) |
| 設定、方案與修正同步 | [SettingsView.swift](../../VolleyIQ/VolleyIQ/Screens/Settings/SettingsView.swift)、[UserTier.swift](../../VolleyIQ/VolleyIQ/Models/UserTier.swift)、[MatchCorrections.swift](../../VolleyIQ/VolleyIQ/Models/MatchCorrections.swift) |
| 精選分數與尚未分局的服務輸出 | [worker/main.py](../../volleyiq-backend/selfhost-worker/worker/main.py) 的 `_segments_to_rallies` |
