# Rally 開頭沒有 serve 的清單

掃描日期：2026-08-28 · 資料：`videos/rally-spot/annotations` × `videos/action/annotations`（200 支影片、8,860 rallies）

重跑：`uv run python scripts/scan_rally_edges.py`

判定：rally span `[start, end]` 內最前的動作事件不是 `serve`。共 32 筆。

分類：`開頭切太晚` = span 外 3 秒內就有 `serve`，標註在、是邊界偏了；`serve 不在最前` = span 內有 `serve`，但前面還有別的動作；`疑似漏標` = 前後都找不到鄰近的 `serve`；`看過畫面的判定` = 導播沒拍到那個 `serve`，或這球是犯規結束、本來就沒有 `serve` 可標。

## 看過畫面的判定 — 30 筆

看過畫面了，這批不是漏標：`導播問題` = 轉播切走（重播、觀眾、板凳），`serve` 不在帶子上；其他判定是犯規結束的球——觸網、越界、持球…是裁判的哨音不是觸球，所以沒有 `serve` 可標。留著是為了下次掃描不用再看一遍。

判定分布：導播問題 29、發球踩線 1。

| 影片 | Rally | 起 | 訖 | 最前動作 | 判定 | 動作序列（前 6） |
|---|---:|---:|---:|---|---|---|
| 03⧸14(六) 16_00｜例行賽G104 #獅子王 vs. #屏東台電｜企業21年甲級男女排球聯賽_set2 | 27 | 15:37 | 15:42 | receive | 導播問題 | receive → set → spike → score |
| 03⧸14(六) 18_00｜例行賽G105 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 1 | 0:00 | 0:06 | receive | 導播問題 | receive → set → score → spike |
| 03⧸14(六) 18_00｜例行賽G105 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 36 | 22:00 | 22:04 | receive | 導播問題 | receive → set → spike → block → score |
| 03⧸15(日) 13_00｜例行賽G106 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set3 | 41 | 24:48 | 24:50 | set | 導播問題 | set → spike → block → score |
| 03⧸15(日) 17_00｜例行賽G108 #新北中纖 vs. #高雄台電｜企業21年甲級男女排球聯賽_set3 | 7 | 3:43 | 3:52 | set | 導播問題 | set → spike → receive → set → spike → block |
| 03⧸20(五) 14_00｜挑戰賽G110 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set2 | 6 | 2:24 | 3:31 | receive | 導播問題 | receive → set → spike → receive → set → spike |
| 03⧸21(六) 18_00｜男子組冠軍賽G114 #雲林美津濃 vs. #屏東台電｜企業21年甲級男女排球聯賽_set1 | 18 | 10:06 | 10:15 | block | 導播問題 | block → block → receive → set → receive → receive |
| 03⧸22(日) 17_00｜男子組冠軍賽 G117 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set1 | 30 | 16:33 | 16:38 | set | 導播問題 | set → spike → block → score |
| 03⧸22(日) 17_00｜男子組冠軍賽 G117 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set2 | 2 | 0:25 | 0:36 | set | 導播問題 | set → spike → receive → set → block → spike |
| 2025-10-11_G8_臺中連莊_vs_桃園雲豹飛將_set1 | 1 | 0:05 | 0:07 | set | 導播問題 | set → spike → score |
| 37-39 Thriller! - Japan 🇯🇵 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 31 | 12:39 | 12:43 | receive | 導播問題 | receive → set → spike → receive → score |
| 37-39 Thriller! - Japan 🇯🇵 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set2 | 12 | 5:07 | 5:09 | spike | 導播問題 | spike → score |
| 37-39 Thriller! - Japan 🇯🇵 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set2 | 13 | 5:21 | 5:28 | receive | 導播問題 | receive → set → spike → receive → set → spike |
| 37-39 Thriller! - Japan 🇯🇵 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set2 | 20 | 8:52 | 8:55 | set | 導播問題 | set → spike → score |
| Brazil 🇧🇷 vs. Italy 🇮🇹  ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 9 | 3:13 | 3:16 | score | 導播問題 | score |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 33 | 16:45 | 16:50 | receive | 導播問題 | receive → set → spike → block → score |
| France vs. Iran - Semi Final 2 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 24 | 11:05 | 11:10 | set | 導播問題 | set → spike → score |
| Full Match ｜ Poland vs. Slovakia ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 21 | 9:10 | 9:22 | receive | 導播問題 | receive → set → receive → receive → set → spike |
| Japan 🇯🇵 vs. Czechia 🇨🇿 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 42 | 20:16 | 20:22 | receive | 導播問題 | receive → set → spike → receive → score |
| Japan 🇯🇵 vs. France 🇫🇷 ｜ VNL 2025 - Full Match ｜ Week 3_set1 | 42 | 24:41 | 24:46 | receive | 導播問題 | receive → set → spike → score |
| Korea vs. Bulgaria - Classification 5-8 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 42 | 19:54 | 19:57 | set | 導播問題 | set → spike → score |
| Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 24 | 13:07 | 13:11 | receive | 導播問題 | receive → set → spike → score |
| Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 27 | 14:05 | 14:10 | receive | 導播問題 | receive → set → spike → block → score |
| Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 44 | 23:00 | 23:06 | receive | 導播問題 | receive → set → spike → block → score |
| Spain vs. Iran - Ranking 3-4 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 31 | 19:58 | 20:07 | receive | 導播問題 | receive → set → spike → block → receive → set |
| Spain vs. Iran - Ranking 3-4 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 46 | 29:39 | 29:49 | receive | 導播問題 | receive → set → spike → block → receive → set |
| Suntory Sunbirds vs. Osaka Bluteon ｜ SV.LEAGUE 2025⧸26 ｜ Full Match - Volleyball_set1 | 5 | 1:56 | 2:18 | receive | 導播問題 | receive → set → spike → block → receive → set |
| Uzbekistan vs. Pakistan - Ranking 5-6 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 16 | 6:35 | 6:44 | receive | 導播問題 | receive → set → spike → receive → set → spike |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G25 11⧸16 15_00 桃園雲豹飛將 vs 台鋼天鷹_set1 | 17 | 8:09 | 8:18 | receive | 導播問題 | receive → set → spike → block → receive → set |
| Bulgaria 🇧🇬 vs. Argentina 🇦🇷 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 23 | 11:12 | 11:14 | score | 發球踩線 | score → serve |

## serve 不在最前 — 2 筆

span 內有 `serve`，但它不是最前的事件。

| 影片 | Rally | 起 | 訖 | 最前動作 | serve 距開頭 | 動作序列（前 6） |
|---|---:|---:|---:|---|---:|---|
| Bulgaria 🇧🇬 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 37 | 18:51 | 18:58 | score | 2.2s | score → serve → receive → set |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 10 | 4:03 | 4:08 | score | 1.5s | score → serve → receive |
