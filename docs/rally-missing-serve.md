# Rally 開頭沒有 serve 的清單

掃描日期：2026-08-28 · 資料：`videos/rally-spot/annotations` × `videos/action/annotations`（200 支影片、8,860 rallies）

重跑：`uv run python scripts/scan_rally_edges.py`

判定：rally span `[start, end]` 內最前的動作事件不是 `serve`。共 32 筆。

分類：`開頭切太晚` = span 外 3 秒內就有 `serve`，標註在、是邊界偏了；`serve 不在最前` = span 內有 `serve`，但前面還有別的動作；`疑似漏標` = 前後都找不到鄰近的 `serve`；`看過畫面的判定` = 導播沒拍到那個 `serve`，或這球是犯規結束、本來就沒有 `serve` 可標。

## 看過畫面的判定 — 32 筆

看過畫面了，這批不是漏標：`導播問題` = 轉播切走（重播、觀眾、板凳），`serve` 不在帶子上；其他判定是犯規結束的球——觸網、越界、持球…是裁判的哨音不是觸球，所以沒有 `serve` 可標。留著是為了下次掃描不用再看一遍。

判定分布：導播問題 29、發球踩線 3。

| 判定 | 筆數 | 意思 |
|---|---:|---|
| 導播問題 | 29 | 轉播切到重播、觀眾或板凳，那個動作不在畫面上，看不到就標不到。 |
| 發球踩線 | 3 | 發球踩線，發球方直接失分；rally 沒有正常展開。 |

| 影片 | Rally | 起 | 訖 | 最前動作 | 判定 |
|---|---:|---:|---:|---|---|
| 03⧸14(六) 16_00｜例行賽G104 #獅子王 vs. #屏東台電｜企業21年甲級男女排球聯賽_set2 | 27 | 15:37 | 15:42 | receive | 導播問題 |
| 03⧸14(六) 18_00｜例行賽G105 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 1 | 0:00 | 0:06 | receive | 導播問題 |
| 03⧸14(六) 18_00｜例行賽G105 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 36 | 22:00 | 22:04 | receive | 導播問題 |
| 03⧸15(日) 13_00｜例行賽G106 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set3 | 41 | 24:48 | 24:50 | set | 導播問題 |
| 03⧸15(日) 17_00｜例行賽G108 #新北中纖 vs. #高雄台電｜企業21年甲級男女排球聯賽_set3 | 7 | 3:43 | 3:52 | set | 導播問題 |
| 03⧸20(五) 14_00｜挑戰賽G110 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set2 | 6 | 2:24 | 3:31 | receive | 導播問題 |
| 03⧸21(六) 18_00｜男子組冠軍賽G114 #雲林美津濃 vs. #屏東台電｜企業21年甲級男女排球聯賽_set1 | 18 | 10:06 | 10:15 | block | 導播問題 |
| 03⧸22(日) 17_00｜男子組冠軍賽 G117 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set1 | 30 | 16:33 | 16:38 | set | 導播問題 |
| 03⧸22(日) 17_00｜男子組冠軍賽 G117 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set2 | 2 | 0:25 | 0:36 | set | 導播問題 |
| 2025-10-11_G8_臺中連莊_vs_桃園雲豹飛將_set1 | 1 | 0:05 | 0:07 | set | 導播問題 |
| 37-39 Thriller! - Japan 🇯🇵 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 31 | 12:39 | 12:43 | receive | 導播問題 |
| 37-39 Thriller! - Japan 🇯🇵 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set2 | 12 | 5:07 | 5:09 | spike | 導播問題 |
| 37-39 Thriller! - Japan 🇯🇵 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set2 | 13 | 5:21 | 5:28 | receive | 導播問題 |
| 37-39 Thriller! - Japan 🇯🇵 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set2 | 20 | 8:52 | 8:55 | set | 導播問題 |
| Brazil 🇧🇷 vs. Italy 🇮🇹  ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 9 | 3:13 | 3:16 | score | 導播問題 |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 33 | 16:45 | 16:50 | receive | 導播問題 |
| France vs. Iran - Semi Final 2 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 24 | 11:05 | 11:10 | set | 導播問題 |
| Full Match ｜ Poland vs. Slovakia ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 21 | 9:10 | 9:22 | receive | 導播問題 |
| Japan 🇯🇵 vs. Czechia 🇨🇿 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 42 | 20:16 | 20:22 | receive | 導播問題 |
| Japan 🇯🇵 vs. France 🇫🇷 ｜ VNL 2025 - Full Match ｜ Week 3_set1 | 42 | 24:41 | 24:46 | receive | 導播問題 |
| Korea vs. Bulgaria - Classification 5-8 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 42 | 19:54 | 19:57 | set | 導播問題 |
| Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 24 | 13:07 | 13:11 | receive | 導播問題 |
| Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 27 | 14:05 | 14:10 | receive | 導播問題 |
| Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 44 | 23:00 | 23:06 | receive | 導播問題 |
| Spain vs. Iran - Ranking 3-4 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 31 | 19:58 | 20:07 | receive | 導播問題 |
| Spain vs. Iran - Ranking 3-4 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 46 | 29:39 | 29:49 | receive | 導播問題 |
| Suntory Sunbirds vs. Osaka Bluteon ｜ SV.LEAGUE 2025⧸26 ｜ Full Match - Volleyball_set1 | 5 | 1:56 | 2:18 | receive | 導播問題 |
| Uzbekistan vs. Pakistan - Ranking 5-6 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 16 | 6:35 | 6:44 | receive | 導播問題 |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G25 11⧸16 15_00 桃園雲豹飛將 vs 台鋼天鷹_set1 | 17 | 8:09 | 8:18 | receive | 導播問題 |
| Bulgaria 🇧🇬 vs. Argentina 🇦🇷 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 23 | 11:12 | 11:14 | score | 發球踩線 |
| Bulgaria 🇧🇬 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 37 | 18:51 | 18:58 | score | 發球踩線 |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 10 | 4:03 | 4:08 | score | 發球踩線 |
