# Rally 開頭沒有 serve 的清單

掃描日期：2026-08-25 · 資料：`videos/rally-spot/annotations` × `videos/action/annotations`（197 支影片、8,722 rallies）

重跑：`uv run python scripts/scan_rally_edges.py`

判定：rally span `[start, end]` 內最前的動作事件不是 `serve`。共 88 筆。

分類：`開頭切太晚` = span 外 3 秒內就有 `serve`，標註在、是邊界偏了；`serve 不在最前` = span 內有 `serve`，但前面還有別的動作；`疑似漏標` = 前後都找不到鄰近的 `serve`。

## 疑似漏標 — 75 筆

| 影片 | Rally | 起 | 訖 | 最前動作 | 動作序列（前 6） |
|---|---:|---:|---:|---|---|
| 03⧸14(六) 16_00｜例行賽G104 #獅子王 vs. #屏東台電｜企業21年甲級男女排球聯賽_set2 | 13 | 8:32 | 8:48 | set | set → receive → spike → receive → receive → receive |
| 03⧸14(六) 16_00｜例行賽G104 #獅子王 vs. #屏東台電｜企業21年甲級男女排球聯賽_set2 | 27 | 15:38 | 15:42 | receive | receive → set → spike → score |
| 03⧸14(六) 18_00｜例行賽G105 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 36 | 22:00 | 22:04 | set | set → spike → block → score |
| 03⧸15(日) 13_00｜例行賽G106 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set3 | 41 | 24:48 | 24:51 | set | set → spike → block → score |
| 03⧸15(日) 17_00｜例行賽G108 #新北中纖 vs. #高雄台電｜企業21年甲級男女排球聯賽_set1 | 19 | 9:18 | 9:32 | receive | receive → set → spike → block → receive → set |
| 03⧸15(日) 17_00｜例行賽G108 #新北中纖 vs. #高雄台電｜企業21年甲級男女排球聯賽_set3 | 7 | 3:43 | 3:52 | set | set → spike → receive → set → spike → block |
| 03⧸20(五) 14_00｜挑戰賽G110 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set2 | 6 | 2:25 | 3:31 | receive | receive → set → spike → receive → set → spike |
| 03⧸21(六) 18_00｜男子組冠軍賽G114 #雲林美津濃 vs. #屏東台電｜企業21年甲級男女排球聯賽_set1 | 18 | 10:06 | 10:16 | block | block → block → receive → receive → receive → receive |
| 03⧸22(日) 17_00｜男子組冠軍賽 G117 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set1 | 30 | 16:33 | 16:38 | set | set → spike → block → score |
| 03⧸22(日) 17_00｜男子組冠軍賽 G117 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set2 | 2 | 0:25 | 0:36 | receive | receive → set → spike → receive → set → spike |
| 0601小窩季打 2 | 1 | 0:04 | 0:59 | receive | receive → set → spike → receive → set → spike |
| 0601小窩季打 2 | 14 | 4:40 | 4:54 | receive | receive → set → spike → receive → set → spike |
| 0601小窩季打 2 | 22 | 6:50 | 7:03 | receive | receive → set → spike → receive → set → spike |
| 10⧸4 2 | 12 | 5:12 | 5:22 | receive | receive → set → spike → block → score |
| 10⧸4 3 | 14 | 3:52 | 4:00 | receive | receive |
| 2025-10-11_G8_臺中連莊_vs_桃園雲豹飛將_set1 | 1 | 0:05 | 0:08 | set | set → spike → score |
| 2025-10-11_G8_臺中連莊_vs_桃園雲豹飛將_set1 | 26 | 15:26 | 15:33 | set | set → score |
| 2025-11-08_G19_臺北伊斯特_vs_臺中連莊_set1 | 43 | 25:57 | 26:03 | receive | receive → set → spike → score |
| 20250424 排島惡館-7 | 9 | 4:38 | 4:48 | set | set → spike → receive → set → spike → score |
| 20250504 大統OB-成功大學vs台北大學B-第二局 | 1 | 0:18 | 0:36 | receive | receive → set → spike → block → receive → set |
| 20250504 大統OB-成功大學vs台北大學B-第二局 | 20 | 9:00 | 9:08 | receive | receive → set → spike → score |
| 20251109-排島本館-03 | 5 | 2:14 | 2:22 | receive | receive → receive → set → receive → spike → score |
| 20251109-排島本館-03 | 11 | 4:38 | 4:50 | spike | spike → receive → set → spike → receive → set |
| 20251109-排島本館-03 | 47 | 16:24 | 16:41 | set | set → receive → set → spike → receive → set |
| 20260403-霖度C-01_set1 | 22 | 8:46 | 9:01 | receive | receive → set → receive → receive → set → spike |
| 20260426-小窩-01 | 15 | 5:52 | 6:26 | receive | receive → receive → set → receive → receive → set |
| 20260426-小窩-01 | 34 | 14:14 | 14:32 | receive | receive → set → spike → receive → set → spike |
| 20260510邷力豹臨打1 | 24 | 12:14 | 12:44 | receive | receive → set → receive → receive → set → spike |
| 2026⧸03⧸25 3 | 40 | 13:54 | 13:59 | receive | receive → score |
| 37-39 Thriller! - Japan 🇯🇵 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 31 | 12:39 | 12:44 | receive | receive → set → spike → receive → score |
| 37-39 Thriller! - Japan 🇯🇵 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set2 | 12 | 5:07 | 5:09 | spike | spike → score |
| 37-39 Thriller! - Japan 🇯🇵 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set2 | 13 | 5:21 | 5:28 | receive | receive → set → spike → receive → set → spike |
| 37-39 Thriller! - Japan 🇯🇵 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set2 | 20 | 8:52 | 8:55 | receive | receive → set → spike → score |
| Brazil 🇧🇷 vs. Italy 🇮🇹  ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 9 | 3:14 | 3:16 | score | score |
| Brazil 🇧🇷 vs. Italy 🇮🇹  ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 10 | 3:38 | 3:44 | set | set → spike → score |
| Bulgaria 🇧🇬 vs. Argentina 🇦🇷 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 23 | 11:13 | 11:16 | score | score |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 33 | 16:45 | 16:49 | receive | receive → set → spike → block → score |
| China 🇨🇳 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 9 | 4:52 | 4:58 | set | set → spike → block → score |
| France vs. Iran - Semi Final 2 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 24 | 11:05 | 11:12 | set | set → spike → score |
| Full Match ｜ Poland vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 37 | 19:36 | 19:42 | receive | receive → set → spike → block → score |
| Full Match ｜ Poland vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 41 | 21:42 | 21:45 | score | score |
| Full Match ｜ Poland vs. Slovakia ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 21 | 9:12 | 9:22 | set | set → receive → receive → set → spike → receive |
| Full Match ｜ Serbia vs. Bulgaria ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set1 | 10 | 3:50 | 4:03 | set | set → spike → block → receive → receive → set |
| Full Match ｜ Slovakia vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 6 | 2:42 | 2:52 | receive | receive → receive → receive → receive → set → spike |
| Full Match ｜ Slovakia vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 40 | 21:58 | 22:06 | receive | receive → set → spike |
| Full Match ｜ Spain vs The Netherlands ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool C_set1 | 26 | 14:32 | 14:38 | score | score |
| Japan 🇯🇵 vs. Czechia 🇨🇿 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 20 | 9:06 | 9:11 | receive | receive → set → spike |
| Japan 🇯🇵 vs. Czechia 🇨🇿 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 22 | 9:50 | 9:56 | receive | receive → set → spike → receive → score → receive |
| Japan 🇯🇵 vs. Czechia 🇨🇿 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 42 | 20:17 | 20:22 | receive | receive → set → spike → receive → score |
| Japan 🇯🇵 vs. France 🇫🇷 ｜ VNL 2025 - Full Match ｜ Week 3_set1 | 42 | 24:41 | 24:46 | receive | receive → set → spike → score |
| Japan 🇯🇵 vs. Netherlands 🇳🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 36 | 18:12 | 18:31 | receive | receive → set → spike → block → receive → set |
| Japan 🇯🇵 vs. Serbia 🇷🇸 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 1 | 0:00 | 0:04 | score | score |
| Korea vs. Bulgaria - Classification 5-8 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 42 | 19:54 | 19:58 | set | set → spike → score |
| Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 16 | 8:22 | 8:38 | receive | receive → set → spike → block → receive → set |
| Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 17 | 8:58 | 9:04 | receive | receive → set → spike → block → receive → score |
| Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 24 | 13:07 | 13:12 | receive | receive → set → spike → score |
| Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 27 | 14:05 | 14:12 | receive | receive → set → spike → block → score |
| Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 36 | 18:57 | 19:04 | receive | receive → set → spike → score |
| Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 37 | 19:24 | 19:40 | spike | spike → receive → set → spike → receive → receive |
| Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 44 | 23:00 | 23:08 | receive | receive → set → spike → block → score |
| Spain vs. Iran - Ranking 3-4 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 31 | 19:58 | 20:08 | receive | receive → set → spike → block → receive → set |
| Spain vs. Iran - Ranking 3-4 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 46 | 29:39 | 29:51 | receive | receive → set → spike → block → receive → set |
| Suntory Sunbirds vs. Osaka Bluteon ｜ SV.LEAGUE 2025⧸26 ｜ Full Match - Volleyball_set1 | 5 | 1:56 | 2:18 | receive | receive → set → spike → block → receive → set |
| Suntory Sunbirds vs. Osaka Bluteon ｜ SV.LEAGUE 2025⧸26 ｜ Full Match - Volleyball_set1 | 8 | 4:28 | 4:34 | spike | spike → receive → set → spike → score |
| Uzbekistan vs. Japan - Ranking 19-20 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 31 | 14:28 | 14:35 | receive | receive → set → spike → score |
| Uzbekistan vs. Pakistan - Ranking 5-6 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 16 | 6:35 | 6:45 | receive | receive → set → spike → receive → set → spike |
| ᴴᴰ114UVL預賽：：中山大學vs國北教大：：男一級 大專排球聯賽 AI網路直播_set1 | 31 | 14:16 | 14:24 | receive | receive → set → spike → score |
| ᴴᴰ114UVL預賽：：清華大學vs嘉義大學：：男一級 大專排球聯賽 AI網路直播_set1 | 21 | 9:14 | 9:20 | receive | receive → set → spike → score |
| ᴴᴰ114UVL預賽：：清華大學vs嘉義大學：：男一級 大專排球聯賽 AI網路直播_set1 | 22 | 9:36 | 9:44 | receive | receive → receive → set → spike → score |
| ᴴᴰ114UVL預賽：：清華大學vs嘉義大學：：男一級 大專排球聯賽 AI網路直播_set1 | 23 | 10:00 | 10:30 | receive | receive → set → spike → receive → receive → receive |
| ᴴᴰ114UVL預賽：：清華大學vs嘉義大學：：男一級 大專排球聯賽 AI網路直播_set1 | 26 | 11:44 | 11:48 | score | score |
| ᴴᴰ114UVL預賽：：清華大學vs嘉義大學：：男一級 大專排球聯賽 AI網路直播_set1 | 35 | 16:22 | 16:28 | receive | receive → spike → score |
| ᴴᴰ114UVL預賽：：清華大學vs嘉義大學：：男一級 大專排球聯賽 AI網路直播_set1 | 37 | 17:06 | 17:17 | receive | receive → set → spike → block → receive → set |
| ᴴᴰ114UVL預賽：：臺灣師大vs中山大學：：男一級 大專排球聯賽 AI網路直播_set1 | 9 | 3:26 | 3:34 | spike | spike → receive → set → spike → block → score |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G25 11⧸16 15_00 桃園雲豹飛將 vs 台鋼天鷹_set1 | 17 | 8:10 | 8:18 | receive | receive → set → spike → block → receive → set |

## serve 不在最前 — 13 筆

span 內有 `serve`，但它不是最前的事件。

| 影片 | Rally | 起 | 訖 | 最前動作 | serve 距開頭 | 動作序列（前 6） |
|---|---:|---:|---:|---|---:|---|
| 03⧸14(六) 14_00｜例行賽G103 #雲林美津濃 vs. #桃園臺灣產險｜企業21年甲級男女排球聯賽_set1 | 40 | 20:01 | 20:08 | spike | 1.5s | spike → spike → serve → receive → set → spike |
| 03⧸14(六) 16_00｜例行賽G104 #獅子王 vs. #屏東台電｜企業21年甲級男女排球聯賽_set1 | 7 | 2:41 | 2:49 | spike | 1.5s | spike → serve → receive → set → spike → block |
| 03⧸14(六) 16_00｜例行賽G104 #獅子王 vs. #屏東台電｜企業21年甲級男女排球聯賽_set1 | 15 | 6:18 | 6:27 | set | 1.5s | set → serve → serve → receive → set → spike |
| 03⧸15(日) 13_00｜例行賽G106 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set1 | 20 | 10:57 | 11:12 | spike | 1.5s | spike → serve → receive → set → spike → block |
| 03⧸15(日) 17_00｜例行賽G108 #新北中纖 vs. #高雄台電｜企業21年甲級男女排球聯賽_set1 | 13 | 6:35 | 6:44 | set | 1.5s | set → serve → receive → spike → receive → receive |
| 03⧸15(日) 17_00｜例行賽G108 #新北中纖 vs. #高雄台電｜企業21年甲級男女排球聯賽_set1 | 39 | 22:37 | 22:46 | spike | 1.5s | spike → set → serve → receive → set → spike |
| 03⧸20(五) 16_00｜挑戰賽G111 #屏東台電 vs. #桃園臺產｜企業21年甲級男女排球聯賽_set1 | 32 | 15:05 | 15:12 | set | 1.5s | set → serve → receive → set → spike → score |
| 2025-10-11_G7_臺北伊斯特_vs_台鋼天鷹_set1 | 40 | 34:31 | 34:34 | receive | 1.5s | receive → receive → receive → serve → score |
| 20250628-霖度C-1 | 40 | 13:05 | 13:13 | spike | 1.5s | spike → serve → receive → set → spike → score |
| Bulgaria 🇧🇬 vs. Argentina 🇦🇷 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 7 | 2:34 | 2:45 | set | 1.5s | set → serve → receive → set → spike → receive |
| Bulgaria 🇧🇬 vs. Canada 🇨🇦 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 17 | 10:30 | 10:34 | score | 1.5s | score → serve → serve → score |
| Bulgaria 🇧🇬 vs. Canada 🇨🇦 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 40 | 23:08 | 23:16 | receive | 1.5s | receive → serve → receive → set → spike → block |
| Bulgaria 🇧🇬 vs. Canada 🇨🇦 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 46 | 27:32 | 27:38 | set | 1.5s | set → serve → receive → set → spike → score |

## 附錄 A：span 內有 2 個以上 `serve` — 147 筆

間隔不到 0.5 秒的，是同一個 `serve` 被標了兩次，不是兩個。

| 影片 | Rally | 起 | 訖 | serve 數 | 最小間隔 |
|---|---:|---:|---:|---:|---:|
| 0225小窩臨打 4 | 27 | 10:33 | 10:47 | 2 | 4.40s |
| 03⧸14(六) 14_00｜例行賽G103 #雲林美津濃 vs. #桃園臺灣產險｜企業21年甲級男女排球聯賽_set2 | 24 | 12:21 | 12:28 | 2 | 0.05s |
| 03⧸14(六) 14_00｜例行賽G103 #雲林美津濃 vs. #桃園臺灣產險｜企業21年甲級男女排球聯賽_set3 | 18 | 12:45 | 12:52 | 5 | 0.00s |
| 03⧸14(六) 14_00｜例行賽G103 #雲林美津濃 vs. #桃園臺灣產險｜企業21年甲級男女排球聯賽_set3 | 24 | 16:28 | 16:34 | 2 | 0.22s |
| 03⧸14(六) 16_00｜例行賽G104 #獅子王 vs. #屏東台電｜企業21年甲級男女排球聯賽_set1 | 15 | 6:18 | 6:27 | 2 | 0.88s |
| 03⧸14(六) 18_00｜例行賽G105 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 5 | 1:44 | 2:08 | 2 | 21.35s |
| 03⧸14(六) 18_00｜例行賽G105 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 11 | 4:46 | 4:53 | 4 | 0.00s |
| 03⧸14(六) 18_00｜例行賽G105 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 28 | 15:58 | 16:28 | 2 | 0.30s |
| 03⧸14(六) 18_00｜例行賽G105 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 30 | 17:23 | 17:29 | 2 | 0.38s |
| 03⧸14(六) 18_00｜例行賽G105 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 39 | 23:24 | 23:32 | 2 | 0.28s |
| 03⧸15(日) 13_00｜例行賽G106 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set1 | 41 | 24:13 | 24:19 | 5 | 0.00s |
| 03⧸15(日) 13_00｜例行賽G106 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set1 | 48 | 10:26 | 10:33 | 2 | 0.02s |
| 03⧸15(日) 13_00｜例行賽G106 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set2 | 13 | 5:36 | 5:44 | 5 | 0.00s |
| 03⧸15(日) 13_00｜例行賽G106 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set2 | 28 | 24:29 | 24:39 | 2 | 1.05s |
| 03⧸15(日) 13_00｜例行賽G106 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set2 | 45 | 29:16 | 29:41 | 2 | 0.63s |
| 03⧸15(日) 13_00｜例行賽G106 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set3 | 45 | 27:42 | 27:50 | 2 | 0.13s |
| 03⧸15(日) 17_00｜例行賽G108 #新北中纖 vs. #高雄台電｜企業21年甲級男女排球聯賽_set3 | 6 | 3:17 | 3:28 | 3 | 0.02s |
| 03⧸15(日) 17_00｜例行賽G108 #新北中纖 vs. #高雄台電｜企業21年甲級男女排球聯賽_set3 | 13 | 7:20 | 7:34 | 2 | 0.10s |
| 03⧸15(日) 17_00｜例行賽G108 #新北中纖 vs. #高雄台電｜企業21年甲級男女排球聯賽_set3 | 17 | 10:06 | 10:21 | 2 | 0.50s |
| 03⧸15(日) 17_00｜例行賽G108 #新北中纖 vs. #高雄台電｜企業21年甲級男女排球聯賽_set3 | 24 | 13:50 | 14:03 | 2 | 0.30s |
| 03⧸15(日) 17_00｜例行賽G108 #新北中纖 vs. #高雄台電｜企業21年甲級男女排球聯賽_set5 | 27 | 20:04 | 20:14 | 2 | 0.15s |
| 03⧸15(日) 17_00｜例行賽G108 #新北中纖 vs. #高雄台電｜企業21年甲級男女排球聯賽_set5 | 29 | 21:51 | 21:58 | 3 | 0.02s |
| 03⧸21(六) 18_00｜男子組冠軍賽G114 #雲林美津濃 vs. #屏東台電｜企業21年甲級男女排球聯賽_set1 | 22 | 13:38 | 13:49 | 2 | 0.07s |
| 03⧸22(日) 17_00｜男子組冠軍賽 G117 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set1 | 26 | 14:38 | 14:46 | 2 | 0.22s |
| 0914小窩季打 2 | 3 | 0:40 | 0:53 | 2 | 6.00s |
| 10⧸4 3 | 15 | 4:02 | 4:18 | 2 | 7.47s |
| 20241103 霖度C-1 | 22 | 9:36 | 9:40 | 2 | 0.20s |
| 20241103 霖度C-1 | 23 | 9:45 | 9:50 | 2 | 0.53s |
| 2025-09-27_G1_臺北伊斯特_vs_臺中連莊_set1 | 40 | 27:34 | 27:42 | 2 | 0.07s |
| 2025-09-28_G2_臺北伊斯特_vs_桃園雲豹飛將_set1 | 26 | 17:22 | 17:37 | 2 | 5.69s |
| 2025-09-28_G2_臺北伊斯特_vs_桃園雲豹飛將_set2 | 39 | 23:12 | 23:21 | 2 | 3.87s |
| 2025-10-12_G10_臺北伊斯特_vs_桃園雲豹飛將_set1 | 5 | 1:57 | 2:01 | 2 | 0.07s |
| 2025-10-25_G11_臺北伊斯特_vs_桃園雲豹飛將_set1 | 16 | 7:52 | 8:09 | 3 | 0.02s |
| 2025-10-25_G12_台鋼天鷹_vs_臺中連莊_set1 | 5 | 2:12 | 2:21 | 2 | 2.20s |
| 20250424 排島惡館-6 | 10 | 3:26 | 3:36 | 2 | 0.27s |
| 20250424 排島惡館-6 | 30 | 9:14 | 9:30 | 3 | 0.03s |
| 20250424 排島惡館-6 | 35 | 11:00 | 11:10 | 3 | 0.00s |
| 20250424 排島惡館-6 | 39 | 12:19 | 12:28 | 2 | 0.63s |
| 20250424 排島惡館-7 | 6 | 3:16 | 3:36 | 3 | 0.00s |
| 20250424 排島惡館-7 | 15 | 6:34 | 6:42 | 2 | 0.03s |
| 20250424 排島惡館-7 | 24 | 9:12 | 9:25 | 7 | 0.00s |
| 20250424 排島惡館-7 | 28 | 10:20 | 10:26 | 3 | 0.00s |
| 20250424 排島惡館-7 | 31 | 11:04 | 11:12 | 3 | 0.03s |
| 20250628-霖度C-1 | 3 | 1:29 | 1:34 | 2 | 0.03s |
| 20250918-排島本館-5 | 16 | 6:03 | 6:16 | 2 | 0.60s |
| 20250918-排島本館-5 | 36 | 13:34 | 13:56 | 2 | 0.17s |
| 20250918-排島本館-5 | 42 | 15:26 | 15:42 | 2 | 0.17s |
| 20251109-排島本館-03 | 24 | 8:16 | 8:24 | 4 | 0.00s |
| 20251109-排島本館-03 | 37 | 12:34 | 12:45 | 5 | 0.00s |
| 20251109-排島本館-03 | 40 | 13:30 | 13:40 | 2 | 0.40s |
| 20251227-排島本館-3 | 1 | 0:17 | 0:24 | 2 | 0.00s |
| 20251227-排島本館-3 | 25 | 7:10 | 7:34 | 2 | 0.33s |
| 20251227-排島本館-3 | 30 | 9:18 | 9:26 | 2 | 0.03s |
| 20251227-排島本館-3 | 34 | 10:14 | 10:22 | 2 | 0.10s |
| 20251227-排島本館-3 | 35 | 10:26 | 10:55 | 2 | 0.47s |
| 20251227-排島本館-3 | 44 | 14:09 | 14:23 | 2 | 0.67s |
| 20251227-排島本館-6 | 19 | 7:44 | 7:52 | 2 | 0.33s |
| 20251227-排島本館-6 | 20 | 7:59 | 8:12 | 2 | 0.83s |
| 20251227-排島本館-6 | 31 | 13:38 | 13:46 | 2 | 0.03s |
| 20251227-排島本館-6 | 32 | 13:54 | 14:03 | 2 | 0.20s |
| 20251227-排島本館-6 | 33 | 14:11 | 14:28 | 2 | 0.27s |
| 20251227-排島本館-6 | 40 | 16:16 | 16:29 | 2 | 0.17s |
| 20260108-排排棧-01 | 4 | 2:52 | 3:02 | 2 | 0.53s |
| 20260108-排排棧-01 | 9 | 4:27 | 5:10 | 2 | 1.20s |
| 20260108-排排棧-01 | 15 | 6:55 | 7:04 | 2 | 0.67s |
| 20260108-排排棧-01 | 34 | 12:48 | 13:08 | 2 | 0.47s |
| 20260321-排島本館-01 | 11 | 3:52 | 4:22 | 2 | 0.43s |
| 20260321-排島本館-01 | 12 | 4:28 | 4:39 | 2 | 0.23s |
| 20260321-排島本館-01 | 20 | 7:01 | 7:30 | 2 | 0.40s |
| 20260321-排島本館-01 | 25 | 9:19 | 9:48 | 2 | 0.20s |
| 20260426-小窩-03 | 23 | 9:02 | 9:20 | 2 | 0.13s |
| 20260426-小窩-03 | 27 | 10:49 | 11:00 | 3 | 0.03s |
| 20260426-小窩-03 | 32 | 12:44 | 12:58 | 2 | 0.07s |
| 20260502-排島本館-01 | 20 | 7:44 | 7:52 | 4 | 0.00s |
| 20260510邷力豹臨打1 | 11 | 6:22 | 6:39 | 2 | 0.10s |
| 20260510邷力豹臨打1 | 19 | 9:50 | 10:04 | 6 | 0.00s |
| 20260510邷力豹臨打1 | 25 | 12:50 | 13:06 | 2 | 0.47s |
| 20260510邷力豹臨打1 | 30 | 14:10 | 14:23 | 4 | 0.00s |
| 20260510邷力豹臨打1 | 31 | 14:40 | 14:45 | 2 | 0.03s |
| 20260510邷力豹臨打1 | 33 | 15:17 | 15:35 | 2 | 0.53s |
| 37-39 Thriller! - Japan 🇯🇵 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 2 | 0:20 | 0:38 | 2 | 1.48s |
| 37-39 Thriller! - Japan 🇯🇵 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 16 | 6:02 | 6:06 | 3 | 0.00s |
| 37-39 Thriller! - Japan 🇯🇵 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set2 | 10 | 3:36 | 3:50 | 2 | 0.24s |
| 37-39 Thriller! - Japan 🇯🇵 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set2 | 35 | 15:14 | 15:34 | 2 | 0.12s |
| Brazil 🇧🇷 vs. Cuba 🇨🇺 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 5 | 1:56 | 1:59 | 2 | 0.06s |
| Brazil 🇧🇷 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 15 | 7:32 | 7:56 | 2 | 0.36s |
| Bulgaria 🇧🇬 vs. Argentina 🇦🇷 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 6 | 2:16 | 2:20 | 2 | 0.64s |
| Bulgaria 🇧🇬 vs. Argentina 🇦🇷 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 10 | 4:02 | 4:06 | 4 | 0.00s |
| Bulgaria 🇧🇬 vs. Argentina 🇦🇷 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 22 | 10:46 | 10:54 | 2 | 5.32s |
| Bulgaria 🇧🇬 vs. Canada 🇨🇦 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 17 | 10:30 | 10:34 | 2 | 0.04s |
| Bulgaria 🇧🇬 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 22 | 11:27 | 11:34 | 2 | 0.24s |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 2 | 0:27 | 0:46 | 2 | 0.52s |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 5 | 1:58 | 2:01 | 4 | 0.00s |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 7 | 2:44 | 2:56 | 2 | 0.12s |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 8 | 3:14 | 3:22 | 3 | 0.04s |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 20 | 8:30 | 8:40 | 2 | 0.36s |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 26 | 13:41 | 13:45 | 4 | 0.00s |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 37 | 19:10 | 19:16 | 2 | 0.44s |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 43 | 24:04 | 24:07 | 2 | 0.04s |
| China 🇨🇳 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 11 | 5:26 | 5:42 | 2 | 0.12s |
| China 🇨🇳 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 34 | 15:22 | 15:28 | 2 | 0.48s |
| France vs. Iran - Semi Final 2 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 35 | 17:26 | 17:47 | 2 | 14.60s |
| France 🇫🇷 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 12 | 5:08 | 5:26 | 2 | 0.40s |
| Full Match ｜ Bulgaria vs Luxembourg ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set1 | 16 | 7:26 | 7:37 | 2 | 4.40s |
| Full Match ｜ Poland vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 1 | 0:01 | 0:16 | 2 | 0.70s |
| Full Match ｜ Poland vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 2 | 0:31 | 0:43 | 2 | 0.83s |
| Full Match ｜ Poland vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 4 | 1:24 | 1:32 | 2 | 0.10s |
| Full Match ｜ Poland vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 19 | 9:30 | 9:46 | 2 | 0.30s |
| Full Match ｜ Poland vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 34 | 17:52 | 18:04 | 2 | 0.27s |
| Full Match ｜ Poland vs. Slovakia ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 11 | 4:48 | 4:56 | 2 | 0.33s |
| Japan 🇯🇵 vs. China 🇨🇳 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 15 | 7:33 | 7:37 | 2 | 0.12s |
| Japan 🇯🇵 vs. China 🇨🇳 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 18 | 8:50 | 8:58 | 2 | 0.28s |
| Japan 🇯🇵 vs. Czechia 🇨🇿 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 26 | 11:20 | 11:26 | 2 | 0.04s |
| Japan 🇯🇵 vs. Dominican Republic 🇩🇴 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 10 | 4:14 | 4:27 | 2 | 0.28s |
| Japan 🇯🇵 vs. Dominican Republic 🇩🇴 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 12 | 5:04 | 5:12 | 2 | 0.04s |
| Japan 🇯🇵 vs. Dominican Republic 🇩🇴 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 15 | 6:32 | 6:40 | 2 | 0.20s |
| Japan 🇯🇵 vs. Dominican Republic 🇩🇴 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 20 | 9:19 | 9:28 | 2 | 0.68s |
| Japan 🇯🇵 vs. Dominican Republic 🇩🇴 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 21 | 9:43 | 9:52 | 2 | 0.56s |
| Japan 🇯🇵 vs. Dominican Republic 🇩🇴 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 25 | 11:32 | 11:42 | 2 | 0.44s |
| Japan 🇯🇵 vs. Dominican Republic 🇩🇴 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 27 | 12:34 | 12:38 | 3 | 0.00s |
| Japan 🇯🇵 vs. Dominican Republic 🇩🇴 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 29 | 13:26 | 13:34 | 2 | 0.44s |
| Japan 🇯🇵 vs. Dominican Republic 🇩🇴 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 31 | 14:34 | 14:42 | 2 | 0.36s |
| Japan 🇯🇵 vs. Dominican Republic 🇩🇴 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 37 | 18:40 | 18:53 | 2 | 0.24s |
| Japan 🇯🇵 vs. Dominican Republic 🇩🇴 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 43 | 23:16 | 23:24 | 2 | 0.08s |
| Japan 🇯🇵 vs. Netherlands 🇳🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 14 | 6:38 | 6:46 | 2 | 0.28s |
| Japan 🇯🇵 vs. Netherlands 🇳🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 30 | 14:41 | 14:48 | 2 | 0.56s |
| Japan 🇯🇵 vs. Serbia 🇷🇸 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 20 | 8:20 | 8:28 | 2 | 0.08s |
| Jtekt Stings 🇯🇵 - Suntory Sunbirds Osaka 🇯🇵 ｜ SV League 2026 ｜ Full Match - Volleyball_set1 | 19 | 10:51 | 11:00 | 2 | 3.34s |
| Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 34 | 17:23 | 17:30 | 2 | 0.03s |
| Poland vs. France - Final ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 47 | 22:14 | 22:23 | 2 | 0.35s |
| Suntory Sunbirds vs. Osaka Bluteon ｜ SV.LEAGUE 2025⧸26 ｜ Full Match - Volleyball_set1 | 4 | 1:20 | 1:41 | 3 | 0.00s |
| Suntory Sunbirds vs. Osaka Bluteon ｜ SV.LEAGUE 2025⧸26 ｜ Full Match - Volleyball_set1 | 25 | 11:42 | 11:50 | 2 | 0.28s |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G22 11⧸9 18_30 臺北伊斯特 vs 台鋼天鷹_set1 | 30 | 16:05 | 16:14 | 2 | 0.07s |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G22 11⧸9 18_30 臺北伊斯特 vs 台鋼天鷹_set1 | 40 | 23:02 | 23:12 | 2 | 0.18s |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G26 11⧸16 18_30 臺中連莊 vs 臺北伊斯特_set1 | 13 | 7:28 | 7:36 | 2 | 0.28s |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G26 11⧸16 18_30 臺中連莊 vs 臺北伊斯特_set1 | 32 | 17:53 | 18:00 | 2 | 0.40s |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G26 11⧸16 18_30 臺中連莊 vs 臺北伊斯特_set1 | 43 | 25:37 | 25:44 | 2 | 0.02s |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G26 11⧸16 18_30 臺中連莊 vs 臺北伊斯特_set1 | 63 | 39:40 | 39:46 | 2 | 0.08s |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G26 11⧸16 18_30 臺中連莊 vs 臺北伊斯特_set1 | 69 | 45:06 | 45:13 | 2 | 0.02s |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G26 11⧸16 18_30 臺中連莊 vs 臺北伊斯特_set1 | 71 | 46:09 | 46:18 | 2 | 0.40s |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G27 11⧸22 15_00 桃園雲豹飛將 vs 台中連莊_set1 | 20 | 9:52 | 10:00 | 5 | 0.00s |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G30 11⧸23 18_30 桃園雲豹飛將 vs 台鋼天鷹_set1 | 9 | 5:28 | 5:41 | 2 | 0.27s |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G30 11⧸23 18_30 桃園雲豹飛將 vs 台鋼天鷹_set1 | 12 | 8:37 | 8:50 | 4 | 0.00s |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G30 11⧸23 18_30 桃園雲豹飛將 vs 台鋼天鷹_set1 | 13 | 9:11 | 9:18 | 2 | 0.15s |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G30 11⧸23 18_30 桃園雲豹飛將 vs 台鋼天鷹_set1 | 21 | 13:47 | 13:54 | 2 | 0.02s |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G30 11⧸23 18_30 桃園雲豹飛將 vs 台鋼天鷹_set1 | 42 | 28:26 | 28:33 | 2 | 0.07s |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G6 10⧸5 18_30 臺中連莊 vs 台鋼天鷹_set1 | 15 | 7:22 | 7:29 | 2 | 0.10s |
