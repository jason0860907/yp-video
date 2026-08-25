# Rally 結尾沒有 score 的清單

掃描日期：2026-08-25 · 資料：`videos/rally-spot/annotations` × `videos/action/annotations`（197 支影片、8,721 rallies）

重跑：`uv run python scripts/scan_rally_edges.py`

判定：rally span `[start, end]` 內最後的動作事件不是 `score`。共 355 筆。

分類：`結尾切太早` = span 外 3 秒內就有 `score`，標註在、是邊界偏了；`score 不在最後` = span 內有 `score`，但後面還有別的動作；`疑似漏標` = 前後都找不到鄰近的 `score`。

## 疑似漏標 — 205 筆

| 影片 | Rally | 起 | 訖 | 最後動作 | 動作序列（後 6） |
|---|---:|---:|---:|---|---|
| 0316小窩季打 3 | 25 | 8:08 | 8:34 | receive | receive → set → spike → receive → receive → receive |
| 0323小窩臨打 3 | 4 | 1:38 | 1:57 | spike | spike → receive → set → receive → receive → spike |
| 0323小窩臨打 3 | 5 | 2:00 | 2:07 | receive | serve → receive |
| 0323小窩臨打 3 | 8 | 2:49 | 2:54 | serve | serve |
| 0323小窩臨打 3 | 10 | 3:18 | 3:35 | spike | receive → set → spike → receive → set → spike |
| 0323小窩臨打 3 | 13 | 4:26 | 4:36 | set | serve → receive → set |
| 0323小窩臨打 3 | 18 | 6:14 | 6:20 | serve | serve |
| 0323小窩臨打 3 | 20 | 6:46 | 6:51 | serve | serve |
| 0323小窩臨打 3 | 22 | 7:18 | 7:23 | serve | serve |
| 0323小窩臨打 3 | 23 | 7:27 | 7:37 | receive | serve → receive → set → receive |
| 0323小窩臨打 3 | 25 | 8:07 | 8:18 | receive | serve → receive → set → spike → receive |
| 0323小窩臨打 3 | 28 | 8:58 | 9:18 | receive | set → spike → receive → set → spike → receive |
| 0323小窩臨打 3 | 29 | 9:23 | 9:32 | receive | serve → receive → set → spike → receive |
| 0323小窩臨打 3 | 40 | 12:55 | 13:00 | serve | serve |
| 0323小窩臨打 3 | 41 | 13:07 | 13:28 | set | spike → receive → set → spike → receive → set |
| 03⧸14(六) 14_00｜例行賽G103 #雲林美津濃 vs. #桃園臺灣產險｜企業21年甲級男女排球聯賽_set1 | 13 | 5:18 | 5:38 | receive | spike → block → receive → set → spike → receive |
| 03⧸14(六) 14_00｜例行賽G103 #雲林美津濃 vs. #桃園臺灣產險｜企業21年甲級男女排球聯賽_set2 | 13 | 5:44 | 5:50 | receive | serve → receive → set → spike → receive |
| 03⧸14(六) 14_00｜例行賽G103 #雲林美津濃 vs. #桃園臺灣產險｜企業21年甲級男女排球聯賽_set2 | 29 | 15:25 | 15:32 | spike | serve → receive → set → spike |
| 03⧸14(六) 14_00｜例行賽G103 #雲林美津濃 vs. #桃園臺灣產險｜企業21年甲級男女排球聯賽_set2 | 31 | 17:05 | 17:15 | spike | receive → set → spike → receive → set → spike |
| 03⧸14(六) 14_00｜例行賽G103 #雲林美津濃 vs. #桃園臺灣產險｜企業21年甲級男女排球聯賽_set2 | 37 | 24:41 | 24:52 | receive | set → spike → receive → set → spike → receive |
| 03⧸14(六) 16_00｜例行賽G104 #獅子王 vs. #屏東台電｜企業21年甲級男女排球聯賽_set1 | 23 | 10:47 | 10:55 | receive | serve → receive → set → spike → receive |
| 03⧸14(六) 16_00｜例行賽G104 #獅子王 vs. #屏東台電｜企業21年甲級男女排球聯賽_set1 | 30 | 13:53 | 14:00 | block | serve → receive → set → spike → block |
| 03⧸14(六) 16_00｜例行賽G104 #獅子王 vs. #屏東台電｜企業21年甲級男女排球聯賽_set1 | 41 | 20:20 | 20:25 | receive | serve → receive |
| 03⧸14(六) 16_00｜例行賽G104 #獅子王 vs. #屏東台電｜企業21年甲級男女排球聯賽_set2 | 3 | 2:10 | 2:21 | block | set → spike → block → set → spike → block |
| 03⧸14(六) 18_00｜例行賽G105 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 1 | 0:00 | 0:06 | spike | receive → set → spike |
| 03⧸15(日) 15_00｜例行賽G107 #桃園臺灣產險 vs. #獅子王｜企業21年甲級男女排球聯賽_set1 | 45 | 25:11 | 25:19 | receive | serve → receive → set → spike → block → receive |
| 03⧸15(日) 17_00｜例行賽G108 #新北中纖 vs. #高雄台電｜企業21年甲級男女排球聯賽_set2 | 35 | 24:29 | 24:48 | receive | block → receive → set → spike → block → receive |
| 03⧸20(五) 14_00｜挑戰賽G110 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 10 | 4:34 | 4:44 | receive | serve → receive → set → spike → receive → receive |
| 03⧸20(五) 14_00｜挑戰賽G110 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 27 | 14:27 | 14:44 | spike | receive → set → spike → receive → set → spike |
| 03⧸20(五) 14_00｜挑戰賽G110 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 29 | 15:31 | 15:40 | set | serve → receive → set → spike → receive → set |
| 03⧸20(五) 14_00｜挑戰賽G110 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 42 | 28:31 | 28:40 | spike | set → spike → block → receive → set → spike |
| 03⧸20(五) 14_00｜挑戰賽G110 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 43 | 28:57 | 29:05 | receive | serve → receive → set → spike → receive |
| 03⧸20(五) 16_00｜挑戰賽G111 #屏東台電 vs. #桃園臺產｜企業21年甲級男女排球聯賽_set1 | 24 | 10:38 | 10:49 | set | serve → receive → set → spike → receive → set |
| 03⧸20(五) 18_00｜挑戰賽G112 #臺北國北獅 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set1 | 20 | 10:04 | 10:16 | block | spike → block → receive → set → spike → block |
| 0601小窩季打 2 | 5 | 1:38 | 2:09 | receive | set → spike → receive → set → spike → receive |
| 0601小窩季打 2 | 7 | 2:28 | 2:41 | receive | receive → set → spike → receive → set → receive |
| 0601小窩季打 2 | 14 | 4:40 | 4:54 | receive | set → spike → receive → set → spike → receive |
| 0601小窩季打 2 | 17 | 5:25 | 5:45 | receive | spike → receive → set → spike → receive → receive |
| 0601小窩季打 2 | 27 | 9:21 | 9:48 | receive | receive → set → receive → receive → receive → receive |
| 0601小窩季打 2 | 37 | 12:47 | 13:00 | receive | set → spike → receive → set → spike → receive |
| 10⧸4 2 | 6 | 3:20 | 3:38 | spike | receive → set → spike → receive → set → spike |
| 10⧸4 2 | 13 | 5:27 | 5:49 | spike | receive → set → spike → receive → set → spike |
| 10⧸4 2 | 14 | 6:03 | 6:08 | serve | serve |
| 10⧸4 2 | 17 | 6:57 | 7:15 | receive | set → spike → receive → set → spike → receive |
| 10⧸4 2 | 18 | 7:22 | 7:28 | serve | serve |
| 10⧸4 2 | 20 | 7:47 | 7:52 | serve | serve |
| 10⧸4 2 | 25 | 9:39 | 9:48 | spike | serve → receive → set → spike |
| 10⧸4 3 | 2 | 0:27 | 0:44 | receive | spike → spike → receive → set → spike → receive |
| 10⧸4 3 | 8 | 2:10 | 2:27 | receive | receive → set → spike → receive → set → receive |
| 10⧸4 3 | 16 | 4:24 | 4:34 | receive | serve → receive → set → spike → receive |
| 10⧸4 3 | 18 | 5:06 | 5:20 | spike | receive → set → receive → receive → set → spike |
| 10⧸4 3 | 23 | 7:00 | 7:12 | receive | serve → receive → receive → set → spike → receive |
| 10⧸4 3 | 31 | 9:01 | 9:23 | receive | receive → set → spike → receive → receive → receive |
| 10⧸4 3 | 42 | 12:40 | 12:59 | receive | set → receive → receive → set → spike → receive |
| 10⧸4 3 | 44 | 13:43 | 13:50 | receive | serve → receive |
| 2025-09-27_G1_臺北伊斯特_vs_臺中連莊_set1 | 17 | 10:42 | 10:55 | receive | spike → block → receive → set → spike → receive |
| 2025-09-27_G1_臺北伊斯特_vs_臺中連莊_set1 | 20 | 12:29 | 12:35 | receive | serve → receive → set → spike → receive |
| 2025-09-27_G1_臺北伊斯特_vs_臺中連莊_set1 | 38 | 23:44 | 23:58 | spike | set → spike → block → receive → set → spike |
| 2025-09-28_G2_臺北伊斯特_vs_桃園雲豹飛將_set1 | 38 | 26:41 | 26:48 | receive | serve → receive → set → spike → receive |
| 2025-09-28_G2_臺北伊斯特_vs_桃園雲豹飛將_set2 | 30 | 17:48 | 17:57 | receive | set → spike → block → receive → receive → receive |
| 2025-10-04_G3_臺中連莊_vs_桃園雲豹飛將_set1 | 13 | 5:53 | 5:59 | receive | serve → receive → set → receive |
| 2025-10-04_G3_臺中連莊_vs_桃園雲豹飛將_set1 | 36 | 15:51 | 15:57 | receive | serve → receive → set → spike → receive |
| 2025-10-04_G3_臺中連莊_vs_桃園雲豹飛將_set1 | 39 | 17:19 | 17:34 | block | set → spike → receive → set → spike → block |
| 2025-10-05_G5_臺北伊斯特_vs_桃園雲豹飛將_set1 | 16 | 7:45 | 7:59 | receive | receive → set → spike → block → receive → receive |
| 2025-10-05_G5_臺北伊斯特_vs_桃園雲豹飛將_set1 | 46 | 35:12 | 35:31 | block | spike → receive → receive → set → spike → block |
| 2025-10-11_G7_臺北伊斯特_vs_台鋼天鷹_set1 | 22 | 15:04 | 15:17 | receive | set → spike → receive → set → spike → receive |
| 2025-10-26_G14_臺北伊斯特_vs_臺中連莊_set1 | 3 | 1:02 | 1:09 | receive | serve → receive → set → spike → receive |
| 2025-10-26_G14_臺北伊斯特_vs_臺中連莊_set1 | 27 | 14:10 | 14:42 | receive | block → receive → receive → set → spike → receive |
| 2025-10-26_G14_臺北伊斯特_vs_臺中連莊_set1 | 37 | 22:35 | 22:44 | set | receive → set → spike → block → receive → set |
| 2025-11-02_G17_桃園雲豹飛將_vs_臺北伊斯特_set1 | 4 | 1:31 | 1:41 | receive | spike → block → receive → set → spike → receive |
| 2025-11-02_G18_臺中連莊_vs_台鋼天鷹_set1 | 12 | 5:32 | 5:40 | receive | serve → receive → set → spike → receive |
| 2025-11-02_G18_臺中連莊_vs_台鋼天鷹_set1 | 29 | 13:33 | 13:40 | receive | set → spike → block → receive → receive → receive |
| 2025-11-02_G18_臺中連莊_vs_台鋼天鷹_set1 | 46 | 28:04 | 28:18 | receive | spike → block → receive → set → spike → receive |
| 2025-11-08_G19_臺北伊斯特_vs_臺中連莊_set1 | 6 | 2:23 | 2:30 | block | serve → receive → set → spike → block |
| 2025-11-08_G19_臺北伊斯特_vs_臺中連莊_set1 | 16 | 9:21 | 9:29 | spike | serve → receive → set → spike |
| 20250424 排島惡館-8 | 2 | 2:03 | 2:12 | receive | serve → receive → set → spike → receive |
| 20250424 排島惡館-8 | 3 | 2:22 | 2:29 | receive | serve → receive → receive |
| 20250424 排島惡館-8 | 5 | 2:59 | 3:12 | receive | set → spike → receive → set → spike → receive |
| 20250424 排島惡館-8 | 20 | 7:57 | 8:10 | receive | set → spike → receive → set → spike → receive |
| 20250504 大統OB-成功大學vs台北大學B-第二局 | 1 | 0:18 | 0:36 | receive | set → spike → receive → set → spike → receive |
| 20250504 大統OB-成功大學vs台北大學B-第二局 | 8 | 3:07 | 3:12 | serve | serve |
| 20250504 大統OB-成功大學vs台北大學B-第二局 | 17 | 7:38 | 7:50 | block | set → receive → receive → set → spike → block |
| 20250504 大統OB-成功大學vs台北大學B-第二局 | 22 | 9:44 | 9:52 | spike | serve → receive → set → spike |
| 20250504 大統OB-成功大學vs台北大學B-第二局 | 23 | 10:03 | 10:22 | receive | set → spike → receive → set → spike → receive |
| 20250504 大統OB-成功大學vs台北大學B-第二局 | 27 | 11:43 | 11:48 | serve | serve |
| 20250504 大統OB-成功大學vs台北大學B-第二局 | 31 | 13:02 | 13:14 | receive | receive → set → spike → block → receive → receive |
| 20250504 大統OB-成功大學vs台北大學B-第二局 | 32 | 13:30 | 13:36 | receive | serve → receive → receive |
| 20250504 大統OB-成功大學vs台北大學B-第二局 | 36 | 14:52 | 15:08 | spike | receive → set → spike → receive → set → spike |
| 20250918-排島本館-5 | 25 | 9:18 | 9:28 | spike | serve → receive → set → spike |
| 20251227-排島本館-3 | 42 | 13:15 | 13:48 | receive | set → spike → receive → receive → receive → receive |
| 20251227-排島本館-6 | 21 | 8:19 | 8:28 | spike | serve → receive → set → spike |
| 20260403-霖度C-01_set1 | 24 | 9:26 | 9:42 | spike | receive → set → spike → receive → set → spike |
| 20260403-霖度C-02 | 6 | 1:39 | 1:44 | receive | serve → receive |
| 20260403-霖度C-02 | 25 | 7:47 | 7:56 | spike | serve → receive → set → spike |
| 20260403-霖度C-02 | 28 | 8:35 | 8:48 | receive | set → spike → receive → set → spike → receive |
| 20260426-小窩-01 | 1 | 0:03 | 0:24 | receive | set → spike → receive → set → spike → receive |
| 20260426-小窩-01 | 2 | 0:37 | 0:50 | spike | set → spike → receive → receive → receive → spike |
| 20260426-小窩-01 | 3 | 1:01 | 1:36 | spike | receive → set → spike → receive → set → spike |
| 20260426-小窩-01 | 21 | 8:23 | 8:32 | receive | serve → receive → set → spike → receive |
| 20260426-小窩-01 | 31 | 12:27 | 13:08 | receive | spike → receive → receive → receive → receive → receive |
| 20260426-小窩-01 | 35 | 14:38 | 15:02 | spike | receive → set → receive → receive → set → spike |
| 20260502-排島本館-02 | 7 | 4:05 | 4:12 | receive | serve → receive → set → spike → block → receive |
| 20260502-排島本館-02 | 9 | 4:45 | 4:49 | serve | serve |
| 20260502-排島本館-02 | 23 | 9:28 | 9:36 | receive | serve → receive → set → spike → block → receive |
| 20260502-排島本館-02 | 24 | 9:47 | 9:52 | receive | serve → receive |
| 20260502-排島本館-02 | 25 | 9:59 | 10:12 | receive | receive → set → spike → receive → set → receive |
| 20260502-排島本館-02 | 38 | 15:41 | 15:49 | receive | serve → receive → receive → receive |
| 20260502-排島本館-02 | 40 | 16:07 | 16:18 | set | serve → receive → set → receive → receive → set |
| 20260502-排島本館-02 | 46 | 18:34 | 18:42 | block | serve → receive → set → spike → block |
| 20260507 工資管友誼賽2 | 10 | 3:19 | 3:28 | receive | serve → receive → set → spike → receive |
| 2026⧸03⧸25 3 | 27 | 9:46 | 10:05 | spike | set → receive → spike → receive → set → spike |
| 37-39 Thriller! - Japan 🇯🇵 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 44 | 17:51 | 17:58 | spike | serve → receive → set → spike |
| Bulgaria 🇧🇬 vs. Argentina 🇦🇷 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 25 | 11:53 | 12:02 | spike | receive → set → spike → receive → set → spike |
| Bulgaria 🇧🇬 vs. Argentina 🇦🇷 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 35 | 18:08 | 18:27 | block | set → spike → block → receive → spike → block |
| Champions crowned in Final 24⧸25 (2⧸2) ｜ Suntory Sunbirds Osaka - Stings Aichi ｜ SV League 24⧸25_set1 | 18 | 8:18 | 8:28 | spike | receive → set → spike → receive → set → spike |
| China vs. Brazil - Ranking 13-14 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 11 | 4:37 | 4:46 | spike | serve → receive → set → spike → block → spike |
| Final - Stings vs. Sunbirds ｜ SVL League 2024⧸25 - Full Match ｜ Volleyball_set1 | 29 | 18:43 | 18:52 | receive | serve → receive → set → spike → receive |
| France vs. Iran - Semi Final 2 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 3 | 0:58 | 1:14 | set | receive → set → spike → block → receive → set |
| France vs. Iran - Semi Final 2 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 5 | 2:16 | 2:26 | receive | block → receive → set → spike → block → receive |
| France vs. Iran - Semi Final 2 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 18 | 8:40 | 8:53 | set | block → receive → set → spike → receive → set |
| Full Match ｜ Slovakia vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 6 | 2:42 | 2:52 | receive | receive → receive → set → spike → block → receive |
| Full Match ｜ Slovakia vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 14 | 6:29 | 6:38 | receive | serve → receive → set → spike → block → receive |
| Full Match ｜ Slovakia vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 20 | 10:12 | 10:16 | serve | serve |
| Full Match ｜ Slovakia vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 25 | 12:08 | 12:14 | receive | serve → receive |
| Full Match ｜ Slovakia vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 26 | 12:29 | 12:38 | spike | serve → receive → set → spike |
| Full Match ｜ Slovakia vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 27 | 13:01 | 13:06 | serve | serve |
| Full Match ｜ Slovakia vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 31 | 15:44 | 15:56 | receive | block → receive → set → spike → receive → receive |
| Full Match ｜ Slovakia vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 32 | 16:19 | 16:32 | block | spike → receive → receive → set → spike → block |
| Full Match ｜ Slovakia vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 40 | 21:58 | 22:06 | spike | receive → set → spike |
| Full Match ｜ Slovakia vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 41 | 23:13 | 23:20 | spike | serve → receive → set → spike |
| Full Match ｜ Spain vs The Netherlands ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool C_set1 | 4 | 1:32 | 1:48 | spike | receive → set → spike → receive → set → spike |
| Full Match ｜ Spain vs The Netherlands ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool C_set1 | 8 | 3:49 | 4:00 | spike | set → spike → block → receive → set → spike |
| Full Match ｜ Spain vs The Netherlands ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool C_set1 | 13 | 7:46 | 8:00 | receive | receive → set → spike → block → receive → receive |
| Full Match ｜ Spain vs The Netherlands ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool C_set1 | 25 | 14:09 | 14:14 | serve | serve |
| Full Match ｜ Spain vs The Netherlands ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool C_set1 | 31 | 17:36 | 17:44 | receive | serve → receive → set → spike → receive |
| Full Match ｜ Spain vs The Netherlands ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool C_set1 | 32 | 18:00 | 18:08 | spike | serve → receive → set → spike |
| Full Match ｜ Spain vs The Netherlands ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool C_set1 | 35 | 19:12 | 19:18 | spike | serve → receive → set → spike |
| Full Match ｜ Spain vs The Netherlands ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool C_set1 | 37 | 20:30 | 20:42 | spike | set → spike → block → receive → set → spike |
| Full Match ｜ Spain vs The Netherlands ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool C_set1 | 39 | 21:37 | 21:48 | receive | spike → receive → set → spike → block → receive |
| Japan 🇯🇵 vs. Czechia 🇨🇿 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 2 | 0:23 | 0:28 | receive | serve → receive |
| Japan 🇯🇵 vs. Czechia 🇨🇿 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 20 | 9:06 | 9:11 | spike | receive → set → spike |
| Japan 🇯🇵 vs. France 🇫🇷 ｜ VNL 2025 - Full Match ｜ Week 3_set1 | 5 | 1:39 | 1:54 | receive | spike → block → receive → set → spike → receive |
| Japan 🇯🇵 vs. France 🇫🇷 ｜ VNL 2025 - Full Match ｜ Week 3_set1 | 18 | 7:57 | 8:04 | receive | serve → receive → set → spike → receive |
| Japan 🇯🇵 vs. France 🇫🇷 ｜ VNL 2025 - Full Match ｜ Week 3_set1 | 23 | 9:48 | 9:54 | receive | serve → receive → set → spike → receive |
| Japan 🇯🇵 vs. France 🇫🇷 ｜ VNL 2025 - Full Match ｜ Week 3_set1 | 26 | 10:58 | 11:02 | receive | serve → receive |
| Japan 🇯🇵 vs. Serbia 🇷🇸 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 7 | 2:54 | 2:58 | receive | serve → receive |
| Japan 🇯🇵 vs. Serbia 🇷🇸 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 36 | 15:48 | 16:00 | receive | receive → receive → receive → set → spike → receive |
| Korea vs. Finland - Ranking 11-12 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 4 | 1:09 | 1:17 | block | serve → receive → set → spike → block |
| Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 9 | 3:21 | 3:30 | spike | serve → receive → set → spike |
| Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 11 | 4:51 | 5:00 | receive | set → spike → block → receive → receive → receive |
| Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 21 | 10:49 | 10:57 | receive | serve → receive → set → spike → block → receive |
| Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 23 | 12:40 | 12:50 | receive | serve → receive → set → spike → receive |
| Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 26 | 13:42 | 13:50 | receive | serve → receive → set → spike → block → receive |
| Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 40 | 20:56 | 21:06 | receive | serve → receive → set → spike → receive |
| Osaka Bluteon vs. Toray Arrows Shizuoka - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 31 | 17:20 | 17:32 | spike | set → spike → block → receive → set → spike |
| Semi Final 3 - Suntory Sunbirds vs. Wolfdogs Nagoya ｜ SVL Playoff - Full Match ｜ Volleyball_set1 | 46 | 27:17 | 27:26 | block | serve → receive → receive → spike → block |
| Spain vs. Iran - Ranking 3-4 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 43 | 26:33 | 27:03 | block | set → spike → receive → set → spike → block |
| Uzbekistan vs. Japan - Ranking 19-20 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 5 | 1:52 | 1:58 | receive | serve → receive → set → spike → receive |
| Uzbekistan vs. Japan - Ranking 19-20 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 10 | 4:08 | 4:12 | receive | serve → receive |
| Uzbekistan vs. Japan - Ranking 19-20 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 11 | 4:32 | 4:40 | spike | serve → set → spike |
| Uzbekistan vs. Japan - Ranking 19-20 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 16 | 7:44 | 7:49 | receive | serve → receive |
| ᴴᴰ114UVL預賽：：中原大學vs實踐大學：：男一級 大專排球聯賽 AI網路直播_set1 | 10 | 3:52 | 3:57 | receive | serve → receive |
| ᴴᴰ114UVL預賽：：中原大學vs實踐大學：：男一級 大專排球聯賽 AI網路直播_set1 | 15 | 6:18 | 6:40 | block | receive → receive → receive → set → spike → block |
| ᴴᴰ114UVL預賽：：中山大學vs國北教大：：男一級 大專排球聯賽 AI網路直播_set1 | 3 | 1:12 | 1:28 | receive | spike → block → receive → set → spike → receive |
| ᴴᴰ114UVL預賽：：中山大學vs國北教大：：男一級 大專排球聯賽 AI網路直播_set1 | 5 | 2:18 | 2:29 | spike | receive → set → spike → receive → set → spike |
| ᴴᴰ114UVL預賽：：中山大學vs國北教大：：男一級 大專排球聯賽 AI網路直播_set1 | 15 | 6:35 | 6:42 | set | serve → receive → spike → receive → set |
| ᴴᴰ114UVL預賽：：中山大學vs國北教大：：男一級 大專排球聯賽 AI網路直播_set1 | 19 | 8:24 | 8:32 | receive | serve → receive → set → spike → block → receive |
| ᴴᴰ114UVL預賽：：中山大學vs國北教大：：男一級 大專排球聯賽 AI網路直播_set1 | 32 | 14:50 | 14:54 | serve | serve |
| ᴴᴰ114UVL預賽：：中山大學vs國北教大：：男一級 大專排球聯賽 AI網路直播_set1 | 34 | 15:24 | 15:28 | serve | serve |
| ᴴᴰ114UVL預賽：：中山大學vs國北教大：：男一級 大專排球聯賽 AI網路直播_set1 | 41 | 18:24 | 18:32 | receive | serve → receive → set → spike → receive |
| ᴴᴰ114UVL預賽：：中山大學vs清華大學：：男一級 大專排球聯賽 AI網路直播_set1 | 9 | 3:07 | 3:16 | receive | serve → receive → set → spike → receive |
| ᴴᴰ114UVL預賽：：中山大學vs清華大學：：男一級 大專排球聯賽 AI網路直播_set1 | 19 | 7:23 | 7:30 | spike | serve → receive → set → spike |
| ᴴᴰ114UVL預賽：：中山大學vs清華大學：：男一級 大專排球聯賽 AI網路直播_set1 | 39 | 17:21 | 17:33 | receive | block → receive → set → spike → block → receive |
| ᴴᴰ114UVL預賽：：中山大學vs清華大學：：男一級 大專排球聯賽 AI網路直播_set1 | 41 | 18:34 | 18:43 | receive | receive → set → spike → receive → receive → receive |
| ᴴᴰ114UVL預賽：：清華大學vs嘉義大學：：男一級 大專排球聯賽 AI網路直播_set1 | 11 | 4:10 | 4:18 | receive | receive → set → spike → block → receive → receive |
| ᴴᴰ114UVL預賽：：清華大學vs嘉義大學：：男一級 大專排球聯賽 AI網路直播_set1 | 20 | 8:49 | 8:56 | block | serve → receive → set → spike → block |
| ᴴᴰ114UVL預賽：：清華大學vs嘉義大學：：男一級 大專排球聯賽 AI網路直播_set1 | 23 | 10:00 | 10:30 | spike | receive → set → spike → receive → set → spike |
| ᴴᴰ114UVL預賽：：清華大學vs嘉義大學：：男一級 大專排球聯賽 AI網路直播_set1 | 24 | 10:48 | 10:54 | receive | serve → receive → set → spike → receive |
| ᴴᴰ114UVL預賽：：清華大學vs嘉義大學：：男一級 大專排球聯賽 AI網路直播_set1 | 32 | 14:57 | 15:08 | block | spike → block → receive → set → spike → block |
| ᴴᴰ114UVL預賽：：清華大學vs嘉義大學：：男一級 大專排球聯賽 AI網路直播_set1 | 36 | 16:43 | 16:50 | receive | serve → receive → set → spike → block → receive |
| ᴴᴰ114UVL預賽：：臺灣師大vs中山大學：：男一級 大專排球聯賽 AI網路直播_set1 | 3 | 0:49 | 1:01 | receive | receive → set → receive → set → spike → receive |
| ᴴᴰ114UVL預賽：：臺灣師大vs中山大學：：男一級 大專排球聯賽 AI網路直播_set1 | 31 | 13:09 | 13:22 | set | receive → set → spike → block → receive → set |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G22 11⧸9 18_30 臺北伊斯特 vs 台鋼天鷹_set1 | 5 | 2:39 | 2:46 | receive | serve → receive → set → spike → receive |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G22 11⧸9 18_30 臺北伊斯特 vs 台鋼天鷹_set1 | 12 | 6:13 | 6:20 | receive | serve → receive → set → spike → receive |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G22 11⧸9 18_30 臺北伊斯特 vs 台鋼天鷹_set1 | 27 | 14:22 | 14:29 | block | serve → receive → set → spike → block |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G22 11⧸9 18_30 臺北伊斯特 vs 台鋼天鷹_set1 | 42 | 24:05 | 24:16 | block | receive → spike → receive → set → spike → block |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G22 11⧸9 18_30 臺北伊斯特 vs 台鋼天鷹_set1 | 43 | 25:22 | 25:32 | receive | set → spike → receive → set → spike → receive |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G22 11⧸9 18_30 臺北伊斯特 vs 台鋼天鷹_set1 | 44 | 26:11 | 26:18 | spike | serve → receive → set → spike |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G23 11⧸15 15_00 臺中連莊 vs 台鋼天鷹_set1 | 30 | 16:52 | 17:00 | receive | serve → receive → set → spike → block → receive |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G23 11⧸15 15_00 臺中連莊 vs 台鋼天鷹_set1 | 32 | 18:10 | 18:18 | receive | serve → receive → set → spike → receive |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G23 11⧸15 15_00 臺中連莊 vs 台鋼天鷹_set1 | 33 | 18:41 | 18:54 | receive | spike → receive → receive → set → spike → receive |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G26 11⧸16 18_30 臺中連莊 vs 臺北伊斯特_set1 | 64 | 40:14 | 40:30 | block | spike → block → receive → set → spike → block |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G28 11⧸22 18_30 臺北伊斯特 vs 台鋼天鷹_set1 | 13 | 5:37 | 5:42 | serve | serve |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G28 11⧸22 18_30 臺北伊斯特 vs 台鋼天鷹_set1 | 15 | 6:21 | 6:36 | spike | block → receive → set → receive → set → spike |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G28 11⧸22 18_30 臺北伊斯特 vs 台鋼天鷹_set1 | 17 | 7:20 | 7:30 | spike | set → spike → block → receive → set → spike |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G28 11⧸22 18_30 臺北伊斯特 vs 台鋼天鷹_set1 | 18 | 8:29 | 8:32 | serve | serve |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G28 11⧸22 18_30 臺北伊斯特 vs 台鋼天鷹_set1 | 19 | 8:51 | 8:58 | spike | serve → receive → set → spike |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G28 11⧸22 18_30 臺北伊斯特 vs 台鋼天鷹_set1 | 31 | 17:16 | 17:22 | block | serve → receive → set → spike → block |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G28 11⧸22 18_30 臺北伊斯特 vs 台鋼天鷹_set1 | 40 | 21:19 | 21:22 | serve | serve |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G29 11⧸23 15_00 臺北伊斯特 vs 台中連莊_set1 | 2 | 1:43 | 1:56 | receive | spike → block → receive → set → spike → receive |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G29 11⧸23 15_00 臺北伊斯特 vs 台中連莊_set1 | 7 | 4:38 | 4:44 | receive | serve → receive → set → spike → receive |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G29 11⧸23 15_00 臺北伊斯特 vs 台中連莊_set1 | 10 | 6:07 | 6:16 | spike | serve → receive → set → spike → receive → spike |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G29 11⧸23 15_00 臺北伊斯特 vs 台中連莊_set1 | 18 | 9:51 | 9:59 | receive | serve → receive → set → spike → block → receive |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G29 11⧸23 15_00 臺北伊斯特 vs 台中連莊_set1 | 22 | 11:37 | 11:47 | spike | receive → block → set → receive → set → spike |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G6 10⧸5 18_30 臺中連莊 vs 台鋼天鷹_set1 | 25 | 12:12 | 12:19 | receive | serve → receive → set → block → spike → receive |

## score 不在最後 — 115 筆

span 內有 `score`，但它不是最後的事件。

| 影片 | Rally | 起 | 訖 | 最後動作 | score 距結尾 | 動作序列（後 6） |
|---|---:|---:|---:|---|---:|---|
| 0104排島臨打 3 | 34 | 12:52 | 13:06 | set | 4.1s | receive → set → score → spike → receive → set |
| 0323小窩臨打 3 | 19 | 6:27 | 6:38 | set | 1.8s | spike → receive → set → spike → score → set |
| 03⧸14(六) 14_00｜例行賽G103 #雲林美津濃 vs. #桃園臺灣產險｜企業21年甲級男女排球聯賽_set3 | 29 | 18:54 | 19:10 | receive | 1.0s | block → receive → receive → receive → score → receive |
| 03⧸21(六) 18_00｜男子組冠軍賽G114 #雲林美津濃 vs. #屏東台電｜企業21年甲級男女排球聯賽_set1 | 11 | 6:03 | 6:12 | set | 3.6s | receive → set → spike → score → receive → set |
| 03⧸21(六) 18_00｜男子組冠軍賽G114 #雲林美津濃 vs. #屏東台電｜企業21年甲級男女排球聯賽_set1 | 34 | 19:35 | 19:42 | receive | 2.5s | serve → receive → score → spike → receive |
| 03⧸21(六) 18_00｜男子組冠軍賽G114 #雲林美津濃 vs. #屏東台電｜企業21年甲級男女排球聯賽_set1 | 39 | 22:18 | 22:26 | receive | 0.8s | set → spike → block → receive → score → receive |
| 03⧸21(六) 18_00｜男子組冠軍賽G114 #雲林美津濃 vs. #屏東台電｜企業21年甲級男女排球聯賽_set1 | 40 | 22:50 | 23:01 | receive | 1.9s | spike → receive → set → spike → score → receive |
| 03⧸22(日) 17_00｜男子組冠軍賽 G117 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set1 | 2 | 0:22 | 0:36 | block | 3.6s | block → receive → set → score → spike → block |
| 03⧸22(日) 17_00｜男子組冠軍賽 G117 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set2 | 30 | 19:22 | 19:28 | receive | 1.7s | receive → set → score → spike → block → receive |
| 03⧸22(日) 17_00｜男子組冠軍賽 G117 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set3 | 4 | 1:19 | 1:35 | receive | 2.2s | receive → set → spike → block → score → receive |
| 0419小窩臨打 2 | 37 | 14:08 | 14:28 | spike | 2.4s | set → spike → receive → set → score → spike |
| 0420小窩季打 1 | 43 | 14:50 | 14:54 | receive | 1.0s | serve → receive → score → receive |
| 0420小窩季打 2 | 11 | 3:38 | 3:58 | receive | 3.1s | spike → receive → set → score → spike → receive |
| 0420小窩季打 3 | 15 | 4:12 | 4:20 | spike | 2.8s | serve → receive → set → score → spike |
| 0727小窩季打 2 | 13 | 4:50 | 5:14 | receive | 0.0s | spike → receive → set → receive → score → receive |
| 0914小窩季打 2 | 22 | 7:40 | 7:47 | set | 1.0s | serve → receive → receive → receive → score → set |
| 20241103 霖度C-1 | 6 | 3:50 | 3:54 | receive | 1.0s | serve → score → receive |
| 2025-10-05_G5_臺北伊斯特_vs_桃園雲豹飛將_set1 | 6 | 2:31 | 2:37 | block | 1.0s | serve → receive → set → spike → score → block |
| 2025-10-05_G5_臺北伊斯特_vs_桃園雲豹飛將_set1 | 35 | 24:48 | 24:56 | receive | 2.7s | serve → receive → set → spike → score → receive |
| 2025-10-12_G9_臺中連莊_vs_台鋼天鷹_set1 | 47 | 28:58 | 29:12 | receive | 2.0s | receive → receive → set → spike → score → receive |
| 2025-10-25_G11_臺北伊斯特_vs_桃園雲豹飛將_set1 | 26 | 14:08 | 14:16 | receive | 1.5s | receive → set → spike → score → block → receive |
| 2025-11-01_G15_臺中連莊_vs_臺北伊斯特_set1 | 32 | 18:36 | 18:43 | receive | 0.9s | serve → receive → set → spike → score → receive |
| 2025-11-01_G15_臺中連莊_vs_臺北伊斯特_set1 | 36 | 20:28 | 20:38 | spike | 0.8s | set → spike → receive → set → score → spike |
| 2025-11-08_G19_臺北伊斯特_vs_臺中連莊_set1 | 47 | 29:31 | 29:37 | receive | 3.2s | serve → score → receive |
| 2025-11-08_G20_桃園雲豹飛將_vs_台鋼天鷹_set1 | 29 | 16:19 | 16:26 | set | 1.0s | set → spike → block → receive → score → set |
| 20260502-排島本館-01 | 44 | 16:03 | 16:10 | serve | 1.0s | receive → set → spike → block → score → serve |
| 20260507 工資管友誼賽2 | 33 | 13:55 | 14:04 | set | 1.0s | receive → set → spike → receive → score → set |
| 20260510邷力豹臨打1 | 12 | 6:53 | 7:15 | receive | 1.0s | receive → receive → set → spike → score → receive |
| Bulgaria 🇧🇬 vs. Canada 🇨🇦 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 46 | 27:32 | 27:38 | receive | 1.2s | serve → receive → set → spike → score → receive |
| Bulgaria 🇧🇬 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 14 | 6:04 | 6:12 | receive | 2.1s | receive → set → spike → receive → score → receive |
| Bulgaria 🇧🇬 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 37 | 18:52 | 18:58 | set | 3.4s | serve → score → receive → set |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 4 | 1:31 | 1:38 | receive | 1.2s | serve → receive → set → spike → score → receive |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 10 | 4:03 | 4:08 | receive | 2.4s | serve → score → receive |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 31 | 15:37 | 15:54 | set | 2.0s | set → spike → block → receive → score → set |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 50 | 34:58 | 35:02 | receive | 1.5s | serve → score → receive |
| China vs. Argentina - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 2 | 0:33 | 0:43 | set | 2.5s | receive → set → spike → receive → score → set |
| China vs. Brazil - Ranking 13-14 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 6 | 2:36 | 2:45 | receive | 2.7s | set → spike → block → score → receive → receive |
| China vs. Brazil - Ranking 13-14 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 19 | 7:53 | 8:04 | receive | 2.6s | receive → set → spike → block → score → receive |
| China 🇨🇳 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 3 | 1:56 | 2:06 | receive | 2.7s | serve → receive → receive → spike → score → receive |
| Cuba vs. Puerto Rico - Ranking 17-18 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 22 | 10:15 | 10:31 | receive | 2.9s | receive → receive → spike → score → receive → receive |
| France 🇫🇷 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 19 | 9:53 | 10:01 | receive | 2.4s | receive → set → spike → block → score → receive |
| France 🇫🇷 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 41 | 21:11 | 21:24 | receive | 3.4s | receive → set → score → spike → receive → receive |
| France 🇫🇷 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 46 | 24:44 | 24:50 | receive | 1.0s | serve → receive → set → spike → score → receive |
| Full Match ｜ Bulgaria vs Luxembourg ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set1 | 31 | 14:42 | 14:58 | receive | 2.7s | receive → set → spike → score → receive → receive |
| Full Match ｜ Croatia vs. Bulgaria ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set1 | 24 | 10:02 | 10:14 | spike | 3.3s | set → spike → receive → set → score → spike |
| Full Match ｜ Croatia vs. Bulgaria ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set1 | 28 | 11:48 | 11:54 | set | 1.0s | serve → receive → set → spike → score → set |
| Full Match ｜ Croatia vs. Serbia ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set1 | 21 | 8:55 | 9:16 | receive | 4.2s | set → spike → block → score → receive → receive |
| Full Match ｜ Denmark vs. England ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool A_set1 | 11 | 7:16 | 7:26 | receive | 2.3s | serve → receive → set → spike → score → receive |
| Full Match ｜ Denmark vs. England ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool A_set1 | 18 | 10:30 | 10:40 | receive | 4.0s | receive → set → spike → score → receive → receive |
| Full Match ｜ Ireland vs. Spain ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool D_set1 | 9 | 3:21 | 3:34 | receive | 1.8s | spike → receive → set → spike → score → receive |
| Full Match ｜ Ireland vs. Spain ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool D_set1 | 11 | 5:06 | 5:16 | set | 4.3s | set → spike → block → score → receive → set |
| Full Match ｜ Ireland vs. Türkiye ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool D_set1 | 13 | 5:29 | 5:46 | receive | 1.7s | spike → receive → set → spike → score → receive |
| Full Match ｜ Ireland vs. Türkiye ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool D_set1 | 14 | 6:58 | 7:05 | receive | 1.6s | serve → receive → set → score → receive |
| Full Match ｜ Ireland vs. Türkiye ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool D_set1 | 19 | 8:48 | 9:02 | set | 3.1s | block → receive → set → spike → score → set |
| Full Match ｜ Italy vs. England ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool A_set1 | 30 | 16:41 | 16:52 | receive | 0.7s | receive → set → spike → block → score → receive |
| Full Match ｜ Luxembourg vs. Croatia - CEV U22 Volleyball European Championship 2026 ｜ Women ｜ Pool E_set1 | 1 | 0:05 | 0:14 | receive | 1.9s | set → spike → receive → receive → score → receive |
| Full Match ｜ Norway vs. Bulgaria ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool C_set1 | 21 | 9:14 | 9:23 | receive | 2.6s | receive → set → spike → score → receive → receive |
| Full Match ｜ Norway vs. Ireland ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool D_set1 | 37 | 19:05 | 19:38 | set | 4.0s | receive → set → spike → receive → score → set |
| Full Match ｜ Poland vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 4 | 1:24 | 1:32 | receive | 2.1s | serve → receive → set → spike → score → receive |
| Full Match ｜ Poland vs. Slovakia ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 26 | 11:57 | 12:06 | receive | 4.0s | serve → receive → score → receive |
| Full Match ｜ Serbia vs. Bulgaria ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set1 | 20 | 8:10 | 8:20 | receive | 1.6s | receive → set → spike → receive → score → receive |
| Full Match ｜ Serbia vs. Luxembourg ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set1 | 34 | 17:25 | 17:34 | receive | 1.6s | receive → set → receive → receive → score → receive |
| Full Match ｜ Türkiye vs. Spain ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool D_set1 | 40 | 21:22 | 21:38 | receive | 4.6s | set → spike → score → set → spike → receive |
| Japan vs. USA - Ranking 15-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 26 | 10:14 | 10:21 | receive | 3.0s | serve → receive → score → receive |
| Japan vs. USA - Ranking 15-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 31 | 13:23 | 13:31 | receive | 2.2s | serve → receive → set → spike → score → receive |
| Japan 🇯🇵 vs. Czechia 🇨🇿 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 22 | 9:50 | 9:56 | receive | 2.2s | receive → set → spike → receive → score → receive |
| Japan 🇯🇵 vs. Dominican Republic 🇩🇴 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 24 | 11:02 | 11:15 | set | 3.6s | receive → set → spike → score → receive → set |
| Japan 🇯🇵 vs. Dominican Republic 🇩🇴 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 25 | 11:32 | 11:42 | receive | 2.7s | spike → receive → receive → receive → score → receive |
| Japan 🇯🇵 vs. Dominican Republic 🇩🇴 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 29 | 13:26 | 13:34 | receive | 2.6s | serve → receive → set → spike → score → receive |
| Japan 🇯🇵 vs. France 🇫🇷 ｜ VNL 2025 - Full Match ｜ Week 3_set1 | 36 | 17:57 | 18:28 | block | 19.9s | receive → receive → receive → set → spike → block |
| Japan 🇯🇵 vs. Netherlands 🇳🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 2 | 0:25 | 0:35 | receive | 3.9s | spike → receive → score → receive → spike → receive |
| Japan 🇯🇵 vs. Netherlands 🇳🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 23 | 10:53 | 11:10 | receive | 3.2s | set → spike → block → score → receive → receive |
| Japan 🇯🇵 vs. Netherlands 🇳🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 31 | 15:06 | 15:14 | set | 2.7s | receive → set → spike → score → receive → set |
| Japan 🇯🇵 vs. Netherlands 🇳🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 43 | 21:20 | 21:32 | receive | 5.2s | receive → set → spike → score → receive → receive |
| Japan 🇯🇵 vs. Serbia 🇷🇸 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 38 | 16:51 | 16:59 | spike | 1.4s | receive → set → spike → receive → score → spike |
| Jtekt Stings 🇯🇵 - Suntory Sunbirds Osaka 🇯🇵 ｜ SV League 2026 ｜ Full Match - Volleyball_set1 | 31 | 17:17 | 17:30 | receive | 1.4s | receive → set → spike → block → score → receive |
| Korea vs. Finland - Ranking 11-12 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 21 | 7:25 | 7:32 | spike | 1.5s | serve → receive → set → score → spike |
| Korea vs. Finland - Ranking 11-12 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 44 | 19:50 | 19:58 | receive | 3.1s | serve → receive → set → spike → score → receive |
| Osaka Bluteon 🇯🇵 vs. JTEKT Stings 🇯🇵 ｜ SV League 2026 ｜ Full Match - Volleyball_set1 | 23 | 11:23 | 11:31 | set | 2.5s | set → spike → block → score → receive → set |
| Osaka Bluteon 🇯🇵 vs. JTEKT Stings 🇯🇵 ｜ SV League 2026 ｜ Full Match - Volleyball_set1 | 42 | 22:28 | 22:36 | receive | 1.9s | serve → receive → set → spike → score → receive |
| Pakistan vs. USA - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 1 | 0:07 | 0:14 | receive | 2.4s | serve → receive → set → score → spike → receive |
| Pakistan vs. USA - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 6 | 2:26 | 2:34 | receive | 2.1s | serve → receive → set → spike → score → receive |
| Pakistan vs. USA - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 16 | 7:38 | 7:55 | receive | 2.1s | set → spike → block → receive → score → receive |
| Poland vs. Spain - Semi Final 1 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 35 | 19:57 | 20:06 | receive | 1.8s | serve → receive → set → spike → score → receive |
| Semi Final 1 - Osaka Bluteon vs. Stings Aichi ｜ SV League - Full Match ｜ Volleyball_set1 | 8 | 3:45 | 3:52 | receive | 1.6s | receive → set → spike → block → score → receive |
| Semi Final 1 - Osaka Bluteon vs. Stings Aichi ｜ SV League - Full Match ｜ Volleyball_set1 | 18 | 8:57 | 9:04 | block | 1.6s | serve → receive → set → spike → score → block |
| Semi Final 1 - Osaka Bluteon vs. Stings Aichi ｜ SV League - Full Match ｜ Volleyball_set1 | 42 | 25:36 | 25:56 | receive | 3.2s | receive → set → spike → block → score → receive |
| Semi Final 2 - Osaka Bluteon vs. Stings Aichi ｜ SV League - Full Match ｜ Volleyball_set1 | 30 | 16:30 | 16:36 | spike | 3.7s | serve → score → set → spike |
| Semi Final 2 - Osaka Bluteon vs. Stings Aichi ｜ SV League - Full Match ｜ Volleyball_set1 | 39 | 23:08 | 23:16 | receive | 2.8s | serve → receive → block → score → receive |
| Spain vs. Iran - Ranking 3-4 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 14 | 10:01 | 10:10 | receive | 3.2s | serve → receive → receive → receive → score → receive |
| Spain vs. Iran - Ranking 3-4 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 16 | 11:03 | 11:16 | spike | 2.8s | receive → block → receive → score → set → spike |
| Suntory Sunbirds vs. Osaka Bluteon ｜ SV.LEAGUE 2025⧸26 ｜ Full Match - Volleyball_set1 | 23 | 10:53 | 11:00 | receive | 2.0s | serve → receive → set → spike → score → receive |
| Suntory Sunbirds 🇯🇵 vs. Stings AICHI 🇯🇵 ｜ SV League 2026 ｜ Full Match - Japan Volleyball_set1 | 6 | 3:19 | 3:27 | receive | 2.4s | receive → set → spike → block → score → receive |
| Suntory Sunbirds 🇯🇵 vs. Stings AICHI 🇯🇵 ｜ SV League 2026 ｜ Full Match - Japan Volleyball_set1 | 30 | 14:25 | 14:37 | receive | 3.1s | receive → set → spike → block → score → receive |
| Taipei vs. Argentina - Playoffs ｜ Girls' U19 World Champs 2025 - Full Match_set1 | 7 | 3:28 | 3:47 | set | 1.9s | receive → set → spike → receive → score → set |
| Taipei vs. Argentina - Playoffs ｜ Girls' U19 World Champs 2025 - Full Match_set1 | 12 | 6:19 | 6:36 | receive | 3.0s | receive → set → spike → receive → score → receive |
| Taipei vs. Argentina - Playoffs ｜ Girls' U19 World Champs 2025 - Full Match_set1 | 22 | 12:57 | 13:06 | set | 2.8s | receive → set → spike → score → receive → set |
| Türkiye vs. Colombia - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 9 | 5:33 | 5:41 | receive | 2.1s | set → spike → block → receive → score → receive |
| Türkiye vs. Colombia - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 27 | 12:41 | 12:50 | receive | 2.5s | set → spike → block → score → receive → receive |
| Türkiye vs. Colombia - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 29 | 13:32 | 13:40 | set | 2.7s | serve → receive → spike → block → score → set |
| Türkiye vs. Colombia - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 31 | 15:39 | 15:51 | set | 4.4s | block → receive → score → receive → receive → set |
| Uzbekistan vs. Pakistan - Ranking 5-6 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 15 | 6:09 | 6:17 | receive | 2.6s | receive → set → spike → block → score → receive |
| ᴴᴰ114UVL預賽：：中原大學vs陽明交大：：男一級 大專排球聯賽 AI網路直播_set1 | 17 | 8:10 | 8:19 | receive | 2.8s | serve → receive → set → spike → score → receive |
| ᴴᴰ114UVL預賽：：中原大學vs陽明交大：：男一級 大專排球聯賽 AI網路直播_set1 | 23 | 11:37 | 11:45 | receive | 4.0s | serve → receive → score → receive |
| ᴴᴰ114UVL預賽：：中原大學vs陽明交大：：男一級 大專排球聯賽 AI網路直播_set1 | 24 | 11:55 | 12:04 | receive | 2.0s | serve → receive → set → spike → score → receive |
| ᴴᴰ114UVL預賽：：中原大學vs陽明交大：：男一級 大專排球聯賽 AI網路直播_set1 | 38 | 18:28 | 18:38 | receive | 2.5s | set → spike → block → receive → score → receive |
| ᴴᴰ114UVL預賽：：中原大學vs陽明交大：：男一級 大專排球聯賽 AI網路直播_set1 | 39 | 18:55 | 19:03 | set | 2.8s | receive → set → spike → score → receive → set |
| ᴴᴰ114UVL預賽：：國北教大vs嘉義大學：：男一級 大專排球聯賽 AI網路直播_set1 | 1 | 0:32 | 0:40 | set | 1.9s | serve → receive → set → spike → score → set |
| ᴴᴰ114UVL預賽：：臺灣師大vs中山大學：：男一級 大專排球聯賽 AI網路直播_set1 | 23 | 9:13 | 9:24 | spike | 0.4s | receive → set → spike → receive → score → spike |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G21 11⧸9 15_00 桃園雲豹飛將 vs 臺中連莊_set1 | 39 | 19:25 | 20:03 | receive | 2.6s | receive → receive → set → spike → score → receive |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G23 11⧸15 15_00 臺中連莊 vs 台鋼天鷹_set1 | 24 | 11:12 | 11:29 | receive | 2.6s | receive → set → spike → receive → score → receive |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G24 11⧸15 18_30 桃園雲豹飛將 vs 臺北伊斯特_set1 | 9 | 4:19 | 4:26 | receive | 0.9s | serve → receive → set → spike → score → receive |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G24 11⧸15 18_30 桃園雲豹飛將 vs 臺北伊斯特_set1 | 25 | 11:40 | 11:50 | block | 1.3s | block → receive → score → set → spike → block |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G24 11⧸15 18_30 桃園雲豹飛將 vs 臺北伊斯特_set1 | 45 | 24:13 | 24:31 | receive | 2.0s | block → receive → set → spike → score → receive |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G26 11⧸16 18_30 臺中連莊 vs 臺北伊斯特_set1 | 17 | 9:20 | 9:29 | receive | 2.7s | set → spike → block → score → receive → receive |

## 結尾切太早 — 35 筆

`score` 就在 span 外不到 3 秒 —— 標註本身在，要動的是 `end`。

| 影片 | Rally | 起 | 訖 | 最後動作 | score 距結尾 | 動作序列（後 6） |
|---|---:|---:|---:|---|---:|---|
| 0104排島臨打 1 | 20 | 8:38 | 8:45 | spike | 0.1s | serve → receive → set → spike |
| 0323小窩臨打 3 | 33 | 10:22 | 10:29 | receive | 0.7s | serve → receive → set → spike → receive |
| 0323小窩臨打 3 | 43 | 14:29 | 14:41 | receive | 0.5s | serve → receive → set → set → spike → receive |
| 03⧸14(六) 18_00｜例行賽G105 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 31 | 17:52 | 18:05 | receive | 0.0s | set → spike → receive → set → spike → receive |
| 2025-10-05_G5_臺北伊斯特_vs_桃園雲豹飛將_set1 | 44 | 33:11 | 33:18 | receive | 0.0s | serve → receive → set → spike → block → receive |
| 20250424 排島惡館-8 | 30 | 11:20 | 11:26 | receive | 3.0s | serve → receive → receive |
| 20250504 大統OB-成功大學vs台北大學B-第二局 | 4 | 1:29 | 1:44 | set | 1.2s | receive → receive → set → spike → receive → set |
| 20260403-霖度C-02 | 7 | 1:50 | 2:00 | receive | 0.7s | receive → set → spike → receive → receive → receive |
| 20260403-霖度C-02 | 11 | 2:53 | 3:04 | spike | 0.0s | receive → set → spike → receive → set → spike |
| 20260403-霖度C-02 | 23 | 7:11 | 7:26 | spike | 0.4s | receive → set → spike → receive → set → spike |
| 20260403-霖度C-02 | 31 | 9:45 | 9:50 | receive | 0.1s | serve → receive → receive |
| 20260426-小窩-01 | 10 | 3:54 | 4:10 | spike | 0.3s | receive → set → spike → receive → set → spike |
| 20260426-小窩-01 | 16 | 6:42 | 6:52 | spike | 0.2s | receive → set → spike → receive → set → spike |
| 20260426-小窩-01 | 18 | 7:32 | 7:44 | receive | 1.3s | set → spike → receive → set → spike → receive |
| 20260502-排島本館-02 | 2 | 1:52 | 2:06 | receive | 0.5s | receive → receive → set → spike → block → receive |
| 20260502-排島本館-02 | 4 | 3:10 | 3:18 | receive | 0.5s | serve → receive → spike → receive → receive |
| 20260502-排島本館-02 | 8 | 4:21 | 4:32 | spike | 0.9s | receive → set → spike → receive → set → spike |
| 20260502-排島本館-02 | 15 | 6:45 | 6:52 | set | 0.0s | serve → receive → set |
| Full Match ｜ Spain vs The Netherlands ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool C_set1 | 3 | 1:04 | 1:10 | spike | 0.1s | serve → receive → set → spike |
| Full Match ｜ Spain vs The Netherlands ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool C_set1 | 9 | 4:19 | 4:38 | receive | 0.0s | set → spike → receive → set → spike → receive |
| Full Match ｜ Spain vs The Netherlands ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool C_set1 | 12 | 6:23 | 6:34 | receive | 0.1s | receive → receive → set → spike → block → receive |
| Japan 🇯🇵 vs. Czechia 🇨🇿 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 12 | 4:42 | 4:53 | receive | 0.1s | set → spike → receive → set → spike → receive |
| Japan 🇯🇵 vs. Czechia 🇨🇿 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 27 | 12:29 | 12:34 | receive | 1.4s | serve → receive → receive |
| Japan 🇯🇵 vs. France 🇫🇷 ｜ VNL 2025 - Full Match ｜ Week 3_set1 | 22 | 9:24 | 9:30 | receive | 0.5s | serve → receive → set → spike → block → receive |
| Semi Final 1 - Osaka Bluteon vs. Stings Aichi ｜ SV League - Full Match ｜ Volleyball_set1 | 50 | 30:53 | 31:02 | set | 0.8s | serve → receive → set → spike → receive → set |
| Uzbekistan vs. Japan - Ranking 19-20 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 18 | 8:28 | 8:35 | receive | 0.3s | serve → receive → set → spike → block → receive |
| ᴴᴰ114UVL預賽：：中山大學vs清華大學：：男一級 大專排球聯賽 AI網路直播_set1 | 16 | 6:07 | 6:18 | set | 1.5s | set → receive → receive → spike → receive → set |
| ᴴᴰ114UVL預賽：：中山大學vs清華大學：：男一級 大專排球聯賽 AI網路直播_set1 | 34 | 14:30 | 14:38 | receive | 0.1s | serve → receive → set → spike → receive |
| ᴴᴰ114UVL預賽：：清華大學vs嘉義大學：：男一級 大專排球聯賽 AI網路直播_set1 | 14 | 5:41 | 5:49 | receive | 0.1s | set → spike → block → receive → spike → receive |
| ᴴᴰ114UVL預賽：：清華大學vs嘉義大學：：男一級 大專排球聯賽 AI網路直播_set1 | 25 | 11:16 | 11:26 | spike | 0.2s | set → spike → block → receive → set → spike |
| ᴴᴰ114UVL預賽：：清華大學vs嘉義大學：：男一級 大專排球聯賽 AI網路直播_set1 | 30 | 14:06 | 14:13 | spike | 0.1s | serve → receive → set → spike |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G28 11⧸22 18_30 臺北伊斯特 vs 台鋼天鷹_set1 | 10 | 4:16 | 4:30 | receive | 0.2s | set → receive → receive → set → spike → receive |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G28 11⧸22 18_30 臺北伊斯特 vs 台鋼天鷹_set1 | 25 | 13:37 | 13:42 | spike | 0.3s | serve → receive → set → spike |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G28 11⧸22 18_30 臺北伊斯特 vs 台鋼天鷹_set1 | 29 | 16:18 | 16:22 | spike | 0.5s | serve → receive → spike |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G6 10⧸5 18_30 臺中連莊 vs 台鋼天鷹_set1 | 13 | 6:29 | 6:36 | receive | 0.1s | serve → receive → receive → receive |

## 附錄 A：span 內有 2 個以上 `score` — 69 筆

間隔不到 0.5 秒的，是同一個 `score` 被標了兩次，不是兩個。

| 影片 | Rally | 起 | 訖 | score 數 | 最小間隔 |
|---|---:|---:|---:|---:|---:|
| 0323小窩臨打 3 | 37 | 11:47 | 11:58 | 2 | 1.10s |
| 03⧸15(日) 17_00｜例行賽G108 #新北中纖 vs. #高雄台電｜企業21年甲級男女排球聯賽_set3 | 21 | 12:23 | 12:30 | 2 | 1.57s |
| 0420小窩季打 3 | 20 | 6:14 | 6:26 | 2 | 0.47s |
| 0427小窩季打 10 | 24 | 7:51 | 8:06 | 2 | 0.63s |
| 0427小窩季打 10 | 42 | 15:07 | 15:31 | 2 | 0.73s |
| 0427小窩季打 11 | 10 | 5:09 | 5:21 | 2 | 0.93s |
| 0427小窩季打 11 | 34 | 13:38 | 13:47 | 2 | 0.93s |
| 0427小窩季打 11 | 43 | 16:52 | 17:13 | 2 | 0.73s |
| 0427小窩季打 12 | 4 | 1:09 | 1:18 | 2 | 1.00s |
| 0427小窩季打 12 | 49 | 17:53 | 18:18 | 2 | 0.93s |
| 0601小窩季打 2 | 10 | 3:34 | 3:39 | 2 | 1.00s |
| 0803小窩季打 1 | 15 | 5:12 | 5:38 | 2 | 1.40s |
| 1005小窩臨打 1 | 4 | 3:53 | 4:03 | 2 | 1.57s |
| 1005小窩臨打 1 | 33 | 13:07 | 13:16 | 2 | 1.23s |
| 2025-09-27_G1_臺北伊斯特_vs_臺中連莊_set1 | 12 | 7:10 | 7:15 | 2 | 0.07s |
| 2025-09-28_G2_臺北伊斯特_vs_桃園雲豹飛將_set2 | 33 | 19:16 | 19:22 | 2 | 0.00s |
| 2025-10-26_G13_台鋼天鷹_vs_桃園雲豹飛將_set1 | 35 | 18:50 | 19:04 | 2 | 0.62s |
| 2025-11-01_G16_桃園雲豹飛將_vs_台鋼天鷹_set1 | 13 | 6:35 | 6:44 | 2 | 0.10s |
| 20250424 排島惡館-6 | 37 | 11:44 | 11:53 | 2 | 0.30s |
| 20250424 排島惡館-7 | 7 | 3:43 | 4:06 | 2 | 1.10s |
| 20250621 排島本館-2 | 35 | 10:22 | 10:31 | 2 | 0.97s |
| 20251227-排島本館-3 | 1 | 0:17 | 0:25 | 2 | 0.17s |
| 20251227-排島本館-3 | 3 | 0:54 | 1:03 | 2 | 0.17s |
| 20251227-排島本館-3 | 5 | 1:33 | 1:41 | 2 | 0.10s |
| 20251227-排島本館-6 | 12 | 5:40 | 6:01 | 2 | 0.20s |
| 20251227-排島本館-6 | 25 | 9:27 | 9:33 | 2 | 0.23s |
| 20260502-排島本館-02 | 45 | 18:17 | 18:26 | 2 | 0.03s |
| 37-39 Thriller! - Japan 🇯🇵 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 14 | 5:12 | 5:23 | 2 | 0.52s |
| 37-39 Thriller! - Japan 🇯🇵 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set2 | 16 | 7:14 | 7:31 | 2 | 0.16s |
| 37-39 Thriller! - Japan 🇯🇵 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set2 | 47 | 22:05 | 22:25 | 2 | 0.48s |
| Brazil 🇧🇷 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 35 | 18:16 | 18:23 | 2 | 1.00s |
| Bulgaria vs. Italy - Ranking 7-8 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 12 | 5:48 | 5:55 | 2 | 0.08s |
| Bulgaria 🇧🇬 vs. Canada 🇨🇦 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 17 | 10:30 | 10:33 | 2 | 0.32s |
| Bulgaria 🇧🇬 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 1 | 0:00 | 0:05 | 2 | 0.24s |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 37 | 19:10 | 19:17 | 2 | 0.36s |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 39 | 21:15 | 21:23 | 2 | 0.04s |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 49 | 34:38 | 34:43 | 2 | 1.36s |
| China 🇨🇳 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 7 | 3:44 | 3:53 | 2 | 0.16s |
| China 🇨🇳 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 15 | 7:19 | 7:27 | 2 | 0.92s |
| France 🇫🇷 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 8 | 3:19 | 3:25 | 2 | 0.36s |
| France 🇫🇷 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 33 | 15:50 | 15:57 | 3 | 0.00s |
| France 🇫🇷 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 44 | 23:48 | 23:51 | 2 | 0.40s |
| Full Match ｜ Bulgaria vs Luxembourg ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set3 | 37 | 18:34 | 18:41 | 2 | 1.13s |
| Full Match ｜ Croatia vs. Bulgaria ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set1 | 33 | 13:50 | 13:59 | 2 | 0.63s |
| Full Match ｜ Croatia vs. Bulgaria ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set1 | 37 | 17:00 | 17:10 | 2 | 0.57s |
| Full Match ｜ Croatia vs. Serbia ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set1 | 19 | 7:35 | 8:00 | 2 | 0.07s |
| Full Match ｜ Denmark vs. England ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool A_set1 | 2 | 0:39 | 0:55 | 2 | 0.23s |
| Full Match ｜ Ireland vs. Spain ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool D_set1 | 28 | 13:27 | 13:31 | 2 | 0.20s |
| Full Match ｜ Luxembourg vs. Croatia - CEV U22 Volleyball European Championship 2026 ｜ Women ｜ Pool E_set1 | 25 | 12:40 | 12:55 | 2 | 0.60s |
| Full Match ｜ Poland vs. Slovakia ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 2 | 0:37 | 0:45 | 2 | 0.47s |
| Full Match ｜ Serbia vs. Bulgaria ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set1 | 6 | 2:27 | 2:31 | 2 | 0.03s |
| Japan vs. USA - Ranking 15-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 45 | 21:39 | 21:49 | 2 | 0.72s |
| Japan 🇯🇵 vs. Czechia 🇨🇿 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 7 | 2:35 | 2:46 | 2 | 1.04s |
| Japan 🇯🇵 vs. Czechia 🇨🇿 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 25 | 10:52 | 11:05 | 2 | 1.08s |
| Japan 🇯🇵 vs. Dominican Republic 🇩🇴 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 7 | 3:03 | 3:11 | 2 | 0.16s |
| Japan 🇯🇵 vs. France 🇫🇷 ｜ VNL 2025 - Full Match ｜ Week 3_set1 | 19 | 8:20 | 8:33 | 2 | 0.92s |
| Semi Final 1 - Osaka Bluteon vs. Stings Aichi ｜ SV League - Full Match ｜ Volleyball_set1 | 14 | 6:36 | 6:45 | 2 | 0.23s |
| Semi Final 3 - Suntory Sunbirds vs. Wolfdogs Nagoya ｜ SVL Playoff - Full Match ｜ Volleyball_set1 | 8 | 3:22 | 3:31 | 2 | 0.87s |
| ᴴᴰ114UVL預賽：：中原大學vs實踐大學：：男一級 大專排球聯賽 AI網路直播_set1 | 21 | 10:27 | 10:35 | 2 | 0.37s |
| ᴴᴰ114UVL預賽：：中山大學vs國北教大：：男一級 大專排球聯賽 AI網路直播_set1 | 35 | 15:42 | 15:57 | 2 | 1.23s |
| ᴴᴰ114UVL預賽：：臺灣師大vs中山大學：：男一級 大專排球聯賽 AI網路直播_set1 | 20 | 7:57 | 8:01 | 2 | 0.27s |
| ᴴᴰ114UVL預賽：：陽明交大vs臺灣體大：：男一級 大專排球聯賽 AI網路直播_set1 | 32 | 14:55 | 15:10 | 2 | 0.10s |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G22 11⧸9 18_30 臺北伊斯特 vs 台鋼天鷹_set1 | 34 | 20:05 | 20:11 | 2 | 0.08s |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G26 11⧸16 18_30 臺中連莊 vs 臺北伊斯特_set1 | 18 | 9:50 | 9:57 | 2 | 0.27s |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G26 11⧸16 18_30 臺中連莊 vs 臺北伊斯特_set1 | 37 | 21:34 | 21:44 | 2 | 0.28s |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G26 11⧸16 18_30 臺中連莊 vs 臺北伊斯特_set1 | 55 | 34:25 | 34:36 | 2 | 0.55s |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G27 11⧸22 15_00 桃園雲豹飛將 vs 台中連莊_set1 | 7 | 3:08 | 3:18 | 3 | 0.40s |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G27 11⧸22 15_00 桃園雲豹飛將 vs 台中連莊_set1 | 14 | 7:11 | 7:15 | 2 | 0.05s |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G27 11⧸22 15_00 桃園雲豹飛將 vs 台中連莊_set1 | 29 | 13:46 | 13:53 | 2 | 0.27s |
