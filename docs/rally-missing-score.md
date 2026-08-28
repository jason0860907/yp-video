# Rally 結尾沒有 score 的清單

掃描日期：2026-08-28 · 資料：`videos/rally-spot/annotations` × `videos/action/annotations`（200 支影片、8,860 rallies）

重跑：`uv run python scripts/scan_rally_edges.py`

判定：rally span `[start, end]` 內最後的動作事件不是 `score`。共 136 筆。

分類：`結尾切太早` = span 外 3 秒內就有 `score`，標註在、是邊界偏了；`score 不在最後` = span 內有 `score`，但後面還有別的動作；`疑似漏標` = 前後都找不到鄰近的 `score`；`導播問題` = 已經看過畫面，導播沒拍到那個 `score`，補不了。

## score 不在最後 — 136 筆

span 內有 `score`，但它不是最後的事件。

| 影片 | Rally | 起 | 訖 | 最後動作 | score 距結尾 | 動作序列（後 6） |
|---|---:|---:|---:|---|---:|---|
| 0104排島臨打 3 | 34 | 12:52 | 13:06 | set | 4.1s | receive → set → score → spike → receive → set |
| 0323小窩臨打 3 | 19 | 6:27 | 6:38 | receive | 1.8s | spike → receive → set → spike → score → receive |
| 03⧸14(六) 14_00｜例行賽G103 #雲林美津濃 vs. #桃園臺灣產險｜企業21年甲級男女排球聯賽_set3 | 29 | 18:54 | 19:10 | receive | 2.0s | block → receive → receive → receive → score → receive |
| 03⧸14(六) 18_00｜例行賽G105 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 1 | 0:00 | 0:06 | spike | 3.6s | receive → set → score → spike |
| 03⧸14(六) 18_00｜例行賽G105 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 31 | 17:52 | 18:05 | set | 4.1s | receive → set → spike → score → receive → set |
| 03⧸20(五) 14_00｜挑戰賽G110 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 10 | 4:34 | 4:44 | receive | 4.4s | receive → set → spike → score → receive → receive |
| 03⧸20(五) 14_00｜挑戰賽G110 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 29 | 15:31 | 15:40 | set | 2.8s | receive → set → spike → receive → score → set |
| 03⧸20(五) 14_00｜挑戰賽G110 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 42 | 28:31 | 28:40 | spike | 2.6s | spike → block → receive → set → score → spike |
| 03⧸20(五) 16_00｜挑戰賽G111 #屏東台電 vs. #桃園臺產｜企業21年甲級男女排球聯賽_set1 | 24 | 10:38 | 10:49 | set | 3.8s | receive → set → spike → score → receive → set |
| 03⧸20(五) 18_00｜挑戰賽G112 #臺北國北獅 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set1 | 20 | 10:04 | 10:16 | block | 3.1s | block → receive → set → spike → score → block |
| 03⧸21(六) 18_00｜男子組冠軍賽G114 #雲林美津濃 vs. #屏東台電｜企業21年甲級男女排球聯賽_set1 | 11 | 6:03 | 6:12 | set | 3.6s | receive → set → spike → score → receive → set |
| 03⧸21(六) 18_00｜男子組冠軍賽G114 #雲林美津濃 vs. #屏東台電｜企業21年甲級男女排球聯賽_set1 | 34 | 19:35 | 19:42 | receive | 2.6s | serve → receive → block → score → spike → receive |
| 03⧸21(六) 18_00｜男子組冠軍賽G114 #雲林美津濃 vs. #屏東台電｜企業21年甲級男女排球聯賽_set1 | 39 | 22:18 | 22:26 | receive | 0.8s | set → spike → block → receive → score → receive |
| 03⧸21(六) 18_00｜男子組冠軍賽G114 #雲林美津濃 vs. #屏東台電｜企業21年甲級男女排球聯賽_set1 | 40 | 22:50 | 23:01 | receive | 1.9s | spike → receive → set → spike → score → receive |
| 03⧸22(日) 17_00｜男子組冠軍賽 G117 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set1 | 2 | 0:22 | 0:36 | block | 3.6s | block → receive → set → score → spike → block |
| 03⧸22(日) 17_00｜男子組冠軍賽 G117 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set2 | 30 | 19:22 | 19:28 | receive | 1.7s | receive → set → score → spike → block → receive |
| 03⧸22(日) 17_00｜男子組冠軍賽 G117 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set3 | 4 | 1:19 | 1:35 | receive | 2.2s | receive → set → spike → block → score → receive |
| 0419小窩臨打 2 | 37 | 14:08 | 14:28 | spike | 2.5s | set → spike → receive → set → score → spike |
| 0420小窩季打 1 | 43 | 14:50 | 14:54 | receive | 1.0s | serve → receive → score → receive |
| 0420小窩季打 2 | 11 | 3:38 | 3:58 | receive | 3.1s | spike → receive → set → score → spike → receive |
| 0420小窩季打 3 | 15 | 4:12 | 4:20 | spike | 2.8s | serve → receive → set → score → spike |
| 2025-10-04_G3_臺中連莊_vs_桃園雲豹飛將_set1 | 13 | 5:53 | 5:59 | receive | 3.2s | serve → receive → score → set → receive |
| 2025-10-05_G5_臺北伊斯特_vs_桃園雲豹飛將_set1 | 16 | 7:45 | 7:59 | receive | 2.2s | set → spike → block → receive → score → receive |
| 2025-10-25_G11_臺北伊斯特_vs_桃園雲豹飛將_set1 | 26 | 14:08 | 14:16 | receive | 1.5s | receive → set → spike → score → block → receive |
| 2025-10-26_G14_臺北伊斯特_vs_臺中連莊_set1 | 37 | 22:35 | 22:44 | set | 3.2s | set → spike → block → receive → score → set |
| 2025-11-01_G15_臺中連莊_vs_臺北伊斯特_set1 | 32 | 18:36 | 18:43 | receive | 1.7s | serve → receive → set → spike → score → receive |
| 2025-11-01_G15_臺中連莊_vs_臺北伊斯特_set1 | 36 | 20:28 | 20:38 | spike | 0.8s | set → spike → receive → set → score → spike |
| 2025-11-01_G16_桃園雲豹飛將_vs_台鋼天鷹_set1 | 13 | 6:35 | 6:44 | receive | 2.9s | serve → receive → set → spike → score → receive |
| 2025-11-02_G17_桃園雲豹飛將_vs_臺北伊斯特_set1 | 4 | 1:31 | 1:41 | receive | 1.5s | spike → block → receive → set → score → receive |
| 2025-11-08_G19_臺北伊斯特_vs_臺中連莊_set1 | 47 | 29:31 | 29:37 | receive | 3.2s | serve → score → receive |
| 2025-11-08_G20_桃園雲豹飛將_vs_台鋼天鷹_set1 | 29 | 16:19 | 16:26 | set | 2.3s | set → spike → block → score → receive → set |
| 20250504 大統OB-成功大學vs台北大學B-第二局 | 4 | 1:29 | 1:44 | set | 2.6s | receive → set → spike → score → receive → set |
| 20260426-小窩-01 | 31 | 12:27 | 13:08 | receive | 15.1s | spike → receive → receive → receive → receive → receive |
| 20260502-排島本館-02 | 7 | 4:05 | 4:13 | receive | 2.6s | receive → set → spike → block → score → receive |
| Bulgaria 🇧🇬 vs. Argentina 🇦🇷 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 23 | 11:12 | 11:14 | serve | 1.5s | score → serve |
| Bulgaria 🇧🇬 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 14 | 6:04 | 6:12 | receive | 3.5s | receive → set → score → spike → receive → receive |
| Bulgaria 🇧🇬 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 37 | 18:51 | 18:58 | set | 4.6s | score → serve → receive → set |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 4 | 1:31 | 1:38 | receive | 1.2s | serve → receive → set → spike → score → receive |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 10 | 4:03 | 4:08 | receive | 3.2s | score → serve → receive |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 31 | 15:37 | 15:54 | set | 3.2s | set → spike → block → score → receive → set |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 50 | 34:58 | 35:02 | receive | 1.5s | serve → score → receive |
| Champions crowned in Final 24⧸25 (2⧸2) ｜ Suntory Sunbirds Osaka - Stings Aichi ｜ SV League 24⧸25_set1 | 18 | 8:18 | 8:29 | spike | 4.2s | receive → set → spike → receive → score → spike |
| China vs. Argentina - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 2 | 0:33 | 0:43 | set | 2.6s | receive → set → spike → receive → score → set |
| China vs. Brazil - Ranking 13-14 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 6 | 2:36 | 2:45 | receive | 2.8s | set → spike → block → score → receive → receive |
| China vs. Brazil - Ranking 13-14 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 19 | 7:53 | 8:04 | receive | 2.7s | receive → set → spike → score → block → receive |
| China 🇨🇳 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 3 | 1:56 | 2:06 | receive | 2.7s | serve → receive → receive → spike → score → receive |
| Cuba vs. Puerto Rico - Ranking 17-18 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 22 | 10:15 | 10:31 | receive | 2.9s | receive → receive → spike → score → receive → receive |
| Final - Stings vs. Sunbirds ｜ SVL League 2024⧸25 - Full Match ｜ Volleyball_set1 | 29 | 18:43 | 18:52 | receive | 3.6s | serve → receive → set → score → spike → receive |
| France vs. Iran - Semi Final 2 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 3 | 0:58 | 1:14 | set | 2.6s | set → spike → block → receive → score → set |
| France vs. Iran - Semi Final 2 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 5 | 2:16 | 2:26 | receive | 2.6s | receive → set → score → spike → block → receive |
| France vs. Iran - Semi Final 2 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 18 | 8:40 | 8:53 | set | 2.9s | receive → set → spike → score → receive → set |
| France 🇫🇷 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 19 | 9:53 | 10:01 | receive | 2.4s | receive → set → spike → block → score → receive |
| France 🇫🇷 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 41 | 21:11 | 21:24 | receive | 3.3s | receive → set → spike → score → receive → receive |
| Full Match ｜ Bulgaria vs Luxembourg ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set1 | 31 | 14:42 | 14:58 | receive | 2.2s | spike → receive → set → spike → score → receive |
| Full Match ｜ Bulgaria vs Luxembourg ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set3 | 37 | 18:34 | 18:41 | spike | 2.9s | serve → receive → set → score → spike |
| Full Match ｜ Croatia vs. Bulgaria ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set1 | 24 | 10:02 | 10:14 | spike | 3.8s | set → spike → receive → set → score → spike |
| Full Match ｜ Croatia vs. Bulgaria ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set1 | 28 | 11:48 | 11:54 | receive | 1.0s | serve → receive → set → spike → score → receive |
| Full Match ｜ Croatia vs. Serbia ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set1 | 21 | 8:55 | 9:16 | receive | 4.2s | set → spike → block → score → receive → receive |
| Full Match ｜ Denmark vs. England ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool A_set1 | 11 | 7:16 | 7:26 | receive | 2.4s | serve → receive → set → spike → score → receive |
| Full Match ｜ Denmark vs. England ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool A_set1 | 18 | 10:30 | 10:40 | receive | 4.0s | receive → set → spike → score → receive → receive |
| Full Match ｜ Ireland vs. Spain ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool D_set1 | 9 | 3:21 | 3:34 | receive | 1.8s | spike → receive → set → spike → score → receive |
| Full Match ｜ Ireland vs. Spain ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool D_set1 | 11 | 5:06 | 5:16 | set | 4.3s | set → spike → block → score → receive → set |
| Full Match ｜ Ireland vs. Türkiye ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool D_set1 | 14 | 6:58 | 7:05 | receive | 1.6s | serve → receive → set → score → receive |
| Full Match ｜ Ireland vs. Türkiye ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool D_set1 | 19 | 8:48 | 9:02 | set | 3.1s | block → receive → set → spike → score → set |
| Full Match ｜ Italy vs. England ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool A_set1 | 30 | 16:41 | 16:52 | receive | 0.7s | receive → set → spike → block → score → receive |
| Full Match ｜ Luxembourg vs. Croatia - CEV U22 Volleyball European Championship 2026 ｜ Women ｜ Pool E_set1 | 1 | 0:05 | 0:14 | receive | 2.0s | set → spike → receive → receive → score → receive |
| Full Match ｜ Norway vs. Bulgaria ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool C_set1 | 21 | 9:14 | 9:23 | receive | 2.6s | receive → set → spike → score → receive → receive |
| Full Match ｜ Poland vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 4 | 1:24 | 1:32 | receive | 2.1s | serve → receive → set → spike → score → receive |
| Full Match ｜ Poland vs. Slovakia ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 26 | 11:57 | 12:06 | receive | 4.0s | serve → receive → set → score → receive |
| Full Match ｜ Serbia vs. Bulgaria ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set1 | 20 | 8:10 | 8:20 | receive | 1.6s | receive → set → spike → receive → score → receive |
| Full Match ｜ Serbia vs. Luxembourg ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set1 | 34 | 17:25 | 17:34 | receive | 2.8s | receive → set → receive → score → receive → receive |
| Full Match ｜ Slovakia vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 31 | 15:44 | 15:56 | receive | 3.1s | receive → set → spike → score → receive → receive |
| Full Match ｜ Spain vs The Netherlands ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool C_set1 | 13 | 7:46 | 8:00 | receive | 2.6s | set → spike → block → score → receive → receive |
| Full Match ｜ Türkiye vs. Spain ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool D_set1 | 40 | 21:22 | 21:38 | receive | 4.6s | set → spike → score → set → spike → receive |
| Japan vs. USA - Ranking 15-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 26 | 10:14 | 10:21 | receive | 3.0s | serve → receive → score → receive |
| Japan vs. USA - Ranking 15-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 45 | 21:39 | 21:49 | receive | 3.6s | receive → set → spike → block → score → receive |
| Japan 🇯🇵 vs. Czechia 🇨🇿 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 22 | 9:48 | 9:56 | receive | 2.2s | receive → set → spike → receive → score → receive |
| Japan 🇯🇵 vs. Dominican Republic 🇩🇴 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 24 | 11:02 | 11:15 | set | 3.6s | receive → set → spike → score → receive → set |
| Japan 🇯🇵 vs. Dominican Republic 🇩🇴 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 29 | 13:26 | 13:34 | receive | 2.6s | serve → receive → set → spike → score → receive |
| Japan 🇯🇵 vs. France 🇫🇷 ｜ VNL 2025 - Full Match ｜ Week 3_set1 | 36 | 17:57 | 18:28 | block | 19.7s | receive → receive → receive → set → spike → block |
| Japan 🇯🇵 vs. Netherlands 🇳🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 2 | 0:25 | 0:35 | receive | 4.5s | spike → score → receive → receive → spike → receive |
| Japan 🇯🇵 vs. Netherlands 🇳🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 23 | 10:53 | 11:10 | receive | 3.2s | set → spike → block → score → receive → receive |
| Japan 🇯🇵 vs. Netherlands 🇳🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 31 | 15:06 | 15:14 | set | 2.7s | receive → set → spike → score → receive → set |
| Japan 🇯🇵 vs. Netherlands 🇳🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 43 | 21:20 | 21:32 | receive | 5.2s | receive → set → spike → score → receive → receive |
| Japan 🇯🇵 vs. Serbia 🇷🇸 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 38 | 16:51 | 16:59 | spike | 1.4s | receive → set → spike → receive → score → spike |
| Jtekt Stings 🇯🇵 - Suntory Sunbirds Osaka 🇯🇵 ｜ SV League 2026 ｜ Full Match - Volleyball_set1 | 31 | 17:17 | 17:30 | receive | 1.4s | receive → set → spike → block → score → receive |
| Korea vs. Finland - Ranking 11-12 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 21 | 7:25 | 7:32 | spike | 2.6s | serve → receive → set → score → spike |
| Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 11 | 4:51 | 5:00 | receive | 2.4s | spike → block → receive → receive → score → receive |
| Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 26 | 13:42 | 13:50 | receive | 1.3s | receive → set → spike → block → score → receive |
| Osaka Bluteon vs. Toray Arrows Shizuoka - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 31 | 17:20 | 17:33 | receive | 8.6s | spike → block → receive → set → spike → receive |
| Osaka Bluteon 🇯🇵 vs. JTEKT Stings 🇯🇵 ｜ SV League 2026 ｜ Full Match - Volleyball_set1 | 23 | 11:23 | 11:31 | set | 2.5s | set → spike → block → score → receive → set |
| Osaka Bluteon 🇯🇵 vs. JTEKT Stings 🇯🇵 ｜ SV League 2026 ｜ Full Match - Volleyball_set1 | 42 | 22:28 | 22:36 | receive | 1.9s | serve → receive → set → spike → score → receive |
| Pakistan vs. USA - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 1 | 0:07 | 0:14 | receive | 2.4s | serve → receive → set → score → spike → receive |
| Pakistan vs. USA - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 6 | 2:26 | 2:34 | receive | 2.1s | serve → receive → set → spike → score → receive |
| Pakistan vs. USA - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 16 | 7:38 | 7:55 | receive | 2.1s | set → spike → block → receive → score → receive |
| Poland vs. Spain - Semi Final 1 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 35 | 19:57 | 20:06 | receive | 1.8s | serve → receive → set → spike → score → receive |
| Semi Final 1 - Osaka Bluteon vs. Stings Aichi ｜ SV League - Full Match ｜ Volleyball_set1 | 8 | 3:45 | 3:52 | receive | 1.6s | receive → set → spike → block → score → receive |
| Semi Final 1 - Osaka Bluteon vs. Stings Aichi ｜ SV League - Full Match ｜ Volleyball_set1 | 18 | 8:57 | 9:04 | block | 2.0s | serve → receive → set → score → spike → block |
| Semi Final 1 - Osaka Bluteon vs. Stings Aichi ｜ SV League - Full Match ｜ Volleyball_set1 | 42 | 25:36 | 25:56 | receive | 3.2s | receive → set → spike → block → score → receive |
| Semi Final 2 - Osaka Bluteon vs. Stings Aichi ｜ SV League - Full Match ｜ Volleyball_set1 | 30 | 16:30 | 16:36 | spike | 3.7s | serve → score → set → spike |
| Semi Final 2 - Osaka Bluteon vs. Stings Aichi ｜ SV League - Full Match ｜ Volleyball_set1 | 39 | 23:08 | 23:16 | receive | 2.8s | serve → receive → score → block → receive |
| Semi Final 3 - Suntory Sunbirds vs. Wolfdogs Nagoya ｜ SVL Playoff - Full Match ｜ Volleyball_set1 | 46 | 27:17 | 27:26 | block | 4.6s | serve → receive → score → spike → block |
| Spain vs. Iran - Ranking 3-4 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 14 | 10:01 | 10:10 | receive | 3.2s | serve → receive → receive → score → block → receive |
| Spain vs. Iran - Ranking 3-4 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 16 | 11:03 | 11:16 | spike | 2.8s | receive → block → receive → score → set → spike |
| Suntory Sunbirds vs. Osaka Bluteon ｜ SV.LEAGUE 2025⧸26 ｜ Full Match - Volleyball_set1 | 23 | 10:53 | 11:00 | receive | 2.0s | serve → receive → set → spike → score → receive |
| Suntory Sunbirds 🇯🇵 vs. Stings AICHI 🇯🇵 ｜ SV League 2026 ｜ Full Match - Japan Volleyball_set1 | 6 | 3:19 | 3:27 | receive | 2.4s | receive → set → spike → block → score → receive |
| Suntory Sunbirds 🇯🇵 vs. Stings AICHI 🇯🇵 ｜ SV League 2026 ｜ Full Match - Japan Volleyball_set1 | 30 | 14:25 | 14:37 | receive | 3.0s | receive → set → spike → block → score → receive |
| Taipei vs. Argentina - Playoffs ｜ Girls' U19 World Champs 2025 - Full Match_set1 | 7 | 3:28 | 3:47 | set | 2.6s | receive → set → spike → score → receive → set |
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
| ᴴᴰ114UVL預賽：：中原大學vs陽明交大：：男一級 大專排球聯賽 AI網路直播_set1 | 39 | 18:55 | 19:03 | set | 3.1s | receive → set → spike → score → receive → set |
| ᴴᴰ114UVL預賽：：中山大學vs國北教大：：男一級 大專排球聯賽 AI網路直播_set1 | 15 | 6:35 | 6:42 | set | 2.8s | serve → receive → score → receive → set |
| ᴴᴰ114UVL預賽：：中山大學vs國北教大：：男一級 大專排球聯賽 AI網路直播_set1 | 19 | 8:24 | 8:32 | receive | 2.8s | receive → set → spike → block → score → receive |
| ᴴᴰ114UVL預賽：：中山大學vs國北教大：：男一級 大專排球聯賽 AI網路直播_set1 | 35 | 15:42 | 15:57 | block | 2.2s | spike → receive → set → spike → score → block |
| ᴴᴰ114UVL預賽：：中山大學vs清華大學：：男一級 大專排球聯賽 AI網路直播_set1 | 16 | 6:07 | 6:18 | set | 2.5s | receive → receive → spike → score → receive → set |
| ᴴᴰ114UVL預賽：：中山大學vs清華大學：：男一級 大專排球聯賽 AI網路直播_set1 | 39 | 17:21 | 17:33 | receive | 3.2s | receive → set → spike → block → score → receive |
| ᴴᴰ114UVL預賽：：中山大學vs清華大學：：男一級 大專排球聯賽 AI網路直播_set1 | 41 | 18:34 | 18:43 | receive | 3.1s | set → spike → score → receive → receive → receive |
| ᴴᴰ114UVL預賽：：國北教大vs嘉義大學：：男一級 大專排球聯賽 AI網路直播_set1 | 1 | 0:32 | 0:40 | set | 1.9s | serve → receive → set → spike → score → set |
| ᴴᴰ114UVL預賽：：臺灣師大vs中山大學：：男一級 大專排球聯賽 AI網路直播_set1 | 3 | 0:49 | 1:01 | receive | 2.7s | set → receive → set → spike → score → receive |
| ᴴᴰ114UVL預賽：：臺灣師大vs中山大學：：男一級 大專排球聯賽 AI網路直播_set1 | 23 | 9:13 | 9:24 | spike | 2.3s | receive → set → spike → score → receive → spike |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G21 11⧸9 15_00 桃園雲豹飛將 vs 臺中連莊_set1 | 39 | 19:25 | 20:03 | receive | 2.6s | receive → receive → set → spike → score → receive |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G23 11⧸15 15_00 臺中連莊 vs 台鋼天鷹_set1 | 24 | 11:12 | 11:29 | receive | 2.6s | receive → set → spike → receive → score → receive |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G24 11⧸15 18_30 桃園雲豹飛將 vs 臺北伊斯特_set1 | 9 | 4:19 | 4:26 | receive | 0.9s | serve → receive → set → spike → score → receive |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G24 11⧸15 18_30 桃園雲豹飛將 vs 臺北伊斯特_set1 | 25 | 11:40 | 11:51 | block | 3.1s | block → receive → score → set → spike → block |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G24 11⧸15 18_30 桃園雲豹飛將 vs 臺北伊斯特_set1 | 45 | 24:13 | 24:31 | receive | 2.0s | block → receive → set → spike → score → receive |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G26 11⧸16 18_30 臺中連莊 vs 臺北伊斯特_set1 | 17 | 9:20 | 9:29 | receive | 2.7s | set → spike → block → score → receive → receive |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G29 11⧸23 15_00 臺北伊斯特 vs 台中連莊_set1 | 18 | 9:51 | 9:59 | receive | 2.3s | receive → set → spike → score → block → receive |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G6 10⧸5 18_30 臺中連莊 vs 台鋼天鷹_set1 | 25 | 12:12 | 12:19 | receive | 2.4s | receive → set → block → score → spike → receive |

## 附錄 A：span 內有 2 個以上 `score` — 1 筆

間隔不到 0.5 秒的，是同一個 `score` 被標了兩次，不是兩個。

| 影片 | Rally | 起 | 訖 | score 數 | 最小間隔 |
|---|---:|---:|---:|---:|---:|
| China 🇨🇳 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 15 | 7:19 | 7:27 | 2 | 0.92s |
