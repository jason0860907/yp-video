# Rally 開頭沒有 serve 的清單

掃描日期：2026-08-24 · 資料：`videos/rally-spot/annotations` × `videos/action/annotations`（197 支影片、8,722 rallies）

判定：rally span `[start, end]` 內第一個動作事件不是 `serve`。共 121 筆。

分類：`切界太緊` = 前一個 serve 落在 rally start 前 3 秒內（發球接觸幀被切在 span 外）；`serve 不在最前` = span 內有 serve，但前面還有別的動作；`疑似漏標` = span 內外都找不到鄰近 serve。


## 切界太緊 — 36 筆

| 影片 | Rally | 起 | 訖 | 首個動作 | 距前一個 serve | 動作序列（前 6） |
|---|---:|---:|---:|---|---:|---|
| 03⧸15(日) 17_00｜例行賽G108 #新北中纖 vs. #高雄台電｜企業21年甲級男女排球聯賽_set1 | 16 | 7:50 | 7:56 | receive | 0.1s | receive → set → spike → score |
| 03⧸15(日) 17_00｜例行賽G108 #新北中纖 vs. #高雄台電｜企業21年甲級男女排球聯賽_set1 | 29 | 15:25 | 15:30 | receive | 0.1s | receive → set → spike → block → score |
| 03⧸20(五) 14_00｜挑戰賽G110 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 23 | 12:15 | 12:28 | receive | 0.0s | receive → set → spike → receive → receive → set |
| 03⧸20(五) 14_00｜挑戰賽G110 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 32 | 17:55 | 18:00 | receive | 0.3s | receive → set → spike → score |
| 03⧸22(日) 17_00｜男子組冠軍賽 G117 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set1 | 30 | 16:35 | 16:39 | set | 0.0s | set → spike → block → score |
| 2025-10-12_G9_臺中連莊_vs_台鋼天鷹_set1 | 26 | 13:06 | 13:23 | receive | 0.0s | receive → receive → receive → serve → receive → set |
| 20250424 排島惡館-8 | 5 | 3:02 | 3:12 | receive | 0.7s | receive → set → spike → receive → set → spike |
| 20250424 排島惡館-8 | 6 | 3:24 | 3:34 | receive | 0.2s | receive → set → spike → receive → set → spike |
| 20250424 排島惡館-8 | 10 | 4:58 | 5:04 | receive | 0.1s | receive → set → spike → score |
| 20250424 排島惡館-8 | 18 | 7:08 | 7:34 | receive | 0.3s | receive → set → spike → receive → set → spike |
| 20250424 排島惡館-8 | 22 | 8:34 | 8:58 | receive | 0.1s | receive → set → spike → receive → set → spike |
| 20250424 排島惡館-8 | 26 | 10:12 | 10:25 | receive | 0.8s | receive → set → spike → receive → set → spike |
| 20251227-排島本館-3 | 22 | 6:38 | 6:44 | receive | 0.0s | receive → set → spike → receive → score |
| 20260403-霖度C-02 | 26 | 8:02 | 8:09 | receive | 0.3s | receive → set → spike → score |
| 20260426-小窩-01 | 1 | 0:05 | 0:24 | receive | 0.2s | receive → set → spike → receive → set → receive |
| 20260426-小窩-01 | 6 | 2:32 | 2:42 | receive | 0.1s | receive → receive → receive → receive → set → spike |
| 20260426-小窩-01 | 7 | 2:58 | 3:08 | receive | 0.0s | receive → set → spike → receive → set → spike |
| 20260426-小窩-01 | 22 | 8:38 | 8:50 | receive | 0.1s | receive → set → spike → receive → set → spike |
| 20260502-排島本館-02 | 2 | 1:54 | 2:06 | receive | 0.1s | receive → set → spike → block → receive → receive |
| 20260510邷力豹臨打1 | 15 | 8:18 | 8:27 | receive | 1.2s | receive → set → spike → receive → set → spike |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 25 | 11:52 | 12:08 | set | 1.1s | set → spike → block → receive → set → spike |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 31 | 15:40 | 15:54 | receive | 1.2s | receive → set → spike → receive → set → spike |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 32 | 16:24 | 16:30 | receive | 0.6s | receive → set → spike → score |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 34 | 17:09 | 17:14 | receive | 0.8s | receive → set → spike → score |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 48 | 34:02 | 34:18 | receive | 0.8s | receive → set → spike → block → receive → set |
| Full Match ｜ Poland vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 14 | 6:14 | 6:28 | receive | 1.1s | receive → set → spike → block → receive → set |
| Full Match ｜ Poland vs. Slovakia ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 1 | 0:18 | 0:24 | receive | 0.5s | receive → set → spike → score |
| Full Match ｜ Slovakia vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 1 | 0:20 | 0:28 | receive | 0.3s | receive → set → spike → block → receive → score |
| Japan 🇯🇵 vs. Czechia 🇨🇿 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 21 | 9:28 | 9:34 | receive | 0.2s | receive → set → spike → score |
| Japan 🇯🇵 vs. Serbia 🇷🇸 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 8 | 3:20 | 3:26 | receive | 0.1s | receive → set → spike → block → score |
| Suntory Sunbirds vs. Osaka Bluteon ｜ SV.LEAGUE 2025⧸26 ｜ Full Match - Volleyball_set1 | 1 | 0:08 | 0:12 | receive | 0.7s | receive → set → spike → block → score |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G22 11⧸9 18_30 臺北伊斯特 vs 台鋼天鷹_set1 | 24 | 12:56 | 13:02 | receive | 0.0s | receive → set → spike → receive → score |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G22 11⧸9 18_30 臺北伊斯特 vs 台鋼天鷹_set1 | 25 | 13:26 | 13:30 | receive | 0.3s | receive → set → spike → score |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G22 11⧸9 18_30 臺北伊斯特 vs 台鋼天鷹_set1 | 44 | 26:14 | 26:18 | receive | 0.6s | receive → set → spike |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G24 11⧸15 18_30 桃園雲豹飛將 vs 臺北伊斯特_set1 | 33 | 15:45 | 15:52 | receive | 1.2s | receive → set → spike → block → receive → set |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G30 11⧸23 18_30 桃園雲豹飛將 vs 台鋼天鷹_set1 | 14 | 10:30 | 10:35 | receive | 0.5s | receive → set → spike → block → receive → receive |

## serve 不在最前 — 7 筆

| 影片 | Rally | 起 | 訖 | 首個動作 | 距前一個 serve | 動作序列（前 6） |
|---|---:|---:|---:|---|---:|---|
| 03⧸14(六) 14_00｜例行賽G103 #雲林美津濃 vs. #桃園臺灣產險｜企業21年甲級男女排球聯賽_set2 | 40 | 26:02 | 26:06 | score | 25.5s | score → serve → score |
| 03⧸15(日) 13_00｜例行賽G106 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set3 | 19 | 10:45 | 10:50 | spike | 26.4s | spike → serve → receive → set → spike → block |
| 03⧸15(日) 17_00｜例行賽G108 #新北中纖 vs. #高雄台電｜企業21年甲級男女排球聯賽_set2 | 21 | 13:12 | 13:23 | spike | 74.1s | spike → serve → receive → set → spike → receive |
| 03⧸15(日) 17_00｜例行賽G108 #新北中纖 vs. #高雄台電｜企業21年甲級男女排球聯賽_set3 | 7 | 3:45 | 3:52 | receive | 25.6s | receive → serve → set → spike → receive → set |
| Final - Stings vs. Sunbirds ｜ SVL League 2024⧸25 - Full Match ｜ Volleyball_set1 | 40 | 26:30 | 26:38 | set | 89.9s | set → serve → receive → set → spike → block |
| Korea vs. Bulgaria - Classification 5-8 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 36 | 17:00 | 17:14 | receive | 26.7s | receive → receive → serve → receive → spike → receive |
| Poland vs. France - Final ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 26 | 11:34 | 11:45 | receive | 21.5s | receive → serve → receive → set → spike → block |

## 疑似漏標 — 78 筆

| 影片 | Rally | 起 | 訖 | 首個動作 | 距前一個 serve | 動作序列（前 6） |
|---|---:|---:|---:|---|---:|---|
| 0323小窩臨打 3 | 9 | 2:59 | 3:12 | receive | 8.0s | receive → set → spike → receive → set → spike |
| 0323小窩臨打 3 | 10 | 3:18 | 3:35 | receive | 27.0s | receive → receive → spike → receive → set → spike |
| 0323小窩臨打 3 | 12 | 4:12 | 4:21 | receive | 30.9s | receive → set → spike → score |
| 0323小窩臨打 3 | 26 | 8:30 | 8:35 | spike | 21.4s | spike → score |
| 0323小窩臨打 3 | 39 | 12:16 | 12:45 | receive | 7.3s | receive → set → spike → receive → set → spike |
| 03⧸14(六) 16_00｜例行賽G104 #獅子王 vs. #屏東台電｜企業21年甲級男女排球聯賽_set2 | 13 | 8:32 | 8:48 | set | 21.2s | set → receive → spike → receive → receive → receive |
| 03⧸14(六) 16_00｜例行賽G104 #獅子王 vs. #屏東台電｜企業21年甲級男女排球聯賽_set2 | 27 | 15:38 | 15:42 | receive | 29.9s | receive → set → spike → score |
| 03⧸14(六) 18_00｜例行賽G105 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 36 | 22:01 | 22:04 | set | 24.4s | set → spike → block → score |
| 03⧸15(日) 13_00｜例行賽G106 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set3 | 41 | 24:49 | 24:51 | set | 24.4s | set → spike → block → score |
| 03⧸15(日) 17_00｜例行賽G108 #新北中纖 vs. #高雄台電｜企業21年甲級男女排球聯賽_set1 | 19 | 9:18 | 9:32 | receive | 28.8s | receive → set → spike → block → receive → set |
| 03⧸20(五) 14_00｜挑戰賽G110 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set2 | 6 | 2:25 | 3:31 | receive | 26.6s | receive → set → spike → receive → set → spike |
| 03⧸21(六) 18_00｜男子組冠軍賽G114 #雲林美津濃 vs. #屏東台電｜企業21年甲級男女排球聯賽_set1 | 18 | 10:06 | 10:16 | block | 21.3s | block → block → receive → receive → receive → receive |
| 03⧸22(日) 17_00｜男子組冠軍賽 G117 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set2 | 2 | 0:25 | 0:36 | receive | 23.2s | receive → set → spike → receive → set → spike |
| 0601小窩季打 2 | 1 | 0:04 | 0:59 | receive | — | receive → set → spike → receive → set → spike |
| 0601小窩季打 2 | 14 | 4:40 | 4:54 | receive | 9.0s | receive → set → spike → receive → set → spike |
| 0601小窩季打 2 | 22 | 6:50 | 7:03 | receive | 11.0s | receive → set → spike → receive → set → spike |
| 10⧸4 2 | 12 | 5:12 | 5:22 | receive | 17.4s | receive → set → spike → block → score |
| 10⧸4 3 | 14 | 3:52 | 4:00 | receive | 6.9s | receive |
| 2025-10-11_G8_臺中連莊_vs_桃園雲豹飛將_set1 | 1 | 0:05 | 0:08 | set | — | set → spike → score |
| 2025-10-11_G8_臺中連莊_vs_桃園雲豹飛將_set1 | 26 | 15:26 | 15:33 | set | 127.3s | set → score |
| 2025-11-08_G19_臺北伊斯特_vs_臺中連莊_set1 | 43 | 25:57 | 26:03 | receive | 32.7s | receive → set → spike → score |
| 20250424 排島惡館-7 | 9 | 4:38 | 4:48 | set | 22.3s | set → spike → receive → set → spike → score |
| 20250504 大統OB-成功大學vs台北大學B-第二局 | 1 | 0:18 | 0:36 | receive | — | receive → set → spike → block → receive → set |
| 20250504 大統OB-成功大學vs台北大學B-第二局 | 20 | 9:00 | 9:08 | receive | 28.6s | receive → set → spike → score |
| 20251109-排島本館-03 | 5 | 2:14 | 2:22 | receive | 16.7s | receive → receive → set → receive → spike → score |
| 20251109-排島本館-03 | 11 | 4:38 | 4:50 | spike | 25.6s | spike → receive → set → spike → receive → set |
| 20251109-排島本館-03 | 47 | 16:24 | 16:41 | set | 19.5s | set → receive → set → spike → receive → set |
| 20260403-霖度C-01_set1 | 22 | 8:46 | 9:01 | receive | 23.4s | receive → set → receive → receive → set → spike |
| 20260426-小窩-01 | 15 | 5:52 | 6:26 | receive | 11.9s | receive → receive → set → receive → receive → set |
| 20260426-小窩-01 | 34 | 14:14 | 14:32 | receive | 8.3s | receive → set → spike → receive → set → spike |
| 20260510邷力豹臨打1 | 24 | 12:14 | 12:44 | receive | 32.7s | receive → set → receive → receive → set → spike |
| 2026⧸03⧸25 3 | 40 | 13:54 | 13:59 | receive | 15.2s | receive → score |
| 37-39 Thriller! - Japan 🇯🇵 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 31 | 12:39 | 12:44 | receive | 41.0s | receive → set → spike → receive → score |
| 37-39 Thriller! - Japan 🇯🇵 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set2 | 12 | 5:07 | 5:09 | spike | 36.9s | spike → score |
| 37-39 Thriller! - Japan 🇯🇵 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set2 | 13 | 5:21 | 5:28 | receive | 50.9s | receive → set → spike → receive → set → spike |
| 37-39 Thriller! - Japan 🇯🇵 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set2 | 20 | 8:53 | 8:56 | receive | 25.9s | receive → set → spike → score |
| Brazil 🇧🇷 vs. Italy 🇮🇹  ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 9 | 3:14 | 3:16 | score | 21.6s | score |
| Brazil 🇧🇷 vs. Italy 🇮🇹  ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 10 | 3:38 | 3:44 | set | 45.1s | set → spike → score |
| Bulgaria 🇧🇬 vs. Argentina 🇦🇷 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 23 | 11:13 | 11:17 | score | 19.9s | score |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 33 | 16:46 | 16:50 | receive | 22.3s | receive → set → spike → block → score |
| China 🇨🇳 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 9 | 4:53 | 4:58 | set | 7.0s | set → spike → block → score |
| France vs. Iran - Semi Final 2 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 24 | 11:05 | 11:12 | set | 28.8s | set → spike → score |
| Full Match ｜ Poland vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 37 | 19:36 | 19:42 | receive | 27.7s | receive → set → spike → block → score |
| Full Match ｜ Poland vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 41 | 21:42 | 21:45 | score | 28.3s | score |
| Full Match ｜ Poland vs. Slovakia ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 21 | 9:12 | 9:22 | set | 23.5s | set → receive → receive → set → spike → receive |
| Full Match ｜ Serbia vs. Bulgaria ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set1 | 10 | 3:50 | 4:03 | set | 17.1s | set → spike → block → receive → receive → set |
| Full Match ｜ Slovakia vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 6 | 2:42 | 2:52 | receive | 19.5s | receive → receive → receive → receive → set → spike |
| Full Match ｜ Slovakia vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 40 | 21:58 | 22:06 | receive | 41.3s | receive → set → spike |
| Full Match ｜ Spain vs The Netherlands ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool C_set1 | 26 | 14:32 | 14:38 | score | 3.9s | score |
| Japan 🇯🇵 vs. Czechia 🇨🇿 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 20 | 9:06 | 9:11 | receive | 18.6s | receive → set → spike |
| Japan 🇯🇵 vs. Czechia 🇨🇿 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 22 | 9:50 | 9:56 | receive | 22.2s | receive → set → spike → receive → score → receive |
| Japan 🇯🇵 vs. Czechia 🇨🇿 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 42 | 20:17 | 20:23 | receive | 32.8s | receive → set → spike → receive → score |
| Japan 🇯🇵 vs. France 🇫🇷 ｜ VNL 2025 - Full Match ｜ Week 3_set1 | 42 | 24:41 | 24:46 | receive | 35.0s | receive → set → spike → score |
| Japan 🇯🇵 vs. Netherlands 🇳🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 36 | 18:12 | 18:31 | receive | 25.2s | receive → set → spike → block → receive → set |
| Japan 🇯🇵 vs. Serbia 🇷🇸 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 1 | 0:00 | 0:04 | score | — | score |
| Korea vs. Bulgaria - Classification 5-8 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 42 | 19:54 | 19:58 | set | 42.0s | set → spike → score |
| Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 16 | 8:22 | 8:38 | receive | 70.2s | receive → set → spike → block → receive → set |
| Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 17 | 8:58 | 9:04 | receive | 106.2s | receive → set → spike → block → receive → score |
| Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 24 | 13:07 | 13:12 | receive | 25.2s | receive → set → spike → score |
| Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 27 | 14:06 | 14:12 | receive | 21.6s | receive → set → spike → block → score |
| Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 36 | 18:57 | 19:04 | receive | 21.0s | receive → set → spike → score |
| Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 37 | 19:24 | 19:40 | spike | 48.0s | spike → receive → set → spike → receive → receive |
| Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 44 | 23:00 | 23:08 | receive | 37.0s | receive → set → spike → block → score |
| Spain vs. Iran - Ranking 3-4 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 31 | 19:58 | 20:08 | receive | 35.8s | receive → set → spike → block → receive → set |
| Spain vs. Iran - Ranking 3-4 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 46 | 29:39 | 29:51 | receive | 56.8s | receive → set → spike → block → receive → set |
| Suntory Sunbirds vs. Osaka Bluteon ｜ SV.LEAGUE 2025⧸26 ｜ Full Match - Volleyball_set1 | 5 | 1:56 | 2:18 | receive | 34.5s | receive → set → spike → block → receive → set |
| Suntory Sunbirds vs. Osaka Bluteon ｜ SV.LEAGUE 2025⧸26 ｜ Full Match - Volleyball_set1 | 8 | 4:28 | 4:34 | spike | 42.7s | spike → receive → set → spike → score |
| Uzbekistan vs. Japan - Ranking 19-20 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 31 | 14:28 | 14:35 | receive | 18.6s | receive → set → spike → score |
| Uzbekistan vs. Pakistan - Ranking 5-6 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 16 | 6:35 | 6:45 | receive | 24.1s | receive → set → spike → receive → set → spike |
| ᴴᴰ114UVL預賽：：中山大學vs國北教大：：男一級 大專排球聯賽 AI網路直播_set1 | 31 | 14:16 | 14:24 | receive | 18.5s | receive → set → spike → score |
| ᴴᴰ114UVL預賽：：清華大學vs嘉義大學：：男一級 大專排球聯賽 AI網路直播_set1 | 21 | 9:14 | 9:20 | receive | 10.0s | receive → set → spike → score |
| ᴴᴰ114UVL預賽：：清華大學vs嘉義大學：：男一級 大專排球聯賽 AI網路直播_set1 | 22 | 9:36 | 9:44 | receive | 32.0s | receive → receive → set → spike → score |
| ᴴᴰ114UVL預賽：：清華大學vs嘉義大學：：男一級 大專排球聯賽 AI網路直播_set1 | 23 | 10:00 | 10:30 | receive | 56.0s | receive → set → spike → receive → receive → receive |
| ᴴᴰ114UVL預賽：：清華大學vs嘉義大學：：男一級 大專排球聯賽 AI網路直播_set1 | 26 | 11:44 | 11:48 | score | 26.4s | score |
| ᴴᴰ114UVL預賽：：清華大學vs嘉義大學：：男一級 大專排球聯賽 AI網路直播_set1 | 35 | 16:22 | 16:28 | receive | 19.6s | receive → spike → score |
| ᴴᴰ114UVL預賽：：清華大學vs嘉義大學：：男一級 大專排球聯賽 AI網路直播_set1 | 37 | 17:06 | 17:17 | receive | 21.3s | receive → set → spike → block → receive → set |
| ᴴᴰ114UVL預賽：：臺灣師大vs中山大學：：男一級 大專排球聯賽 AI網路直播_set1 | 9 | 3:26 | 3:34 | spike | 21.6s | spike → receive → set → spike → block → score |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G25 11⧸16 15_00 桃園雲豹飛將 vs 台鋼天鷹_set1 | 17 | 8:10 | 8:18 | receive | 32.2s | receive → set → spike → block → receive → set |

## 附錄 A：span 內有 2 個 serve — 22 筆

| 影片 | Rally | 起 | 訖 | serve 距起點 |
|---|---:|---:|---:|---|
| 0225小窩臨打 4 | 27 | 10:33 | 10:47 | 2.1s, 6.5s |
| 03⧸14(六) 14_00｜例行賽G103 #雲林美津濃 vs. #桃園臺灣產險｜企業21年甲級男女排球聯賽_set2 | 24 | 12:22 | 12:28 | 1.1s, 1.1s |
| 03⧸14(六) 18_00｜例行賽G105 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 5 | 1:46 | 2:08 | 0.0s, 21.4s |
| 03⧸14(六) 18_00｜例行賽G105 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 39 | 23:25 | 23:32 | 0.8s, 1.1s |
| 03⧸15(日) 13_00｜例行賽G106 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set1 | 48 | 10:28 | 10:34 | 0.4s, 0.4s |
| 03⧸15(日) 13_00｜例行賽G106 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set3 | 45 | 27:43 | 27:50 | 1.0s, 1.1s |
| 0914小窩季打 2 | 3 | 0:40 | 0:53 | 2.4s, 8.4s |
| 10⧸4 3 | 15 | 4:02 | 4:18 | 2.2s, 9.6s |
| 2025-09-27_G1_臺北伊斯特_vs_臺中連莊_set1 | 40 | 27:34 | 27:43 | 2.0s, 2.1s |
| 2025-09-28_G2_臺北伊斯特_vs_桃園雲豹飛將_set1 | 26 | 17:23 | 17:38 | 1.5s, 7.1s |
| 2025-09-28_G2_臺北伊斯特_vs_桃園雲豹飛將_set2 | 39 | 23:12 | 23:22 | 2.0s, 5.9s |
| 2025-10-12_G10_臺北伊斯特_vs_桃園雲豹飛將_set1 | 5 | 1:58 | 2:01 | 1.3s, 1.4s |
| 2025-10-25_G12_台鋼天鷹_vs_臺中連莊_set1 | 5 | 2:12 | 2:21 | 1.8s, 4.0s |
| 20251227-排島本館-3 | 1 | 0:16 | 0:24 | 2.6s, 2.6s |
| Bulgaria 🇧🇬 vs. Argentina 🇦🇷 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 22 | 10:48 | 10:54 | 0.0s, 5.3s |
| France vs. Iran - Semi Final 2 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 35 | 17:23 | 17:47 | 5.3s, 19.9s |
| Full Match ｜ Bulgaria vs Luxembourg ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set1 | 16 | 7:26 | 7:37 | 1.9s, 6.3s |
| Jtekt Stings 🇯🇵 - Suntory Sunbirds Osaka 🇯🇵 ｜ SV League 2026 ｜ Full Match - Volleyball_set1 | 19 | 10:52 | 11:00 | 1.2s, 4.5s |
| Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 34 | 17:24 | 17:30 | 1.0s, 1.0s |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G22 11⧸9 18_30 臺北伊斯特 vs 台鋼天鷹_set1 | 30 | 16:06 | 16:14 | 1.4s, 1.5s |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G22 11⧸9 18_30 臺北伊斯特 vs 台鋼天鷹_set1 | 40 | 23:02 | 23:12 | 1.7s, 1.9s |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G6 10⧸5 18_30 臺中連莊 vs 台鋼天鷹_set1 | 15 | 7:22 | 7:29 | 1.7s, 1.8s |

## 附錄 B：span 內完全沒有動作標註 — 3 筆

| 影片 | Rally | 起 | 訖 |
|---|---:|---:|---:|
| 10⧸4 3 | 29 | 8:36 | 8:42 |
| 10⧸4 3 | 33 | 10:09 | 10:15 |
| 2025-09-27_G1_臺北伊斯特_vs_臺中連莊_set1 | 42 | 28:36 | 28:40 |
