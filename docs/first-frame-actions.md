# Rally 第一幀就有動作的事件清單

掃描日期：2026-07-31 · 判定條件：`relative_frame == 0`（動作接觸幀 = rally 起始幀）

這些事件的 actor 特徵窗（±16 幀）前半段完全落在 rally 範圍外，tracking 沒有涵蓋，
候選人的助跑歷史是空的 — 監督品質天生較差。絕大多數是 serve（rally 切分從發球附近
起算，切太緊時發球接觸幀剛好是第 0 幀）。

## 人工標記（action-annotations）— 107 筆 / 8843 rallies（約 1.2%）

| 影片 | Rally | Frame | 時間 | 動作 | 球可見 |
|---|---:|---:|---:|---|:---:|
| 03⧸14(六) 16_00｜例行賽G104 #獅子王 vs. #屏東台電｜企業21年甲級男女排球聯賽_set2 | 3 | 7920 | 2:12 | serve | ✓ |
| 03⧸14(六) 16_00｜例行賽G104 #獅子王 vs. #屏東台電｜企業21年甲級男女排球聯賽_set2 | 4 | 9600 | 2:40 | serve | ✓ |
| 03⧸14(六) 18_00｜例行賽G105 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 39 | 84360 | 23:26 | serve | ✗ |
| 03⧸15(日) 13_00｜例行賽G106 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set1 | 8 | 12900 | 3:35 | serve | ✓ |
| 03⧸15(日) 13_00｜例行賽G106 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set1 | 19 | 37680 | 10:28 | serve | ✓ |
| 03⧸15(日) 13_00｜例行賽G106 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set1 | 27 | 51840 | 14:24 | serve | ✓ |
| 03⧸15(日) 13_00｜例行賽G106 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set2 | 37 | 105540 | 29:19 | serve | ✗ |
| 03⧸15(日) 13_00｜例行賽G106 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set3 | 16 | 33775 | 9:22 | serve | ✓ |
| 03⧸15(日) 13_00｜例行賽G106 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set3 | 19 | 38760 | 10:46 | serve | ✓ |
| 03⧸15(日) 13_00｜例行賽G106 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set3 | 44 | 94260 | 26:11 | serve | ✓ |
| 03⧸15(日) 13_00｜例行賽G106 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set3 | 45 | 99840 | 27:44 | serve | ✓ |
| 03⧸15(日) 17_00｜例行賽G108 #新北中纖 vs. #高雄台電｜企業21年甲級男女排球聯賽_set2 | 1 | 0 | 0:00 | serve | ✗ |
| 03⧸15(日) 17_00｜例行賽G108 #新北中纖 vs. #高雄台電｜企業21年甲級男女排球聯賽_set2 | 20 | 43073 | 11:57 | serve | ✗ |
| 03⧸15(日) 17_00｜例行賽G108 #新北中纖 vs. #高雄台電｜企業21年甲級男女排球聯賽_set2 | 21 | 47603 | 13:13 | serve | ✓ |
| 03⧸15(日) 17_00｜例行賽G108 #新北中纖 vs. #高雄台電｜企業21年甲級男女排球聯賽_set3 | 2 | 1752 | 0:29 | serve | ✓ |
| 03⧸15(日) 17_00｜例行賽G108 #新北中纖 vs. #高雄台電｜企業21年甲級男女排球聯賽_set3 | 7 | 13475 | 3:44 | receive | ✗ |
| 03⧸15(日) 17_00｜例行賽G108 #新北中纖 vs. #高雄台電｜企業21年甲級男女排球聯賽_set3 | 7 | 13475 | 3:44 | serve | ✗ |
| 03⧸15(日) 17_00｜例行賽G108 #新北中纖 vs. #高雄台電｜企業21年甲級男女排球聯賽_set3 | 13 | 26503 | 7:21 | serve | ✓ |
| 03⧸15(日) 17_00｜例行賽G108 #新北中纖 vs. #高雄台電｜企業21年甲級男女排球聯賽_set3 | 24 | 49962 | 13:52 | serve | ✓ |
| 03⧸15(日) 17_00｜例行賽G108 #新北中纖 vs. #高雄台電｜企業21年甲級男女排球聯賽_set5 | 1 | 0 | 0:00 | serve | ✓ |
| 03⧸15(日) 17_00｜例行賽G108 #新北中纖 vs. #高雄台電｜企業21年甲級男女排球聯賽_set5 | 8 | 11400 | 3:10 | serve | ✓ |
| 03⧸15(日) 17_00｜例行賽G108 #新北中纖 vs. #高雄台電｜企業21年甲級男女排球聯賽_set5 | 27 | 72359 | 20:05 | serve | ✓ |
| 03⧸15(日) 17_00｜例行賽G108 #新北中纖 vs. #高雄台電｜企業21年甲級男女排球聯賽_set5 | 29 | 78765 | 21:52 | serve | ✓ |
| 03⧸20(五) 16_00｜挑戰賽G111 #屏東台電 vs. #桃園臺產｜企業21年甲級男女排球聯賽_set1 | 1 | 0 | 0:00 | serve | ✓ |
| 03⧸21(六) 18_00｜男子組冠軍賽G114 #雲林美津濃 vs. #屏東台電｜企業21年甲級男女排球聯賽_set1 | 22 | 49200 | 13:40 | serve | ✓ |
| 0518小窩季打 1 | 1 | 240 | 0:08 | set | ✗ |
| 0518小窩季打 2 | 1 | 120 | 0:04 | serve | ✗ |
| 0518小窩季打 3 | 19 | 10440 | 5:48 | serve | ✗ |
| 20241103 霖度C-1 | 35 | 24540 | 13:38 | serve | ✗ |
| 20241103 霖度C-1 | 36 | 24960 | 13:52 | serve | ✗ |
| 20241103 霖度C-1 | 37 | 26400 | 14:40 | serve | ✗ |
| 20241103 霖度C-2 | 16 | 11040 | 6:08 | serve | ✗ |
| 20241103 霖度C-2 | 20 | 13200 | 7:20 | serve | ✗ |
| 20241103 霖度C-2 | 26 | 16680 | 9:16 | serve | ✗ |
| 2025-09-28_G2_臺北伊斯特_vs_桃園雲豹飛將_set2 | 10 | 15927 | 4:25 | serve | ✓ |
| 2025-10-04_G4_臺北伊斯特_vs_台鋼天鷹_set1 | 9 | 15038 | 4:10 | serve | ✗ |
| 2025-10-12_G9_臺中連莊_vs_台鋼天鷹_set1 | 9 | 13401 | 3:43 | serve | ✗ |
| 20250413 霖度C-5 | 39 | 29970 | 16:39 | serve | ✗ |
| 20250420 霖度C-1 | 27 | 19500 | 10:50 | serve | ✗ |
| 20250424 排島惡館-6 | 10 | 6234 | 3:28 | serve | ✗ |
| 20250424 排島惡館-7 | 22 | 15405 | 8:34 | serve | ✓ |
| 20250424 排島惡館-7 | 25 | 17203 | 9:34 | serve | ✗ |
| 20250424 排島惡館-7 | 35 | 22358 | 12:26 | serve | ✗ |
| 20250424 排島惡館-7 | 36 | 23017 | 12:48 | serve | ✗ |
| 20250628-霖度C-1 | 40 | 23590 | 13:07 | serve | ✗ |
| 20251227-排島本館-3 | 34 | 18462 | 10:16 | serve | ✗ |
| 20260321-排島本館-01 | 9 | 5657 | 3:08 | serve | ✓ |
| 20260321-排島本館-01 | 10 | 6387 | 3:33 | serve | ✗ |
| 20260426-小窩-01 | 17 | 12720 | 7:04 | serve | ✗ |
| 20260426-小窩-03 | 23 | 16304 | 9:04 | serve | ✗ |
| 20260510邷力豹臨打1 | 25 | 23137 | 12:52 | serve | ✗ |
| 20260510邷力豹臨打1 | 31 | 26434 | 14:42 | serve | ✓ |
| 37-39 Thriller! - Japan 🇯🇵 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 1 | 100 | 0:04 | serve | ✗ |
| 37-39 Thriller! - Japan 🇯🇵 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 30 | 17950 | 11:58 | serve | ✓ |
| 37-39 Thriller! - Japan 🇯🇵 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 35 | 21100 | 14:04 | serve | ✗ |
| 37-39 Thriller! - Japan 🇯🇵 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 37 | 22400 | 14:56 | serve | ✓ |
| 37-39 Thriller! - Japan 🇯🇵 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set2 | 10 | 5450 | 3:38 | serve | ✓ |
| 37-39 Thriller! - Japan 🇯🇵 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set2 | 35 | 22900 | 15:16 | serve | ✓ |
| Brazil 🇧🇷 vs. Cuba 🇨🇺 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 2 | 1200 | 0:24 | serve | ✓ |
| Brazil 🇧🇷 vs. Cuba 🇨🇺 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 47 | 84900 | 28:18 | serve | ✗ |
| Brazil 🇧🇷 vs. Italy 🇮🇹  ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 22 | 27250 | 9:05 | serve | ✓ |
| Brazil 🇧🇷 vs. Italy 🇮🇹  ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 26 | 34683 | 11:33 | serve | ✗ |
| Brazil 🇧🇷 vs. Italy 🇮🇹  ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 29 | 38150 | 12:43 | serve | ✓ |
| Bulgaria 🇧🇬 vs. Argentina 🇦🇷 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 8 | 4602 | 3:04 | serve | ✗ |
| Bulgaria 🇧🇬 vs. Argentina 🇦🇷 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 11 | 6650 | 4:26 | serve | ✗ |
| Bulgaria 🇧🇬 vs. Argentina 🇦🇷 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 15 | 9325 | 6:13 | serve | ✓ |
| Bulgaria 🇧🇬 vs. Argentina 🇦🇷 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 22 | 16200 | 10:48 | serve | ✓ |
| Bulgaria 🇧🇬 vs. Argentina 🇦🇷 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 34 | 26650 | 17:46 | serve | ✓ |
| Bulgaria 🇧🇬 vs. Canada 🇨🇦 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 11 | 11325 | 7:33 | serve | ✓ |
| Bulgaria 🇧🇬 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 22 | 17225 | 11:29 | serve | ✓ |
| Bulgaria 🇧🇬 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 37 | 28350 | 18:54 | serve | ✓ |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 2 | 750 | 0:30 | serve | ✗ |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 7 | 4150 | 2:46 | serve | ✓ |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 8 | 4900 | 3:16 | serve | ✗ |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 20 | 12800 | 8:32 | serve | ✗ |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 33 | 25143 | 16:45 | receive | ✗ |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 43 | 36162 | 24:06 | serve | ✓ |
| China 🇨🇳 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 11 | 8200 | 5:28 | serve | ✗ |
| France vs. Iran - Semi Final 2 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 25 | 20640 | 11:28 | serve | ✗ |
| France 🇫🇷 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 5 | 3200 | 2:08 | serve | ✗ |
| France 🇫🇷 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 45 | 36150 | 24:06 | serve | ✓ |
| Full Match ｜ Poland vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 1 | 120 | 0:04 | serve | ✗ |
| Full Match ｜ Poland vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 2 | 1020 | 0:34 | serve | ✗ |
| Full Match ｜ Poland vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 14 | 11220 | 6:14 | receive | ✓ |
| Full Match ｜ Poland vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 34 | 32220 | 17:54 | serve | ✓ |
| Japan 🇯🇵 vs. China 🇨🇳 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 18 | 13300 | 8:52 | serve | ✗ |
| Japan 🇯🇵 vs. Czechia 🇨🇿 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 26 | 17050 | 11:22 | serve | ✓ |
| Japan 🇯🇵 vs. Dominican Republic 🇩🇴 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 10 | 6400 | 4:16 | serve | ✗ |
| Japan 🇯🇵 vs. Dominican Republic 🇩🇴 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 12 | 7650 | 5:06 | serve | ✓ |
| Japan 🇯🇵 vs. Dominican Republic 🇩🇴 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 15 | 9850 | 6:34 | serve | ✗ |
| Japan 🇯🇵 vs. Dominican Republic 🇩🇴 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 21 | 14650 | 9:46 | serve | ✗ |
| Japan 🇯🇵 vs. Dominican Republic 🇩🇴 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 25 | 17350 | 11:34 | serve | ✗ |
| Japan 🇯🇵 vs. Dominican Republic 🇩🇴 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 29 | 20200 | 13:28 | serve | ✗ |
| Japan 🇯🇵 vs. Dominican Republic 🇩🇴 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 37 | 28050 | 18:42 | serve | ✗ |
| Japan 🇯🇵 vs. Dominican Republic 🇩🇴 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 43 | 34950 | 23:18 | serve | ✓ |
| Japan 🇯🇵 vs. Netherlands 🇳🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 14 | 10000 | 6:40 | serve | ✓ |
| Japan 🇯🇵 vs. Netherlands 🇳🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 30 | 22100 | 14:44 | serve | ✗ |
| Poland vs. France - Final ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 4 | 5700 | 1:35 | serve | ✗ |
| Poland vs. France - Final ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 38 | 59340 | 16:29 | serve | ✓ |
| Spain vs. Iran - Ranking 3-4 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 31 | 71880 | 19:58 | receive | ✗ |
| ᴴᴰ114UVL預賽：：陽明交大vs臺灣體大：：男一級 大專排球聯賽 AI網路直播_set1 | 17 | 13860 | 7:42 | serve | ✗ |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G24 11⧸15 18_30 桃園雲豹飛將 vs 臺北伊斯特_set1 | 33 | 56630 | 15:44 | receive | ✗ |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G26 11⧸16 18_30 臺中連莊 vs 臺北伊斯特_set1 | 13 | 27009 | 7:30 | serve | ✗ |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G26 11⧸16 18_30 臺中連莊 vs 臺北伊斯特_set1 | 32 | 64454 | 17:55 | serve | ✗ |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G26 11⧸16 18_30 臺中連莊 vs 臺北伊斯特_set1 | 43 | 92260 | 25:39 | serve | ✓ |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G26 11⧸16 18_30 臺中連莊 vs 臺北伊斯特_set1 | 55 | 123916 | 34:27 | serve | ✗ |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G26 11⧸16 18_30 臺中連莊 vs 臺北伊斯特_set1 | 63 | 142759 | 39:41 | serve | ✓ |

### 特別注意

- 有 3 個 rally 的動作落在**影片第 0 幀**（影片就是從發球瞬間切出）：
  `03⧸15(日) 17_00 G108_set2` rally 1、同場 `_set5` rally 1、`03⧸20(五) 16_00 G111_set1` rally 1
- `03⧸15(日) 17_00 G108_set3` **rally 7 在同一幀 13475 同時標了 serve 和 receive**，
  疑似標記異常，建議回看
  （id：`act_926acbcd063f4ad8` / `act_42267ee694ad4ad3`）

## 預測標記（pre-annotations）— 12 筆 / 6722 rallies（約 0.2%）

| 影片 | Rally | Frame | 時間 | 動作 | 球可見 |
|---|---:|---:|---:|---|:---:|
| 03⧸20(五) 14_00｜挑戰賽G110 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 23 | 44100 | 12:15 | serve | ✓ |
| 0518小窩季打 3 | 19 | 10440 | 5:48 | serve | ✓ |
| 20250413 霖度C-6 | 27 | 21420 | 11:54 | serve | ✓ |
| 20250621 排島本館-1 | 8 | 4740 | 2:38 | receive | ✓ |
| 20250628-霖度C-2 | 26 | 16830 | 9:21 | serve | ✓ |
| 20251227-排島本館-1 | 29 | 15105 | 8:24 | serve | ✓ |
| 20260321-排島本館-03 | 35 | 21219 | 11:48 | serve | ✓ |
| 20260426-小窩-01 | 17 | 12720 | 7:04 | serve | ✓ |
| 20260507 工資管友誼賽3 | 55 | 37980 | 21:06 | serve | ✓ |
| France 🇫🇷 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 45 | 36150 | 24:06 | serve | ✓ |
| Poland vs. France - Final ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 38 | 59340 | 16:29 | serve | ✓ |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G26 11⧸16 18_30 臺中連莊 vs 臺北伊斯特_set1 | 43 | 92260 | 25:39 | serve | ✓ |
