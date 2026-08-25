# Rally 開頭沒有 serve 的清單

掃描日期：2026-08-25 · 資料：`videos/rally-spot/annotations` × `videos/action/annotations`（197 支影片、8,721 rallies）

重跑：`uv run python scripts/scan_rally_edges.py`

判定：rally span `[start, end]` 內最前的動作事件不是 `serve`。共 56 筆。

分類：`開頭切太晚` = span 外 3 秒內就有 `serve`，標註在、是邊界偏了；`serve 不在最前` = span 內有 `serve`，但前面還有別的動作；`疑似漏標` = 前後都找不到鄰近的 `serve`；`導播問題` = 已經看過畫面，導播沒拍到那個 `serve`，補不了。

## 疑似漏標 — 45 筆

| 影片 | Rally | 起 | 訖 | 最前動作 | 動作序列（前 6） |
|---|---:|---:|---:|---|---|
| 37-39 Thriller! - Japan 🇯🇵 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set2 | 12 | 5:07 | 5:09 | spike | spike → score |
| 37-39 Thriller! - Japan 🇯🇵 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set2 | 13 | 5:21 | 5:28 | receive | receive → set → spike → receive → set → spike |
| 37-39 Thriller! - Japan 🇯🇵 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set2 | 20 | 8:52 | 8:55 | receive | receive → set → spike → score |
| Brazil 🇧🇷 vs. Italy 🇮🇹  ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 9 | 3:14 | 3:16 | score | score |
| Brazil 🇧🇷 vs. Italy 🇮🇹  ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 10 | 3:38 | 3:42 | set | set → spike → score |
| Bulgaria 🇧🇬 vs. Argentina 🇦🇷 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 23 | 11:13 | 11:14 | score | score |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 33 | 16:45 | 16:50 | receive | receive → set → spike → block → score |
| China 🇨🇳 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 9 | 4:52 | 4:59 | set | set → spike → block → score |
| France vs. Iran - Semi Final 2 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 24 | 11:05 | 11:10 | set | set → spike → score |
| Full Match ｜ Poland vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 37 | 19:36 | 19:40 | receive | receive → set → spike → block → score |
| Full Match ｜ Poland vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 41 | 21:42 | 21:45 | score | score |
| Full Match ｜ Poland vs. Slovakia ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 21 | 9:12 | 9:22 | set | set → receive → receive → set → spike → receive |
| Full Match ｜ Serbia vs. Bulgaria ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set1 | 10 | 3:50 | 4:03 | set | set → spike → block → receive → receive → set |
| Full Match ｜ Slovakia vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 6 | 2:42 | 2:52 | receive | receive → receive → receive → receive → set → spike |
| Full Match ｜ Slovakia vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 40 | 21:58 | 22:06 | receive | receive → set → spike |
| Full Match ｜ Spain vs The Netherlands ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool C_set1 | 26 | 14:32 | 14:37 | score | score |
| Japan 🇯🇵 vs. Czechia 🇨🇿 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 20 | 9:06 | 9:11 | receive | receive → set → spike |
| Japan 🇯🇵 vs. Czechia 🇨🇿 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 22 | 9:50 | 9:56 | receive | receive → set → spike → receive → score → receive |
| Japan 🇯🇵 vs. Czechia 🇨🇿 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 42 | 20:17 | 20:22 | receive | receive → set → spike → receive → score |
| Japan 🇯🇵 vs. France 🇫🇷 ｜ VNL 2025 - Full Match ｜ Week 3_set1 | 42 | 24:41 | 24:46 | receive | receive → set → spike → score |
| Japan 🇯🇵 vs. Netherlands 🇳🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 36 | 18:12 | 18:30 | receive | receive → set → spike → block → receive → set |
| Japan 🇯🇵 vs. Serbia 🇷🇸 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 1 | 0:00 | 0:04 | score | score |
| Korea vs. Bulgaria - Classification 5-8 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 42 | 19:54 | 19:57 | set | set → spike → score |
| Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 16 | 8:22 | 8:37 | receive | receive → set → spike → block → receive → set |
| Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 17 | 8:58 | 9:03 | receive | receive → set → spike → block → receive → score |
| Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 24 | 13:07 | 13:11 | receive | receive → set → spike → score |
| Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 27 | 14:05 | 14:10 | receive | receive → set → spike → block → score |
| Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 36 | 18:57 | 19:03 | receive | receive → set → spike → score |
| Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 37 | 19:24 | 19:38 | spike | spike → receive → set → spike → receive → receive |
| Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 44 | 23:00 | 23:06 | receive | receive → set → spike → block → score |
| Spain vs. Iran - Ranking 3-4 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 31 | 19:58 | 20:07 | receive | receive → set → spike → block → receive → set |
| Spain vs. Iran - Ranking 3-4 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 46 | 29:39 | 29:49 | receive | receive → set → spike → block → receive → set |
| Suntory Sunbirds vs. Osaka Bluteon ｜ SV.LEAGUE 2025⧸26 ｜ Full Match - Volleyball_set1 | 5 | 1:56 | 2:18 | receive | receive → set → spike → block → receive → set |
| Suntory Sunbirds vs. Osaka Bluteon ｜ SV.LEAGUE 2025⧸26 ｜ Full Match - Volleyball_set1 | 8 | 4:28 | 4:34 | spike | spike → receive → set → spike → score |
| Uzbekistan vs. Japan - Ranking 19-20 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 31 | 14:28 | 14:35 | receive | receive → set → spike → score |
| Uzbekistan vs. Pakistan - Ranking 5-6 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 16 | 6:35 | 6:44 | receive | receive → set → spike → receive → set → spike |
| ᴴᴰ114UVL預賽：：中山大學vs國北教大：：男一級 大專排球聯賽 AI網路直播_set1 | 31 | 14:16 | 14:23 | receive | receive → set → spike → score |
| ᴴᴰ114UVL預賽：：清華大學vs嘉義大學：：男一級 大專排球聯賽 AI網路直播_set1 | 21 | 9:14 | 9:20 | receive | receive → set → spike → score |
| ᴴᴰ114UVL預賽：：清華大學vs嘉義大學：：男一級 大專排球聯賽 AI網路直播_set1 | 22 | 9:36 | 9:44 | receive | receive → receive → set → spike → score |
| ᴴᴰ114UVL預賽：：清華大學vs嘉義大學：：男一級 大專排球聯賽 AI網路直播_set1 | 23 | 10:00 | 10:30 | receive | receive → set → spike → receive → receive → receive |
| ᴴᴰ114UVL預賽：：清華大學vs嘉義大學：：男一級 大專排球聯賽 AI網路直播_set1 | 26 | 11:44 | 11:48 | score | score |
| ᴴᴰ114UVL預賽：：清華大學vs嘉義大學：：男一級 大專排球聯賽 AI網路直播_set1 | 35 | 16:22 | 16:28 | receive | receive → spike → score |
| ᴴᴰ114UVL預賽：：清華大學vs嘉義大學：：男一級 大專排球聯賽 AI網路直播_set1 | 37 | 17:06 | 17:16 | receive | receive → set → spike → block → receive → set |
| ᴴᴰ114UVL預賽：：臺灣師大vs中山大學：：男一級 大專排球聯賽 AI網路直播_set1 | 9 | 3:26 | 3:34 | spike | spike → receive → set → spike → block → score |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G25 11⧸16 15_00 桃園雲豹飛將 vs 台鋼天鷹_set1 | 17 | 8:10 | 8:18 | receive | receive → set → spike → block → receive → set |

## 導播問題 — 11 筆

看過畫面了：轉播切走（重播、觀眾、板凳），`serve` 不在帶子上。這批不是漏標，標不出來，留著只是為了下次掃描不用再看一遍。

| 影片 | Rally | 起 | 訖 | 最前動作 | 動作序列（前 6） |
|---|---:|---:|---:|---|---|
| 03⧸14(六) 16_00｜例行賽G104 #獅子王 vs. #屏東台電｜企業21年甲級男女排球聯賽_set2 | 27 | 15:37 | 15:42 | receive | receive → set → spike → score |
| 03⧸14(六) 18_00｜例行賽G105 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 1 | 0:00 | 0:06 | receive | receive → set → spike |
| 03⧸14(六) 18_00｜例行賽G105 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 36 | 22:00 | 22:04 | receive | receive → set → spike → block → score |
| 03⧸15(日) 13_00｜例行賽G106 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set3 | 41 | 24:48 | 24:50 | set | set → spike → block → score |
| 03⧸15(日) 17_00｜例行賽G108 #新北中纖 vs. #高雄台電｜企業21年甲級男女排球聯賽_set3 | 7 | 3:43 | 3:52 | set | set → spike → receive → set → spike → block |
| 03⧸20(五) 14_00｜挑戰賽G110 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set2 | 6 | 2:24 | 3:31 | receive | receive → set → spike → receive → set → spike |
| 03⧸21(六) 18_00｜男子組冠軍賽G114 #雲林美津濃 vs. #屏東台電｜企業21年甲級男女排球聯賽_set1 | 18 | 10:06 | 10:15 | block | block → block → receive → set → receive → receive |
| 03⧸22(日) 17_00｜男子組冠軍賽 G117 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set1 | 30 | 16:33 | 16:38 | set | set → spike → block → score |
| 03⧸22(日) 17_00｜男子組冠軍賽 G117 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set2 | 2 | 0:25 | 0:36 | set | set → spike → receive → set → block → spike |
| 2025-10-11_G8_臺中連莊_vs_桃園雲豹飛將_set1 | 1 | 0:05 | 0:07 | set | set → spike → score |
| 37-39 Thriller! - Japan 🇯🇵 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 31 | 12:39 | 12:43 | receive | receive → set → spike → receive → score |

## 附錄 A：span 內有 2 個以上 `serve` — 72 筆

間隔不到 0.5 秒的，是同一個 `serve` 被標了兩次，不是兩個。

| 影片 | Rally | 起 | 訖 | serve 數 | 最小間隔 |
|---|---:|---:|---:|---:|---:|
| 20260321-排島本館-01 | 12 | 4:28 | 4:37 | 2 | 0.23s |
| 20260321-排島本館-01 | 20 | 7:01 | 7:27 | 2 | 0.40s |
| 20260321-排島本館-01 | 25 | 9:19 | 9:46 | 2 | 0.20s |
| 20260426-小窩-03 | 23 | 9:02 | 9:20 | 2 | 0.13s |
| 20260426-小窩-03 | 27 | 10:49 | 11:00 | 3 | 0.03s |
| 20260426-小窩-03 | 32 | 12:44 | 12:52 | 2 | 0.07s |
| 20260502-排島本館-01 | 20 | 7:44 | 7:52 | 4 | 0.00s |
| 20260502-排島本館-01 | 44 | 16:03 | 16:10 | 2 | 5.64s |
| 20260510邷力豹臨打1 | 11 | 6:22 | 6:39 | 2 | 0.10s |
| 20260510邷力豹臨打1 | 19 | 9:50 | 10:04 | 6 | 0.00s |
| 20260510邷力豹臨打1 | 25 | 12:50 | 13:05 | 2 | 0.47s |
| 20260510邷力豹臨打1 | 30 | 14:10 | 14:22 | 4 | 0.00s |
| 20260510邷力豹臨打1 | 31 | 14:40 | 14:45 | 2 | 0.03s |
| 20260510邷力豹臨打1 | 33 | 15:17 | 15:34 | 2 | 0.53s |
| 37-39 Thriller! - Japan 🇯🇵 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 2 | 0:20 | 0:37 | 2 | 1.48s |
| 37-39 Thriller! - Japan 🇯🇵 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 16 | 6:02 | 6:05 | 3 | 0.00s |
| 37-39 Thriller! - Japan 🇯🇵 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set2 | 10 | 3:36 | 3:49 | 2 | 0.24s |
| 37-39 Thriller! - Japan 🇯🇵 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set2 | 35 | 15:14 | 15:34 | 2 | 0.12s |
| Brazil 🇧🇷 vs. Cuba 🇨🇺 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 5 | 1:56 | 1:59 | 2 | 0.06s |
| Brazil 🇧🇷 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 15 | 7:32 | 7:56 | 2 | 0.36s |
| Bulgaria 🇧🇬 vs. Argentina 🇦🇷 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 6 | 2:16 | 2:20 | 2 | 0.64s |
| Bulgaria 🇧🇬 vs. Argentina 🇦🇷 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 10 | 4:02 | 4:06 | 4 | 0.00s |
| Bulgaria 🇧🇬 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 22 | 11:27 | 11:33 | 2 | 0.24s |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 2 | 0:27 | 0:46 | 2 | 0.52s |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 5 | 1:58 | 2:01 | 4 | 0.00s |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 7 | 2:44 | 2:55 | 2 | 0.12s |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 8 | 3:14 | 3:20 | 3 | 0.04s |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 20 | 8:30 | 8:40 | 2 | 0.36s |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 26 | 13:41 | 13:46 | 4 | 0.00s |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 37 | 19:10 | 19:17 | 2 | 0.44s |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 43 | 24:04 | 24:07 | 2 | 0.04s |
| China 🇨🇳 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 11 | 5:26 | 5:41 | 2 | 0.12s |
| China 🇨🇳 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 34 | 15:22 | 15:27 | 2 | 0.48s |
| France 🇫🇷 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 12 | 5:08 | 5:25 | 2 | 0.40s |
| Full Match ｜ Poland vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 1 | 0:01 | 0:15 | 2 | 0.70s |
| Full Match ｜ Poland vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 2 | 0:31 | 0:42 | 2 | 0.83s |
| Full Match ｜ Poland vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 4 | 1:24 | 1:32 | 2 | 0.10s |
| Full Match ｜ Poland vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 19 | 9:30 | 9:45 | 2 | 0.30s |
| Full Match ｜ Poland vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 34 | 17:52 | 18:02 | 2 | 0.27s |
| Full Match ｜ Poland vs. Slovakia ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 11 | 4:48 | 4:55 | 2 | 0.33s |
| Japan 🇯🇵 vs. China 🇨🇳 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 15 | 7:33 | 7:37 | 2 | 0.12s |
| Japan 🇯🇵 vs. China 🇨🇳 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 18 | 8:50 | 8:58 | 2 | 0.28s |
| Japan 🇯🇵 vs. Czechia 🇨🇿 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 26 | 11:20 | 11:25 | 2 | 0.04s |
| Japan 🇯🇵 vs. Dominican Republic 🇩🇴 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 10 | 4:14 | 4:26 | 2 | 0.28s |
| Japan 🇯🇵 vs. Dominican Republic 🇩🇴 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 12 | 5:04 | 5:11 | 2 | 0.04s |
| Japan 🇯🇵 vs. Dominican Republic 🇩🇴 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 15 | 6:32 | 6:38 | 2 | 0.20s |
| Japan 🇯🇵 vs. Dominican Republic 🇩🇴 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 20 | 9:19 | 9:27 | 2 | 0.68s |
| Japan 🇯🇵 vs. Dominican Republic 🇩🇴 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 21 | 9:43 | 9:51 | 2 | 0.56s |
| Japan 🇯🇵 vs. Dominican Republic 🇩🇴 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 25 | 11:32 | 11:42 | 2 | 0.44s |
| Japan 🇯🇵 vs. Dominican Republic 🇩🇴 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 27 | 12:34 | 12:37 | 3 | 0.00s |
| Japan 🇯🇵 vs. Dominican Republic 🇩🇴 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 29 | 13:26 | 13:34 | 2 | 0.44s |
| Japan 🇯🇵 vs. Dominican Republic 🇩🇴 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 31 | 14:34 | 14:41 | 2 | 0.36s |
| Japan 🇯🇵 vs. Dominican Republic 🇩🇴 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 37 | 18:40 | 18:51 | 2 | 0.24s |
| Japan 🇯🇵 vs. Dominican Republic 🇩🇴 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 43 | 23:16 | 23:22 | 2 | 0.08s |
| Japan 🇯🇵 vs. Netherlands 🇳🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 14 | 6:38 | 6:45 | 2 | 0.28s |
| Japan 🇯🇵 vs. Netherlands 🇳🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 30 | 14:41 | 14:47 | 2 | 0.56s |
| Japan 🇯🇵 vs. Serbia 🇷🇸 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 20 | 8:20 | 8:27 | 2 | 0.08s |
| Poland vs. France - Final ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 47 | 22:14 | 22:22 | 2 | 0.35s |
| Suntory Sunbirds vs. Osaka Bluteon ｜ SV.LEAGUE 2025⧸26 ｜ Full Match - Volleyball_set1 | 4 | 1:20 | 1:41 | 3 | 0.00s |
| Suntory Sunbirds vs. Osaka Bluteon ｜ SV.LEAGUE 2025⧸26 ｜ Full Match - Volleyball_set1 | 25 | 11:42 | 11:49 | 2 | 0.28s |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G26 11⧸16 18_30 臺中連莊 vs 臺北伊斯特_set1 | 13 | 7:28 | 7:36 | 2 | 0.28s |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G26 11⧸16 18_30 臺中連莊 vs 臺北伊斯特_set1 | 32 | 17:53 | 17:59 | 2 | 0.40s |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G26 11⧸16 18_30 臺中連莊 vs 臺北伊斯特_set1 | 43 | 25:37 | 25:44 | 2 | 0.02s |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G26 11⧸16 18_30 臺中連莊 vs 臺北伊斯特_set1 | 63 | 39:40 | 39:46 | 2 | 0.08s |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G26 11⧸16 18_30 臺中連莊 vs 臺北伊斯特_set1 | 69 | 45:06 | 45:12 | 2 | 0.02s |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G26 11⧸16 18_30 臺中連莊 vs 臺北伊斯特_set1 | 71 | 46:09 | 46:18 | 2 | 0.40s |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G27 11⧸22 15_00 桃園雲豹飛將 vs 台中連莊_set1 | 20 | 9:52 | 9:59 | 5 | 0.00s |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G30 11⧸23 18_30 桃園雲豹飛將 vs 台鋼天鷹_set1 | 9 | 5:28 | 5:41 | 2 | 0.27s |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G30 11⧸23 18_30 桃園雲豹飛將 vs 台鋼天鷹_set1 | 12 | 8:37 | 8:50 | 4 | 0.00s |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G30 11⧸23 18_30 桃園雲豹飛將 vs 台鋼天鷹_set1 | 13 | 9:11 | 9:18 | 2 | 0.15s |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G30 11⧸23 18_30 桃園雲豹飛將 vs 台鋼天鷹_set1 | 21 | 13:47 | 13:54 | 2 | 0.02s |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G30 11⧸23 18_30 桃園雲豹飛將 vs 台鋼天鷹_set1 | 42 | 28:26 | 28:33 | 2 | 0.07s |
