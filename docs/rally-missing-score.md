# Rally 結尾沒有 score 的清單

掃描日期：2026-08-28 · 資料：`videos/rally-spot/annotations` × `videos/action/annotations`（200 支影片、8,860 rallies）

重跑：`uv run python scripts/scan_rally_edges.py`

判定：rally span `[start, end]` 內最後的動作事件不是 `score`。共 136 筆。

分類：`結尾切太早` = span 外 3 秒內就有 `score`，標註在、是邊界偏了；`score 不在最後` = span 內有 `score`，但後面還有別的動作；`疑似漏標` = 前後都找不到鄰近的 `score`；`看過畫面的判定` = 導播沒拍到那個 `score`，或這球是犯規結束、本來就沒有 `score` 可標。

## 看過畫面的判定 — 122 筆

看過畫面了，這批不是漏標：`導播問題` = 轉播切走（重播、觀眾、板凳），`score` 不在帶子上；其他判定是犯規結束的球——觸網、越界、持球…是裁判的哨音不是觸球，所以沒有 `score` 可標。留著是為了下次掃描不用再看一遍。

判定分布：觸網 60、落地後 17、打到標竿 9、後排踩線 6、持球 6、越界 6、舉球後排越界 3、越網擊球 3、發球踩線 3、二擊 3、no in 2、標竿外 1、公正 1、阻擋舉球 1、越界救球 1。

| 判定 | 筆數 | 意思 |
|---|---:|---|
| 觸網 | 60 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| 落地後 | 17 | 球已經落地得分，之後球員又碰到球；最後那個觸球是落地後的多餘動作，不是 score。 |
| 打到標竿 | 9 | 球打到標竿出界，犯規結束。 |
| 後排踩線 | 6 | 後排球員踩線起跳攻擊或攔網，位置違例。 |
| 持球 | 6 | 持球（catch / hold）犯規。 |
| 越界 | 6 | 球出界或球員越過中線，裁判判定結束。 |
| 舉球後排越界 | 3 | 後排球員在前區把球舉過網，位置違例。 |
| 越網擊球 | 3 | 越過球網擊球（reaching over）犯規。 |
| 發球踩線 | 3 | 發球踩線，發球方直接失分；rally 沒有正常展開。 |
| 二擊 | 3 | 連擊（double contact）犯規。 |
| no in | 2 | 球沒進——未過網或落在界外（標註員原文 no in）。 |
| 標竿外 | 1 | 球從標竿外側過網，視同出界。 |
| 公正 | 1 | 裁判／挑戰判決結束這球（標註員原文「公正」）。 |
| 阻擋舉球 | 1 | 攔網時觸碰到對方正在舉球的球，犯規。 |
| 越界救球 | 1 | 救球時越過中線或從界外把球救回，判犯規。 |

| 影片 | Rally | 起 | 訖 | 最後動作 | 判定 | 為什麼最後不是 score |
|---|---:|---:|---:|---|---|---|
| Full Match ｜ Türkiye vs. Spain ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool D_set1 | 40 | 21:22 | 21:38 | receive | no in | 球沒進——未過網或落在界外（標註員原文 no in）。 |
| Poland vs. Spain - Semi Final 1 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 35 | 19:57 | 20:06 | receive | no in | 球沒進——未過網或落在界外（標註員原文 no in）。 |
| Champions crowned in Final 24⧸25 (2⧸2) ｜ Suntory Sunbirds Osaka - Stings Aichi ｜ SV League 24⧸25_set1 | 18 | 8:18 | 8:29 | spike | 二擊 | 連擊（double contact）犯規。 |
| Full Match ｜ Croatia vs. Bulgaria ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set1 | 24 | 10:02 | 10:14 | spike | 二擊 | 連擊（double contact）犯規。 |
| Türkiye vs. Colombia - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 9 | 5:33 | 5:41 | receive | 二擊 | 連擊（double contact）犯規。 |
| 0420小窩季打 1 | 43 | 14:50 | 14:54 | receive | 公正 | 裁判／挑戰判決結束這球（標註員原文「公正」）。 |
| 0104排島臨打 3 | 34 | 12:52 | 13:06 | set | 後排踩線 | 後排球員踩線起跳攻擊或攔網，位置違例。 |
| 0420小窩季打 2 | 11 | 3:38 | 3:58 | receive | 後排踩線 | 後排球員踩線起跳攻擊或攔網，位置違例。 |
| 0420小窩季打 3 | 15 | 4:12 | 4:20 | spike | 後排踩線 | 後排球員踩線起跳攻擊或攔網，位置違例。 |
| Bulgaria 🇧🇬 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 14 | 6:04 | 6:12 | receive | 後排踩線 | 後排球員踩線起跳攻擊或攔網，位置違例。 |
| Final - Stings vs. Sunbirds ｜ SVL League 2024⧸25 - Full Match ｜ Volleyball_set1 | 29 | 18:43 | 18:52 | receive | 後排踩線 | 後排球員踩線起跳攻擊或攔網，位置違例。 |
| Semi Final 1 - Osaka Bluteon vs. Stings Aichi ｜ SV League - Full Match ｜ Volleyball_set1 | 18 | 8:57 | 9:04 | block | 後排踩線 | 後排球員踩線起跳攻擊或攔網，位置違例。 |
| 03⧸20(五) 18_00｜挑戰賽G112 #臺北國北獅 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set1 | 20 | 10:04 | 10:16 | block | 打到標竿 | 球打到標竿出界，犯規結束。 |
| 2025-10-25_G11_臺北伊斯特_vs_桃園雲豹飛將_set1 | 26 | 14:08 | 14:16 | receive | 打到標竿 | 球打到標竿出界，犯規結束。 |
| 2025-11-08_G19_臺北伊斯特_vs_臺中連莊_set1 | 47 | 29:31 | 29:37 | receive | 打到標竿 | 球打到標竿出界，犯規結束。 |
| China vs. Brazil - Ranking 13-14 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 19 | 7:53 | 8:04 | receive | 打到標竿 | 球打到標竿出界，犯規結束。 |
| Full Match ｜ Poland vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 4 | 1:24 | 1:32 | receive | 打到標竿 | 球打到標竿出界，犯規結束。 |
| Japan 🇯🇵 vs. France 🇫🇷 ｜ VNL 2025 - Full Match ｜ Week 3_set1 | 36 | 17:57 | 18:28 | block | 打到標竿 | 球打到標竿出界，犯規結束。 |
| Suntory Sunbirds 🇯🇵 vs. Stings AICHI 🇯🇵 ｜ SV League 2026 ｜ Full Match - Japan Volleyball_set1 | 30 | 14:25 | 14:37 | receive | 打到標竿 | 球打到標竿出界，犯規結束。 |
| ᴴᴰ114UVL預賽：：中原大學vs陽明交大：：男一級 大專排球聯賽 AI網路直播_set1 | 24 | 11:55 | 12:04 | receive | 打到標竿 | 球打到標竿出界，犯規結束。 |
| ᴴᴰ114UVL預賽：：中山大學vs國北教大：：男一級 大專排球聯賽 AI網路直播_set1 | 35 | 15:42 | 15:57 | block | 打到標竿 | 球打到標竿出界，犯規結束。 |
| 03⧸20(五) 14_00｜挑戰賽G110 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 42 | 28:31 | 28:40 | spike | 持球 | 持球（catch / hold）犯規。 |
| 2025-10-05_G5_臺北伊斯特_vs_桃園雲豹飛將_set1 | 16 | 7:45 | 7:59 | receive | 持球 | 持球（catch / hold）犯規。 |
| Full Match ｜ Luxembourg vs. Croatia - CEV U22 Volleyball European Championship 2026 ｜ Women ｜ Pool E_set1 | 1 | 0:05 | 0:14 | receive | 持球 | 持球（catch / hold）犯規。 |
| Full Match ｜ Poland vs. Slovakia ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 26 | 11:57 | 12:06 | receive | 持球 | 持球（catch / hold）犯規。 |
| Türkiye vs. Colombia - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 29 | 13:32 | 13:40 | set | 持球 | 持球（catch / hold）犯規。 |
| ᴴᴰ114UVL預賽：：中原大學vs陽明交大：：男一級 大專排球聯賽 AI網路直播_set1 | 23 | 11:37 | 11:45 | receive | 持球 | 持球（catch / hold）犯規。 |
| 03⧸14(六) 14_00｜例行賽G103 #雲林美津濃 vs. #桃園臺灣產險｜企業21年甲級男女排球聯賽_set3 | 29 | 18:54 | 19:10 | receive | 標竿外 | 球從標竿外側過網，視同出界。 |
| Bulgaria 🇧🇬 vs. Argentina 🇦🇷 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 23 | 11:12 | 11:14 | serve | 發球踩線 | 發球踩線，發球方直接失分；rally 沒有正常展開。 |
| Bulgaria 🇧🇬 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 37 | 18:51 | 18:58 | set | 發球踩線 | 發球踩線，發球方直接失分；rally 沒有正常展開。 |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 10 | 4:03 | 4:08 | receive | 發球踩線 | 發球踩線，發球方直接失分；rally 沒有正常展開。 |
| 03⧸21(六) 18_00｜男子組冠軍賽G114 #雲林美津濃 vs. #屏東台電｜企業21年甲級男女排球聯賽_set1 | 34 | 19:35 | 19:42 | receive | 舉球後排越界 | 後排球員在前區把球舉過網，位置違例。 |
| Japan vs. USA - Ranking 15-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 26 | 10:14 | 10:21 | receive | 舉球後排越界 | 後排球員在前區把球舉過網，位置違例。 |
| Semi Final 2 - Osaka Bluteon vs. Stings Aichi ｜ SV League - Full Match ｜ Volleyball_set1 | 39 | 23:08 | 23:16 | receive | 舉球後排越界 | 後排球員在前區把球舉過網，位置違例。 |
| 0323小窩臨打 3 | 19 | 6:27 | 6:38 | receive | 落地後 | 球已經落地得分，之後球員又碰到球；最後那個觸球是落地後的多餘動作，不是 score。 |
| 03⧸21(六) 18_00｜男子組冠軍賽G114 #雲林美津濃 vs. #屏東台電｜企業21年甲級男女排球聯賽_set1 | 39 | 22:18 | 22:26 | receive | 落地後 | 球已經落地得分，之後球員又碰到球；最後那個觸球是落地後的多餘動作，不是 score。 |
| 03⧸21(六) 18_00｜男子組冠軍賽G114 #雲林美津濃 vs. #屏東台電｜企業21年甲級男女排球聯賽_set1 | 40 | 22:50 | 23:01 | receive | 落地後 | 球已經落地得分，之後球員又碰到球；最後那個觸球是落地後的多餘動作，不是 score。 |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 4 | 1:31 | 1:38 | receive | 落地後 | 球已經落地得分，之後球員又碰到球；最後那個觸球是落地後的多餘動作，不是 score。 |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 50 | 34:58 | 35:02 | receive | 落地後 | 球已經落地得分，之後球員又碰到球；最後那個觸球是落地後的多餘動作，不是 score。 |
| France 🇫🇷 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 19 | 9:53 | 10:01 | receive | 落地後 | 球已經落地得分，之後球員又碰到球；最後那個觸球是落地後的多餘動作，不是 score。 |
| Full Match ｜ Croatia vs. Bulgaria ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set1 | 28 | 11:48 | 11:54 | receive | 落地後 | 球已經落地得分，之後球員又碰到球；最後那個觸球是落地後的多餘動作，不是 score。 |
| Full Match ｜ Denmark vs. England ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool A_set1 | 11 | 7:16 | 7:26 | receive | 落地後 | 球已經落地得分，之後球員又碰到球；最後那個觸球是落地後的多餘動作，不是 score。 |
| Full Match ｜ Ireland vs. Spain ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool D_set1 | 9 | 3:21 | 3:34 | receive | 落地後 | 球已經落地得分，之後球員又碰到球；最後那個觸球是落地後的多餘動作，不是 score。 |
| Full Match ｜ Ireland vs. Türkiye ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool D_set1 | 14 | 6:58 | 7:05 | receive | 落地後 | 球已經落地得分，之後球員又碰到球；最後那個觸球是落地後的多餘動作，不是 score。 |
| Full Match ｜ Italy vs. England ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool A_set1 | 30 | 16:41 | 16:52 | receive | 落地後 | 球已經落地得分，之後球員又碰到球；最後那個觸球是落地後的多餘動作，不是 score。 |
| Japan 🇯🇵 vs. Netherlands 🇳🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 23 | 10:53 | 11:10 | receive | 落地後 | 球已經落地得分，之後球員又碰到球；最後那個觸球是落地後的多餘動作，不是 score。 |
| Osaka Bluteon 🇯🇵 vs. JTEKT Stings 🇯🇵 ｜ SV League 2026 ｜ Full Match - Volleyball_set1 | 42 | 22:28 | 22:36 | receive | 落地後 | 球已經落地得分，之後球員又碰到球；最後那個觸球是落地後的多餘動作，不是 score。 |
| Semi Final 2 - Osaka Bluteon vs. Stings Aichi ｜ SV League - Full Match ｜ Volleyball_set1 | 30 | 16:30 | 16:36 | spike | 落地後 | 球已經落地得分，之後球員又碰到球；最後那個觸球是落地後的多餘動作，不是 score。 |
| Suntory Sunbirds vs. Osaka Bluteon ｜ SV.LEAGUE 2025⧸26 ｜ Full Match - Volleyball_set1 | 23 | 10:53 | 11:00 | receive | 落地後 | 球已經落地得分，之後球員又碰到球；最後那個觸球是落地後的多餘動作，不是 score。 |
| Türkiye vs. Colombia - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 31 | 15:39 | 15:51 | set | 落地後 | 球已經落地得分，之後球員又碰到球；最後那個觸球是落地後的多餘動作，不是 score。 |
| ᴴᴰ114UVL預賽：：國北教大vs嘉義大學：：男一級 大專排球聯賽 AI網路直播_set1 | 1 | 0:32 | 0:40 | set | 落地後 | 球已經落地得分，之後球員又碰到球；最後那個觸球是落地後的多餘動作，不是 score。 |
| 03⧸14(六) 18_00｜例行賽G105 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 1 | 0:00 | 0:06 | spike | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| 03⧸14(六) 18_00｜例行賽G105 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 31 | 17:52 | 18:05 | set | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| 03⧸20(五) 14_00｜挑戰賽G110 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 10 | 4:34 | 4:44 | receive | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| 03⧸20(五) 14_00｜挑戰賽G110 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 29 | 15:31 | 15:40 | set | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| 03⧸20(五) 16_00｜挑戰賽G111 #屏東台電 vs. #桃園臺產｜企業21年甲級男女排球聯賽_set1 | 24 | 10:38 | 10:49 | set | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| 03⧸21(六) 18_00｜男子組冠軍賽G114 #雲林美津濃 vs. #屏東台電｜企業21年甲級男女排球聯賽_set1 | 11 | 6:03 | 6:12 | set | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| 03⧸22(日) 17_00｜男子組冠軍賽 G117 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set1 | 2 | 0:22 | 0:36 | block | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| 03⧸22(日) 17_00｜男子組冠軍賽 G117 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set2 | 30 | 19:22 | 19:28 | receive | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| 03⧸22(日) 17_00｜男子組冠軍賽 G117 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set3 | 4 | 1:19 | 1:35 | receive | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| 0419小窩臨打 2 | 37 | 14:08 | 14:28 | spike | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| 2025-10-26_G14_臺北伊斯特_vs_臺中連莊_set1 | 37 | 22:35 | 22:44 | set | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| 2025-11-01_G15_臺中連莊_vs_臺北伊斯特_set1 | 32 | 18:36 | 18:43 | receive | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| 2025-11-01_G16_桃園雲豹飛將_vs_台鋼天鷹_set1 | 13 | 6:35 | 6:44 | receive | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| 2025-11-08_G20_桃園雲豹飛將_vs_台鋼天鷹_set1 | 29 | 16:19 | 16:26 | set | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| 20250504 大統OB-成功大學vs台北大學B-第二局 | 4 | 1:29 | 1:44 | set | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| 20260426-小窩-01 | 31 | 12:27 | 13:08 | receive | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| 20260502-排島本館-02 | 7 | 4:05 | 4:13 | receive | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 31 | 15:37 | 15:54 | set | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| China vs. Argentina - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 2 | 0:33 | 0:43 | set | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| China vs. Brazil - Ranking 13-14 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 6 | 2:36 | 2:45 | receive | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| China 🇨🇳 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 3 | 1:56 | 2:06 | receive | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| Cuba vs. Puerto Rico - Ranking 17-18 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 22 | 10:15 | 10:31 | receive | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| France vs. Iran - Semi Final 2 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 3 | 0:58 | 1:14 | set | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| France vs. Iran - Semi Final 2 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 5 | 2:16 | 2:26 | receive | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| France vs. Iran - Semi Final 2 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 18 | 8:40 | 8:53 | set | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| France 🇫🇷 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 41 | 21:11 | 21:24 | receive | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| Full Match ｜ Bulgaria vs Luxembourg ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set3 | 37 | 18:34 | 18:41 | spike | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| Full Match ｜ Croatia vs. Serbia ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set1 | 21 | 8:55 | 9:16 | receive | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| Full Match ｜ Ireland vs. Spain ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool D_set1 | 11 | 5:06 | 5:16 | set | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| Full Match ｜ Norway vs. Bulgaria ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool C_set1 | 21 | 9:14 | 9:23 | receive | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| Full Match ｜ Serbia vs. Luxembourg ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set1 | 34 | 17:25 | 17:34 | receive | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| Full Match ｜ Slovakia vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 31 | 15:44 | 15:56 | receive | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| Full Match ｜ Spain vs The Netherlands ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool C_set1 | 13 | 7:46 | 8:00 | receive | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| Japan vs. USA - Ranking 15-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 45 | 21:39 | 21:49 | receive | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| Japan 🇯🇵 vs. Dominican Republic 🇩🇴 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 24 | 11:02 | 11:15 | set | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| Japan 🇯🇵 vs. Dominican Republic 🇩🇴 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 29 | 13:26 | 13:34 | receive | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| Japan 🇯🇵 vs. Netherlands 🇳🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 2 | 0:25 | 0:35 | receive | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| Japan 🇯🇵 vs. Netherlands 🇳🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 31 | 15:06 | 15:14 | set | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| Japan 🇯🇵 vs. Netherlands 🇳🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 43 | 21:20 | 21:32 | receive | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| Jtekt Stings 🇯🇵 - Suntory Sunbirds Osaka 🇯🇵 ｜ SV League 2026 ｜ Full Match - Volleyball_set1 | 31 | 17:17 | 17:30 | receive | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| Korea vs. Finland - Ranking 11-12 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 21 | 7:25 | 7:32 | spike | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 11 | 4:51 | 5:00 | receive | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 26 | 13:42 | 13:50 | receive | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| Osaka Bluteon 🇯🇵 vs. JTEKT Stings 🇯🇵 ｜ SV League 2026 ｜ Full Match - Volleyball_set1 | 23 | 11:23 | 11:31 | set | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| Pakistan vs. USA - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 6 | 2:26 | 2:34 | receive | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| Pakistan vs. USA - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 16 | 7:38 | 7:55 | receive | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| Semi Final 1 - Osaka Bluteon vs. Stings Aichi ｜ SV League - Full Match ｜ Volleyball_set1 | 8 | 3:45 | 3:52 | receive | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| Semi Final 1 - Osaka Bluteon vs. Stings Aichi ｜ SV League - Full Match ｜ Volleyball_set1 | 42 | 25:36 | 25:56 | receive | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| Suntory Sunbirds 🇯🇵 vs. Stings AICHI 🇯🇵 ｜ SV League 2026 ｜ Full Match - Japan Volleyball_set1 | 6 | 3:19 | 3:27 | receive | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| Taipei vs. Argentina - Playoffs ｜ Girls' U19 World Champs 2025 - Full Match_set1 | 7 | 3:28 | 3:47 | set | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| Taipei vs. Argentina - Playoffs ｜ Girls' U19 World Champs 2025 - Full Match_set1 | 22 | 12:57 | 13:06 | set | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| Türkiye vs. Colombia - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 27 | 12:41 | 12:50 | receive | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| Uzbekistan vs. Pakistan - Ranking 5-6 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 15 | 6:09 | 6:17 | receive | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| ᴴᴰ114UVL預賽：：中原大學vs陽明交大：：男一級 大專排球聯賽 AI網路直播_set1 | 39 | 18:55 | 19:03 | set | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| ᴴᴰ114UVL預賽：：中山大學vs國北教大：：男一級 大專排球聯賽 AI網路直播_set1 | 19 | 8:24 | 8:32 | receive | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| ᴴᴰ114UVL預賽：：中山大學vs清華大學：：男一級 大專排球聯賽 AI網路直播_set1 | 16 | 6:07 | 6:18 | set | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| ᴴᴰ114UVL預賽：：中山大學vs清華大學：：男一級 大專排球聯賽 AI網路直播_set1 | 39 | 17:21 | 17:33 | receive | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| ᴴᴰ114UVL預賽：：中山大學vs清華大學：：男一級 大專排球聯賽 AI網路直播_set1 | 41 | 18:34 | 18:43 | receive | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| ᴴᴰ114UVL預賽：：臺灣師大vs中山大學：：男一級 大專排球聯賽 AI網路直播_set1 | 3 | 0:49 | 1:01 | receive | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| ᴴᴰ114UVL預賽：：臺灣師大vs中山大學：：男一級 大專排球聯賽 AI網路直播_set1 | 23 | 9:13 | 9:24 | spike | 觸網 | 球員觸網犯規，裁判哨音結束這球；沒有得分的觸球。 |
| 2025-10-04_G3_臺中連莊_vs_桃園雲豹飛將_set1 | 13 | 5:53 | 5:59 | receive | 越界 | 球出界或球員越過中線，裁判判定結束。 |
| 2025-11-01_G15_臺中連莊_vs_臺北伊斯特_set1 | 36 | 20:28 | 20:38 | spike | 越界 | 球出界或球員越過中線，裁判判定結束。 |
| Full Match ｜ Denmark vs. England ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool A_set1 | 18 | 10:30 | 10:40 | receive | 越界 | 球出界或球員越過中線，裁判判定結束。 |
| Spain vs. Iran - Ranking 3-4 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 16 | 11:03 | 11:16 | spike | 越界 | 球出界或球員越過中線，裁判判定結束。 |
| ᴴᴰ114UVL預賽：：中原大學vs陽明交大：：男一級 大專排球聯賽 AI網路直播_set1 | 17 | 8:10 | 8:19 | receive | 越界 | 球出界或球員越過中線，裁判判定結束。 |
| ᴴᴰ114UVL預賽：：中原大學vs陽明交大：：男一級 大專排球聯賽 AI網路直播_set1 | 38 | 18:28 | 18:38 | receive | 越界 | 球出界或球員越過中線，裁判判定結束。 |
| Semi Final 3 - Suntory Sunbirds vs. Wolfdogs Nagoya ｜ SVL Playoff - Full Match ｜ Volleyball_set1 | 46 | 27:17 | 27:26 | block | 越界救球 | 救球時越過中線或從界外把球救回，判犯規。 |
| 2025-11-02_G17_桃園雲豹飛將_vs_臺北伊斯特_set1 | 4 | 1:31 | 1:41 | receive | 越網擊球 | 越過球網擊球（reaching over）犯規。 |
| Spain vs. Iran - Ranking 3-4 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 14 | 10:01 | 10:10 | receive | 越網擊球 | 越過球網擊球（reaching over）犯規。 |
| ᴴᴰ114UVL預賽：：中山大學vs國北教大：：男一級 大專排球聯賽 AI網路直播_set1 | 15 | 6:35 | 6:42 | set | 越網擊球 | 越過球網擊球（reaching over）犯規。 |
| Pakistan vs. USA - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 1 | 0:07 | 0:14 | receive | 阻擋舉球 | 攔網時觸碰到對方正在舉球的球，犯規。 |

## score 不在最後 — 14 筆

span 內有 `score`，但它不是最後的事件。

| 影片 | Rally | 起 | 訖 | 最後動作 | score 距結尾 |
|---|---:|---:|---:|---|---:|
| Full Match ｜ Bulgaria vs Luxembourg ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set1 | 31 | 14:42 | 14:58 | receive | 2.2s |
| Full Match ｜ Ireland vs. Türkiye ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool D_set1 | 19 | 8:48 | 9:02 | set | 3.1s |
| Full Match ｜ Serbia vs. Bulgaria ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set1 | 20 | 8:10 | 8:20 | receive | 1.6s |
| Japan 🇯🇵 vs. Czechia 🇨🇿 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 22 | 9:48 | 9:56 | receive | 2.2s |
| Japan 🇯🇵 vs. Serbia 🇷🇸 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 38 | 16:51 | 16:59 | spike | 1.4s |
| Osaka Bluteon vs. Toray Arrows Shizuoka - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 31 | 17:20 | 17:33 | receive | 8.6s |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G21 11⧸9 15_00 桃園雲豹飛將 vs 臺中連莊_set1 | 39 | 19:25 | 20:03 | receive | 2.6s |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G23 11⧸15 15_00 臺中連莊 vs 台鋼天鷹_set1 | 24 | 11:12 | 11:29 | receive | 2.6s |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G24 11⧸15 18_30 桃園雲豹飛將 vs 臺北伊斯特_set1 | 9 | 4:19 | 4:26 | receive | 0.9s |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G24 11⧸15 18_30 桃園雲豹飛將 vs 臺北伊斯特_set1 | 25 | 11:40 | 11:51 | block | 3.1s |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G24 11⧸15 18_30 桃園雲豹飛將 vs 臺北伊斯特_set1 | 45 | 24:13 | 24:31 | receive | 2.0s |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G26 11⧸16 18_30 臺中連莊 vs 臺北伊斯特_set1 | 17 | 9:20 | 9:29 | receive | 2.7s |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G29 11⧸23 15_00 臺北伊斯特 vs 台中連莊_set1 | 18 | 9:51 | 9:59 | receive | 2.3s |
| 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G6 10⧸5 18_30 臺中連莊 vs 台鋼天鷹_set1 | 25 | 12:12 | 12:19 | receive | 2.4s |
