# Rally 前後各 1 秒：調整結果與編號問題清單

產生時間：2026-09-16 01:28:32；狀態：已套用。

## 規則與範圍

以原 span 內第一個 serve − 1 秒、最後一個 score + 1 秒設定起訖（毫秒精度）。缺少某端動作時保留該端原值；邊界限制在 [0, metadata duration]。所有動作標註、rally ID、winner 與其他欄位均保留。多個動作的問題仍須人工複查。

全部 806 支 / 35,418 rallies；有人工 action 標註 200 支 / 8,861 rallies。已修改 200 支 / 8,861 rallies / 17,686 個邊界。

可留滿 1 秒：start 8,819、end 8,852；受影片頭尾限制：start 14、end 5；缺少 serve 28、缺少 score 4。

R2 同步：200/200 檔案成功。調整過的 rally fingerprint 會改變，未偽造 tracking 新鮮度；既有 tracking 可能需要重新執行。

本次為標註資料檢查，未重新觀看影片；既有人工判定只作參考。問題按影片/rally/類型固定排序編號；同一 rally 可有多個問題。保留調整前與調整後問題的聯集，避免縮短範圍後把問題隱藏。

## 問題統計

| 問題 | 筆數 |
|---|---:|
| score 不在最後 | 148 |
| 調整後移出 span 的動作 | 84 |
| 缺少 serve | 28 |
| start 小於 0 | 22 |
| 多個 score | 20 |
| 影片頭尾不足 1 秒 | 19 |
| end 超出影片 | 15 |
| 缺少 score | 4 |
| 多個 serve | 3 |
| serve 不在最前 | 3 |

## Rally 問題清單

時間以影片內秒數表示。

| 編號 | 影片 | Rally | 調整後起訖 | 問題 | 狀態 | 詳細資料 |
|---|---|---:|---|---|---|---|
| Q0001 | 0104排島臨打 1 | 35 | 832.133–843.400 | 缺少 score | 調整後仍存在 | span 內沒有 score |
| Q0002 | 0104排島臨打 1 | 37 | 874.833–878.867 | 缺少 score | 調整後仍存在 | span 內沒有 score |
| Q0003 | 0104排島臨打 1 | 39 | 916.033–920.300 | 缺少 score | 調整後仍存在 | span 內沒有 score |
| Q0004 | 0104排島臨打 2 | 35 | 793.267–804.733 | score 不在最後 | 調整後未出現；原始問題保留供複查 | 事件序列：serve@794.267s, receive@795.633s, set@797.033s, spike@798.567s, receive@799.267s, set@801.000s, spike@802.600s, block@802.833s, score@803.733s, set@805.033s |
| Q0005 | 0104排島臨打 2 | 35 | 793.267–804.733 | 調整後移出 span 的動作 | 需複查；action 標註保留 | set@805.033s |
| Q0006 | 0104排島臨打 3 | 2 | 70.500–109.900 | 多個 serve | 調整後仍存在 | 2 個：71.500s, 81.433s |
| Q0007 | 0323小窩臨打 3 | 19 | 387.533–397.167 | score 不在最後 | 調整後未出現；原始問題保留供複查 | 事件序列：serve@388.533s, receive@389.500s, spike@391.167s, receive@392.400s, set@393.900s, spike@395.333s, score@396.167s, receive@397.567s；既有人工紀錄：落地後（本次未重看影片） |
| Q0008 | 0323小窩臨打 3 | 19 | 387.533–397.167 | 調整後移出 span 的動作 | 需複查；action 標註保留 | receive@397.567s |
| Q0009 | 03⧸14(六) 14_00｜例行賽G103 #雲林美津濃 vs. #桃園臺灣產險｜企業21年甲級男女排球聯賽_set1 | 47 | 1513.333–1516.133 | end 超出影片 | 調整後未出現；原始問題保留供複查 | end=1516.633s；metadata duration=1516.301s |
| Q0010 | 03⧸14(六) 14_00｜例行賽G103 #雲林美津濃 vs. #桃園臺灣產險｜企業21年甲級男女排球聯賽_set2 | 1 | 0.017–5.967 | start 小於 0 | 調整後未出現；原始問題保留供複查 | start=-0.483s |
| Q0011 | 03⧸14(六) 14_00｜例行賽G103 #雲林美津濃 vs. #桃園臺灣產險｜企業21年甲級男女排球聯賽_set2 | 46 | 1774.233–1781.117 | end 超出影片 | 調整後未出現；原始問題保留供複查 | end=1781.617s；metadata duration=1781.495s |
| Q0012 | 03⧸14(六) 14_00｜例行賽G103 #雲林美津濃 vs. #桃園臺灣產險｜企業21年甲級男女排球聯賽_set3 | 1 | 0.283–16.383 | start 小於 0 | 調整後未出現；原始問題保留供複查 | start=-0.217s |
| Q0013 | 03⧸14(六) 14_00｜例行賽G103 #雲林美津濃 vs. #桃園臺灣產險｜企業21年甲級男女排球聯賽_set3 | 29 | 1134.817–1149.700 | score 不在最後 | 調整後未出現；原始問題保留供複查 | 事件序列：serve@1135.817s, receive@1136.783s, receive@1137.933s, receive@1139.667s, receive@1140.583s, set@1142.217s, spike@1142.633s, block@1142.683s, receive@1143.183s, receive@1145.300s, receive@1147.800s, score@1148.700s, receive@1150.067s；既有人工紀錄：標竿外（本次未重看影片） |
| Q0014 | 03⧸14(六) 14_00｜例行賽G103 #雲林美津濃 vs. #桃園臺灣產險｜企業21年甲級男女排球聯賽_set3 | 29 | 1134.817–1149.700 | 調整後移出 span 的動作 | 需複查；action 標註保留 | receive@1150.067s |
| Q0015 | 03⧸14(六) 16_00｜例行賽G104 #獅子王 vs. #屏東台電｜企業21年甲級男女排球聯賽_set1 | 44 | 1465.133–1471.917 | end 超出影片 | 調整後未出現；原始問題保留供複查 | end=1472.433s；metadata duration=1472.430s |
| Q0016 | 03⧸14(六) 16_00｜例行賽G104 #獅子王 vs. #屏東台電｜企業21年甲級男女排球聯賽_set2 | 1 | 0.267–3.717 | start 小於 0 | 調整後未出現；原始問題保留供複查 | start=-0.233s |
| Q0017 | 03⧸14(六) 16_00｜例行賽G104 #獅子王 vs. #屏東台電｜企業21年甲級男女排球聯賽_set2 | 20 | 719.550–727.317 | score 不在最後 | 調整後仍存在 | 事件序列：serve@720.550s, receive@721.200s, spike@723.417s, block@723.450s, receive@723.783s, set@724.800s, score@726.317s, receive@726.900s |
| Q0018 | 03⧸14(六) 16_00｜例行賽G104 #獅子王 vs. #屏東台電｜企業21年甲級男女排球聯賽_set2 | 27 | 937.500–942.250 | 缺少 serve | 調整後仍存在 | span 內沒有 serve；既有人工紀錄：導播問題（本次未重看影片） |
| Q0019 | 03⧸14(六) 18_00｜例行賽G105 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 1 | 0.000–3.417 | score 不在最後 | 調整後未出現；原始問題保留供複查 | 事件序列：serve@0.000s, receive@0.783s, set@2.400s, score@2.417s, spike@3.650s；既有人工紀錄：觸網（本次未重看影片） |
| Q0020 | 03⧸14(六) 18_00｜例行賽G105 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 1 | 0.000–3.417 | 調整後移出 span 的動作 | 需複查；action 標註保留 | spike@3.650s |
| Q0021 | 03⧸14(六) 18_00｜例行賽G105 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 1 | 0.000–3.417 | 影片頭尾不足 1 秒 | 已截在影片範圍 | start 無法留滿 1 秒；duration=1720.441s |
| Q0022 | 03⧸14(六) 18_00｜例行賽G105 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 31 | 1073.167–1082.933 | score 不在最後 | 調整後仍存在 | 事件序列：serve@1074.167s, receive@1075.267s, set@1076.617s, spike@1077.950s, receive@1078.367s, set@1080.433s, spike@1081.717s, score@1081.933s, receive@1082.400s；既有人工紀錄：觸網（本次未重看影片） |
| Q0023 | 03⧸14(六) 18_00｜例行賽G105 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 31 | 1073.167–1082.933 | 調整後移出 span 的動作 | 需複查；action 標註保留 | set@1083.833s |
| Q0024 | 03⧸14(六) 18_00｜例行賽G105 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 36 | 1320.227–1324.383 | 缺少 serve | 調整後仍存在 | span 內沒有 serve；既有人工紀錄：導播問題（本次未重看影片） |
| Q0025 | 03⧸14(六) 18_00｜例行賽G105 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 45 | 1713.567–1720.441 | end 超出影片 | 調整後未出現；原始問題保留供複查 | end=1721.267s；metadata duration=1720.441s |
| Q0026 | 03⧸14(六) 18_00｜例行賽G105 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 45 | 1713.567–1720.441 | 影片頭尾不足 1 秒 | 已截在影片範圍 | end 無法留滿 1 秒；duration=1720.441s |
| Q0027 | 03⧸15(日) 13_00｜例行賽G106 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set2 | 1 | 0.000–5.967 | start 小於 0 | 調整後未出現；原始問題保留供複查 | start=-0.533s |
| Q0028 | 03⧸15(日) 13_00｜例行賽G106 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set2 | 1 | 0.000–5.967 | 影片頭尾不足 1 秒 | 已截在影片範圍 | start 無法留滿 1 秒；duration=1987.807s |
| Q0029 | 03⧸15(日) 13_00｜例行賽G106 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set3 | 41 | 1488.102–1490.550 | 缺少 serve | 調整後仍存在 | span 內沒有 serve；既有人工紀錄：導播問題（本次未重看影片） |
| Q0030 | 03⧸15(日) 13_00｜例行賽G106 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set3 | 47 | 1718.267–1729.997 | end 超出影片 | 調整後未出現；原始問題保留供複查 | end=1730.700s；metadata duration=1729.997s |
| Q0031 | 03⧸15(日) 13_00｜例行賽G106 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set3 | 47 | 1718.267–1729.997 | 影片頭尾不足 1 秒 | 已截在影片範圍 | end 無法留滿 1 秒；duration=1729.997s |
| Q0032 | 03⧸15(日) 15_00｜例行賽G107 #桃園臺灣產險 vs. #獅子王｜企業21年甲級男女排球聯賽_set1 | 1 | 0.000–4.883 | start 小於 0 | 調整後未出現；原始問題保留供複查 | start=-1.433s |
| Q0033 | 03⧸15(日) 15_00｜例行賽G107 #桃園臺灣產險 vs. #獅子王｜企業21年甲級男女排球聯賽_set1 | 1 | 0.000–4.883 | 影片頭尾不足 1 秒 | 已截在影片範圍 | start 無法留滿 1 秒；duration=1708.966s |
| Q0034 | 03⧸15(日) 15_00｜例行賽G107 #桃園臺灣產險 vs. #獅子王｜企業21年甲級男女排球聯賽_set1 | 47 | 1694.717–1708.966 | end 超出影片 | 調整後未出現；原始問題保留供複查 | end=1710.433s；metadata duration=1708.966s |
| Q0035 | 03⧸15(日) 15_00｜例行賽G107 #桃園臺灣產險 vs. #獅子王｜企業21年甲級男女排球聯賽_set1 | 47 | 1694.717–1708.966 | 影片頭尾不足 1 秒 | 已截在影片範圍 | end 無法留滿 1 秒；duration=1708.966s |
| Q0036 | 03⧸15(日) 17_00｜例行賽G108 #新北中纖 vs. #高雄台電｜企業21年甲級男女排球聯賽_set1 | 1 | 0.383–6.933 | start 小於 0 | 調整後未出現；原始問題保留供複查 | start=-0.117s |
| Q0037 | 03⧸15(日) 17_00｜例行賽G108 #新北中纖 vs. #高雄台電｜企業21年甲級男女排球聯賽_set1 | 48 | 1851.017–1868.181 | end 超出影片 | 調整後未出現；原始問題保留供複查 | end=1868.783s；metadata duration=1868.181s |
| Q0038 | 03⧸15(日) 17_00｜例行賽G108 #新北中纖 vs. #高雄台電｜企業21年甲級男女排球聯賽_set1 | 48 | 1851.017–1868.181 | 影片頭尾不足 1 秒 | 已截在影片範圍 | end 無法留滿 1 秒；duration=1868.181s |
| Q0039 | 03⧸15(日) 17_00｜例行賽G108 #新北中纖 vs. #高雄台電｜企業21年甲級男女排球聯賽_set2 | 1 | 0.000–5.200 | start 小於 0 | 調整後未出現；原始問題保留供複查 | start=-1.500s |
| Q0040 | 03⧸15(日) 17_00｜例行賽G108 #新北中纖 vs. #高雄台電｜企業21年甲級男女排球聯賽_set2 | 1 | 0.000–5.200 | 影片頭尾不足 1 秒 | 已截在影片範圍 | start 無法留滿 1 秒；duration=1845.805s |
| Q0041 | 03⧸15(日) 17_00｜例行賽G108 #新北中纖 vs. #高雄台電｜企業21年甲級男女排球聯賽_set2 | 44 | 1831.617–1845.600 | end 超出影片 | 調整後未出現；原始問題保留供複查 | end=1846.100s；metadata duration=1845.805s |
| Q0042 | 03⧸15(日) 17_00｜例行賽G108 #新北中纖 vs. #高雄台電｜企業21年甲級男女排球聯賽_set3 | 1 | 0.117–7.100 | start 小於 0 | 調整後未出現；原始問題保留供複查 | start=-0.383s |
| Q0043 | 03⧸15(日) 17_00｜例行賽G108 #新北中纖 vs. #高雄台電｜企業21年甲級男女排球聯賽_set3 | 7 | 223.280–232.300 | 缺少 serve | 調整後仍存在 | span 內沒有 serve；既有人工紀錄：導播問題（本次未重看影片） |
| Q0044 | 03⧸15(日) 17_00｜例行賽G108 #新北中纖 vs. #高雄台電｜企業21年甲級男女排球聯賽_set5 | 1 | 0.000–9.733 | start 小於 0 | 調整後未出現；原始問題保留供複查 | start=-1.500s |
| Q0045 | 03⧸15(日) 17_00｜例行賽G108 #新北中纖 vs. #高雄台電｜企業21年甲級男女排球聯賽_set5 | 1 | 0.000–9.733 | 影片頭尾不足 1 秒 | 已截在影片範圍 | start 無法留滿 1 秒；duration=1376.469s |
| Q0046 | 03⧸15(日) 17_00｜例行賽G108 #新北中纖 vs. #高雄台電｜企業21年甲級男女排球聯賽_set5 | 30 | 1371.317–1376.469 | end 超出影片 | 調整後未出現；原始問題保留供複查 | end=1377.317s；metadata duration=1376.469s |
| Q0047 | 03⧸15(日) 17_00｜例行賽G108 #新北中纖 vs. #高雄台電｜企業21年甲級男女排球聯賽_set5 | 30 | 1371.317–1376.469 | 影片頭尾不足 1 秒 | 已截在影片範圍 | end 無法留滿 1 秒；duration=1376.469s |
| Q0048 | 03⧸20(五) 14_00｜挑戰賽G110 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 10 | 275.117–280.617 | score 不在最後 | 調整後仍存在 | 事件序列：serve@276.117s, receive@277.183s, set@278.767s, spike@279.417s, score@279.617s, receive@279.733s；既有人工紀錄：觸網（本次未重看影片） |
| Q0049 | 03⧸20(五) 14_00｜挑戰賽G110 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 10 | 275.117–280.617 | 調整後移出 span 的動作 | 需複查；action 標註保留 | receive@281.433s |
| Q0050 | 03⧸20(五) 14_00｜挑戰賽G110 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 29 | 932.300–938.200 | score 不在最後 | 調整後未出現；原始問題保留供複查 | 事件序列：serve@933.300s, receive@934.467s, set@935.600s, spike@936.733s, receive@937.200s, score@937.200s, set@939.250s；既有人工紀錄：觸網（本次未重看影片） |
| Q0051 | 03⧸20(五) 14_00｜挑戰賽G110 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 29 | 932.300–938.200 | 調整後移出 span 的動作 | 需複查；action 標註保留 | set@939.250s |
| Q0052 | 03⧸20(五) 14_00｜挑戰賽G110 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 33 | 1151.550–1159.300 | 多個 score | 調整後仍存在 | 2 個：1156.900s, 1158.300s |
| Q0053 | 03⧸20(五) 14_00｜挑戰賽G110 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1 | 42 | 1711.933–1718.433 | score 不在最後 | 調整後仍存在 | 事件序列：serve@1712.933s, receive@1713.883s, set@1715.283s, spike@1715.750s, block@1715.783s, receive@1716.400s, set@1717.417s, score@1717.433s, spike@1718.067s；既有人工紀錄：持球（本次未重看影片） |
| Q0054 | 03⧸20(五) 14_00｜挑戰賽G110 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set2 | 1 | 0.000–8.967 | start 小於 0 | 調整後未出現；原始問題保留供複查 | start=-1.417s |
| Q0055 | 03⧸20(五) 14_00｜挑戰賽G110 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set2 | 1 | 0.000–8.967 | 影片頭尾不足 1 秒 | 已截在影片範圍 | start 無法留滿 1 秒；duration=1769.679s |
| Q0056 | 03⧸20(五) 14_00｜挑戰賽G110 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set2 | 6 | 144.800–211.117 | 缺少 serve | 調整後仍存在 | span 內沒有 serve；既有人工紀錄：導播問題（本次未重看影片） |
| Q0057 | 03⧸20(五) 14_00｜挑戰賽G110 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set2 | 42 | 1763.500–1769.617 | end 超出影片 | 調整後未出現；原始問題保留供複查 | end=1770.117s；metadata duration=1769.679s |
| Q0058 | 03⧸20(五) 16_00｜挑戰賽G111 #屏東台電 vs. #桃園臺產｜企業21年甲級男女排球聯賽_set1 | 1 | 0.000–9.167 | start 小於 0 | 調整後未出現；原始問題保留供複查 | start=-1.500s |
| Q0059 | 03⧸20(五) 16_00｜挑戰賽G111 #屏東台電 vs. #桃園臺產｜企業21年甲級男女排球聯賽_set1 | 1 | 0.000–9.167 | 影片頭尾不足 1 秒 | 已截在影片範圍 | start 無法留滿 1 秒；duration=1379.764s |
| Q0060 | 03⧸20(五) 16_00｜挑戰賽G111 #屏東台電 vs. #桃園臺產｜企業21年甲級男女排球聯賽_set1 | 24 | 639.000–646.233 | score 不在最後 | 調整後仍存在 | 事件序列：serve@640.000s, receive@640.900s, spike@642.133s, receive@642.883s, set@644.350s, spike@644.933s, score@645.233s, receive@645.367s；既有人工紀錄：觸網（本次未重看影片） |
| Q0061 | 03⧸20(五) 16_00｜挑戰賽G111 #屏東台電 vs. #桃園臺產｜企業21年甲級男女排球聯賽_set1 | 24 | 639.000–646.233 | 調整後移出 span 的動作 | 需複查；action 標註保留 | set@647.300s |
| Q0062 | 03⧸20(五) 16_00｜挑戰賽G111 #屏東台電 vs. #桃園臺產｜企業21年甲級男女排球聯賽_set1 | 43 | 1364.667–1379.617 | end 超出影片 | 調整後未出現；原始問題保留供複查 | end=1380.117s；metadata duration=1379.764s |
| Q0063 | 03⧸20(五) 18_00｜挑戰賽G112 #臺北國北獅 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set1 | 20 | 604.883–613.900 | score 不在最後 | 調整後仍存在 | 事件序列：serve@605.883s, receive@606.750s, set@608.233s, spike@609.200s, block@609.300s, receive@610.350s, set@611.983s, spike@612.817s, score@612.900s, block@612.983s；既有人工紀錄：打到標竿（本次未重看影片） |
| Q0064 | 03⧸20(五) 18_00｜挑戰賽G112 #臺北國北獅 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set1 | 43 | 1504.117–1506.917 | end 超出影片 | 調整後未出現；原始問題保留供複查 | end=1507.417s；metadata duration=1507.382s |
| Q0065 | 03⧸21(六) 18_00｜男子組冠軍賽G114 #雲林美津濃 vs. #屏東台電｜企業21年甲級男女排球聯賽_set1 | 1 | 0.000–19.767 | start 小於 0 | 調整後未出現；原始問題保留供複查 | start=-1.200s |
| Q0066 | 03⧸21(六) 18_00｜男子組冠軍賽G114 #雲林美津濃 vs. #屏東台電｜企業21年甲級男女排球聯賽_set1 | 1 | 0.000–19.767 | 影片頭尾不足 1 秒 | 已截在影片範圍 | start 無法留滿 1 秒；duration=1760.447s |
| Q0067 | 03⧸21(六) 18_00｜男子組冠軍賽G114 #雲林美津濃 vs. #屏東台電｜企業21年甲級男女排球聯賽_set1 | 11 | 363.567–369.350 | score 不在最後 | 調整後仍存在 | 事件序列：serve@364.567s, receive@365.567s, set@367.133s, spike@368.100s, score@368.350s, receive@368.767s；既有人工紀錄：觸網（本次未重看影片） |
| Q0068 | 03⧸21(六) 18_00｜男子組冠軍賽G114 #雲林美津濃 vs. #屏東台電｜企業21年甲級男女排球聯賽_set1 | 11 | 363.567–369.350 | 調整後移出 span 的動作 | 需複查；action 標註保留 | set@370.700s |
| Q0069 | 03⧸21(六) 18_00｜男子組冠軍賽G114 #雲林美津濃 vs. #屏東台電｜企業21年甲級男女排球聯賽_set1 | 18 | 606.300–615.617 | 缺少 serve | 調整後仍存在 | span 內沒有 serve；既有人工紀錄：導播問題（本次未重看影片） |
| Q0070 | 03⧸21(六) 18_00｜男子組冠軍賽G114 #雲林美津濃 vs. #屏東台電｜企業21年甲級男女排球聯賽_set1 | 34 | 1176.333–1180.533 | score 不在最後 | 調整後仍存在 | 事件序列：serve@1177.333s, receive@1178.033s, block@1179.517s, score@1179.533s, spike@1179.933s；既有人工紀錄：舉球後排越界（本次未重看影片） |
| Q0071 | 03⧸21(六) 18_00｜男子組冠軍賽G114 #雲林美津濃 vs. #屏東台電｜企業21年甲級男女排球聯賽_set1 | 34 | 1176.333–1180.533 | 調整後移出 span 的動作 | 需複查；action 標註保留 | receive@1180.633s |
| Q0072 | 03⧸21(六) 18_00｜男子組冠軍賽G114 #雲林美津濃 vs. #屏東台電｜企業21年甲級男女排球聯賽_set1 | 39 | 1339.100–1346.250 | score 不在最後 | 調整後仍存在 | 事件序列：serve@1340.100s, receive@1341.133s, set@1342.533s, spike@1343.683s, block@1343.783s, receive@1344.233s, score@1345.250s, receive@1345.867s；既有人工紀錄：落地後（本次未重看影片） |
| Q0073 | 03⧸21(六) 18_00｜男子組冠軍賽G114 #雲林美津濃 vs. #屏東台電｜企業21年甲級男女排球聯賽_set1 | 40 | 1371.017–1380.133 | score 不在最後 | 調整後未出現；原始問題保留供複查 | 事件序列：serve@1372.017s, receive@1373.100s, spike@1374.733s, receive@1375.150s, set@1376.867s, spike@1378.300s, score@1379.133s, receive@1380.300s；既有人工紀錄：落地後（本次未重看影片） |
| Q0074 | 03⧸21(六) 18_00｜男子組冠軍賽G114 #雲林美津濃 vs. #屏東台電｜企業21年甲級男女排球聯賽_set1 | 40 | 1371.017–1380.133 | 調整後移出 span 的動作 | 需複查；action 標註保留 | receive@1380.300s |
| Q0075 | 03⧸22(日) 17_00｜男子組冠軍賽 G117 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set1 | 2 | 23.233–33.383 | score 不在最後 | 調整後仍存在 | 事件序列：serve@24.233s, receive@25.167s, spike@26.500s, receive@27.233s, set@28.500s, spike@29.717s, block@29.800s, receive@30.717s, set@31.967s, score@32.383s, spike@33.167s, block@33.233s；既有人工紀錄：觸網（本次未重看影片） |
| Q0076 | 03⧸22(日) 17_00｜男子組冠軍賽 G117 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set1 | 30 | 993.638–998.283 | 缺少 serve | 調整後仍存在 | span 內沒有 serve；既有人工紀錄：導播問題（本次未重看影片） |
| Q0077 | 03⧸22(日) 17_00｜男子組冠軍賽 G117 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set2 | 2 | 25.000–36.300 | 缺少 serve | 調整後仍存在 | span 內沒有 serve；既有人工紀錄：導播問題（本次未重看影片） |
| Q0078 | 03⧸22(日) 17_00｜男子組冠軍賽 G117 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set2 | 30 | 1163.067–1167.283 | score 不在最後 | 調整後仍存在 | 事件序列：serve@1164.067s, receive@1164.733s, set@1166.133s, score@1166.283s, spike@1166.650s, block@1166.683s, receive@1166.833s；既有人工紀錄：觸網（本次未重看影片） |
| Q0079 | 03⧸22(日) 17_00｜男子組冠軍賽 G117 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set3 | 1 | 0.000–3.383 | start 小於 0 | 調整後未出現；原始問題保留供複查 | start=-0.533s |
| Q0080 | 03⧸22(日) 17_00｜男子組冠軍賽 G117 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set3 | 1 | 0.000–3.383 | 影片頭尾不足 1 秒 | 已截在影片範圍 | start 無法留滿 1 秒；duration=1761.195s |
| Q0081 | 03⧸22(日) 17_00｜男子組冠軍賽 G117 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set3 | 4 | 79.517–93.750 | score 不在最後 | 調整後仍存在 | 事件序列：serve@80.517s, receive@81.700s, set@82.783s, spike@83.850s, block@83.983s, receive@84.833s, set@86.733s, spike@88.517s, receive@89.233s, set@90.983s, spike@92.217s, block@92.300s, score@92.750s, receive@92.967s；既有人工紀錄：觸網（本次未重看影片） |
| Q0082 | 0419小窩臨打 1 | 27 | 646.030–664.763 | score 不在最後 | 調整後未出現；原始問題保留供複查 | 事件序列：serve@647.030s, receive@648.331s, set@649.732s, spike@651.034s, receive@652.068s, set@653.319s, spike@654.804s, receive@655.588s, set@657.240s, spike@658.825s, receive@659.492s, set@661.361s, spike@662.829s, score@663.763s, serve@665.064s |
| Q0083 | 0419小窩臨打 1 | 27 | 646.030–664.763 | 多個 serve | 調整後未出現；原始問題保留供複查 | 2 個：647.030s, 665.064s |
| Q0084 | 0419小窩臨打 1 | 27 | 646.030–664.763 | 調整後移出 span 的動作 | 需複查；action 標註保留 | serve@665.064s |
| Q0085 | 0419小窩臨打 2 | 37 | 849.249–866.515 | score 不在最後 | 調整後仍存在 | 事件序列：serve@850.249s, receive@851.167s, set@852.435s, spike@853.886s, receive@854.437s, receive@855.972s, receive@857.757s, receive@859.509s, set@860.977s, spike@862.445s, receive@863.062s, set@864.897s, score@865.515s, spike@866.449s；既有人工紀錄：觸網（本次未重看影片） |
| Q0086 | 0420小窩季打 1 | 43 | 890.567–894.400 | score 不在最後 | 調整後仍存在 | 事件序列：serve@891.567s, receive@892.667s, score@893.400s, receive@894.067s；既有人工紀錄：公正（本次未重看影片） |
| Q0087 | 0420小窩季打 2 | 11 | 219.300–235.933 | score 不在最後 | 調整後仍存在 | 事件序列：serve@220.300s, receive@221.433s, set@222.967s, spike@224.767s, receive@225.633s, set@227.200s, spike@228.867s, receive@229.500s, spike@230.767s, receive@231.800s, set@233.733s, score@234.933s, spike@235.367s；既有人工紀錄：後排踩線（本次未重看影片） |
| Q0088 | 0420小窩季打 2 | 11 | 219.300–235.933 | 調整後移出 span 的動作 | 需複查；action 標註保留 | receive@236.000s |
| Q0089 | 0420小窩季打 2 | 47 | 931.433–946.200 | 多個 score | 調整後仍存在 | 2 個：943.700s, 945.200s |
| Q0090 | 0420小窩季打 3 | 15 | 252.933–258.200 | score 不在最後 | 調整後仍存在 | 事件序列：serve@253.933s, receive@254.967s, set@256.167s, score@257.200s, spike@257.567s；既有人工紀錄：後排踩線（本次未重看影片） |
| Q0091 | 0427小窩季打 11 | 18 | 462.100–480.767 | score 不在最後 | 調整後未出現；原始問題保留供複查 | 事件序列：serve@463.100s, receive@464.200s, set@465.900s, spike@467.367s, receive@468.633s, set@470.100s, spike@471.867s, receive@472.333s, set@474.067s, spike@475.600s, receive@476.333s, set@478.200s, score@479.767s, receive@480.900s |
| Q0092 | 0427小窩季打 11 | 18 | 462.100–480.767 | 調整後移出 span 的動作 | 需複查；action 標註保留 | receive@480.900s |
| Q0093 | 0427小窩季打 12 | 17 | 371.133–394.767 | 多個 score | 調整後仍存在 | 2 個：392.267s, 393.767s |
| Q0094 | 0427小窩季打 12 | 53 | 1190.833–1204.333 | 多個 score | 調整後仍存在 | 2 個：1202.000s, 1203.333s |
| Q0095 | 0914小窩季打 2 | 22 | 461.167–468.067 | score 不在最後 | 調整後未出現；原始問題保留供複查 | 事件序列：serve@462.167s, receive@463.033s, receive@464.333s, receive@465.667s, score@467.067s, receive@468.300s |
| Q0096 | 0914小窩季打 2 | 22 | 461.167–468.067 | 調整後移出 span 的動作 | 需複查；action 標註保留 | receive@468.300s |
| Q0097 | 2025-10-04_G3_臺中連莊_vs_桃園雲豹飛將_set1 | 13 | 353.888–357.323 | score 不在最後 | 調整後仍存在 | 事件序列：serve@354.888s, receive@355.822s, score@356.323s, set@356.957s；既有人工紀錄：越界（本次未重看影片） |
| Q0098 | 2025-10-04_G3_臺中連莊_vs_桃園雲豹飛將_set1 | 13 | 353.888–357.323 | 調整後移出 span 的動作 | 需複查；action 標註保留 | receive@358.341s |
| Q0099 | 2025-10-04_G4_臺北伊斯特_vs_台鋼天鷹_set1 | 30 | 951.101–954.486 | score 不在最後 | 調整後未出現；原始問題保留供複查 | 事件序列：serve@952.101s, score@953.486s, receive@954.854s |
| Q0100 | 2025-10-04_G4_臺北伊斯特_vs_台鋼天鷹_set1 | 30 | 951.101–954.486 | 調整後移出 span 的動作 | 需複查；action 標註保留 | receive@954.854s |
| Q0101 | 2025-10-05_G5_臺北伊斯特_vs_桃園雲豹飛將_set1 | 16 | 466.117–478.010 | score 不在最後 | 調整後仍存在 | 事件序列：serve@467.117s, receive@468.184s, set@469.452s, spike@469.836s, receive@470.854s, set@472.455s, spike@473.006s, block@473.189s, receive@473.740s, set@475.058s, spike@476.393s, block@476.459s, receive@477.010s, score@477.010s, receive@477.727s；既有人工紀錄：持球（本次未重看影片） |
| Q0102 | 2025-10-11_G8_臺中連莊_vs_桃園雲豹飛將_set1 | 1 | 5.300–7.590 | 缺少 serve | 調整後仍存在 | span 內沒有 serve；既有人工紀錄：導播問題（本次未重看影片） |
| Q0103 | 2025-10-25_G11_臺北伊斯特_vs_桃園雲豹飛將_set1 | 26 | 849.083–855.537 | score 不在最後 | 調整後仍存在 | 事件序列：serve@850.083s, receive@851.117s, set@852.952s, spike@854.420s, score@854.537s, block@854.554s, receive@855.071s；既有人工紀錄：打到標竿（本次未重看影片） |
| Q0104 | 2025-10-26_G14_臺北伊斯特_vs_臺中連莊_set1 | 37 | 1355.805–1362.110 | score 不在最後 | 調整後未出現；原始問題保留供複查 | 事件序列：serve@1356.805s, receive@1357.690s, set@1359.508s, spike@1360.693s, block@1360.743s, receive@1361.110s, score@1361.110s, set@1362.928s；既有人工紀錄：觸網（本次未重看影片） |
| Q0105 | 2025-10-26_G14_臺北伊斯特_vs_臺中連莊_set1 | 37 | 1355.805–1362.110 | 調整後移出 span 的動作 | 需複查；action 標註保留 | set@1362.928s |
| Q0106 | 2025-11-01_G15_臺中連莊_vs_臺北伊斯特_set1 | 32 | 1117.150–1123.104 | score 不在最後 | 調整後仍存在 | 事件序列：serve@1118.150s, receive@1119.352s, set@1120.686s, spike@1121.620s, score@1122.104s, receive@1122.171s；既有人工紀錄：觸網（本次未重看影片） |
| Q0107 | 2025-11-01_G15_臺中連莊_vs_臺北伊斯特_set1 | 36 | 1229.329–1238.236 | score 不在最後 | 調整後仍存在 | 事件序列：serve@1230.329s, receive@1231.330s, set@1232.631s, spike@1233.666s, receive@1234.016s, set@1236.135s, score@1237.236s, spike@1237.403s；既有人工紀錄：越界（本次未重看影片） |
| Q0108 | 2025-11-01_G16_桃園雲豹飛將_vs_台鋼天鷹_set1 | 13 | 396.000–402.133 | score 不在最後 | 調整後仍存在 | 事件序列：serve@397.000s, receive@398.067s, set@399.733s, spike@400.867s, score@401.133s, receive@401.167s；既有人工紀錄：觸網（本次未重看影片） |
| Q0109 | 2025-11-02_G17_桃園雲豹飛將_vs_臺北伊斯特_set1 | 4 | 91.593–100.916 | score 不在最後 | 調整後仍存在 | 事件序列：serve@92.593s, receive@93.460s, set@94.695s, spike@95.912s, block@95.963s, receive@97.114s, set@98.532s, score@99.916s, receive@100.584s；既有人工紀錄：越網擊球（本次未重看影片） |
| Q0110 | 2025-11-08_G19_臺北伊斯特_vs_臺中連莊_set1 | 1 | 0.101–5.922 | start 小於 0 | 調整後未出現；原始問題保留供複查 | start=-0.399s |
| Q0111 | 2025-11-08_G19_臺北伊斯特_vs_臺中連莊_set1 | 47 | 1772.205–1774.772 | score 不在最後 | 調整後仍存在 | 事件序列：serve@1773.205s, score@1773.772s, receive@1774.439s；既有人工紀錄：打到標竿（本次未重看影片） |
| Q0112 | 2025-11-08_G20_桃園雲豹飛將_vs_台鋼天鷹_set1 | 29 | 979.680–985.734 | score 不在最後 | 調整後仍存在 | 事件序列：serve@980.680s, receive@981.380s, set@983.249s, spike@984.350s, block@984.734s, score@984.734s, receive@985.184s；既有人工紀錄：觸網（本次未重看影片） |
| Q0113 | 2025-11-08_G20_桃園雲豹飛將_vs_台鋼天鷹_set1 | 29 | 979.680–985.734 | 調整後移出 span 的動作 | 需複查；action 標註保留 | set@986.486s |
| Q0114 | 20250504 大統OB-成功大學vs台北大學B-第二局 | 4 | 90.367–102.433 | score 不在最後 | 調整後仍存在 | 事件序列：serve@91.367s, receive@92.800s, set@94.200s, receive@95.867s, receive@97.900s, set@99.267s, spike@100.900s, score@101.433s, receive@101.800s；既有人工紀錄：觸網（本次未重看影片） |
| Q0115 | 20250504 大統OB-成功大學vs台北大學B-第二局 | 4 | 90.367–102.433 | 調整後移出 span 的動作 | 需複查；action 標註保留 | set@103.200s |
| Q0116 | 20251227-排島本館-6 | 19 | 464.732–473.238 | 多個 score | 調整後仍存在 | 2 個：470.804s, 472.238s |
| Q0117 | 20260108-排排棧-01 | 31 | 710.644–727.192 | 多個 score | 調整後仍存在 | 2 個：724.924s, 726.192s |
| Q0118 | 20260321-排島本館-01 | 1 | 0.401–28.427 | start 小於 0 | 調整後未出現；原始問題保留供複查 | start=-0.099s |
| Q0119 | 20260426-小窩-01 | 7 | 176.967–189.067 | 多個 score | 調整後仍存在 | 2 個：186.833s, 188.067s |
| Q0120 | 20260426-小窩-01 | 31 | 748.467–773.867 | score 不在最後 | 調整後未出現；原始問題保留供複查 | 事件序列：serve@749.467s, receive@750.367s, set@752.000s, spike@753.400s, receive@753.933s, set@755.633s, receive@757.233s, receive@759.067s, set@760.633s, spike@762.033s, receive@762.733s, spike@764.467s, receive@765.000s, set@766.733s, spike@768.033s, receive@768.533s, set@770.567s, receive@772.333s, score@772.867s, receive@774.567s, receive@776.067s, receive@776.933s, receive@778.200s, spike@779.467s, receive@779.867s, receive@781.933s, receive@783.267s, receive@784.367s, receive@785.767s；既有人工紀錄：觸網（本次未重看影片） |
| Q0121 | 20260426-小窩-01 | 31 | 748.467–773.867 | 調整後移出 span 的動作 | 需複查；action 標註保留 | receive@774.567s, receive@776.067s, receive@776.933s, receive@778.200s, spike@779.467s, receive@779.867s, receive@781.933s, receive@783.267s, receive@784.367s, receive@785.767s |
| Q0122 | 20260502-排島本館-01 | 2 | 134.402–142.208 | 多個 score | 調整後仍存在 | 2 個：140.140s, 141.208s |
| Q0123 | 20260502-排島本館-01 | 20 | 464.966–473.739 | 多個 score | 調整後仍存在 | 2 個：471.738s, 472.739s |
| Q0124 | 20260502-排島本館-02 | 7 | 245.913–251.984 | score 不在最後 | 調整後仍存在 | 事件序列：serve@246.913s, receive@248.281s, set@249.716s, spike@250.884s, block@250.951s, score@250.984s, receive@251.751s；既有人工紀錄：觸網（本次未重看影片） |
| Q0125 | 20260502-排島本館-02 | 8 | 262.396–274.774 | 多個 score | 調整後仍存在 | 2 個：272.839s, 273.774s |
| Q0126 | 20260502-排島本館-02 | 29 | 706.006–721.920 | 多個 score | 調整後仍存在 | 2 個：719.919s, 720.920s |
| Q0127 | 20260502-排島本館-02 | 34 | 828.962–836.535 | 多個 score | 調整後仍存在 | 2 個：834.534s, 835.535s |
| Q0128 | 37-39 Thriller! - Japan 🇯🇵 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 31 | 759.000–763.640 | 缺少 serve | 調整後仍存在 | span 內沒有 serve；既有人工紀錄：導播問題（本次未重看影片） |
| Q0129 | 37-39 Thriller! - Japan 🇯🇵 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set2 | 12 | 307.600–309.040 | 缺少 serve | 調整後仍存在 | span 內沒有 serve；既有人工紀錄：導播問題（本次未重看影片） |
| Q0130 | 37-39 Thriller! - Japan 🇯🇵 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set2 | 13 | 321.000–328.880 | 缺少 serve | 調整後仍存在 | span 內沒有 serve；既有人工紀錄：導播問題（本次未重看影片） |
| Q0131 | 37-39 Thriller! - Japan 🇯🇵 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set2 | 20 | 532.252–535.440 | 缺少 serve | 調整後仍存在 | span 內沒有 serve；既有人工紀錄：導播問題（本次未重看影片） |
| Q0132 | Brazil 🇧🇷 vs. Cuba 🇨🇺 ｜ VNL 2025 - Full Match ｜ Week 1_set5 | 1 | 0.000–5.600 | start 小於 0 | 調整後未出現；原始問題保留供複查 | start=-1.220s |
| Q0133 | Brazil 🇧🇷 vs. Cuba 🇨🇺 ｜ VNL 2025 - Full Match ｜ Week 1_set5 | 1 | 0.000–5.600 | 影片頭尾不足 1 秒 | 已截在影片範圍 | start 無法留滿 1 秒；duration=1099.634s |
| Q0134 | Brazil 🇧🇷 vs. Italy 🇮🇹  ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 9 | 193.658–196.000 | 缺少 serve | 調整後仍存在 | span 內沒有 serve；既有人工紀錄：導播問題（本次未重看影片） |
| Q0135 | Brazil 🇧🇷 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 39 | 1345.320–1351.800 | score 不在最後 | 調整後未出現；原始問題保留供複查 | 事件序列：serve@1346.320s, receive@1347.160s, set@1348.720s, spike@1349.800s, receive@1350.240s, score@1350.800s, set@1352.240s |
| Q0136 | Brazil 🇧🇷 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 39 | 1345.320–1351.800 | 調整後移出 span 的動作 | 需複查；action 標註保留 | set@1352.240s |
| Q0137 | Bulgaria 🇧🇬 vs. Argentina 🇦🇷 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 23 | 672.400–673.880 | score 不在最後 | 調整後仍存在 | 事件序列：score@672.880s, serve@673.400s；既有人工紀錄：發球踩線（本次未重看影片） |
| Q0138 | Bulgaria 🇧🇬 vs. Argentina 🇦🇷 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 23 | 672.400–673.880 | serve 不在最前 | 調整後仍存在 | 事件序列：score@672.880s, serve@673.400s；既有人工紀錄：發球踩線（本次未重看影片） |
| Q0139 | Bulgaria 🇧🇬 vs. Canada 🇨🇦 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 44 | 1552.920–1559.640 | score 不在最後 | 調整後未出現；原始問題保留供複查 | 事件序列：serve@1553.920s, receive@1554.720s, set@1555.760s, spike@1557.720s, block@1557.800s, receive@1558.280s, score@1558.640s, set@1560.120s |
| Q0140 | Bulgaria 🇧🇬 vs. Canada 🇨🇦 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 44 | 1552.920–1559.640 | 調整後移出 span 的動作 | 需複查；action 標註保留 | set@1560.120s |
| Q0141 | Bulgaria 🇧🇬 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 1 | 0.000–5.200 | start 小於 0 | 調整後未出現；原始問題保留供複查 | start=-1.260s |
| Q0142 | Bulgaria 🇧🇬 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 1 | 0.000–5.200 | 影片頭尾不足 1 秒 | 已截在影片範圍 | start 無法留滿 1 秒；duration=1801.481s |
| Q0143 | Bulgaria 🇧🇬 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 14 | 364.640–369.520 | score 不在最後 | 調整後仍存在 | 事件序列：serve@365.640s, receive@366.520s, set@368.080s, score@368.520s, spike@368.920s, receive@369.240s；既有人工紀錄：後排踩線（本次未重看影片） |
| Q0144 | Bulgaria 🇧🇬 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 14 | 364.640–369.520 | 調整後移出 span 的動作 | 需複查；action 標註保留 | receive@370.440s |
| Q0145 | Bulgaria 🇧🇬 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 26 | 793.120–799.840 | 多個 score | 調整後仍存在 | 2 個：797.760s, 798.840s |
| Q0146 | Bulgaria 🇧🇬 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 37 | 1133.000–1134.360 | score 不在最後 | 調整後仍存在 | 事件序列：score@1133.360s, serve@1134.000s；既有人工紀錄：發球踩線（本次未重看影片） |
| Q0147 | Bulgaria 🇧🇬 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 37 | 1133.000–1134.360 | serve 不在最前 | 調整後仍存在 | 事件序列：score@1133.360s, serve@1134.000s；既有人工紀錄：發球踩線（本次未重看影片） |
| Q0148 | Bulgaria 🇧🇬 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 37 | 1133.000–1134.360 | 調整後移出 span 的動作 | 需複查；action 標註保留 | receive@1134.680s, set@1136.400s |
| Q0149 | Bulgaria 🇧🇬 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 52 | 1793.240–1801.200 | end 超出影片 | 調整後未出現；原始問題保留供複查 | end=1801.700s；metadata duration=1801.481s |
| Q0150 | Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 1 | 0.000–9.840 | start 小於 0 | 調整後未出現；原始問題保留供複查 | start=-0.780s |
| Q0151 | Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 1 | 0.000–9.840 | 影片頭尾不足 1 秒 | 已截在影片範圍 | start 無法留滿 1 秒；duration=2106.026s |
| Q0152 | Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 4 | 91.880–97.760 | score 不在最後 | 調整後仍存在 | 事件序列：serve@92.880s, receive@93.960s, set@95.640s, spike@95.880s, score@96.760s, receive@97.600s；既有人工紀錄：落地後（本次未重看影片） |
| Q0153 | Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 10 | 244.240–245.800 | score 不在最後 | 調整後仍存在 | 事件序列：score@244.800s, serve@245.240s；既有人工紀錄：發球踩線（本次未重看影片） |
| Q0154 | Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 10 | 244.240–245.800 | serve 不在最前 | 調整後仍存在 | 事件序列：score@244.800s, serve@245.240s；既有人工紀錄：發球踩線（本次未重看影片） |
| Q0155 | Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 10 | 244.240–245.800 | 調整後移出 span 的動作 | 需複查；action 標註保留 | receive@245.880s |
| Q0156 | Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 31 | 937.880–951.760 | score 不在最後 | 調整後仍存在 | 事件序列：serve@938.880s, receive@939.840s, set@941.400s, spike@943.000s, receive@943.880s, set@945.600s, spike@947.160s, receive@947.960s, set@949.560s, spike@950.600s, block@950.720s, score@950.760s, receive@951.480s；既有人工紀錄：觸網（本次未重看影片） |
| Q0157 | Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 31 | 937.880–951.760 | 調整後移出 span 的動作 | 需複查；action 標註保留 | set@952.600s |
| Q0158 | Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 33 | 1005.416–1010.000 | 缺少 serve | 調整後仍存在 | span 內沒有 serve；既有人工紀錄：導播問題（本次未重看影片） |
| Q0159 | Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 50 | 2098.520–2101.520 | score 不在最後 | 調整後未出現；原始問題保留供複查 | 事件序列：serve@2099.520s, score@2100.520s, receive@2101.560s；既有人工紀錄：落地後（本次未重看影片） |
| Q0160 | Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 50 | 2098.520–2101.520 | 調整後移出 span 的動作 | 需複查；action 標註保留 | receive@2101.560s |
| Q0161 | Champions crowned in Final 24⧸25 (2⧸2) ｜ Suntory Sunbirds Osaka - Stings Aichi ｜ SV League 24⧸25_set1 | 18 | 498.900–506.000 | score 不在最後 | 調整後未出現；原始問題保留供複查 | 事件序列：serve@499.900s, receive@500.767s, set@502.500s, spike@503.533s, receive@504.167s, score@505.000s, spike@506.500s；既有人工紀錄：二擊（本次未重看影片） |
| Q0162 | Champions crowned in Final 24⧸25 (2⧸2) ｜ Suntory Sunbirds Osaka - Stings Aichi ｜ SV League 24⧸25_set1 | 18 | 498.900–506.000 | 調整後移出 span 的動作 | 需複查；action 標註保留 | spike@506.500s |
| Q0163 | China vs. Argentina - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 2 | 34.467–41.400 | score 不在最後 | 調整後仍存在 | 事件序列：serve@35.467s, receive@36.567s, set@38.167s, spike@39.567s, receive@40.300s, score@40.400s, set@41.367s；既有人工紀錄：觸網（本次未重看影片） |
| Q0164 | China vs. Brazil - Ranking 13-14 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 6 | 157.000–163.183 | score 不在最後 | 調整後仍存在 | 事件序列：serve@158.000s, receive@159.367s, set@160.800s, spike@162.000s, block@162.083s, score@162.183s, receive@162.967s；既有人工紀錄：觸網（本次未重看影片） |
| Q0165 | China vs. Brazil - Ranking 13-14 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 6 | 157.000–163.183 | 調整後移出 span 的動作 | 需複查；action 標註保留 | receive@164.900s |
| Q0166 | China vs. Brazil - Ranking 13-14 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 19 | 473.800–482.333 | score 不在最後 | 調整後仍存在 | 事件序列：serve@474.800s, receive@475.733s, set@477.367s, spike@477.750s, block@477.833s, receive@478.267s, set@479.633s, spike@481.300s, score@481.333s, block@481.350s, receive@482.100s；既有人工紀錄：打到標竿（本次未重看影片） |
| Q0167 | China 🇨🇳 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 3 | 117.360–124.280 | score 不在最後 | 調整後仍存在 | 事件序列：serve@118.360s, receive@119.000s, receive@121.000s, spike@122.760s, score@123.280s, receive@123.960s；既有人工紀錄：觸網（本次未重看影片） |
| Q0168 | China 🇨🇳 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 7 | 225.200–234.440 | 多個 score | 調整後仍存在 | 2 個：232.120s, 233.440s |
| Q0169 | Cuba vs. Puerto Rico - Ranking 17-18 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 22 | 616.333–629.133 | score 不在最後 | 調整後仍存在 | 事件序列：serve@617.333s, receive@618.467s, set@619.500s, spike@620.633s, block@620.750s, receive@621.100s, set@623.400s, spike@624.633s, block@624.733s, receive@625.167s, receive@626.100s, spike@627.967s, score@628.133s, receive@628.600s；既有人工紀錄：觸網（本次未重看影片） |
| Q0170 | Cuba vs. Puerto Rico - Ranking 17-18 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 22 | 616.333–629.133 | 調整後移出 span 的動作 | 需複查；action 標註保留 | receive@630.450s |
| Q0171 | Final - Stings vs. Sunbirds ｜ SVL League 2024⧸25 - Full Match ｜ Volleyball_set1 | 1 | 9.567–12.967 | score 不在最後 | 調整後未出現；原始問題保留供複查 | 事件序列：serve@10.567s, score@11.967s, serve@13.067s |
| Q0172 | Final - Stings vs. Sunbirds ｜ SVL League 2024⧸25 - Full Match ｜ Volleyball_set1 | 1 | 9.567–12.967 | 多個 serve | 調整後未出現；原始問題保留供複查 | 2 個：10.567s, 13.067s |
| Q0173 | Final - Stings vs. Sunbirds ｜ SVL League 2024⧸25 - Full Match ｜ Volleyball_set1 | 1 | 9.567–12.967 | 調整後移出 span 的動作 | 需複查；action 標註保留 | serve@13.067s |
| Q0174 | Final - Stings vs. Sunbirds ｜ SVL League 2024⧸25 - Full Match ｜ Volleyball_set1 | 29 | 1124.467–1129.367 | score 不在最後 | 調整後仍存在 | 事件序列：serve@1125.467s, receive@1125.933s, set@1127.833s, score@1128.367s, spike@1128.800s；既有人工紀錄：後排踩線（本次未重看影片） |
| Q0175 | Final - Stings vs. Sunbirds ｜ SVL League 2024⧸25 - Full Match ｜ Volleyball_set1 | 29 | 1124.467–1129.367 | 調整後移出 span 的動作 | 需複查；action 標註保留 | receive@1129.400s |
| Q0176 | France vs. Iran - Semi Final 2 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 3 | 58.700–72.433 | score 不在最後 | 調整後未出現；原始問題保留供複查 | 事件序列：serve@59.700s, receive@60.633s, set@62.267s, spike@63.267s, receive@64.067s, set@65.267s, spike@67.267s, receive@67.867s, set@69.333s, spike@70.567s, block@70.633s, receive@71.200s, score@71.433s, set@72.733s；既有人工紀錄：觸網（本次未重看影片） |
| Q0177 | France vs. Iran - Semi Final 2 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 3 | 58.700–72.433 | 調整後移出 span 的動作 | 需複查；action 標註保留 | set@72.733s |
| Q0178 | France vs. Iran - Semi Final 2 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 5 | 136.933–144.400 | score 不在最後 | 調整後仍存在 | 事件序列：serve@137.933s, receive@138.867s, set@140.333s, spike@140.800s, block@140.867s, receive@141.300s, set@142.200s, score@143.400s, spike@143.767s, block@143.833s；既有人工紀錄：觸網（本次未重看影片） |
| Q0179 | France vs. Iran - Semi Final 2 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 5 | 136.933–144.400 | 調整後移出 span 的動作 | 需複查；action 標註保留 | receive@144.433s |
| Q0180 | France vs. Iran - Semi Final 2 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 18 | 520.933–531.133 | score 不在最後 | 調整後仍存在 | 事件序列：serve@521.933s, receive@522.567s, receive@524.233s, set@525.633s, spike@526.800s, block@526.967s, receive@527.533s, set@528.400s, spike@530.000s, score@530.133s, receive@530.400s；既有人工紀錄：觸網（本次未重看影片） |
| Q0181 | France vs. Iran - Semi Final 2 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 18 | 520.933–531.133 | 調整後移出 span 的動作 | 需複查；action 標註保留 | set@531.400s |
| Q0182 | France vs. Iran - Semi Final 2 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 24 | 665.600–670.933 | 缺少 serve | 調整後仍存在 | span 內沒有 serve；既有人工紀錄：導播問題（本次未重看影片） |
| Q0183 | France 🇫🇷 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 19 | 593.680–599.640 | score 不在最後 | 調整後仍存在 | 事件序列：serve@594.680s, receive@595.720s, set@596.880s, spike@598.000s, block@598.120s, score@598.640s, receive@599.600s；既有人工紀錄：落地後（本次未重看影片） |
| Q0184 | France 🇫🇷 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 41 | 1272.240–1281.680 | score 不在最後 | 調整後仍存在 | 事件序列：serve@1273.240s, receive@1274.360s, set@1275.760s, spike@1276.960s, receive@1277.200s, receive@1278.440s, set@1280.080s, spike@1280.640s, score@1280.680s, receive@1281.080s；既有人工紀錄：觸網（本次未重看影片） |
| Q0185 | France 🇫🇷 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 41 | 1272.240–1281.680 | 調整後移出 span 的動作 | 需複查；action 標註保留 | receive@1282.040s |
| Q0186 | Full Match ｜ Bulgaria vs Luxembourg ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set1 | 31 | 882.633–896.767 | score 不在最後 | 調整後未出現；原始問題保留供複查 | 事件序列：serve@883.633s, receive@884.600s, set@886.767s, spike@888.567s, receive@889.433s, set@891.233s, spike@891.900s, receive@892.367s, set@893.833s, spike@895.100s, score@895.767s, receive@897.033s |
| Q0187 | Full Match ｜ Bulgaria vs Luxembourg ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set1 | 31 | 882.633–896.767 | 調整後移出 span 的動作 | 需複查；action 標註保留 | receive@897.033s |
| Q0188 | Full Match ｜ Bulgaria vs Luxembourg ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set3 | 37 | 1114.867–1119.933 | score 不在最後 | 調整後仍存在 | 事件序列：serve@1115.867s, receive@1116.967s, set@1118.433s, score@1118.933s, spike@1119.667s；既有人工紀錄：觸網（本次未重看影片） |
| Q0189 | Full Match ｜ Croatia vs. Bulgaria ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set1 | 24 | 603.200–611.200 | score 不在最後 | 調整後未出現；原始問題保留供複查 | 事件序列：serve@604.200s, receive@605.333s, set@606.333s, spike@607.667s, receive@608.300s, set@610.167s, score@610.200s, spike@612.000s；既有人工紀錄：二擊（本次未重看影片） |
| Q0190 | Full Match ｜ Croatia vs. Bulgaria ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set1 | 24 | 603.200–611.200 | 調整後移出 span 的動作 | 需複查；action 標註保留 | spike@612.000s |
| Q0191 | Full Match ｜ Croatia vs. Bulgaria ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set1 | 28 | 708.500–714.333 | score 不在最後 | 調整後仍存在 | 事件序列：serve@709.500s, receive@710.367s, set@711.867s, spike@712.533s, score@713.333s, receive@714.167s；既有人工紀錄：落地後（本次未重看影片） |
| Q0192 | Full Match ｜ Croatia vs. Serbia ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set1 | 21 | 536.033–552.833 | score 不在最後 | 調整後仍存在 | 事件序列：serve@537.033s, receive@537.967s, set@539.667s, spike@541.167s, block@541.200s, receive@541.800s, set@543.367s, spike@544.567s, block@544.633s, receive@545.133s, set@546.400s, spike@547.867s, block@548.000s, receive@548.767s, set@549.667s, spike@551.200s, block@551.267s, score@551.833s, receive@551.867s；既有人工紀錄：觸網（本次未重看影片） |
| Q0193 | Full Match ｜ Croatia vs. Serbia ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set1 | 21 | 536.033–552.833 | 調整後移出 span 的動作 | 需複查；action 標註保留 | receive@553.700s |
| Q0194 | Full Match ｜ Denmark vs. England ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool A_set1 | 11 | 437.200–444.633 | score 不在最後 | 調整後仍存在 | 事件序列：serve@438.200s, receive@438.933s, set@441.167s, spike@442.600s, score@443.633s, receive@444.433s；既有人工紀錄：落地後（本次未重看影片） |
| Q0195 | Full Match ｜ Denmark vs. England ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool A_set1 | 18 | 630.667–637.033 | score 不在最後 | 調整後仍存在 | 事件序列：serve@631.667s, receive@632.367s, set@634.367s, spike@635.233s, score@636.033s, receive@636.300s；既有人工紀錄：越界（本次未重看影片） |
| Q0196 | Full Match ｜ Denmark vs. England ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool A_set1 | 18 | 630.667–637.033 | 調整後移出 span 的動作 | 需複查；action 標註保留 | receive@637.733s |
| Q0197 | Full Match ｜ Ireland vs. Spain ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool D_set1 | 9 | 202.233–213.200 | score 不在最後 | 調整後仍存在 | 事件序列：serve@203.233s, receive@204.267s, set@206.067s, spike@207.467s, receive@208.533s, set@210.233s, spike@211.233s, score@212.200s, receive@213.067s；既有人工紀錄：落地後（本次未重看影片） |
| Q0198 | Full Match ｜ Ireland vs. Spain ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool D_set1 | 11 | 307.000–312.733 | score 不在最後 | 調整後仍存在 | 事件序列：serve@308.000s, receive@309.000s, set@310.933s, spike@311.367s, block@311.400s, score@311.733s, receive@311.767s, set@312.600s；既有人工紀錄：觸網（本次未重看影片） |
| Q0199 | Full Match ｜ Ireland vs. Türkiye ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool D_set1 | 14 | 418.833–424.367 | score 不在最後 | 調整後仍存在 | 事件序列：serve@419.833s, receive@420.967s, set@422.300s, score@423.367s, receive@424.200s；既有人工紀錄：落地後（本次未重看影片） |
| Q0200 | Full Match ｜ Ireland vs. Türkiye ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool D_set1 | 19 | 528.833–539.933 | score 不在最後 | 調整後未出現；原始問題保留供複查 | 事件序列：serve@529.833s, receive@531.200s, set@532.433s, spike@534.000s, block@534.100s, receive@534.833s, set@536.600s, spike@538.100s, score@538.933s, set@540.233s |
| Q0201 | Full Match ｜ Ireland vs. Türkiye ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool D_set1 | 19 | 528.833–539.933 | 調整後移出 span 的動作 | 需複查；action 標註保留 | set@540.233s |
| Q0202 | Full Match ｜ Italy vs. England ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool A_set1 | 30 | 1002.433–1012.300 | score 不在最後 | 調整後仍存在 | 事件序列：serve@1003.433s, receive@1004.000s, set@1005.767s, spike@1007.100s, receive@1008.067s, set@1009.233s, spike@1010.400s, block@1010.700s, score@1011.300s, receive@1011.900s；既有人工紀錄：落地後（本次未重看影片） |
| Q0203 | Full Match ｜ Luxembourg vs. Croatia - CEV U22 Volleyball European Championship 2026 ｜ Women ｜ Pool E_set1 | 1 | 5.533–13.000 | score 不在最後 | 調整後仍存在 | 事件序列：serve@6.533s, receive@7.600s, set@9.300s, spike@10.500s, receive@10.867s, receive@12.000s, score@12.000s, receive@12.733s；既有人工紀錄：持球（本次未重看影片） |
| Q0204 | Full Match ｜ Norway vs. Bulgaria ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool C_set1 | 21 | 555.200–561.433 | score 不在最後 | 調整後仍存在 | 事件序列：serve@556.200s, receive@556.833s, set@559.033s, spike@560.200s, score@560.433s, receive@560.633s, receive@561.200s；既有人工紀錄：觸網（本次未重看影片） |
| Q0205 | Full Match ｜ Poland vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 4 | 84.933–90.900 | score 不在最後 | 調整後仍存在 | 事件序列：serve@85.933s, receive@86.900s, set@88.767s, spike@89.833s, score@89.900s, receive@90.300s；既有人工紀錄：打到標竿（本次未重看影片） |
| Q0206 | Full Match ｜ Poland vs. Slovakia ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 21 | 550.700–562.900 | 缺少 serve | 調整後仍存在 | span 內沒有 serve；既有人工紀錄：導播問題（本次未重看影片） |
| Q0207 | Full Match ｜ Poland vs. Slovakia ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 26 | 718.467–722.967 | score 不在最後 | 調整後仍存在 | 事件序列：serve@719.467s, receive@720.600s, set@721.933s, score@721.967s, receive@722.933s；既有人工紀錄：持球（本次未重看影片） |
| Q0208 | Full Match ｜ Serbia vs. Bulgaria ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set1 | 20 | 491.367–499.367 | score 不在最後 | 調整後仍存在 | 事件序列：serve@492.367s, receive@493.400s, set@494.633s, spike@496.300s, receive@496.633s, score@498.367s, receive@498.933s |
| Q0209 | Full Match ｜ Serbia vs. Luxembourg ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set1 | 34 | 1045.967–1052.233 | score 不在最後 | 調整後仍存在 | 事件序列：serve@1046.967s, receive@1048.067s, set@1049.500s, receive@1050.833s, score@1051.233s, receive@1051.467s；既有人工紀錄：觸網（本次未重看影片） |
| Q0210 | Full Match ｜ Serbia vs. Luxembourg ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set1 | 34 | 1045.967–1052.233 | 調整後移出 span 的動作 | 需複查；action 標註保留 | receive@1052.500s |
| Q0211 | Full Match ｜ Slovakia vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 31 | 944.667–953.867 | score 不在最後 | 調整後仍存在 | 事件序列：serve@945.667s, receive@946.667s, set@948.400s, spike@949.567s, block@949.667s, receive@950.333s, set@951.200s, spike@952.767s, score@952.867s, receive@953.300s；既有人工紀錄：觸網（本次未重看影片） |
| Q0212 | Full Match ｜ Slovakia vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1 | 31 | 944.667–953.867 | 調整後移出 span 的動作 | 需複查；action 標註保留 | receive@954.033s |
| Q0213 | Full Match ｜ Spain vs The Netherlands ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool C_set1 | 13 | 466.933–478.400 | score 不在最後 | 調整後仍存在 | 事件序列：serve@467.933s, receive@468.867s, set@470.767s, spike@471.900s, receive@472.267s, spike@473.867s, block@473.900s, receive@474.400s, set@475.300s, spike@477.100s, block@477.267s, score@477.400s, receive@477.700s；既有人工紀錄：觸網（本次未重看影片） |
| Q0214 | Full Match ｜ Spain vs The Netherlands ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool C_set1 | 13 | 466.933–478.400 | 調整後移出 span 的動作 | 需複查；action 標註保留 | receive@478.867s |
| Q0215 | Full Match ｜ Türkiye vs. Spain ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool D_set1 | 40 | 1283.000–1294.367 | score 不在最後 | 調整後未出現；原始問題保留供複查 | 事件序列：serve@1284.000s, receive@1285.000s, set@1286.967s, spike@1288.000s, block@1288.067s, receive@1289.300s, set@1290.967s, spike@1292.000s, score@1293.367s, set@1294.800s, spike@1295.367s, receive@1295.533s；既有人工紀錄：no in（本次未重看影片） |
| Q0216 | Full Match ｜ Türkiye vs. Spain ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool D_set1 | 40 | 1283.000–1294.367 | 調整後移出 span 的動作 | 需複查；action 標註保留 | set@1294.800s, spike@1295.367s, receive@1295.533s |
| Q0217 | JT Thunders 🇯🇵 vs. Wolfdogs Nagoya 🇯🇵 ｜ SV League 2026 ｜ Full Match - Volleyball_set1 | 1 | 0.235–15.615 | start 小於 0 | 調整後未出現；原始問題保留供複查 | start=-0.265s |
| Q0218 | Japan vs. USA - Ranking 15-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 26 | 614.767–618.967 | score 不在最後 | 調整後未出現；原始問題保留供複查 | 事件序列：serve@615.767s, receive@616.400s, score@617.967s, receive@619.200s；既有人工紀錄：舉球後排越界（本次未重看影片） |
| Q0219 | Japan vs. USA - Ranking 15-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 26 | 614.767–618.967 | 調整後移出 span 的動作 | 需複查；action 標註保留 | receive@619.200s |
| Q0220 | Japan vs. USA - Ranking 15-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 45 | 1300.233–1306.417 | score 不在最後 | 調整後仍存在 | 事件序列：serve@1301.233s, receive@1302.367s, set@1304.033s, spike@1305.200s, block@1305.350s, score@1305.417s, receive@1306.167s；既有人工紀錄：觸網（本次未重看影片） |
| Q0221 | Japan 🇯🇵 vs. China 🇨🇳 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 18 | 530.720–538.960 | score 不在最後 | 調整後未出現；原始問題保留供複查 | 事件序列：serve@531.720s, receive@532.800s, set@534.240s, spike@535.760s, receive@536.440s, score@537.960s, set@539.120s |
| Q0222 | Japan 🇯🇵 vs. China 🇨🇳 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 18 | 530.720–538.960 | 調整後移出 span 的動作 | 需複查；action 標註保留 | set@539.120s |
| Q0223 | Japan 🇯🇵 vs. China 🇨🇳 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 40 | 1249.120–1263.880 | end 超出影片 | 調整後未出現；原始問題保留供複查 | end=1264.380s；metadata duration=1264.359s |
| Q0224 | Japan 🇯🇵 vs. Czechia 🇨🇿 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 22 | 588.880–594.840 | score 不在最後 | 調整後未出現；原始問題保留供複查 | 事件序列：serve@589.880s, receive@590.920s, set@592.440s, spike@593.360s, receive@593.760s, score@593.840s, receive@596.000s |
| Q0225 | Japan 🇯🇵 vs. Czechia 🇨🇿 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 22 | 588.880–594.840 | 調整後移出 span 的動作 | 需複查；action 標註保留 | receive@596.000s |
| Q0226 | Japan 🇯🇵 vs. Czechia 🇨🇿 ｜ VNL 2025 - Full Match ｜ Week 2_set1 | 42 | 1216.703–1222.160 | 缺少 serve | 調整後仍存在 | span 內沒有 serve；既有人工紀錄：導播問題（本次未重看影片） |
| Q0227 | Japan 🇯🇵 vs. Dominican Republic 🇩🇴 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 24 | 662.680–672.400 | score 不在最後 | 調整後仍存在 | 事件序列：serve@663.680s, receive@664.480s, set@666.560s, spike@667.880s, receive@668.320s, set@670.120s, spike@671.240s, score@671.400s, receive@671.680s；既有人工紀錄：觸網（本次未重看影片） |
| Q0228 | Japan 🇯🇵 vs. Dominican Republic 🇩🇴 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 24 | 662.680–672.400 | 調整後移出 span 的動作 | 需複查；action 標註保留 | set@673.520s |
| Q0229 | Japan 🇯🇵 vs. Dominican Republic 🇩🇴 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 29 | 806.560–812.360 | score 不在最後 | 調整後仍存在 | 事件序列：serve@807.560s, receive@808.520s, set@810.160s, spike@811.200s, score@811.360s, receive@811.520s；既有人工紀錄：觸網（本次未重看影片） |
| Q0230 | Japan 🇯🇵 vs. France 🇫🇷 ｜ VNL 2025 - Full Match ｜ Week 3_set1 | 1 | 0.160–6.680 | start 小於 0 | 調整後未出現；原始問題保留供複查 | start=-0.380s |
| Q0231 | Japan 🇯🇵 vs. France 🇫🇷 ｜ VNL 2025 - Full Match ｜ Week 3_set1 | 36 | 1077.600–1089.320 | score 不在最後 | 調整後仍存在 | 事件序列：serve@1078.600s, receive@1079.640s, set@1081.160s, spike@1082.240s, receive@1082.640s, set@1084.640s, spike@1086.480s, block@1086.560s, score@1088.320s, receive@1089.240s；既有人工紀錄：打到標竿（本次未重看影片） |
| Q0232 | Japan 🇯🇵 vs. France 🇫🇷 ｜ VNL 2025 - Full Match ｜ Week 3_set1 | 36 | 1077.600–1089.320 | 調整後移出 span 的動作 | 需複查；action 標註保留 | receive@1090.640s, set@1092.360s, spike@1093.480s, block@1093.560s, receive@1094.400s, set@1096.280s, spike@1097.480s, receive@1097.880s, receive@1099.880s, receive@1101.160s, receive@1103.320s, set@1104.720s, spike@1105.760s, block@1105.840s |
| Q0233 | Japan 🇯🇵 vs. France 🇫🇷 ｜ VNL 2025 - Full Match ｜ Week 3_set1 | 42 | 1481.300–1486.560 | 缺少 serve | 調整後仍存在 | span 內沒有 serve；既有人工紀錄：導播問題（本次未重看影片） |
| Q0234 | Japan 🇯🇵 vs. Netherlands 🇳🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 1 | 0.000–5.880 | start 小於 0 | 調整後未出現；原始問題保留供複查 | start=-0.980s |
| Q0235 | Japan 🇯🇵 vs. Netherlands 🇳🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 1 | 0.000–5.880 | 影片頭尾不足 1 秒 | 已截在影片範圍 | start 無法留滿 1 秒；duration=1291.792s |
| Q0236 | Japan 🇯🇵 vs. Netherlands 🇳🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 2 | 25.960–31.520 | score 不在最後 | 調整後仍存在 | 事件序列：serve@26.960s, receive@27.880s, set@29.320s, spike@30.400s, score@30.520s, receive@30.760s, receive@31.520s；既有人工紀錄：觸網（本次未重看影片） |
| Q0237 | Japan 🇯🇵 vs. Netherlands 🇳🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 2 | 25.960–31.520 | 調整後移出 span 的動作 | 需複查；action 標註保留 | spike@33.000s, receive@33.440s |
| Q0238 | Japan 🇯🇵 vs. Netherlands 🇳🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 23 | 654.480–667.840 | score 不在最後 | 調整後仍存在 | 事件序列：serve@655.480s, receive@656.080s, set@658.120s, spike@659.280s, receive@660.360s, set@661.840s, spike@663.040s, receive@663.400s, set@665.760s, spike@666.120s, block@666.240s, score@666.840s, receive@667.600s；既有人工紀錄：落地後（本次未重看影片） |
| Q0239 | Japan 🇯🇵 vs. Netherlands 🇳🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 23 | 654.480–667.840 | 調整後移出 span 的動作 | 需複查；action 標註保留 | receive@669.560s |
| Q0240 | Japan 🇯🇵 vs. Netherlands 🇳🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 31 | 907.160–912.320 | score 不在最後 | 調整後仍存在 | 事件序列：serve@908.160s, receive@909.120s, set@910.080s, spike@911.160s, score@911.320s, receive@911.520s；既有人工紀錄：觸網（本次未重看影片） |
| Q0241 | Japan 🇯🇵 vs. Netherlands 🇳🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 31 | 907.160–912.320 | 調整後移出 span 的動作 | 需複查；action 標註保留 | set@913.320s |
| Q0242 | Japan 🇯🇵 vs. Netherlands 🇳🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 43 | 1281.040–1287.840 | end 超出影片 | 調整後未出現；原始問題保留供複查 | end=1292.000s；metadata duration=1291.792s |
| Q0243 | Japan 🇯🇵 vs. Netherlands 🇳🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 43 | 1281.040–1287.840 | score 不在最後 | 調整後仍存在 | 事件序列：serve@1282.040s, receive@1283.000s, set@1284.680s, spike@1286.160s, score@1286.840s, receive@1287.200s；既有人工紀錄：觸網（本次未重看影片） |
| Q0244 | Japan 🇯🇵 vs. Netherlands 🇳🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 43 | 1281.040–1287.840 | 調整後移出 span 的動作 | 需複查；action 標註保留 | receive@1289.000s |
| Q0245 | Japan 🇯🇵 vs. Serbia 🇷🇸 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 38 | 1012.360–1018.640 | score 不在最後 | 調整後未出現；原始問題保留供複查 | 事件序列：serve@1013.360s, receive@1014.400s, set@1015.800s, spike@1016.720s, receive@1017.120s, score@1017.640s, spike@1019.000s |
| Q0246 | Japan 🇯🇵 vs. Serbia 🇷🇸 ｜ VNL 2025 - Full Match ｜ Week 1_set1 | 38 | 1012.360–1018.640 | 調整後移出 span 的動作 | 需複查；action 標註保留 | spike@1019.000s |
| Q0247 | Jtekt Stings 🇯🇵 - Suntory Sunbirds Osaka 🇯🇵 ｜ SV League 2026 ｜ Full Match - Volleyball_set1 | 31 | 1038.071–1049.614 | score 不在最後 | 調整後仍存在 | 事件序列：serve@1039.071s, receive@1039.972s, set@1041.257s, spike@1041.824s, set@1042.942s, spike@1044.477s, block@1044.577s, receive@1045.778s, set@1047.580s, spike@1048.014s, block@1048.080s, score@1048.614s, receive@1049.215s；既有人工紀錄：觸網（本次未重看影片） |
| Q0248 | Korea vs. Bulgaria - Classification 5-8 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 42 | 1194.300–1197.633 | 缺少 serve | 調整後仍存在 | span 內沒有 serve；既有人工紀錄：導播問題（本次未重看影片） |
| Q0249 | Korea vs. Finland - Ranking 11-12 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 21 | 445.833–450.417 | score 不在最後 | 調整後未出現；原始問題保留供複查 | 事件序列：serve@446.833s, receive@447.800s, set@449.267s, score@449.417s, spike@450.500s；既有人工紀錄：觸網（本次未重看影片） |
| Q0250 | Korea vs. Finland - Ranking 11-12 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 21 | 445.833–450.417 | 調整後移出 span 的動作 | 需複查；action 標註保留 | spike@450.500s |
| Q0251 | Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 11 | 292.493–298.631 | score 不在最後 | 調整後未出現；原始問題保留供複查 | 事件序列：serve@293.493s, receive@294.494s, set@295.796s, spike@296.830s, block@296.997s, receive@297.130s, receive@297.631s, score@297.631s, receive@299.032s；既有人工紀錄：觸網（本次未重看影片） |
| Q0252 | Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 11 | 292.493–298.631 | 調整後移出 span 的動作 | 需複查；action 標註保留 | receive@299.032s |
| Q0253 | Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 24 | 787.000–791.423 | 缺少 serve | 調整後仍存在 | span 內沒有 serve；既有人工紀錄：導播問題（本次未重看影片） |
| Q0254 | Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 26 | 823.190–829.661 | score 不在最後 | 調整後仍存在 | 事件序列：serve@824.190s, receive@825.325s, set@826.826s, spike@827.994s, block@828.227s, score@828.661s, receive@828.695s；既有人工紀錄：觸網（本次未重看影片） |
| Q0255 | Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 27 | 845.766–850.882 | 缺少 serve | 調整後仍存在 | span 內沒有 serve；既有人工紀錄：導播問題（本次未重看影片） |
| Q0256 | Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 44 | 1380.000–1386.951 | 缺少 serve | 調整後仍存在 | span 內沒有 serve；既有人工紀錄：導播問題（本次未重看影片） |
| Q0257 | Osaka Bluteon vs. Toray Arrows Shizuoka - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 31 | 1040.741–1046.044 | score 不在最後 | 調整後仍存在 | 事件序列：serve@1041.741s, receive@1042.742s, spike@1044.310s, score@1045.044s, set@1046.012s |
| Q0258 | Osaka Bluteon vs. Toray Arrows Shizuoka - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1 | 31 | 1040.741–1046.044 | 調整後移出 span 的動作 | 需複查；action 標註保留 | spike@1047.680s, block@1047.847s, receive@1048.447s, set@1049.882s, spike@1051.384s, receive@1052.184s |
| Q0259 | Osaka Bluteon 🇯🇵 vs. JTEKT Stings 🇯🇵 ｜ SV League 2026 ｜ Full Match - Volleyball_set1 | 23 | 683.651–689.471 | score 不在最後 | 調整後仍存在 | 事件序列：serve@684.651s, receive@685.218s, set@686.953s, spike@687.954s, block@688.137s, score@688.471s, receive@688.755s；既有人工紀錄：觸網（本次未重看影片） |
| Q0260 | Osaka Bluteon 🇯🇵 vs. JTEKT Stings 🇯🇵 ｜ SV League 2026 ｜ Full Match - Volleyball_set1 | 23 | 683.651–689.471 | 調整後移出 span 的動作 | 需複查；action 標註保留 | set@690.456s |
| Q0261 | Osaka Bluteon 🇯🇵 vs. JTEKT Stings 🇯🇵 ｜ SV League 2026 ｜ Full Match - Volleyball_set1 | 42 | 1348.515–1355.053 | score 不在最後 | 調整後仍存在 | 事件序列：serve@1349.515s, receive@1350.649s, set@1352.017s, spike@1353.052s, score@1354.053s, receive@1354.737s；既有人工紀錄：落地後（本次未重看影片） |
| Q0262 | Pakistan vs. USA - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 1 | 8.200–12.550 | score 不在最後 | 調整後仍存在 | 事件序列：serve@9.200s, receive@10.000s, set@11.533s, score@11.550s, spike@11.967s, receive@12.367s；既有人工紀錄：阻擋舉球（本次未重看影片） |
| Q0263 | Pakistan vs. USA - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 6 | 146.917–152.950 | score 不在最後 | 調整後仍存在 | 事件序列：serve@147.917s, receive@148.917s, set@150.700s, spike@151.767s, score@151.950s, receive@152.200s；既有人工紀錄：觸網（本次未重看影片） |
| Q0264 | Pakistan vs. USA - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 16 | 459.467–473.900 | score 不在最後 | 調整後未出現；原始問題保留供複查 | 事件序列：serve@460.467s, receive@461.300s, set@462.933s, spike@465.000s, receive@465.533s, set@466.900s, receive@468.967s, set@470.667s, spike@472.350s, block@472.417s, receive@472.900s, score@472.900s, receive@473.967s；既有人工紀錄：觸網（本次未重看影片） |
| Q0265 | Pakistan vs. USA - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 16 | 459.467–473.900 | 調整後移出 span 的動作 | 需複查；action 標註保留 | receive@473.967s |
| Q0266 | Poland vs. Spain - Semi Final 1 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 35 | 1198.200–1205.200 | score 不在最後 | 調整後仍存在 | 事件序列：serve@1199.200s, receive@1200.033s, set@1201.900s, spike@1203.700s, score@1204.200s, receive@1204.800s；既有人工紀錄：no in（本次未重看影片） |
| Q0267 | Semi Final 1 - Osaka Bluteon vs. Stings Aichi ｜ SV League - Full Match ｜ Volleyball_set1 | 8 | 225.867–231.400 | score 不在最後 | 調整後仍存在 | 事件序列：serve@226.867s, receive@227.500s, set@228.700s, spike@229.767s, block@229.833s, score@230.400s, receive@231.100s；既有人工紀錄：觸網（本次未重看影片） |
| Q0268 | Semi Final 1 - Osaka Bluteon vs. Stings Aichi ｜ SV League - Full Match ｜ Volleyball_set1 | 18 | 537.800–542.967 | score 不在最後 | 調整後仍存在 | 事件序列：serve@538.800s, receive@539.667s, set@541.533s, score@541.967s, spike@542.333s, block@542.433s；既有人工紀錄：後排踩線（本次未重看影片） |
| Q0269 | Semi Final 1 - Osaka Bluteon vs. Stings Aichi ｜ SV League - Full Match ｜ Volleyball_set1 | 34 | 1081.167–1095.600 | 多個 score | 調整後仍存在 | 2 個：1093.533s, 1094.600s |
| Q0270 | Semi Final 1 - Osaka Bluteon vs. Stings Aichi ｜ SV League - Full Match ｜ Volleyball_set1 | 42 | 1536.833–1553.833 | score 不在最後 | 調整後仍存在 | 事件序列：serve@1537.833s, receive@1538.767s, set@1540.467s, spike@1542.100s, block@1542.167s, receive@1543.767s, set@1545.467s, spike@1546.300s, block@1546.333s, receive@1546.667s, spike@1548.167s, block@1548.200s, receive@1549.133s, set@1550.800s, spike@1552.167s, block@1552.300s, score@1552.833s, receive@1553.233s；既有人工紀錄：觸網（本次未重看影片） |
| Q0271 | Semi Final 2 - Osaka Bluteon vs. Stings Aichi ｜ SV League - Full Match ｜ Volleyball_set1 | 30 | 990.500–993.300 | score 不在最後 | 調整後未出現；原始問題保留供複查 | 事件序列：serve@991.500s, score@992.300s, set@994.000s, spike@995.800s；既有人工紀錄：落地後（本次未重看影片） |
| Q0272 | Semi Final 2 - Osaka Bluteon vs. Stings Aichi ｜ SV League - Full Match ｜ Volleyball_set1 | 30 | 990.500–993.300 | 調整後移出 span 的動作 | 需複查；action 標註保留 | set@994.000s, spike@995.800s |
| Q0273 | Semi Final 2 - Osaka Bluteon vs. Stings Aichi ｜ SV League - Full Match ｜ Volleyball_set1 | 39 | 1389.467–1394.167 | score 不在最後 | 調整後仍存在 | 事件序列：serve@1390.467s, receive@1391.233s, score@1393.167s, block@1393.200s, receive@1393.933s；既有人工紀錄：舉球後排越界（本次未重看影片） |
| Q0274 | Semi Final 3 - Suntory Sunbirds vs. Wolfdogs Nagoya ｜ SVL Playoff - Full Match ｜ Volleyball_set1 | 46 | 1637.667–1642.433 | score 不在最後 | 調整後未出現；原始問題保留供複查 | 事件序列：serve@1638.667s, receive@1639.200s, score@1641.433s, spike@1643.367s, block@1643.467s；既有人工紀錄：越界救球（本次未重看影片） |
| Q0275 | Semi Final 3 - Suntory Sunbirds vs. Wolfdogs Nagoya ｜ SVL Playoff - Full Match ｜ Volleyball_set1 | 46 | 1637.667–1642.433 | 調整後移出 span 的動作 | 需複查；action 標註保留 | spike@1643.367s, block@1643.467s |
| Q0276 | Spain vs. Iran - Ranking 3-4 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 14 | 601.583–607.750 | score 不在最後 | 調整後仍存在 | 事件序列：serve@602.583s, receive@603.633s, receive@605.467s, score@606.750s, block@606.783s；既有人工紀錄：越網擊球（本次未重看影片） |
| Q0277 | Spain vs. Iran - Ranking 3-4 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 14 | 601.583–607.750 | 調整後移出 span 的動作 | 需複查；action 標註保留 | receive@607.767s |
| Q0278 | Spain vs. Iran - Ranking 3-4 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 16 | 663.967–674.167 | score 不在最後 | 調整後仍存在 | 事件序列：serve@664.967s, receive@665.933s, set@667.900s, spike@668.867s, block@669.017s, receive@669.400s, receive@670.000s, block@671.767s, receive@672.400s, score@673.167s, set@673.167s；既有人工紀錄：越界（本次未重看影片） |
| Q0279 | Spain vs. Iran - Ranking 3-4 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 16 | 663.967–674.167 | 調整後移出 span 的動作 | 需複查；action 標註保留 | spike@674.700s |
| Q0280 | Spain vs. Iran - Ranking 3-4 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 31 | 1198.000–1207.400 | 缺少 serve | 調整後仍存在 | span 內沒有 serve；既有人工紀錄：導播問題（本次未重看影片） |
| Q0281 | Spain vs. Iran - Ranking 3-4 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 46 | 1779.000–1789.050 | 缺少 serve | 調整後仍存在 | span 內沒有 serve；既有人工紀錄：導播問題（本次未重看影片） |
| Q0282 | Suntory Sunbirds vs. Osaka Bluteon ｜ SV.LEAGUE 2025⧸26 ｜ Full Match - Volleyball_set1 | 5 | 116.000–138.621 | 缺少 serve | 調整後仍存在 | span 內沒有 serve；既有人工紀錄：導播問題（本次未重看影片） |
| Q0283 | Suntory Sunbirds vs. Osaka Bluteon ｜ SV.LEAGUE 2025⧸26 ｜ Full Match - Volleyball_set1 | 23 | 653.654–659.041 | score 不在最後 | 調整後仍存在 | 事件序列：serve@654.654s, receive@655.355s, set@656.756s, spike@657.223s, score@658.041s, receive@658.975s；既有人工紀錄：落地後（本次未重看影片） |
| Q0284 | Suntory Sunbirds vs. Osaka Bluteon ｜ SV.LEAGUE 2025⧸26 ｜ Full Match - Volleyball_set1 | 33 | 955.789–972.037 | 多個 score | 調整後仍存在 | 2 個：969.986s, 971.037s |
| Q0285 | Suntory Sunbirds 🇯🇵 vs. Stings AICHI 🇯🇵 ｜ SV League 2026 ｜ Full Match - Japan Volleyball_set1 | 6 | 199.600–205.638 | score 不在最後 | 調整後仍存在 | 事件序列：serve@200.600s, receive@201.401s, set@202.853s, spike@204.037s, block@204.071s, score@204.638s, receive@205.539s；既有人工紀錄：觸網（本次未重看影片） |
| Q0286 | Suntory Sunbirds 🇯🇵 vs. Stings AICHI 🇯🇵 ｜ SV League 2026 ｜ Full Match - Japan Volleyball_set1 | 30 | 866.133–875.173 | score 不在最後 | 調整後仍存在 | 事件序列：serve@867.133s, receive@868.167s, set@869.502s, spike@869.969s, block@870.036s, receive@871.437s, set@872.772s, spike@873.706s, block@874.023s, score@874.173s, receive@874.474s；既有人工紀錄：打到標竿（本次未重看影片） |
| Q0287 | Taipei vs. Argentina - Playoffs ｜ Girls' U19 World Champs 2025 - Full Match_set1 | 7 | 209.233–225.417 | score 不在最後 | 調整後仍存在 | 事件序列：serve@210.233s, receive@211.200s, spike@212.567s, receive@213.133s, set@214.517s, spike@216.383s, receive@217.033s, set@218.633s, spike@219.833s, receive@220.300s, set@222.400s, spike@223.633s, score@224.417s, receive@224.617s；既有人工紀錄：觸網（本次未重看影片） |
| Q0288 | Taipei vs. Argentina - Playoffs ｜ Girls' U19 World Champs 2025 - Full Match_set1 | 7 | 209.233–225.417 | 調整後移出 span 的動作 | 需複查；action 標註保留 | set@225.967s |
| Q0289 | Taipei vs. Argentina - Playoffs ｜ Girls' U19 World Champs 2025 - Full Match_set1 | 22 | 777.967–784.217 | score 不在最後 | 調整後仍存在 | 事件序列：serve@778.967s, receive@780.100s, set@781.733s, spike@782.733s, score@783.217s, receive@783.733s；既有人工紀錄：觸網（本次未重看影片） |
| Q0290 | Taipei vs. Argentina - Playoffs ｜ Girls' U19 World Champs 2025 - Full Match_set1 | 22 | 777.967–784.217 | 調整後移出 span 的動作 | 需複查；action 標註保留 | set@785.333s |
| Q0291 | Türkiye vs. Colombia - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 9 | 334.167–339.900 | score 不在最後 | 調整後未出現；原始問題保留供複查 | 事件序列：serve@335.167s, receive@336.233s, set@337.633s, spike@338.200s, block@338.267s, receive@338.400s, score@338.900s, receive@340.733s；既有人工紀錄：二擊（本次未重看影片） |
| Q0292 | Türkiye vs. Colombia - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 9 | 334.167–339.900 | 調整後移出 span 的動作 | 需複查；action 標註保留 | receive@340.733s |
| Q0293 | Türkiye vs. Colombia - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 27 | 762.400–768.450 | score 不在最後 | 調整後仍存在 | 事件序列：serve@763.400s, receive@764.333s, set@765.833s, spike@766.433s, block@766.500s, score@767.450s, receive@767.467s；既有人工紀錄：觸網（本次未重看影片） |
| Q0294 | Türkiye vs. Colombia - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 27 | 762.400–768.450 | 調整後移出 span 的動作 | 需複查；action 標註保留 | receive@768.933s |
| Q0295 | Türkiye vs. Colombia - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 29 | 813.333–818.317 | score 不在最後 | 調整後未出現；原始問題保留供複查 | 事件序列：serve@814.333s, receive@815.133s, spike@816.783s, block@816.900s, score@817.317s, set@818.633s；既有人工紀錄：持球（本次未重看影片） |
| Q0296 | Türkiye vs. Colombia - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 29 | 813.333–818.317 | 調整後移出 span 的動作 | 需複查；action 標註保留 | set@818.633s |
| Q0297 | Türkiye vs. Colombia - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 31 | 939.833–947.567 | score 不在最後 | 調整後仍存在 | 事件序列：serve@940.833s, receive@941.800s, set@943.533s, spike@944.883s, block@945.000s, receive@945.600s, score@946.567s, receive@946.950s；既有人工紀錄：落地後（本次未重看影片） |
| Q0298 | Türkiye vs. Colombia - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 31 | 939.833–947.567 | 調整後移出 span 的動作 | 需複查；action 標註保留 | receive@948.333s, set@949.967s |
| Q0299 | Uzbekistan vs. Pakistan - Ranking 5-6 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 15 | 369.900–375.383 | score 不在最後 | 調整後未出現；原始問題保留供複查 | 事件序列：serve@370.900s, receive@371.933s, set@373.433s, spike@374.100s, block@374.183s, score@374.383s, receive@376.133s；既有人工紀錄：觸網（本次未重看影片） |
| Q0300 | Uzbekistan vs. Pakistan - Ranking 5-6 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 15 | 369.900–375.383 | 調整後移出 span 的動作 | 需複查；action 標註保留 | receive@376.133s |
| Q0301 | Uzbekistan vs. Pakistan - Ranking 5-6 ｜ Boys' U19 World Champs 2025 - Full Match_set1 | 16 | 395.000–404.150 | 缺少 serve | 調整後仍存在 | span 內沒有 serve；既有人工紀錄：導播問題（本次未重看影片） |
| Q0302 | ᴴᴰ114UVL預賽：：中原大學vs實踐大學：：男一級 大專排球聯賽 AI網路直播_set1 | 2 | 37.267–41.233 | 多個 score | 調整後仍存在 | 2 個：39.167s, 40.233s |
| Q0303 | ᴴᴰ114UVL預賽：：中原大學vs陽明交大：：男一級 大專排球聯賽 AI網路直播_set1 | 17 | 490.567–497.200 | score 不在最後 | 調整後仍存在 | 事件序列：serve@491.567s, receive@492.200s, set@494.167s, spike@495.667s, score@496.200s, receive@496.600s；既有人工紀錄：越界（本次未重看影片） |
| Q0304 | ᴴᴰ114UVL預賽：：中原大學vs陽明交大：：男一級 大專排球聯賽 AI網路直播_set1 | 23 | 698.033–701.967 | score 不在最後 | 調整後未出現；原始問題保留供複查 | 事件序列：serve@699.033s, receive@700.133s, score@700.967s, receive@702.200s；既有人工紀錄：持球（本次未重看影片） |
| Q0305 | ᴴᴰ114UVL預賽：：中原大學vs陽明交大：：男一級 大專排球聯賽 AI網路直播_set1 | 23 | 698.033–701.967 | 調整後移出 span 的動作 | 需複查；action 標註保留 | receive@702.200s |
| Q0306 | ᴴᴰ114UVL預賽：：中原大學vs陽明交大：：男一級 大專排球聯賽 AI網路直播_set1 | 24 | 716.367–723.033 | score 不在最後 | 調整後仍存在 | 事件序列：serve@717.367s, receive@718.433s, set@720.600s, spike@722.000s, score@722.033s, receive@722.367s；既有人工紀錄：打到標竿（本次未重看影片） |
| Q0307 | ᴴᴰ114UVL預賽：：中原大學vs陽明交大：：男一級 大專排球聯賽 AI網路直播_set1 | 38 | 1108.667–1116.533 | score 不在最後 | 調整後仍存在 | 事件序列：serve@1109.667s, receive@1110.800s, set@1112.433s, spike@1114.167s, block@1114.233s, receive@1115.033s, score@1115.533s, receive@1115.800s；既有人工紀錄：越界（本次未重看影片） |
| Q0308 | ᴴᴰ114UVL預賽：：中原大學vs陽明交大：：男一級 大專排球聯賽 AI網路直播_set1 | 39 | 1135.700–1140.933 | score 不在最後 | 調整後仍存在 | 事件序列：serve@1136.700s, receive@1137.667s, set@1139.233s, spike@1139.733s, score@1139.933s, receive@1140.433s；既有人工紀錄：觸網（本次未重看影片） |
| Q0309 | ᴴᴰ114UVL預賽：：中原大學vs陽明交大：：男一級 大專排球聯賽 AI網路直播_set1 | 39 | 1135.700–1140.933 | 調整後移出 span 的動作 | 需複查；action 標註保留 | set@1141.833s |
| Q0310 | ᴴᴰ114UVL預賽：：中山大學vs國北教大：：男一級 大專排球聯賽 AI網路直播_set1 | 15 | 395.767–400.233 | score 不在最後 | 調整後仍存在 | 事件序列：serve@396.767s, receive@398.200s, score@399.233s, receive@399.533s；既有人工紀錄：越網擊球（本次未重看影片） |
| Q0311 | ᴴᴰ114UVL預賽：：中山大學vs國北教大：：男一級 大專排球聯賽 AI網路直播_set1 | 15 | 395.767–400.233 | 調整後移出 span 的動作 | 需複查；action 標註保留 | set@401.067s |
| Q0312 | ᴴᴰ114UVL預賽：：中山大學vs國北教大：：男一級 大專排球聯賽 AI網路直播_set1 | 19 | 504.600–510.167 | score 不在最後 | 調整後仍存在 | 事件序列：serve@505.600s, receive@506.967s, set@507.967s, spike@509.033s, block@509.167s, score@509.167s, receive@510.033s；既有人工紀錄：觸網（本次未重看影片） |
| Q0313 | ᴴᴰ114UVL預賽：：中山大學vs國北教大：：男一級 大專排球聯賽 AI網路直播_set1 | 35 | 942.933–956.567 | score 不在最後 | 調整後仍存在 | 事件序列：serve@943.933s, receive@945.200s, set@946.367s, spike@946.867s, block@946.933s, receive@948.700s, set@950.633s, spike@952.033s, receive@953.100s, set@953.833s, spike@955.533s, score@955.567s, block@955.600s；既有人工紀錄：打到標竿（本次未重看影片） |
| Q0314 | ᴴᴰ114UVL預賽：：中山大學vs清華大學：：男一級 大專排球聯賽 AI網路直播_set1 | 16 | 367.733–376.500 | score 不在最後 | 調整後仍存在 | 事件序列：serve@368.733s, receive@369.967s, set@371.500s, receive@372.933s, receive@374.167s, spike@375.400s, score@375.500s, receive@376.233s；既有人工紀錄：觸網（本次未重看影片） |
| Q0315 | ᴴᴰ114UVL預賽：：中山大學vs清華大學：：男一級 大專排球聯賽 AI網路直播_set1 | 16 | 367.733–376.500 | 調整後移出 span 的動作 | 需複查；action 標註保留 | set@377.833s |
| Q0316 | ᴴᴰ114UVL預賽：：中山大學vs清華大學：：男一級 大專排球聯賽 AI網路直播_set1 | 39 | 1041.833–1050.833 | score 不在最後 | 調整後仍存在 | 事件序列：serve@1042.833s, receive@1043.900s, set@1045.633s, spike@1046.100s, block@1046.167s, receive@1046.900s, set@1048.367s, spike@1049.667s, block@1049.733s, score@1049.833s, receive@1050.333s；既有人工紀錄：觸網（本次未重看影片） |
| Q0317 | ᴴᴰ114UVL預賽：：中山大學vs清華大學：：男一級 大專排球聯賽 AI網路直播_set1 | 41 | 1114.600–1120.933 | score 不在最後 | 調整後仍存在 | 事件序列：serve@1115.600s, receive@1116.767s, set@1118.333s, spike@1119.600s, score@1119.933s, receive@1120.267s, receive@1120.733s；既有人工紀錄：觸網（本次未重看影片） |
| Q0318 | ᴴᴰ114UVL預賽：：中山大學vs清華大學：：男一級 大專排球聯賽 AI網路直播_set1 | 41 | 1114.600–1120.933 | 調整後移出 span 的動作 | 需複查；action 標註保留 | receive@1122.533s |
| Q0319 | ᴴᴰ114UVL預賽：：國北教大vs嘉義大學：：男一級 大專排球聯賽 AI網路直播_set1 | 1 | 32.800–39.100 | score 不在最後 | 調整後仍存在 | 事件序列：serve@33.800s, receive@35.000s, set@36.367s, spike@37.200s, score@38.100s, set@38.833s；既有人工紀錄：落地後（本次未重看影片） |
| Q0320 | ᴴᴰ114UVL預賽：：臺灣師大vs中山大學：：男一級 大專排球聯賽 AI網路直播_set1 | 3 | 49.567–60.200 | score 不在最後 | 調整後仍存在 | 事件序列：serve@50.567s, receive@51.600s, set@54.433s, receive@56.400s, set@57.700s, spike@58.733s, score@59.200s, receive@59.333s；既有人工紀錄：觸網（本次未重看影片） |
| Q0321 | ᴴᴰ114UVL預賽：：臺灣師大vs中山大學：：男一級 大專排球聯賽 AI網路直播_set1 | 23 | 554.300–562.700 | score 不在最後 | 調整後仍存在 | 事件序列：serve@555.300s, receive@556.067s, set@557.433s, spike@557.800s, block@557.833s, receive@558.367s, set@560.200s, spike@561.533s, score@561.700s, receive@562.000s；既有人工紀錄：觸網（本次未重看影片） |
| Q0322 | ᴴᴰ114UVL預賽：：臺灣師大vs中山大學：：男一級 大專排球聯賽 AI網路直播_set1 | 23 | 554.300–562.700 | 調整後移出 span 的動作 | 需複查；action 標註保留 | spike@563.633s |
| Q0323 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G21 11⧸9 15_00 桃園雲豹飛將 vs 臺中連莊_set1 | 39 | 1166.283–1201.433 | score 不在最後 | 調整後仍存在 | 事件序列：serve@1167.283s, receive@1167.883s, set@1169.602s, spike@1170.853s, receive@1171.337s, receive@1173.239s, set@1174.907s, spike@1175.941s, block@1176.058s, receive@1176.375s, receive@1177.126s, spike@1178.277s, receive@1179.228s, set@1180.813s, spike@1181.747s, receive@1182.148s, receive@1183.099s, spike@1184.884s, receive@1185.301s, set@1186.752s, spike@1188.220s, receive@1189.121s, set@1190.589s, spike@1191.674s, receive@1192.024s, set@1194.043s, receive@1196.328s, receive@1197.329s, set@1198.964s, spike@1200.282s, score@1200.433s, receive@1201.083s |
| Q0324 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G23 11⧸15 15_00 臺中連莊 vs 台鋼天鷹_set1 | 24 | 672.606–687.419 | score 不在最後 | 調整後未出現；原始問題保留供複查 | 事件序列：serve@673.606s, receive@674.374s, set@676.476s, spike@677.460s, receive@677.794s, set@679.913s, spike@681.831s, receive@683.032s, set@684.851s, spike@685.685s, receive@686.052s, score@686.419s, receive@688.471s |
| Q0325 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G23 11⧸15 15_00 臺中連莊 vs 台鋼天鷹_set1 | 24 | 672.606–687.419 | 調整後移出 span 的動作 | 需複查；action 標註保留 | receive@688.471s |
| Q0326 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G24 11⧸15 18_30 桃園雲豹飛將 vs 臺北伊斯特_set1 | 1 | 0.000–11.744 | start 小於 0 | 調整後未出現；原始問題保留供複查 | start=-1.050s |
| Q0327 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G24 11⧸15 18_30 桃園雲豹飛將 vs 臺北伊斯特_set1 | 1 | 0.000–11.744 | 影片頭尾不足 1 秒 | 已截在影片範圍 | start 無法留滿 1 秒；duration=1561.054s |
| Q0328 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G24 11⧸15 18_30 桃園雲豹飛將 vs 臺北伊斯特_set1 | 9 | 260.328–266.132 | score 不在最後 | 調整後仍存在 | 事件序列：serve@261.328s, receive@262.262s, set@263.597s, spike@264.631s, score@265.132s, receive@265.332s |
| Q0329 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G24 11⧸15 18_30 桃園雲豹飛將 vs 臺北伊斯特_set1 | 24 | 661.545–669.701 | 多個 score | 調整後仍存在 | 2 個：667.267s, 668.701s |
| Q0330 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G24 11⧸15 18_30 桃園雲豹飛將 vs 臺北伊斯特_set1 | 25 | 701.235–709.741 | score 不在最後 | 調整後仍存在 | 事件序列：serve@702.235s, receive@703.069s, set@704.370s, spike@705.538s, block@705.605s, receive@706.973s, score@708.741s, set@708.741s, spike@709.142s, block@709.192s |
| Q0331 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G24 11⧸15 18_30 桃園雲豹飛將 vs 臺北伊斯特_set1 | 45 | 1454.187–1470.001 | score 不在最後 | 調整後仍存在 | 事件序列：serve@1455.187s, receive@1456.088s, set@1457.923s, spike@1459.525s, block@1459.591s, receive@1460.592s, set@1462.094s, spike@1463.729s, block@1463.862s, receive@1464.964s, set@1466.765s, spike@1468.417s, score@1469.001s, receive@1469.335s |
| Q0332 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G25 11⧸16 15_00 桃園雲豹飛將 vs 台鋼天鷹_set1 | 17 | 489.800–498.480 | 缺少 serve | 調整後仍存在 | span 內沒有 serve；既有人工紀錄：導播問題（本次未重看影片） |
| Q0333 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G26 11⧸16 18_30 臺中連莊 vs 臺北伊斯特_set1 | 17 | 561.228–567.266 | score 不在最後 | 調整後仍存在 | 事件序列：serve@562.228s, receive@563.329s, set@564.731s, spike@565.799s, block@565.865s, score@566.266s, receive@566.800s |
| Q0334 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G26 11⧸16 18_30 臺中連莊 vs 臺北伊斯特_set1 | 17 | 561.228–567.266 | 調整後移出 span 的動作 | 需複查；action 標註保留 | receive@568.768s |
| Q0335 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G26 11⧸16 18_30 臺中連莊 vs 臺北伊斯特_set1 | 35 | 1235.318–1241.506 | score 不在最後 | 調整後未出現；原始問題保留供複查 | 事件序列：serve@1236.318s, receive@1237.436s, set@1238.637s, spike@1239.805s, score@1240.506s, receive@1241.507s |
| Q0336 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G26 11⧸16 18_30 臺中連莊 vs 臺北伊斯特_set1 | 35 | 1235.318–1241.506 | 調整後移出 span 的動作 | 需複查；action 標註保留 | receive@1241.507s |
| Q0337 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G26 11⧸16 18_30 臺中連莊 vs 臺北伊斯特_set1 | 58 | 2172.004–2178.876 | 多個 score | 調整後仍存在 | 2 個：2176.725s, 2177.876s |
| Q0338 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G27 11⧸22 15_00 桃園雲豹飛將 vs 台中連莊_set1 | 35 | 1086.887–1090.989 | score 不在最後 | 調整後未出現；原始問題保留供複查 | 事件序列：serve@1087.887s, receive@1089.221s, score@1089.989s, set@1091.123s |
| Q0339 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G27 11⧸22 15_00 桃園雲豹飛將 vs 台中連莊_set1 | 35 | 1086.887–1090.989 | 調整後移出 span 的動作 | 需複查；action 標註保留 | set@1091.123s |
| Q0340 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G29 11⧸23 15_00 臺北伊斯特 vs 台中連莊_set1 | 18 | 591.776–597.746 | score 不在最後 | 調整後仍存在 | 事件序列：serve@592.776s, receive@593.409s, set@595.528s, spike@596.713s, score@596.746s, block@596.813s, receive@597.380s |
| Q0341 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G30 11⧸23 18_30 桃園雲豹飛將 vs 台鋼天鷹_set1 | 36 | 1489.122–1495.994 | score 不在最後 | 調整後未出現；原始問題保留供複查 | 事件序列：serve@1490.122s, receive@1490.956s, spike@1492.508s, receive@1492.891s, score@1494.994s, receive@1496.128s |
| Q0342 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G30 11⧸23 18_30 桃園雲豹飛將 vs 台鋼天鷹_set1 | 36 | 1489.122–1495.994 | 調整後移出 span 的動作 | 需複查；action 標註保留 | receive@1496.128s |
| Q0343 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G30 11⧸23 18_30 桃園雲豹飛將 vs 台鋼天鷹_set1 | 44 | 1857.557–1863.027 | 多個 score | 調整後仍存在 | 2 個：1860.926s, 1862.027s |
| Q0344 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G6 10⧸5 18_30 臺中連莊 vs 台鋼天鷹_set1 | 6 | 172.490–183.465 | 缺少 score | 調整後仍存在 | span 內沒有 score |
| Q0345 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G6 10⧸5 18_30 臺中連莊 vs 台鋼天鷹_set1 | 25 | 732.983–737.619 | score 不在最後 | 調整後仍存在 | 事件序列：serve@733.983s, receive@735.118s, set@736.569s, block@736.619s, score@736.619s, spike@737.220s |
| Q0346 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G6 10⧸5 18_30 臺中連莊 vs 台鋼天鷹_set1 | 25 | 732.983–737.619 | 調整後移出 span 的動作 | 需複查；action 標註保留 | receive@738.037s |

## 無 action 標註：未調整的影片

共 606 支 / 26,557 rallies。這些影片無法得知正確 serve/score 時間，不能宣稱已改為前後 1 秒。

| 編號 | 影片 | Rally 數 | 處理結果 |
|---|---|---:|---|
| U0001 | 0104排島臨打 4 | 37 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0002 | 0104排島臨打 5 | 41 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0003 | 0104排島臨打 6 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0004 | 0104排島臨打 7 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0005 | 0104排島臨打 8 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0006 | 0104排島臨打 9 | 39 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0007 | 0112小窩季打 12 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0008 | 0112小窩季打 2 | 43 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0009 | 0112小窩季打 3 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0010 | 0112小窩季打 4 | 50 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0011 | 0112小窩季打 5 | 40 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0012 | 0112小窩季打 7 | 43 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0013 | 0112小窩季打 8 | 39 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0014 | 0112小窩季打 9 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0015 | 0225小窩臨打 5 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0016 | 0225小窩臨打 6 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0017 | 0225小窩臨打 7 | 52 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0018 | 0225小窩臨打 | 39 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0019 | 0316 小窩季打 12 | 41 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0020 | 0316小窩季打 10 | 31 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0021 | 0316小窩季打 1 | 42 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0022 | 0316小窩季打 2 | 43 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0023 | 0316小窩季打 4 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0024 | 0316小窩季打 5 | 44 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0025 | 0316小窩季打 6 | 44 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0026 | 0316小窩季打 7 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0027 | 0316小窩季打 8 | 50 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0028 | 0316小窩季打 9 | 75 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0029 | 0323小窩臨打 4 | 40 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0030 | 0323小窩臨打 5 | 42 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0031 | 0323小窩臨打 6 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0032 | 0323小窩臨打 7 | 50 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0033 | 0323小窩臨打 8 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0034 | 0323小窩臨打 9 | 41 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0035 | 03⧸14(六) 16_00｜例行賽G104 #獅子王 vs. #屏東台電｜企業21年甲級男女排球聯賽_set3 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0036 | 03⧸14(六) 16_00｜例行賽G104 #獅子王 vs. #屏東台電｜企業21年甲級男女排球聯賽_set4 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0037 | 03⧸14(六) 18_00｜例行賽G105 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set2 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0038 | 03⧸14(六) 18_00｜例行賽G105 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set3 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0039 | 03⧸14(六) 18_00｜例行賽G105 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set4 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0040 | 03⧸15(日) 15_00｜例行賽G107 #桃園臺灣產險 vs. #獅子王｜企業21年甲級男女排球聯賽_set2 | 43 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0041 | 03⧸15(日) 15_00｜例行賽G107 #桃園臺灣產險 vs. #獅子王｜企業21年甲級男女排球聯賽_set3 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0042 | 03⧸15(日) 15_00｜例行賽G107 #桃園臺灣產險 vs. #獅子王｜企業21年甲級男女排球聯賽_set4 | 44 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0043 | 03⧸20(五) 14_00｜挑戰賽G110 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set3 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0044 | 03⧸20(五) 14_00｜挑戰賽G110 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set4 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0045 | 03⧸20(五) 16_00｜挑戰賽G111 #屏東台電 vs. #桃園臺產｜企業21年甲級男女排球聯賽_set2 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0046 | 03⧸20(五) 16_00｜挑戰賽G111 #屏東台電 vs. #桃園臺產｜企業21年甲級男女排球聯賽_set3 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0047 | 03⧸20(五) 18_00｜挑戰賽G112 #臺北國北獅 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set2 | 50 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0048 | 03⧸20(五) 18_00｜挑戰賽G112 #臺北國北獅 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set3 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0049 | 03⧸21(六) 18_00｜男子組冠軍賽G114 #雲林美津濃 vs. #屏東台電｜企業21年甲級男女排球聯賽_set2 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0050 | 03⧸21(六) 18_00｜男子組冠軍賽G114 #雲林美津濃 vs. #屏東台電｜企業21年甲級男女排球聯賽_set4 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0051 | 0419小窩臨打 4 | 40 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0052 | 0419小窩臨打 5 | 41 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0053 | 0419小窩臨打 6 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0054 | 0419小窩臨打 7 | 52 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0055 | 0419小窩臨打 8 | 39 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0056 | 0420小窩季打 10 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0057 | 0420小窩季打 11 | 52 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0058 | 0420小窩季打 12 | 44 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0059 | 0420小窩季打 4 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0060 | 0420小窩季打 5 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0061 | 0420小窩季打 6 | 41 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0062 | 0420小窩季打 7 | 40 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0063 | 0420小窩季打 8 | 42 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0064 | 0420小窩季打 9 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0065 | 0601小窩季打 10 | 43 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0066 | 0601小窩季打 11 | 43 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0067 | 0601小窩季打 12 | 43 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0068 | 0601小窩季打 4 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0069 | 0601小窩季打 5 | 34 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0070 | 0601小窩季打 6 | 41 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0071 | 0601小窩季打 7 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0072 | 0601小窩季打 8 | 37 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0073 | 0601小窩季打 9 | 39 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0074 | 0727小窩季打 10 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0075 | 0727小窩季打 11 | 35 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0076 | 0727小窩季打 3 | 57 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0077 | 0727小窩季打 4 | 41 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0078 | 0727小窩季打 5 | 41 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0079 | 0727小窩季打 6 | 54 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0080 | 0727小窩季打 7 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0081 | 0727小窩季打 9 | 43 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0082 | 0803小窩季打 11 | 51 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0083 | 0803小窩季打 12 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0084 | 0803小窩季打 3 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0085 | 0803小窩季打 4 | 41 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0086 | 0803小窩季打 5 | 50 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0087 | 0803小窩季打 6 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0088 | 0803小窩季打 7 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0089 | 0803小窩季打 8 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0090 | 0803小窩季打 9 | 40 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0091 | 0914小窩季打 11 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0092 | 0914小窩季打 12 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0093 | 0914小窩季打 3 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0094 | 0914小窩季打 4 | 52 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0095 | 0914小窩季打 5 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0096 | 0914小窩季打 6 | 43 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0097 | 0914小窩季打 7 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0098 | 0914小窩季打 8 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0099 | 0914小窩季打 9 | 38 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0100 | 1005小窩臨打 10 | 42 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0101 | 1005小窩臨打 11 | 51 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0102 | 1005小窩臨打 12 | 37 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0103 | 1005小窩臨打 2 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0104 | 1005小窩臨打 3 | 50 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0105 | 1005小窩臨打 5 | 56 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0106 | 1005小窩臨打 6 | 40 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0107 | 1005小窩臨打 7 | 39 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0108 | 1005小窩臨打 8 | 43 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0109 | 1005小窩臨打 9 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0110 | 10⧸4 4 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0111 | 10⧸4 5 | 36 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0112 | 10⧸4 6 | 40 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0113 | 1102小窩季打 10 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0114 | 1102小窩季打 11 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0115 | 1102小窩季打 12 | 54 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0116 | 1102小窩季打 3 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0117 | 1102小窩季打 4 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0118 | 1102小窩季打 5 | 57 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0119 | 1102小窩季打 6 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0120 | 1102小窩季打 7 | 18 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0121 | 1102小窩季打 8 | 36 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0122 | 1102小窩季打 9 | 36 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0123 | 11⧸23小窩臨打1 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0124 | 11⧸23小窩臨打2 | 60 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0125 | 11⧸23小窩臨打3 | 49 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0126 | 11⧸23小窩臨打4 | 42 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0127 | 1203小窩全女臨打1 | 42 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0128 | 1203小窩全女臨打2 | 36 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0129 | 1203小窩全女臨打3 | 41 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0130 | 1203小窩全女臨打4 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0131 | 1203小窩全女臨打6 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0132 | 1203小窩全女臨打7 | 39 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0133 | 20241103 霖度C-3 | 35 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0134 | 20241103 霖度C-4 | 39 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0135 | 20241103 霖度C-5 | 55 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0136 | 20241103 霖度C-6 | 34 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0137 | 20241103 霖度C-7 | 44 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0138 | 20241103 霖度C-8 | 42 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0139 | 20241103 霖度C-9 | 40 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0140 | 2025-09-27_G1_臺北伊斯特_vs_臺中連莊_set2 | 43 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0141 | 2025-09-27_G1_臺北伊斯特_vs_臺中連莊_set3 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0142 | 2025-09-28_G2_臺北伊斯特_vs_桃園雲豹飛將_set3 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0143 | 2025-09-28_G2_臺北伊斯特_vs_桃園雲豹飛將_set4 | 64 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0144 | 2025-09-28_G2_臺北伊斯特_vs_桃園雲豹飛將_set5 | 26 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0145 | 2025-10-04_G3_臺中連莊_vs_桃園雲豹飛將_set2 | 39 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0146 | 2025-10-04_G3_臺中連莊_vs_桃園雲豹飛將_set3 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0147 | 2025-10-04_G4_臺北伊斯特_vs_台鋼天鷹_set2 | 44 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0148 | 2025-10-04_G4_臺北伊斯特_vs_台鋼天鷹_set3 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0149 | 2025-10-05_G5_臺北伊斯特_vs_桃園雲豹飛將_set2 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0150 | 2025-10-05_G5_臺北伊斯特_vs_桃園雲豹飛將_set3 | 42 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0151 | 2025-10-11_G7_臺北伊斯特_vs_台鋼天鷹_set2 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0152 | 2025-10-11_G7_臺北伊斯特_vs_台鋼天鷹_set3 | 50 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0153 | 2025-10-11_G7_臺北伊斯特_vs_台鋼天鷹_set4 | 44 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0154 | 2025-10-11_G7_臺北伊斯特_vs_台鋼天鷹_set5 | 25 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0155 | 2025-10-11_G8_臺中連莊_vs_桃園雲豹飛將_set2 | 50 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0156 | 2025-10-11_G8_臺中連莊_vs_桃園雲豹飛將_set3 | 40 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0157 | 2025-10-12_G10_臺北伊斯特_vs_桃園雲豹飛將_set2 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0158 | 2025-10-12_G10_臺北伊斯特_vs_桃園雲豹飛將_set3 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0159 | 2025-10-12_G10_臺北伊斯特_vs_桃園雲豹飛將_set4 | 44 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0160 | 2025-10-12_G9_臺中連莊_vs_台鋼天鷹_set2 | 50 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0161 | 2025-10-12_G9_臺中連莊_vs_台鋼天鷹_set3 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0162 | 2025-10-25_G11_臺北伊斯特_vs_桃園雲豹飛將_set2 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0163 | 2025-10-25_G11_臺北伊斯特_vs_桃園雲豹飛將_set3 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0164 | 2025-10-25_G11_臺北伊斯特_vs_桃園雲豹飛將_set4 | 52 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0165 | 2025-10-25_G11_臺北伊斯特_vs_桃園雲豹飛將_set5 | 26 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0166 | 2025-10-25_G12_台鋼天鷹_vs_臺中連莊_set2 | 49 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0167 | 2025-10-25_G12_台鋼天鷹_vs_臺中連莊_set3 | 41 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0168 | 2025-10-25_G12_台鋼天鷹_vs_臺中連莊_set4 | 40 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0169 | 2025-10-25_G12_台鋼天鷹_vs_臺中連莊_set5 | 25 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0170 | 2025-10-26_G13_台鋼天鷹_vs_桃園雲豹飛將_set2 | 40 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0171 | 2025-10-26_G13_台鋼天鷹_vs_桃園雲豹飛將_set3 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0172 | 2025-10-26_G13_台鋼天鷹_vs_桃園雲豹飛將_set4 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0173 | 2025-10-26_G14_臺北伊斯特_vs_臺中連莊_set2 | 38 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0174 | 2025-10-26_G14_臺北伊斯特_vs_臺中連莊_set3 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0175 | 2025-10-26_G14_臺北伊斯特_vs_臺中連莊_set4 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0176 | 2025-10-26_G14_臺北伊斯特_vs_臺中連莊_set5 | 27 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0177 | 2025-11-01_G15_臺中連莊_vs_臺北伊斯特_set2 | 52 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0178 | 2025-11-01_G15_臺中連莊_vs_臺北伊斯特_set3 | 42 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0179 | 2025-11-01_G15_臺中連莊_vs_臺北伊斯特_set4 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0180 | 2025-11-01_G16_桃園雲豹飛將_vs_台鋼天鷹_set2 | 42 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0181 | 2025-11-01_G16_桃園雲豹飛將_vs_台鋼天鷹_set3 | 44 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0182 | 2025-11-01_G16_桃園雲豹飛將_vs_台鋼天鷹_set4 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0183 | 2025-11-01_G16_桃園雲豹飛將_vs_台鋼天鷹_set5 | 28 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0184 | 2025-11-02_G17_桃園雲豹飛將_vs_臺北伊斯特_set2 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0185 | 2025-11-02_G17_桃園雲豹飛將_vs_臺北伊斯特_set3 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0186 | 2025-11-02_G17_桃園雲豹飛將_vs_臺北伊斯特_set4 | 44 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0187 | 2025-11-02_G18_臺中連莊_vs_台鋼天鷹_set2 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0188 | 2025-11-02_G18_臺中連莊_vs_台鋼天鷹_set3 | 40 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0189 | 2025-11-08_G19_臺北伊斯特_vs_臺中連莊_set2 | 44 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0190 | 2025-11-08_G19_臺北伊斯特_vs_臺中連莊_set3 | 58 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0191 | 2025-11-08_G19_臺北伊斯特_vs_臺中連莊_set4 | 42 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0192 | 2025-11-08_G20_桃園雲豹飛將_vs_台鋼天鷹_set2 | 44 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0193 | 2025-11-08_G20_桃園雲豹飛將_vs_台鋼天鷹_set3 | 52 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0194 | 2025-11-08_G20_桃園雲豹飛將_vs_台鋼天鷹_set4 | 70 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0195 | 2025-11-08_G20_桃園雲豹飛將_vs_台鋼天鷹_set5 | 28 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0196 | 20250413 霖度C-1 | 50 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0197 | 20250413 霖度C-2 | 24 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0198 | 20250413 霖度C-3 | 40 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0199 | 20250413 霖度C-4 | 40 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0200 | 20250413 霖度C-6 | 38 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0201 | 20250413 霖度C-7 | 39 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0202 | 20250413 霖度C-8 | 72 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0203 | 20250420 霖度C-10 | 57 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0204 | 20250420 霖度C-2 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0205 | 20250420 霖度C-3 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0206 | 20250420 霖度C-4 | 51 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0207 | 20250420 霖度C-5 | 40 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0208 | 20250420 霖度C-6 | 51 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0209 | 20250420 霖度C-7 | 50 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0210 | 20250420 霖度C-8 | 41 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0211 | 20250420 霖度C-9 | 51 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0212 | 20250424 排島惡館-1 | 24 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0213 | 20250424 排島惡館-2 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0214 | 20250424 排島惡館-3 | 41 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0215 | 20250424 排島惡館-4 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0216 | 20250424 排島惡館-5 | 50 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0217 | 20250503 大統OB-成功大學vs中科大-友誼賽 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0218 | 20250503 大統OB-成功大學vs台北大學A-第一局 | 44 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0219 | 20250503 大統OB-成功大學vs台北大學B-第一局 | 39 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0220 | 20250503 大統OB-成功大學vs台北大學B-第二局 | 42 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0221 | 20250503 大統OB-成功大學vs輔仁大學-第一局 | 39 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0222 | 20250503 大統OB-成功大學vs輔仁大學-第二局 | 35 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0223 | 20250503 大統OB-成功大學vs靜宜大學-第一局 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0224 | 20250503 大統OB-成功大學vs靜宜大學-第三局-1 | 14 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0225 | 20250503 大統OB-成功大學vs靜宜大學-第三局-2 | 16 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0226 | 20250503 大統OB-成功大學vs靜宜大學-第二局 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0227 | 20250504 大統OB-成功大學vs中科大-第一局 | 39 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0228 | 20250504 大統OB-成功大學vs中科大-第二局 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0229 | 20250504 大統OB-成功大學vs台北大學A-第二局 | 42 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0230 | 20250504 大統OB-成功大學vs台北大學B-第一局 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0231 | 20250621 排島本館-1 | 40 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0232 | 20250621 排島本館-3 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0233 | 20250621 排島本館-4 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0234 | 20250621 排島本館-5 | 53 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0235 | 20250621 排島本館-6 | 44 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0236 | 20250621 排島本館-7 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0237 | 20250628-霖度C-2 | 44 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0238 | 20250628-霖度C-3 | 36 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0239 | 20250628-霖度C-4 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0240 | 20250628-霖度C-5 | 37 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0241 | 20250628-霖度C-6 | 43 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0242 | 20250628-霖度C-7 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0243 | 20250918-排島本館-1 | 41 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0244 | 20250918-排島本館-2 | 35 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0245 | 20250918-排島本館-3 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0246 | 20250918-排島本館-4 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0247 | 20250918-排島本館-6 | 39 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0248 | 20250918-排島本館-7 | 38 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0249 | 20250918-排島本館-8 | 49 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0250 | 20250918-排島本館-9 | 42 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0251 | 20251012史派克(1) | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0252 | 20251012史派克(2) | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0253 | 20251012史派克(3) | 42 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0254 | 20251012史派克(4) | 44 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0255 | 20251012史派克(5) | 44 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0256 | 20251012史派克(6) | 53 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0257 | 20251109-排島本館-01 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0258 | 20251109-排島本館-02 | 39 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0259 | 20251109-排島本館-04 | 38 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0260 | 20251109-排島本館-05 | 43 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0261 | 20251109-排島本館-06 | 54 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0262 | 20251109-排島本館-07 | 35 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0263 | 20251109-排島本館-08 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0264 | 20251109-排島本館-09 | 21 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0265 | 20251109-排島本館-10 | 26 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0266 | 20251109-排島本館-11 | 20 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0267 | 20251227-排島本館-1 | 40 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0268 | 20251227-排島本館-2 | 42 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0269 | 20251227-排島本館-4 | 41 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0270 | 20251227-排島本館-5 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0271 | 20251227-排島本館-7 | 33 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0272 | 20251227-排島本館-8 | 42 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0273 | 20251227-排島本館-9 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0274 | 20260108-排排棧-03 | 43 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0275 | 20260108-排排棧-04 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0276 | 20260108-排排棧-05 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0277 | 20260108-排排棧-06 | 50 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0278 | 20260108-排排棧-07 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0279 | 20260321-排島本館-02 | 38 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0280 | 20260321-排島本館-03 | 42 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0281 | 20260321-排島本館-04 | 37 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0282 | 20260321-排島本館-05 | 36 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0283 | 20260321-排島本館-06 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0284 | 20260321-排島本館-07 | 43 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0285 | 20260321-排島本館-08 | 42 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0286 | 20260321-排島本館-09 | 29 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0287 | 20260321-排島本館-10 | 42 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0288 | 20260321-排島本館-11 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0289 | 20260403-霖度C-03 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0290 | 20260403-霖度C-04 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0291 | 20260426-小窩-02 | 35 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0292 | 20260426-小窩-04 | 44 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0293 | 20260426-小窩-05 | 40 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0294 | 20260426-小窩-06 | 40 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0295 | 20260426-小窩-07 | 50 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0296 | 20260426-小窩-08 | 40 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0297 | 20260426-小窩-09 | 38 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0298 | 20260426-小窩-10 | 43 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0299 | 20260426-小窩-11 | 40 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0300 | 20260426-小窩-12 | 52 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0301 | 20260502-排島本館-03 | 42 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0302 | 20260502-排島本館-04 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0303 | 20260502-排島本館-05 | 49 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0304 | 20260502-排島本館-06 | 38 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0305 | 20260502-排島本館-07 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0306 | 20260502-排島本館-08 | 42 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0307 | 20260502-排島本館-09 | 43 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0308 | 20260502-排島本館-11 | 43 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0309 | 20260502-排島本館-12 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0310 | 20260507 工資管友誼賽1 | 51 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0311 | 20260507 工資管友誼賽3 | 60 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0312 | 20260510邷力豹臨打2 | 44 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0313 | 20260510邷力豹臨打3 | 43 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0314 | 20260510邷力豹臨打4 | 43 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0315 | 20260510邷力豹臨打5 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0316 | 20260510邷力豹臨打6 | 42 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0317 | 20260510邷力豹臨打7 | 39 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0318 | 20260510邷力豹臨打8 | 38 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0319 | 20260510邷力豹臨打9 | 43 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0320 | 20260514-排排棧-01 | 44 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0321 | 20260514-排排棧-02 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0322 | 20260514-排排棧-03 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0323 | 20260514-排排棧-04 | 52 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0324 | 2026⧸02⧸25 1 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0325 | 2026⧸02⧸25 2 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0326 | 2026⧸02⧸25 3 | 39 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0327 | 2026⧸02⧸25 5 | 41 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0328 | 2026⧸02⧸25 6 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0329 | 2026⧸02⧸25 7 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0330 | 2026⧸02⧸25 8 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0331 | 2026⧸02⧸25 9 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0332 | 2026⧸03⧸25 1 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0333 | 2026⧸03⧸25 2 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0334 | 2026⧸03⧸25 4 | 42 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0335 | 2026⧸03⧸25 5 | 42 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0336 | 2026⧸03⧸25 6 | 37 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0337 | 2026⧸03⧸25 7 | 44 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0338 | 2026⧸03⧸25 8 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0339 | 2026⧸03⧸25 9 | 24 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0340 | 37-39 Thriller! - Japan 🇯🇵 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set3 | 76 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0341 | 7⧸30（三）小窩季打團play-1🦖 | 34 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0342 | Algeria vs. Canada - Ranking 23-24 ｜ Boys' U19 World Champs 2025 - Full Match_set2 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0343 | Algeria vs. Canada - Ranking 23-24 ｜ Boys' U19 World Champs 2025 - Full Match_set3 | 58 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0344 | Argentina vs. Belgium - Ranking 9-10 ｜ Boys' U19 World Champs 2025 - Full Match_set2 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0345 | Argentina vs. Belgium - Ranking 9-10 ｜ Boys' U19 World Champs 2025 - Full Match_set3 | 49 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0346 | Argentina vs. Belgium - Ranking 9-10 ｜ Boys' U19 World Champs 2025 - Full Match_set4 | 43 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0347 | Belgium vs. Thailand - Playoffs ｜ Girls' U19 World Champs 2025 - Full Match_set2 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0348 | Belgium vs. Thailand - Playoffs ｜ Girls' U19 World Champs 2025 - Full Match_set3 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0349 | Belgium vs. Thailand - Playoffs ｜ Girls' U19 World Champs 2025 - Full Match_set4 | 42 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0350 | Brazil vs. Belgium - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set2 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0351 | Brazil vs. Belgium - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set3 | 41 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0352 | Brazil vs. Belgium - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set4 | 41 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0353 | Brazil 🇧🇷 vs. Cuba 🇨🇺 ｜ VNL 2025 - Full Match ｜ Week 1_set2 | 50 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0354 | Brazil 🇧🇷 vs. Cuba 🇨🇺 ｜ VNL 2025 - Full Match ｜ Week 1_set3 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0355 | Brazil 🇧🇷 vs. Cuba 🇨🇺 ｜ VNL 2025 - Full Match ｜ Week 1_set4 | 43 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0356 | Brazil 🇧🇷 vs. Italy 🇮🇹  ｜ VNL 2025 - Full Match ｜ Week 1_set2 | 43 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0357 | Brazil 🇧🇷 vs. Italy 🇮🇹  ｜ VNL 2025 - Full Match ｜ Week 1_set3 | 56 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0358 | Brazil 🇧🇷 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 2_set2 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0359 | Brazil 🇧🇷 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 2_set3 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0360 | Brazil 🇧🇷 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 2_set4 | 54 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0361 | Bulgaria vs. Italy - Ranking 7-8 ｜ Boys' U19 World Champs 2025 - Full Match_set2 | 43 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0362 | Bulgaria vs. Italy - Ranking 7-8 ｜ Boys' U19 World Champs 2025 - Full Match_set3 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0363 | Bulgaria 🇧🇬 vs. Argentina 🇦🇷 ｜ VNL 2025 - Full Match ｜ Week 1_set2 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0364 | Bulgaria 🇧🇬 vs. Argentina 🇦🇷 ｜ VNL 2025 - Full Match ｜ Week 1_set3 | 54 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0365 | Bulgaria 🇧🇬 vs. Canada 🇨🇦 ｜ VNL 2025 - Full Match ｜ Week 1_set2 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0366 | Bulgaria 🇧🇬 vs. Canada 🇨🇦 ｜ VNL 2025 - Full Match ｜ Week 1_set3 | 44 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0367 | Bulgaria 🇧🇬 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set2 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0368 | Bulgaria 🇧🇬 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set3 | 43 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0369 | Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set2 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0370 | Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set3 | 44 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0371 | Champions crowned in Final 24⧸25 (2⧸2) ｜ Suntory Sunbirds Osaka - Stings Aichi ｜ SV League 24⧸25_set2 | 41 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0372 | Champions crowned in Final 24⧸25 (2⧸2) ｜ Suntory Sunbirds Osaka - Stings Aichi ｜ SV League 24⧸25_set3 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0373 | China vs. Argentina - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set2 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0374 | China vs. Argentina - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set3 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0375 | China vs. Argentina - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set4 | 67 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0376 | China vs. Argentina - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set5 | 28 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0377 | China vs. Brazil - Ranking 13-14 ｜ Boys' U19 World Champs 2025 - Full Match_set2 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0378 | China vs. Brazil - Ranking 13-14 ｜ Boys' U19 World Champs 2025 - Full Match_set3 | 50 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0379 | China 🇨🇳 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set2 | 39 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0380 | China 🇨🇳 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set3 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0381 | Cuba vs. Puerto Rico - Ranking 17-18 ｜ Boys' U19 World Champs 2025 - Full Match_set2 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0382 | Cuba vs. Puerto Rico - Ranking 17-18 ｜ Boys' U19 World Champs 2025 - Full Match_set3 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0383 | Final - Stings vs. Sunbirds ｜ SVL League 2024⧸25 - Full Match ｜ Volleyball_set2 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0384 | Final - Stings vs. Sunbirds ｜ SVL League 2024⧸25 - Full Match ｜ Volleyball_set3 | 50 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0385 | Final - Stings vs. Sunbirds ｜ SVL League 2024⧸25 - Full Match ｜ Volleyball_set4 | 62 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0386 | Final - Stings vs. Sunbirds ｜ SVL League 2024⧸25 - Full Match ｜ Volleyball_set5 | 36 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0387 | France vs. Iran - Semi Final 2 ｜ Boys' U19 World Champs 2025 - Full Match_set2 | 39 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0388 | France vs. Iran - Semi Final 2 ｜ Boys' U19 World Champs 2025 - Full Match_set3 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0389 | France vs. Iran - Semi Final 2 ｜ Boys' U19 World Champs 2025 - Full Match_set4 | 44 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0390 | France 🇫🇷 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 2_set2 | 44 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0391 | France 🇫🇷 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 2_set3 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0392 | France 🇫🇷 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 2_set4 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0393 | France 🇫🇷 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 2_set5 | 26 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0394 | Full Match ｜ Croatia vs. Bulgaria ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set2 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0395 | Full Match ｜ Croatia vs. Bulgaria ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set3 | 44 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0396 | Full Match ｜ Croatia vs. Bulgaria ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set4 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0397 | Full Match ｜ Croatia vs. Serbia ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set2 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0398 | Full Match ｜ Croatia vs. Serbia ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set3 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0399 | Full Match ｜ Croatia vs. Serbia ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set4 | 44 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0400 | Full Match ｜ Croatia vs. Serbia ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set5 | 25 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0401 | Full Match ｜ Denmark vs. England ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool A_set2 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0402 | Full Match ｜ Denmark vs. England ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool A_set3 | 38 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0403 | Full Match ｜ Ireland vs. Spain ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool D_set2 | 37 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0404 | Full Match ｜ Ireland vs. Spain ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool D_set3 | 33 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0405 | Full Match ｜ Ireland vs. Türkiye ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool D_set2 | 33 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0406 | Full Match ｜ Ireland vs. Türkiye ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool D_set3 | 32 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0407 | Full Match ｜ Italy vs. England ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool A_set2 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0408 | Full Match ｜ Italy vs. England ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool A_set3 | 40 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0409 | Full Match ｜ Luxembourg vs. Croatia - CEV U22 Volleyball European Championship 2026 ｜ Women ｜ Pool E_set2 | 41 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0410 | Full Match ｜ Luxembourg vs. Croatia - CEV U22 Volleyball European Championship 2026 ｜ Women ｜ Pool E_set3 | 34 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0411 | Full Match ｜ Norway vs. Bulgaria ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool C_set2 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0412 | Full Match ｜ Norway vs. Bulgaria ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool C_set3 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0413 | Full Match ｜ Norway vs. Ireland ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool D_set2 | 40 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0414 | Full Match ｜ Norway vs. Ireland ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool D_set3 | 43 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0415 | Full Match ｜ Poland vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set2 | 39 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0416 | Full Match ｜ Poland vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set3 | 34 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0417 | Full Match ｜ Poland vs. Slovakia ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set2 | 39 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0418 | Full Match ｜ Poland vs. Slovakia ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set3 | 40 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0419 | Full Match ｜ Poland vs. Slovakia ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set4 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0420 | Full Match ｜ Serbia vs. Bulgaria ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set2 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0421 | Full Match ｜ Serbia vs. Bulgaria ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set3 | 38 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0422 | Full Match ｜ Serbia vs. Bulgaria ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set4 | 41 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0423 | Full Match ｜ Serbia vs. Luxembourg ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set2 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0424 | Full Match ｜ Serbia vs. Luxembourg ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set3 | 35 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0425 | Full Match ｜ Slovakia vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set2 | 41 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0426 | Full Match ｜ Slovakia vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set3 | 32 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0427 | Full Match ｜ Spain vs The Netherlands ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool C_set2 | 43 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0428 | Full Match ｜ Spain vs The Netherlands ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool C_set3 | 41 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0429 | Full Match ｜ Spain vs The Netherlands ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool C_set4 | 42 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0430 | Full Match ｜ Spain vs The Netherlands ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool C_set5 | 30 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0431 | Full Match ｜ Türkiye vs. Spain ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool D_set2 | 41 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0432 | Full Match ｜ Türkiye vs. Spain ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool D_set3 | 49 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0433 | Full Match ｜ Türkiye vs. Spain ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool D_set4 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0434 | Italy vs. Finland - Classificationn 5-8 ｜ Boys' U19 World Champs 2025 - Full Match_set2 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0435 | JT Thunders 🇯🇵 vs. Wolfdogs Nagoya 🇯🇵 ｜ SV League 2026 ｜ Full Match - Volleyball_set2 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0436 | JT Thunders 🇯🇵 vs. Wolfdogs Nagoya 🇯🇵 ｜ SV League 2026 ｜ Full Match - Volleyball_set3 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0437 | JT Thunders 🇯🇵 vs. Wolfdogs Nagoya 🇯🇵 ｜ SV League 2026 ｜ Full Match - Volleyball_set4 | 44 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0438 | JT Thunders 🇯🇵 vs. Wolfdogs Nagoya 🇯🇵 ｜ SV League 2026 ｜ Full Match - Volleyball_set5 | 30 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0439 | Japan vs. USA - Ranking 15-16 ｜ Boys' U19 World Champs 2025 - Full Match_set2 | 41 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0440 | Japan vs. USA - Ranking 15-16 ｜ Boys' U19 World Champs 2025 - Full Match_set3 | 40 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0441 | Japan vs. USA - Ranking 15-16 ｜ Boys' U19 World Champs 2025 - Full Match_set4 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0442 | Japan vs. USA - Ranking 15-16 ｜ Boys' U19 World Champs 2025 - Full Match_set5 | 27 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0443 | Japan 🇯🇵 vs. China 🇨🇳 ｜ VNL 2025 - Full Match ｜ Week 2_set2 | 37 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0444 | Japan 🇯🇵 vs. China 🇨🇳 ｜ VNL 2025 - Full Match ｜ Week 2_set3 | 43 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0445 | Japan 🇯🇵 vs. China 🇨🇳 ｜ VNL 2025 - Full Match ｜ Week 2_set4 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0446 | Japan 🇯🇵 vs. Czechia 🇨🇿 ｜ VNL 2025 - Full Match ｜ Week 2_set2 | 42 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0447 | Japan 🇯🇵 vs. Czechia 🇨🇿 ｜ VNL 2025 - Full Match ｜ Week 2_set3 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0448 | Japan 🇯🇵 vs. Dominican Republic 🇩🇴 ｜ VNL 2025 - Full Match ｜ Week 1_set2 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0449 | Japan 🇯🇵 vs. Dominican Republic 🇩🇴 ｜ VNL 2025 - Full Match ｜ Week 1_set3 | 34 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0450 | Japan 🇯🇵 vs. France 🇫🇷 ｜ VNL 2025 - Full Match ｜ Week 3_set2 | 41 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0451 | Japan 🇯🇵 vs. France 🇫🇷 ｜ VNL 2025 - Full Match ｜ Week 3_set3 | 44 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0452 | Japan 🇯🇵 vs. Netherlands 🇳🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set2 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0453 | Japan 🇯🇵 vs. Netherlands 🇳🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set3 | 43 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0454 | Japan 🇯🇵 vs. Serbia 🇷🇸 ｜ VNL 2025 - Full Match ｜ Week 1_set2 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0455 | Japan 🇯🇵 vs. Serbia 🇷🇸 ｜ VNL 2025 - Full Match ｜ Week 1_set3 | 50 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0456 | Jtekt Stings 🇯🇵 - Suntory Sunbirds Osaka 🇯🇵 ｜ SV League 2026 ｜ Full Match - Volleyball_set2 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0457 | Jtekt Stings 🇯🇵 - Suntory Sunbirds Osaka 🇯🇵 ｜ SV League 2026 ｜ Full Match - Volleyball_set3 | 44 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0458 | Jtekt Stings 🇯🇵 - Suntory Sunbirds Osaka 🇯🇵 ｜ SV League 2026 ｜ Full Match - Volleyball_set4 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0459 | Korea vs. Bulgaria - Classification 5-8 ｜ Boys' U19 World Champs 2025 - Full Match_set2 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0460 | Korea vs. Bulgaria - Classification 5-8 ｜ Boys' U19 World Champs 2025 - Full Match_set3 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0461 | Korea vs. Finland - Ranking 11-12 ｜ Boys' U19 World Champs 2025 - Full Match_set2 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0462 | Korea vs. Finland - Ranking 11-12 ｜ Boys' U19 World Champs 2025 - Full Match_set3 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0463 | Korea vs. Finland - Ranking 11-12 ｜ Boys' U19 World Champs 2025 - Full Match_set4 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0464 | Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set2 | 43 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0465 | Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set3 | 58 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0466 | Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set4 | 52 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0467 | Osaka Bluteon vs. Toray Arrows Shizuoka - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set2 | 51 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0468 | Osaka Bluteon vs. Toray Arrows Shizuoka - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set3 | 43 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0469 | Osaka Bluteon 🇯🇵 vs. JTEKT Stings 🇯🇵 ｜ SV League 2026 ｜ Full Match - Volleyball_set2 | 43 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0470 | Osaka Bluteon 🇯🇵 vs. JTEKT Stings 🇯🇵 ｜ SV League 2026 ｜ Full Match - Volleyball_set3 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0471 | Pakistan vs. USA - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set2 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0472 | Pakistan vs. USA - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set3 | 43 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0473 | Pakistan vs. USA - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set4 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0474 | Poland vs. France - Final ｜ Boys' U19 World Champs 2025 - Full Match_set2 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0475 | Poland vs. France - Final ｜ Boys' U19 World Champs 2025 - Full Match_set3 | 40 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0476 | Poland vs. France - Final ｜ Boys' U19 World Champs 2025 - Full Match_set4 | 36 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0477 | Poland vs. Spain - Semi Final 1 ｜ Boys' U19 World Champs 2025 - Full Match_set2 | 41 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0478 | Poland vs. Spain - Semi Final 1 ｜ Boys' U19 World Champs 2025 - Full Match_set3 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0479 | Semi Final 1 - Osaka Bluteon vs. Stings Aichi ｜ SV League - Full Match ｜ Volleyball_set2 | 43 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0480 | Semi Final 1 - Osaka Bluteon vs. Stings Aichi ｜ SV League - Full Match ｜ Volleyball_set3 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0481 | Semi Final 1 - Osaka Bluteon vs. Stings Aichi ｜ SV League - Full Match ｜ Volleyball_set4 | 40 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0482 | Semi Final 1 - Suntory Sunbirds vs. Wolfdogs Nagoya ｜ SVL Playoff - Full Match ｜ Volleyball_set2 | 39 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0483 | Semi Final 1 - Suntory Sunbirds vs. Wolfdogs Nagoya ｜ SVL Playoff - Full Match ｜ Volleyball_set3 | 52 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0484 | Semi Final 1 - Suntory Sunbirds vs. Wolfdogs Nagoya ｜ SVL Playoff - Full Match ｜ Volleyball_set4 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0485 | Semi Final 1 - Suntory Sunbirds vs. Wolfdogs Nagoya ｜ SVL Playoff - Full Match ｜ Volleyball_set5 | 32 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0486 | Semi Final 2 - Osaka Bluteon vs. Stings Aichi ｜ SV League - Full Match ｜ Volleyball_set2 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0487 | Semi Final 2 - Osaka Bluteon vs. Stings Aichi ｜ SV League - Full Match ｜ Volleyball_set3 | 40 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0488 | Semi Final 2 - Osaka Bluteon vs. Stings Aichi ｜ SV League - Full Match ｜ Volleyball_set4 | 52 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0489 | Semi Final 2 - Osaka Bluteon vs. Stings Aichi ｜ SV League - Full Match ｜ Volleyball_set5 | 27 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0490 | Semi Final 3 - Suntory Sunbirds vs. Wolfdogs Nagoya ｜ SVL Playoff - Full Match ｜ Volleyball_set2 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0491 | Semi Final 3 - Suntory Sunbirds vs. Wolfdogs Nagoya ｜ SVL Playoff - Full Match ｜ Volleyball_set3 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0492 | Semi Final 3 - Suntory Sunbirds vs. Wolfdogs Nagoya ｜ SVL Playoff - Full Match ｜ Volleyball_set4 | 53 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0493 | Spain vs. Iran - Ranking 3-4 ｜ Boys' U19 World Champs 2025 - Full Match_set2 | 40 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0494 | Spain vs. Iran - Ranking 3-4 ｜ Boys' U19 World Champs 2025 - Full Match_set3 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0495 | Spain vs. Iran - Ranking 3-4 ｜ Boys' U19 World Champs 2025 - Full Match_set4 | 25 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0496 | Suntory Sunbirds vs. Osaka Bluteon ｜ SV.LEAGUE 2025⧸26 ｜ Full Match - Volleyball_set2 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0497 | Suntory Sunbirds vs. Osaka Bluteon ｜ SV.LEAGUE 2025⧸26 ｜ Full Match - Volleyball_set3 | 50 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0498 | Suntory Sunbirds vs. Osaka Bluteon ｜ SV.LEAGUE 2025⧸26 ｜ Full Match - Volleyball_set4 | 44 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0499 | Suntory Sunbirds 🇯🇵 vs. Stings AICHI 🇯🇵 ｜ SV League 2026 ｜ Full Match - Japan Volleyball_set2 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0500 | Suntory Sunbirds 🇯🇵 vs. Stings AICHI 🇯🇵 ｜ SV League 2026 ｜ Full Match - Japan Volleyball_set3 | 43 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0501 | Taipei vs. Argentina - Playoffs ｜ Girls' U19 World Champs 2025 - Full Match_set2 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0502 | Taipei vs. Argentina - Playoffs ｜ Girls' U19 World Champs 2025 - Full Match_set3 | 41 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0503 | Taipei vs. Argentina - Playoffs ｜ Girls' U19 World Champs 2025 - Full Match_set4 | 43 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0504 | Toray Arrows Shizuoka vs Phitsanulok Volleyball Club - Full Match ｜ SV. League World Tour 2025_set2 | 38 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0505 | Toray Arrows Shizuoka vs Phitsanulok Volleyball Club - Full Match ｜ SV. League World Tour 2025_set3 | 40 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0506 | Türkiye vs. Colombia - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set2 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0507 | Türkiye vs. Colombia - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set3 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0508 | Türkiye vs. Colombia - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set4 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0509 | Türkiye vs. Colombia - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set5 | 27 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0510 | Uzbekistan vs. Japan - Ranking 19-20 ｜ Boys' U19 World Champs 2025 - Full Match_set2 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0511 | Uzbekistan vs. Japan - Ranking 19-20 ｜ Boys' U19 World Champs 2025 - Full Match_set3 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0512 | Uzbekistan vs. Japan - Ranking 19-20 ｜ Boys' U19 World Champs 2025 - Full Match_set4 | 39 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0513 | Uzbekistan vs. Pakistan - Ranking 5-6 ｜ Boys' U19 World Champs 2025 - Full Match_set2 | 44 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0514 | Uzbekistan vs. Pakistan - Ranking 5-6 ｜ Boys' U19 World Champs 2025 - Full Match_set3 | 43 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0515 | Wolfdogs Nagoya 🇯🇵 vs. Osaka Bluteon 🇯🇵 ｜ SV League 2026 ｜ Full Match - Japan Volleyball_set1 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0516 | Wolfdogs Nagoya 🇯🇵 vs. Osaka Bluteon 🇯🇵 ｜ SV League 2026 ｜ Full Match - Japan Volleyball_set2 | 40 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0517 | Wolfdogs Nagoya 🇯🇵 vs. Osaka Bluteon 🇯🇵 ｜ SV League 2026 ｜ Full Match - Japan Volleyball_set3 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0518 | Wolfdogs Nagoya 🇯🇵 vs. Osaka Bluteon 🇯🇵 ｜ SV League 2026 ｜ Full Match - Japan Volleyball_set4 | 44 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0519 | Wolfdogs Nagoya 🇯🇵 vs. Sakai Blazers 🇯🇵 ｜ SV League 2026 ｜ Full Match - Japan Volleyball_set1 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0520 | Wolfdogs Nagoya 🇯🇵 vs. Sakai Blazers 🇯🇵 ｜ SV League 2026 ｜ Full Match - Japan Volleyball_set2 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0521 | Wolfdogs Nagoya 🇯🇵 vs. Sakai Blazers 🇯🇵 ｜ SV League 2026 ｜ Full Match - Japan Volleyball_set3 | 43 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0522 | ᴴᴰ114UVL預賽：：中原大學vs實踐大學：：男一級 大專排球聯賽 AI網路直播_set2 | 41 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0523 | ᴴᴰ114UVL預賽：：中原大學vs實踐大學：：男一級 大專排球聯賽 AI網路直播_set3 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0524 | ᴴᴰ114UVL預賽：：中原大學vs實踐大學：：男一級 大專排球聯賽 AI網路直播_set4 | 41 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0525 | ᴴᴰ114UVL預賽：：中原大學vs陽明交大：：男一級 大專排球聯賽 AI網路直播_set2 | 40 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0526 | ᴴᴰ114UVL預賽：：中原大學vs陽明交大：：男一級 大專排球聯賽 AI網路直播_set3 | 40 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0527 | ᴴᴰ114UVL預賽：：中山大學vs國北教大：：男一級 大專排球聯賽 AI網路直播_set2 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0528 | ᴴᴰ114UVL預賽：：中山大學vs國北教大：：男一級 大專排球聯賽 AI網路直播_set3 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0529 | ᴴᴰ114UVL預賽：：中山大學vs清華大學：：男一級 大專排球聯賽 AI網路直播_set2 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0530 | ᴴᴰ114UVL預賽：：中山大學vs清華大學：：男一級 大專排球聯賽 AI網路直播_set3 | 42 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0531 | ᴴᴰ114UVL預賽：：中山大學vs清華大學：：男一級 大專排球聯賽 AI網路直播_set4 | 40 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0532 | ᴴᴰ114UVL預賽：：嘉義大學vs中山大學：：男一級 大專排球聯賽 AI網路直播_set1 | 44 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0533 | ᴴᴰ114UVL預賽：：嘉義大學vs中山大學：：男一級 大專排球聯賽 AI網路直播_set2 | 39 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0534 | ᴴᴰ114UVL預賽：：嘉義大學vs中山大學：：男一級 大專排球聯賽 AI網路直播_set3 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0535 | ᴴᴰ114UVL預賽：：嘉義大學vs中山大學：：男一級 大專排球聯賽 AI網路直播_set4 | 74 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0536 | ᴴᴰ114UVL預賽：：嘉義大學vs中山大學：：男一級 大專排球聯賽 AI網路直播_set5 | 32 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0537 | ᴴᴰ114UVL預賽：：嘉義大學vs臺灣師大：：男一級 大專排球聯賽 AI網路直播_set1 | 43 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0538 | ᴴᴰ114UVL預賽：：嘉義大學vs臺灣師大：：男一級 大專排球聯賽 AI網路直播_set2 | 38 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0539 | ᴴᴰ114UVL預賽：：嘉義大學vs臺灣師大：：男一級 大專排球聯賽 AI網路直播_set3 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0540 | ᴴᴰ114UVL預賽：：國北教大vs嘉義大學：：男一級 大專排球聯賽 AI網路直播_set2 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0541 | ᴴᴰ114UVL預賽：：國北教大vs嘉義大學：：男一級 大專排球聯賽 AI網路直播_set3 | 42 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0542 | ᴴᴰ114UVL預賽：：國北教大vs大仁科大：：男一級 大專排球聯賽 AI網路直播_set1 | 42 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0543 | ᴴᴰ114UVL預賽：：國北教大vs大仁科大：：男一級 大專排球聯賽 AI網路直播_set2 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0544 | ᴴᴰ114UVL預賽：：國北教大vs大仁科大：：男一級 大專排球聯賽 AI網路直播_set3 | 43 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0545 | ᴴᴰ114UVL預賽：：國北教大vs大仁科大：：男一級 大專排球聯賽 AI網路直播_set4 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0546 | ᴴᴰ114UVL預賽：：國北教大vs清華大學：：男一級 大專排球聯賽 AI網路直播_set1 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0547 | ᴴᴰ114UVL預賽：：國北教大vs清華大學：：男一級 大專排球聯賽 AI網路直播_set2 | 42 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0548 | ᴴᴰ114UVL預賽：：國北教大vs清華大學：：男一級 大專排球聯賽 AI網路直播_set3 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0549 | ᴴᴰ114UVL預賽：：國北教大vs清華大學：：男一級 大專排球聯賽 AI網路直播_set4 | 35 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0550 | ᴴᴰ114UVL預賽：：國北教大vs清華大學：：男一級 大專排球聯賽 AI網路直播_set5 | 21 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0551 | ᴴᴰ114UVL預賽：：大仁科大vs嘉義大學：：男一級 大專排球聯賽 AI網路直播_set1 | 39 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0552 | ᴴᴰ114UVL預賽：：大仁科大vs嘉義大學：：男一級 大專排球聯賽 AI網路直播_set2 | 50 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0553 | ᴴᴰ114UVL預賽：：大仁科大vs嘉義大學：：男一級 大專排球聯賽 AI網路直播_set3  | 42 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0554 | ᴴᴰ114UVL預賽：：實踐大學vs臺灣體大：：男一級 大專排球聯賽 AI網路直播_set2 | 40 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0555 | ᴴᴰ114UVL預賽：：實踐大學vs臺灣體大：：男一級 大專排球聯賽 AI網路直播_set3 | 40 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0556 | ᴴᴰ114UVL預賽：：彰化師大vs臺灣體大：：男一級 大專排球聯賽 AI網路直播_set1 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0557 | ᴴᴰ114UVL預賽：：彰化師大vs臺灣體大：：男一級 大專排球聯賽 AI網路直播_set2 | 39 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0558 | ᴴᴰ114UVL預賽：：彰化師大vs臺灣體大：：男一級 大專排球聯賽 AI網路直播_set3 | 43 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0559 | ᴴᴰ114UVL預賽：：清華大學vs嘉義大學：：男一級 大專排球聯賽 AI網路直播_set2 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0560 | ᴴᴰ114UVL預賽：：清華大學vs嘉義大學：：男一級 大專排球聯賽 AI網路直播_set3 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0561 | ᴴᴰ114UVL預賽：：清華大學vs嘉義大學：：男一級 大專排球聯賽 AI網路直播_set4 | 39 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0562 | ᴴᴰ114UVL預賽：：臺北大學vs中原大學：：男一級 大專排球聯賽 AI網路直播_set1 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0563 | ᴴᴰ114UVL預賽：：臺北大學vs中原大學：：男一級 大專排球聯賽 AI網路直播_set2 | 39 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0564 | ᴴᴰ114UVL預賽：：臺北大學vs中原大學：：男一級 大專排球聯賽 AI網路直播_set3 | 39 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0565 | ᴴᴰ114UVL預賽：：臺灣師大vs中山大學：：男一級 大專排球聯賽 AI網路直播_set2 | 42 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0566 | ᴴᴰ114UVL預賽：：臺灣師大vs中山大學：：男一級 大專排球聯賽 AI網路直播_set3 | 43 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0567 | ᴴᴰ114UVL預賽：：臺灣師大vs大仁科大：：男一級 大專排球聯賽 AI網路直播_set1 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0568 | ᴴᴰ114UVL預賽：：臺灣師大vs大仁科大：：男一級 大專排球聯賽 AI網路直播_set2 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0569 | ᴴᴰ114UVL預賽：：臺灣師大vs大仁科大：：男一級 大專排球聯賽 AI網路直播_set3 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0570 | ᴴᴰ114UVL預賽：：臺灣師大vs大仁科大：：男一級 大專排球聯賽 AI網路直播_set4 | 51 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0571 | ᴴᴰ114UVL預賽：：臺灣師大vs大仁科大：：男一級 大專排球聯賽 AI網路直播_set5 | 28 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0572 | ᴴᴰ114UVL預賽：：陽明交大vs臺北大學：：男一級 大專排球聯賽 AI網路直播_set1 | 42 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0573 | ᴴᴰ114UVL預賽：：陽明交大vs臺北大學：：男一級 大專排球聯賽 AI網路直播_set2 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0574 | ᴴᴰ114UVL預賽：：陽明交大vs臺北大學：：男一級 大專排球聯賽 AI網路直播_set3 | 50 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0575 | ᴴᴰ114UVL預賽：：陽明交大vs臺灣體大：：男一級 大專排球聯賽 AI網路直播_set2 | 40 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0576 | ᴴᴰ114UVL預賽：：陽明交大vs臺灣體大：：男一級 大專排球聯賽 AI網路直播_set3 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0577 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G21 11⧸9 15_00 桃園雲豹飛將 vs 臺中連莊_set2 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0578 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G21 11⧸9 15_00 桃園雲豹飛將 vs 臺中連莊_set3 | 39 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0579 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G22 11⧸9 18_30 臺北伊斯特 vs 台鋼天鷹_set2 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0580 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G22 11⧸9 18_30 臺北伊斯特 vs 台鋼天鷹_set3 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0581 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G22 11⧸9 18_30 臺北伊斯特 vs 台鋼天鷹_set4 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0582 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G22 11⧸9 18_30 臺北伊斯特 vs 台鋼天鷹_set5 | 30 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0583 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G23 11⧸15 15_00 臺中連莊 vs 台鋼天鷹_set2 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0584 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G23 11⧸15 15_00 臺中連莊 vs 台鋼天鷹_set3 | 52 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0585 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G23 11⧸15 15_00 臺中連莊 vs 台鋼天鷹_set4 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0586 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G24 11⧸15 18_30 桃園雲豹飛將 vs 臺北伊斯特_set2 | 43 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0587 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G24 11⧸15 18_30 桃園雲豹飛將 vs 臺北伊斯特_set3 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0588 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G24 11⧸15 18_30 桃園雲豹飛將 vs 臺北伊斯特_set4 | 64 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0589 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G25 11⧸16 15_00 桃園雲豹飛將 vs 台鋼天鷹_set2 | 58 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0590 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G25 11⧸16 15_00 桃園雲豹飛將 vs 台鋼天鷹_set3 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0591 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G26 11⧸16 18_30 臺中連莊 vs 臺北伊斯特_set2 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0592 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G26 11⧸16 18_30 臺中連莊 vs 臺北伊斯特_set3 | 44 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0593 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G26 11⧸16 18_30 臺中連莊 vs 臺北伊斯特_set4 | 39 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0594 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G27 11⧸22 15_00 桃園雲豹飛將 vs 台中連莊_set2 | 41 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0595 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G27 11⧸22 15_00 桃園雲豹飛將 vs 台中連莊_set3 | 50 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0596 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G28 11⧸22 18_30 臺北伊斯特 vs 台鋼天鷹_set2 | 42 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0597 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G28 11⧸22 18_30 臺北伊斯特 vs 台鋼天鷹_set3 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0598 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G28 11⧸22 18_30 臺北伊斯特 vs 台鋼天鷹_set4 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0599 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G29 11⧸23 15_00 臺北伊斯特 vs 台中連莊_set2 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0600 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G29 11⧸23 15_00 臺北伊斯特 vs 台中連莊_set3 | 44 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0601 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G30 11⧸23 18_30 桃園雲豹飛將 vs 台鋼天鷹_set2 | 47 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0602 | 【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G30 11⧸23 18_30 桃園雲豹飛將 vs 台鋼天鷹_set3 | 46 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0603 | 忠孝街大戲院 對 MK叭叭靈 第一局 | 43 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0604 | 忠孝街大戲院 對 MK叭叭靈 第二局 | 40 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0605 | 忠孝街大戲院 對 澎康 第一局 | 48 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |
| U0606 | 忠孝街大戲院 對 澎康 第二局 | 45 | 無對應人工 action 標註，無法計算 serve/score ±1 秒；未修改 |

## 附件

- [編號問題 CSV](issues.csv)
- [未驗證影片 CSV](unverified-videos.csv)
- [逐筆邊界修改 CSV](changes.csv)
- [摘要 JSON](summary.json)

重跑：`uv run python scripts/snap_rally_lead_in.py`（預覽）。
