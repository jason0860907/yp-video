"""Audit: rally spans whose edges do not land on the action that defines them.

A rally opens on a `serve` and closes on a `score`. Both audits ask the same
question of opposite ends, so they are one scan with one set of categories:

- ``邊界切偏``   the action exists just outside the span. The annotation is
                there, the boundary stops short of it, and the fix is the
                boundary.
- ``疑似漏標``   the action is nowhere near the span in either direction. The
                fix is labelling one.
- ``看過畫面的判定`` ...unless the footage says otherwise. A rally in an
                edge's ``verdicts`` table has been watched: the director was
                elsewhere and the action is not on tape, or the point ended
                on a fault (net touch, ball out, held ball...) and there was
                never a touch to label. The scan cannot see either, so the
                verdict is kept in the script and survives every re-run.

Appendices carry the near misses: the action is inside but not at the edge
(a serve with something before it, a score with something after it), the span
holds two of them, or it holds no action at all and no rule can speak to it.

Membership is the rule every reader uses (extraction/store._within):
``start <= frame / fps <= end``, inclusive at both ends.

Read-only. It writes both reports and nothing else, and re-running it after
the annotations change reproduces them — which the first serve audit could
not, having been a throwaway script that never reached the repo.

    uv run python scripts/scan_rally_edges.py
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date

from yp_video.config import (
    ACTION_ANNOTATIONS_DIR,
    PROJECT_ROOT,
    RALLY_ANNOTATIONS_DIR,
)
from yp_video.contracts.action import LABEL_FILE_SUFFIX
from yp_video.core.jsonl import read_jsonl
from yp_video.core.rallies import annotation_name

#: How close an action outside the span has to be for the boundary — not the
#: labelling — to be the thing that is wrong.
NEAR_S = 3.0
#: How many actions of the sequence to show, from the edge being audited.
WINDOW = 6


@dataclass(frozen=True)
class Edge:
    """One end of a rally and the action that should sit on it."""

    label: str
    #: True for the opening edge (serve at `start`), False for the close.
    opening: bool
    filename: str
    title: str
    #: What a mis-set boundary is called at this end.
    boundary: str
    #: What "inside, but not at the edge" is called.
    displaced: str
    #: Rallies reviewed by eye and answered, (stem, rally_id) → verdict: the
    #: broadcast cut away so the action never reached the tape, or the point
    #: ended on a fault so there is no action to label. Neither is labelling
    #: work, and the scan cannot tell them from a real omission, so the
    #: verdict is carried here rather than re-made by hand each re-run.
    verdicts: Mapping[tuple[str, int], str] = field(default_factory=dict)

    @property
    def edge_word(self) -> str:
        return "開頭" if self.opening else "結尾"

    @property
    def seq_word(self) -> str:
        return "前" if self.opening else "後"


#: Rallies reviewed by eye and answered. ``導播問題``: the director was on a
#: replay or the crowd when the serve went up, so it never reached the tape.
#: Anything else names the fault that ended the point at the serve. First
#: batch reviewed 2026-08-25 in-house; later batches from the labeller's
#: passes (隱藏任務0826 notion1.docx, 隱藏任務0827 notion2.docx).
SERVE_VERDICTS: dict[tuple[str, int], str] = {
    ('03⧸14(六) 16_00｜例行賽G104 #獅子王 vs. #屏東台電｜企業21年甲級男女排球聯賽_set2', 27): '導播問題',
    ('03⧸14(六) 18_00｜例行賽G105 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1', 1): '導播問題',
    ('03⧸14(六) 18_00｜例行賽G105 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1', 36): '導播問題',
    ('03⧸15(日) 13_00｜例行賽G106 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set3', 41): '導播問題',
    ('03⧸15(日) 17_00｜例行賽G108 #新北中纖 vs. #高雄台電｜企業21年甲級男女排球聯賽_set3', 7): '導播問題',
    ('03⧸20(五) 14_00｜挑戰賽G110 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set2', 6): '導播問題',
    ('03⧸21(六) 18_00｜男子組冠軍賽G114 #雲林美津濃 vs. #屏東台電｜企業21年甲級男女排球聯賽_set1', 18): '導播問題',
    ('03⧸22(日) 17_00｜男子組冠軍賽 G117 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set1', 30): '導播問題',
    ('03⧸22(日) 17_00｜男子組冠軍賽 G117 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set2', 2): '導播問題',
    ('2025-10-11_G8_臺中連莊_vs_桃園雲豹飛將_set1', 1): '導播問題',
    ('37-39 Thriller! - Japan 🇯🇵 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1', 31): '導播問題',
    ('37-39 Thriller! - Japan 🇯🇵 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set2', 12): '導播問題',
    ('37-39 Thriller! - Japan 🇯🇵 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set2', 13): '導播問題',
    ('37-39 Thriller! - Japan 🇯🇵 vs. Poland 🇵🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set2', 20): '導播問題',
    ('Brazil 🇧🇷 vs. Italy 🇮🇹  ｜ VNL 2025 - Full Match ｜ Week 1_set1', 9): '導播問題',
    ('Bulgaria 🇧🇬 vs. Argentina 🇦🇷 ｜ VNL 2025 - Full Match ｜ Week 1_set1', 23): '發球踩線',
    ('Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1', 33): '導播問題',
    ("France vs. Iran - Semi Final 2 ｜ Boys' U19 World Champs 2025 - Full Match_set1", 24): '導播問題',
    ('Full Match ｜ Poland vs. Slovakia ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1', 21): '導播問題',
    ('Japan 🇯🇵 vs. Czechia 🇨🇿 ｜ VNL 2025 - Full Match ｜ Week 2_set1', 42): '導播問題',
    ('Japan 🇯🇵 vs. France 🇫🇷 ｜ VNL 2025 - Full Match ｜ Week 3_set1', 42): '導播問題',
    ("Korea vs. Bulgaria - Classification 5-8 ｜ Boys' U19 World Champs 2025 - Full Match_set1", 42): '導播問題',
    ('Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1', 24): '導播問題',
    ('Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1', 27): '導播問題',
    ('Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1', 44): '導播問題',
    ("Spain vs. Iran - Ranking 3-4 ｜ Boys' U19 World Champs 2025 - Full Match_set1", 31): '導播問題',
    ("Spain vs. Iran - Ranking 3-4 ｜ Boys' U19 World Champs 2025 - Full Match_set1", 46): '導播問題',
    ('Suntory Sunbirds vs. Osaka Bluteon ｜ SV.LEAGUE 2025⧸26 ｜ Full Match - Volleyball_set1', 5): '導播問題',
    ("Uzbekistan vs. Pakistan - Ranking 5-6 ｜ Boys' U19 World Champs 2025 - Full Match_set1", 16): '導播問題',
    ('【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL  2025-26 例行賽 G25 11⧸16 15_00 桃園雲豹飛將 vs 台鋼天鷹_set1', 17): '導播問題',
}

#: Rallies whose last action is not a ``score`` because the point ended on a
#: fault — net touch, ball out, held ball, foot fault... — which is a referee
#: call, not a touch, so there is nothing to label. Verdicts from the
#: labeller's passes (隱藏任務0826 notion1.docx, 隱藏任務0827 notion2.docx).
SCORE_VERDICTS: dict[tuple[str, int], str] = {
    ('0104排島臨打 3', 34): '後排踩線',
    ('0323小窩臨打 3', 19): '落地後',
    ('03⧸14(六) 14_00｜例行賽G103 #雲林美津濃 vs. #桃園臺灣產險｜企業21年甲級男女排球聯賽_set3', 29): '標竿外',
    ('03⧸14(六) 18_00｜例行賽G105 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1', 1): '觸網',
    ('03⧸14(六) 18_00｜例行賽G105 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1', 31): '觸網',
    ('03⧸20(五) 14_00｜挑戰賽G110 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1', 10): '觸網',
    ('03⧸20(五) 14_00｜挑戰賽G110 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1', 29): '觸網',
    ('03⧸20(五) 14_00｜挑戰賽G110 #高雄台電 vs. #新北中纖｜企業21年甲級男女排球聯賽_set1', 42): '持球',
    ('03⧸20(五) 16_00｜挑戰賽G111 #屏東台電 vs. #桃園臺產｜企業21年甲級男女排球聯賽_set1', 24): '觸網',
    ('03⧸20(五) 18_00｜挑戰賽G112 #臺北國北獅 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set1', 20): '打到標竿',
    ('03⧸21(六) 18_00｜男子組冠軍賽G114 #雲林美津濃 vs. #屏東台電｜企業21年甲級男女排球聯賽_set1', 11): '觸網',
    ('03⧸21(六) 18_00｜男子組冠軍賽G114 #雲林美津濃 vs. #屏東台電｜企業21年甲級男女排球聯賽_set1', 34): '舉球後排越界',
    ('03⧸21(六) 18_00｜男子組冠軍賽G114 #雲林美津濃 vs. #屏東台電｜企業21年甲級男女排球聯賽_set1', 39): '落地後',
    ('03⧸21(六) 18_00｜男子組冠軍賽G114 #雲林美津濃 vs. #屏東台電｜企業21年甲級男女排球聯賽_set1', 40): '落地後',
    ('03⧸22(日) 17_00｜男子組冠軍賽 G117 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set1', 2): '觸網',
    ('03⧸22(日) 17_00｜男子組冠軍賽 G117 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set2', 30): '觸網',
    ('03⧸22(日) 17_00｜男子組冠軍賽 G117 #屏東台電 vs. #雲林美津濃｜企業21年甲級男女排球聯賽_set3', 4): '觸網',
    ('0419小窩臨打 2', 37): '觸網',
    ('0420小窩季打 1', 43): '公正',
    ('0420小窩季打 2', 11): '後排踩線',
    ('0420小窩季打 3', 15): '後排踩線',
    ('2025-10-04_G3_臺中連莊_vs_桃園雲豹飛將_set1', 13): '越界',
    ('2025-10-05_G5_臺北伊斯特_vs_桃園雲豹飛將_set1', 16): '持球',
    ('2025-10-25_G11_臺北伊斯特_vs_桃園雲豹飛將_set1', 26): '打到標竿',
    ('2025-10-26_G14_臺北伊斯特_vs_臺中連莊_set1', 37): '觸網',
    ('2025-11-01_G15_臺中連莊_vs_臺北伊斯特_set1', 32): '觸網',
    ('2025-11-01_G15_臺中連莊_vs_臺北伊斯特_set1', 36): '越界',
    ('2025-11-01_G16_桃園雲豹飛將_vs_台鋼天鷹_set1', 13): '觸網',
    ('2025-11-02_G17_桃園雲豹飛將_vs_臺北伊斯特_set1', 4): '越網擊球',
    ('2025-11-02_G18_臺中連莊_vs_台鋼天鷹_set1', 29): '導播問題',
    ('2025-11-08_G19_臺北伊斯特_vs_臺中連莊_set1', 47): '打到標竿',
    ('2025-11-08_G20_桃園雲豹飛將_vs_台鋼天鷹_set1', 29): '觸網',
    ('20250504 大統OB-成功大學vs台北大學B-第二局', 4): '觸網',
    ('20260426-小窩-01', 31): '觸網',
    ('20260502-排島本館-02', 7): '觸網',
    ('20260502-排島本館-02', 40): '公正',
    ('20260507 工資管友誼賽2', 10): '公正',
    ('Bulgaria 🇧🇬 vs. Argentina 🇦🇷 ｜ VNL 2025 - Full Match ｜ Week 1_set1', 23): '發球踩線',
    ('Bulgaria 🇧🇬 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1', 14): '後排踩線',
    ('Bulgaria 🇧🇬 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1', 37): '發球踩線',
    ('Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1', 4): '落地後',
    ('Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1', 10): '發球踩線',
    ('Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1', 31): '觸網',
    ('Canada 🇨🇦 vs Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1', 50): '落地後',
    ('Champions crowned in Final 24⧸25 (2⧸2) ｜ Suntory Sunbirds Osaka - Stings Aichi ｜ SV League 24⧸25_set1', 18): '二擊',
    ("China vs. Argentina - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1", 2): '觸網',
    ("China vs. Brazil - Ranking 13-14 ｜ Boys' U19 World Champs 2025 - Full Match_set1", 6): '觸網',
    ("China vs. Brazil - Ranking 13-14 ｜ Boys' U19 World Champs 2025 - Full Match_set1", 19): '打到標竿',
    ('China 🇨🇳 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 1_set1', 3): '觸網',
    ("Cuba vs. Puerto Rico - Ranking 17-18 ｜ Boys' U19 World Champs 2025 - Full Match_set1", 22): '觸網',
    ('Final - Stings vs. Sunbirds ｜ SVL League 2024⧸25 - Full Match ｜ Volleyball_set1', 29): '後排踩線',
    ("France vs. Iran - Semi Final 2 ｜ Boys' U19 World Champs 2025 - Full Match_set1", 3): '觸網',
    ("France vs. Iran - Semi Final 2 ｜ Boys' U19 World Champs 2025 - Full Match_set1", 5): '觸網',
    ("France vs. Iran - Semi Final 2 ｜ Boys' U19 World Champs 2025 - Full Match_set1", 18): '觸網',
    ('France 🇫🇷 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 2_set1', 19): '落地後',
    ('France 🇫🇷 vs. Japan 🇯🇵 ｜ VNL 2025 - Full Match ｜ Week 2_set1', 41): '觸網',
    ('Full Match ｜ Bulgaria vs Luxembourg ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set3', 37): '觸網',
    ('Full Match ｜ Croatia vs. Bulgaria ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set1', 24): '二擊',
    ('Full Match ｜ Croatia vs. Bulgaria ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set1', 28): '落地後',
    ('Full Match ｜ Croatia vs. Serbia ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set1', 21): '觸網',
    ('Full Match ｜ Denmark vs. England ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool A_set1', 11): '落地後',
    ('Full Match ｜ Denmark vs. England ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool A_set1', 18): '越界',
    ('Full Match ｜ Ireland vs. Spain ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool D_set1', 9): '落地後',
    ('Full Match ｜ Ireland vs. Spain ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool D_set1', 11): '觸網',
    ('Full Match ｜ Ireland vs. Türkiye ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool D_set1', 13): '打到標竿',
    ('Full Match ｜ Ireland vs. Türkiye ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool D_set1', 14): '落地後',
    ('Full Match ｜ Italy vs. England ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool A_set1', 30): '落地後',
    ('Full Match ｜ Luxembourg vs. Croatia - CEV U22 Volleyball European Championship 2026 ｜ Women ｜ Pool E_set1', 1): '持球',
    ('Full Match ｜ Norway vs. Bulgaria ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool C_set1', 21): '觸網',
    ('Full Match ｜ Norway vs. Ireland ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool D_set1', 37): '二擊',
    ('Full Match ｜ Poland vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1', 4): '打到標竿',
    ('Full Match ｜ Poland vs. Slovakia ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1', 26): '持球',
    ('Full Match ｜ Serbia vs. Luxembourg ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool E_set1', 34): '觸網',
    ('Full Match ｜ Slovakia vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1', 25): '輪轉錯誤',
    ('Full Match ｜ Slovakia vs. England ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool C_set1', 31): '觸網',
    ('Full Match ｜ Spain vs The Netherlands ｜ CEV U22 Volleyball European Championship 2026 Men ｜ Pool C_set1', 13): '觸網',
    ('Full Match ｜ Türkiye vs. Spain ｜ CEV U22 Volleyball European Championship 2026 Women ｜ Pool D_set1', 40): 'no in',
    ("Japan vs. USA - Ranking 15-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1", 26): '舉球後排越界',
    ("Japan vs. USA - Ranking 15-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1", 31): 'no in',
    ("Japan vs. USA - Ranking 15-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1", 45): '觸網',
    ('Japan 🇯🇵 vs. Dominican Republic 🇩🇴 ｜ VNL 2025 - Full Match ｜ Week 1_set1', 24): '觸網',
    ('Japan 🇯🇵 vs. Dominican Republic 🇩🇴 ｜ VNL 2025 - Full Match ｜ Week 1_set1', 25): '4次',
    ('Japan 🇯🇵 vs. Dominican Republic 🇩🇴 ｜ VNL 2025 - Full Match ｜ Week 1_set1', 29): '觸網',
    ('Japan 🇯🇵 vs. France 🇫🇷 ｜ VNL 2025 - Full Match ｜ Week 3_set1', 36): '打到標竿',
    ('Japan 🇯🇵 vs. Netherlands 🇳🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1', 2): '觸網',
    ('Japan 🇯🇵 vs. Netherlands 🇳🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1', 23): '落地後',
    ('Japan 🇯🇵 vs. Netherlands 🇳🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1', 31): '觸網',
    ('Japan 🇯🇵 vs. Netherlands 🇳🇱 ｜ VNL 2025 - Full Match ｜ Week 1_set1', 43): '觸網',
    ('Jtekt Stings 🇯🇵 - Suntory Sunbirds Osaka 🇯🇵 ｜ SV League 2026 ｜ Full Match - Volleyball_set1', 31): '觸網',
    ("Korea vs. Finland - Ranking 11-12 ｜ Boys' U19 World Champs 2025 - Full Match_set1", 21): '觸網',
    ("Korea vs. Finland - Ranking 11-12 ｜ Boys' U19 World Champs 2025 - Full Match_set1", 44): 'no in',
    ('Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1', 11): '觸網',
    ('Osaka Bluteon vs Diamond Food Fine Chef - Full Match ｜ SV. League World Tour 2025 ｜ Volleyball_set1', 26): '觸網',
    ('Osaka Bluteon 🇯🇵 vs. JTEKT Stings 🇯🇵 ｜ SV League 2026 ｜ Full Match - Volleyball_set1', 23): '觸網',
    ('Osaka Bluteon 🇯🇵 vs. JTEKT Stings 🇯🇵 ｜ SV League 2026 ｜ Full Match - Volleyball_set1', 42): '落地後',
    ("Pakistan vs. USA - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1", 1): '阻擋舉球',
    ("Pakistan vs. USA - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1", 6): '觸網',
    ("Pakistan vs. USA - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1", 16): '觸網',
    ("Poland vs. Spain - Semi Final 1 ｜ Boys' U19 World Champs 2025 - Full Match_set1", 35): 'no in',
    ('Semi Final 1 - Osaka Bluteon vs. Stings Aichi ｜ SV League - Full Match ｜ Volleyball_set1', 8): '觸網',
    ('Semi Final 1 - Osaka Bluteon vs. Stings Aichi ｜ SV League - Full Match ｜ Volleyball_set1', 18): '後排踩線',
    ('Semi Final 1 - Osaka Bluteon vs. Stings Aichi ｜ SV League - Full Match ｜ Volleyball_set1', 42): '觸網',
    ('Semi Final 2 - Osaka Bluteon vs. Stings Aichi ｜ SV League - Full Match ｜ Volleyball_set1', 30): '落地後',
    ('Semi Final 2 - Osaka Bluteon vs. Stings Aichi ｜ SV League - Full Match ｜ Volleyball_set1', 39): '舉球後排越界',
    ('Semi Final 3 - Suntory Sunbirds vs. Wolfdogs Nagoya ｜ SVL Playoff - Full Match ｜ Volleyball_set1', 46): '越界救球',
    ("Spain vs. Iran - Ranking 3-4 ｜ Boys' U19 World Champs 2025 - Full Match_set1", 14): '越網擊球',
    ("Spain vs. Iran - Ranking 3-4 ｜ Boys' U19 World Champs 2025 - Full Match_set1", 16): '越界',
    ('Suntory Sunbirds vs. Osaka Bluteon ｜ SV.LEAGUE 2025⧸26 ｜ Full Match - Volleyball_set1', 23): '落地後',
    ('Suntory Sunbirds 🇯🇵 vs. Stings AICHI 🇯🇵 ｜ SV League 2026 ｜ Full Match - Japan Volleyball_set1', 6): '觸網',
    ('Suntory Sunbirds 🇯🇵 vs. Stings AICHI 🇯🇵 ｜ SV League 2026 ｜ Full Match - Japan Volleyball_set1', 30): '打到標竿',
    ("Taipei vs. Argentina - Playoffs ｜ Girls' U19 World Champs 2025 - Full Match_set1", 7): '觸網',
    ("Taipei vs. Argentina - Playoffs ｜ Girls' U19 World Champs 2025 - Full Match_set1", 22): '觸網',
    ("Türkiye vs. Colombia - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1", 9): '二擊',
    ("Türkiye vs. Colombia - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1", 27): '觸網',
    ("Türkiye vs. Colombia - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1", 29): '持球',
    ("Türkiye vs. Colombia - Classification 13-16 ｜ Boys' U19 World Champs 2025 - Full Match_set1", 31): '落地後',
    ("Uzbekistan vs. Japan - Ranking 19-20 ｜ Boys' U19 World Champs 2025 - Full Match_set1", 16): '輪轉錯誤',
    ("Uzbekistan vs. Pakistan - Ranking 5-6 ｜ Boys' U19 World Champs 2025 - Full Match_set1", 15): '觸網',
    ('ᴴᴰ114UVL預賽：：中原大學vs實踐大學：：男一級 大專排球聯賽 AI網路直播_set1', 10): '輪轉錯誤',
    ('ᴴᴰ114UVL預賽：：中原大學vs陽明交大：：男一級 大專排球聯賽 AI網路直播_set1', 17): '越界',
    ('ᴴᴰ114UVL預賽：：中原大學vs陽明交大：：男一級 大專排球聯賽 AI網路直播_set1', 23): '持球',
    ('ᴴᴰ114UVL預賽：：中原大學vs陽明交大：：男一級 大專排球聯賽 AI網路直播_set1', 24): '打到標竿',
    ('ᴴᴰ114UVL預賽：：中原大學vs陽明交大：：男一級 大專排球聯賽 AI網路直播_set1', 38): '越界',
    ('ᴴᴰ114UVL預賽：：中原大學vs陽明交大：：男一級 大專排球聯賽 AI網路直播_set1', 39): '觸網',
    ('ᴴᴰ114UVL預賽：：中山大學vs國北教大：：男一級 大專排球聯賽 AI網路直播_set1', 15): '越網擊球',
    ('ᴴᴰ114UVL預賽：：中山大學vs國北教大：：男一級 大專排球聯賽 AI網路直播_set1', 19): '觸網',
    ('ᴴᴰ114UVL預賽：：中山大學vs國北教大：：男一級 大專排球聯賽 AI網路直播_set1', 35): '打到標竿',
    ('ᴴᴰ114UVL預賽：：中山大學vs清華大學：：男一級 大專排球聯賽 AI網路直播_set1', 16): '觸網',
    ('ᴴᴰ114UVL預賽：：中山大學vs清華大學：：男一級 大專排球聯賽 AI網路直播_set1', 39): '觸網',
    ('ᴴᴰ114UVL預賽：：中山大學vs清華大學：：男一級 大專排球聯賽 AI網路直播_set1', 41): '觸網',
    ('ᴴᴰ114UVL預賽：：國北教大vs嘉義大學：：男一級 大專排球聯賽 AI網路直播_set1', 1): '落地後',
    ('ᴴᴰ114UVL預賽：：臺灣師大vs中山大學：：男一級 大專排球聯賽 AI網路直播_set1', 3): '觸網',
    ('ᴴᴰ114UVL預賽：：臺灣師大vs中山大學：：男一級 大專排球聯賽 AI網路直播_set1', 23): '觸網',
    ('【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL 2025-26 例行賽 G21 11⧸9 15_00 桃園雲豹飛將 vs 臺中連莊_set1', 39): '觸網',
    ('【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL 2025-26 例行賽 G23 11⧸15 15_00 臺中連莊 vs 台鋼天鷹_set1', 24): '越界',
    ('【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL 2025-26 例行賽 G24 11⧸15 18_30 桃園雲豹飛將 vs 臺北伊斯特_set1', 9): '越界',
    ('【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL 2025-26 例行賽 G24 11⧸15 18_30 桃園雲豹飛將 vs 臺北伊斯特_set1', 25): '越網擊球',
    ('【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL 2025-26 例行賽 G24 11⧸15 18_30 桃園雲豹飛將 vs 臺北伊斯特_set1', 45): '觸網',
    ('【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL 2025-26 例行賽 G26 11⧸16 18_30 臺中連莊 vs 臺北伊斯特_set1', 17): '觸網',
    ('【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL 2025-26 例行賽 G27 11⧸22 15_00 桃園雲豹飛將 vs 台中連莊_set1', 7): '打到標竿',
    ('【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL 2025-26 例行賽 G6 10⧸5 18_30 臺中連莊 vs 台鋼天鷹_set1', 13): '4次',
    ('【LIVE】𝗙𝗨𝗟𝗟 𝗠𝗔𝗧𝗖𝗛｜TPVL 2025-26 例行賽 G6 10⧸5 18_30 臺中連莊 vs 台鋼天鷹_set1', 25): '阻擋舉球',
}

SERVE = Edge(
    label="serve",
    opening=True,
    filename="rally-missing-serve.md",
    title="Rally 開頭沒有 serve 的清單",
    boundary="開頭切太晚",
    displaced="serve 不在最前",
    verdicts=SERVE_VERDICTS,
)
SCORE = Edge(
    label="score",
    opening=False,
    filename="rally-missing-score.md",
    title="Rally 結尾沒有 score 的清單",
    boundary="結尾切太早",
    displaced="score 不在最後",
    verdicts=SCORE_VERDICTS,
)


def ts(seconds: float) -> str:
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


@dataclass(frozen=True)
class Row:
    stem: str
    rally_id: int
    start: float
    end: float
    #: Action labels at the audited edge, always in time order.
    seq: tuple[str, ...]
    #: Seconds to the nearest matching action outside the span, or None.
    outside: float | None
    #: Distance from the edge to the matching action inside it, or None.
    inside_gap: float | None
    #: How many actions sit between the edge and the matching one inside.
    displaced_by: int
    matches: int
    #: Closest gap between two matching actions inside the span. Under half a
    #: second means one action annotated twice, not two of them.
    match_gap: float | None = None


@dataclass(frozen=True)
class Scan:
    """One edge's audit, in the buckets the report prints."""

    unlabelled: list[Row]
    #: Eye-checked rallies with their verdict, whatever bucket the rule chose.
    verdicts: list[tuple[Row, str]]
    displaced: list[Row]
    boundary: list[Row]
    extra: list[Row]
    empty: list[Row]
    videos: int
    rallies: int

    @property
    def flagged(self) -> int:
        """Every rally the rule caught, whatever the report does with it."""
        return sum(
            len(rows) for rows in
            (self.unlabelled, self.verdicts, self.displaced, self.boundary, self.empty)
        )


def scan(edge: Edge) -> Scan:
    unlabelled: list[Row] = []
    verdicts: list[tuple[Row, str]] = []
    boundary: list[Row] = []
    displaced: list[Row] = []
    empty: list[Row] = []
    extra: list[Row] = []
    videos = seen = 0

    for rally_path in sorted(RALLY_ANNOTATIONS_DIR.glob(annotation_name("*"))):
        stem = rally_path.name[: -len(annotation_name(""))]
        action_path = ACTION_ANNOTATIONS_DIR / f"{stem}{LABEL_FILE_SUFFIX}"
        if not action_path.exists():
            continue
        videos += 1
        meta, events = read_jsonl(action_path)
        fps = float(meta.get("fps") or 30.0) or 30.0
        timeline = sorted(
            (int(e["frame"]) / fps, str(e.get("label") or ""))
            for e in events
            if e.get("frame") is not None
        )
        hits = [t for t, label in timeline if label == edge.label]
        _, rows = read_jsonl(rally_path)

        for row in rows:
            seen += 1
            start, end = float(row["start"]), float(row["end"])
            rid = int(row.get("rally_id") or 0)
            inside = [x for x in timeline if start <= x[0] <= end]
            # Always read the sequence from the edge under audit.
            window = inside[:WINDOW] if edge.opening else inside[-WINDOW:]
            seq = tuple(label for _, label in window)
            at = [t for t, label in inside if label == edge.label]
            n = len(at)
            gap = min((b - a for a, b in zip(at, at[1:])), default=None)

            def make(outside=None, inside_gap=None, displaced_by=0) -> Row:
                return Row(stem, rid, start, end, seq, outside, inside_gap,
                           displaced_by, n, gap)

            if not inside:
                empty.append(make())
                continue
            if n == 0:
                after = [t - end for t in hits if t > end]
                before = [start - t for t in hits if t < start]
                near = [min(x) for x in (after, before) if x]
                nearest = min(near) if near else None
                # The eye-checked verdict outranks the distance rule: it was
                # made on the footage, which is the only place the answer is.
                if (stem, rid) in edge.verdicts:
                    verdicts.append((make(outside=nearest), edge.verdicts[(stem, rid)]))
                elif nearest is not None and nearest <= NEAR_S:
                    boundary.append(make(outside=nearest))
                else:
                    unlabelled.append(make(outside=nearest))
                continue
            if n > 1:
                extra.append(make())
            at_edge = inside[0] if edge.opening else inside[-1]
            if at_edge[1] != edge.label:
                picked = at[0] if edge.opening else at[-1]
                gap = picked - start if edge.opening else end - picked
                between = sum(
                    1 for t, _ in inside
                    if (t < picked if edge.opening else t > picked)
                )
                row = make(inside_gap=round(gap, 1), displaced_by=between)
                if (stem, rid) in edge.verdicts:
                    verdicts.append((row, edge.verdicts[(stem, rid)]))
                else:
                    displaced.append(row)
    return Scan(unlabelled, verdicts, displaced, boundary, extra, empty, videos, seen)


def render(edge: Edge, found: Scan) -> str:
    at = "最前" if edge.opening else "最後"

    out = [
        f"# {edge.title}",
        "",
        f"掃描日期：{date.today():%Y-%m-%d} · 資料：`videos/rally-spot/annotations` × "
        f"`videos/action/annotations`（{found.videos} 支影片、{found.rallies:,} rallies）",
        "",
        "重跑：`uv run python scripts/scan_rally_edges.py`",
        "",
        f"判定：rally span `[start, end]` 內{at}的動作事件不是 `{edge.label}`。"
        f"共 {found.flagged} 筆。",
        "",
        f"分類：`{edge.boundary}` = span 外 {NEAR_S:.0f} 秒內就有 `{edge.label}`，"
        f"標註在、是邊界偏了；`{edge.displaced}` = span 內有 `{edge.label}`，"
        f"但{'前' if edge.opening else '後'}面還有別的動作；"
        f"`疑似漏標` = 前後都找不到鄰近的 `{edge.label}`；"
        f"`看過畫面的判定` = 導播沒拍到那個 `{edge.label}`，或這球是犯規結束、"
        f"本來就沒有 `{edge.label}` 可標。",
        "",
    ]
    unlabelled, verdicts, displaced = found.unlabelled, found.verdicts, found.displaced
    boundary, extra, empty = found.boundary, found.extra, found.empty

    def table(rows: list[Row], gap_col: str | None, verdict_of=None) -> None:
        head = f"| 影片 | Rally | 起 | 訖 | {at}動作 |"
        sep = "|---|---:|---:|---:|---|"
        if verdict_of:
            head += " 判定 |"
            sep += "---|"
        if gap_col:
            head += f" {gap_col} |"
            sep += "---:|"
        out.extend([head + f" 動作序列（{edge.seq_word} {WINDOW}） |", sep + "---|"])
        for r in rows:
            edge_action = (r.seq[0] if edge.opening else r.seq[-1]) if r.seq else "—"
            cells = [r.stem, str(r.rally_id), ts(r.start), ts(r.end), edge_action]
            if verdict_of:
                cells.append(verdict_of(r))
            if gap_col:
                value = r.outside if r.inside_gap is None else r.inside_gap
                cells.append(f"{value:.1f}s" if value is not None else "—")
            cells.append(" → ".join(r.seq) if r.seq else "—")
            out.append("| " + " | ".join(cells) + " |")
        out.append("")

    key = lambda r: (r.stem, r.rally_id)  # noqa: E731 — one sort key, used thrice
    if unlabelled:
        out += [f"## 疑似漏標 — {len(unlabelled)} 筆", ""]
        table(sorted(unlabelled, key=key), None)
    if verdicts:
        reasons = Counter(reason for _, reason in verdicts)
        summary = "、".join(f"{reason} {n}" for reason, n in reasons.most_common())
        out += [
            f"## 看過畫面的判定 — {len(verdicts)} 筆",
            "",
            f"看過畫面了，這批不是漏標：`導播問題` = 轉播切走（重播、觀眾、板凳），"
            f"`{edge.label}` 不在帶子上；其他判定是犯規結束的球——觸網、越界、持球…"
            f"是裁判的哨音不是觸球，所以沒有 `{edge.label}` 可標。"
            "留著是為了下次掃描不用再看一遍。",
            "",
            f"判定分布：{summary}。",
            "",
        ]
        by_row = {key(r): reason for r, reason in verdicts}
        table(sorted((r for r, _ in verdicts), key=lambda r: (by_row[key(r)], key(r))),
              None, verdict_of=lambda r: by_row[key(r)])
    if displaced:
        out += [
            f"## {edge.displaced} — {len(displaced)} 筆",
            "",
            f"span 內有 `{edge.label}`，但它不是{at}的事件。",
            "",
        ]
        table(sorted(displaced, key=key), f"{edge.label} 距{edge.edge_word}")
    if boundary:
        out += [
            f"## {edge.boundary} — {len(boundary)} 筆",
            "",
            f"`{edge.label}` 就在 span 外不到 {NEAR_S:.0f} 秒 —— 標註本身在，"
            f"要動的是 `{'start' if edge.opening else 'end'}`。",
            "",
        ]
        table(sorted(boundary, key=key), f"{edge.label} 距{edge.edge_word}")
    if extra:
        out += [
            f"## 附錄 A：span 內有 2 個以上 `{edge.label}` — {len(extra)} 筆",
            "",
            f"間隔不到 0.5 秒的，是同一個 `{edge.label}` 被標了兩次，不是兩個。",
            "",
            f"| 影片 | Rally | 起 | 訖 | {edge.label} 數 | 最小間隔 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for r in sorted(extra, key=key):
            gap = f"{r.match_gap:.2f}s" if r.match_gap is not None else "—"
            out.append(
                f"| {r.stem} | {r.rally_id} | {ts(r.start)} | {ts(r.end)} | "
                f"{r.matches} | {gap} |"
            )
        out.append("")
    if empty:
        out += [
            f"## 附錄 B：span 內完全沒有動作 — {len(empty)} 筆",
            "",
            "沒有事件可以判定，兩份清單裡的是同一批。",
            "",
            "| 影片 | Rally | 起 | 訖 |",
            "|---|---:|---:|---:|",
        ]
        for r in sorted(empty, key=key):
            out.append(f"| {r.stem} | {r.rally_id} | {ts(r.start)} | {ts(r.end)} |")
        out.append("")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    docs = PROJECT_ROOT / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    for edge in (SERVE, SCORE):
        found = scan(edge)
        path = docs / edge.filename
        path.write_text(render(edge, found), encoding="utf-8")
        print(f"{edge.label}: {found.videos} video(s), {found.rallies:,} rallies")
        print(f"  疑似漏標            {len(found.unlabelled)}")
        print(f"  看過畫面的判定       {len(found.verdicts)}")
        print(f"  {edge.displaced:<18s}{len(found.displaced)}")
        print(f"  {edge.boundary:<18s}{len(found.boundary)}")
        print(f"  span 內無動作        {len(found.empty)}")
        print(f"  2 個以上            {len(found.extra)}")
        print(f"  → {path.relative_to(PROJECT_ROOT)}\n")


if __name__ == "__main__":
    main()
