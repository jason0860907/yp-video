-- Turn a coalesced row into a work session with a start and an end.
--
-- `at`已經是「最後一次操作」的時間;這裡補上第一次。兩者之間就是這段
-- session 的長度,而 audit.py 的摺疊窗口(5 分鐘沒有新動作就開新的一列)
-- 決定了 session 如何切分。目的是每週結算每個人的標註工時。
--
-- 既有列的 first_at 直接取 at:在補這個欄位之前,每一列只知道一個時間點。
ALTER TABLE audit_events ADD COLUMN first_at TIMESTAMPTZ;
UPDATE audit_events SET first_at = at WHERE first_at IS NULL;
ALTER TABLE audit_events ALTER COLUMN first_at SET NOT NULL;
