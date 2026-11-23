/** 加入訓練資料: copy this match into a cuts-user-<angle> cut with the App's
 *  corrected analysis as pre-labels (web/app_review/training.py). A match
 *  already in training says so and under which angle. */

import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { api, errMsg } from '@/lib/api';
import { fieldCls } from '@/components/form/Field';
import { Button } from '@/components/ui/Button';
import { toast } from '@/components/feedback/toast';
import { cn } from '@/lib/cn';
import { Badge } from '@/components/ui/Badge';
import type { LibraryMatch } from './types';

const ANGLES = [
  { key: 'baseline', label: '底線' },
  { key: 'corner', label: '角落' },
  { key: 'high', label: '高處' },
  { key: 'center', label: '中線側面' },
] as const;
type Angle = (typeof ANGLES)[number]['key'];

const SENT_KEY = ['app-review', 'training'];

export function SendToTraining({ user, match }: { user: string; match: LibraryMatch }) {
  const queryClient = useQueryClient();
  // One listing for every row: match id -> user-<angle> kind.
  const sent = useQuery({ queryKey: SENT_KEY, queryFn: () => api.get('/api/app-review/training') });
  const sentAngle = ANGLES.find((a) => `user-${a.key}` === sent.data?.[match.id]);
  const own = ANGLES.find((a) => a.key === match.angle)?.key;
  const [picked, setAngle] = useState<Angle>();
  const angle = picked ?? sentAngle?.key ?? own ?? 'corner';
  const [busy, setBusy] = useState(false);

  const send = async () => {
    setBusy(true);
    try {
      const out = await api.post('/api/app-review/users/{user}/matches/{match}/training', {
        params: { path: { user, match: match.id } },
        body: { angle },
      });
      if (out.status === 'added') toast.success(`已加入 ${out.kind}（${out.rallies} 回合、${out.events} 動作）`);
      else if (out.status === 'moved') toast.success(`已搬到 ${out.kind}`);
      else toast.success(`已在 ${out.kind}`);
    } catch (e) {
      toast.error(errMsg(e));
    } finally {
      void queryClient.invalidateQueries({ queryKey: SENT_KEY });
      setBusy(false);
    }
  };

  return (
    <span className="inline-flex items-center gap-1.5">
      {sentAngle ? (
        <Badge tone="success">已加入 · {sentAngle.label}</Badge>
      ) : (
        sent.data && <Badge>未加入</Badge>
      )}
      <select
        value={angle}
        onChange={(e) => setAngle(e.target.value as Angle)}
        className={cn(fieldCls, 'h-8 py-0 text-xs')}
        title="拍攝角度（預設為上傳時選的角度）"
      >
        {ANGLES.map((a) => (
          <option key={a.key} value={a.key}>
            {a.label}
          </option>
        ))}
      </select>
      <Button size="sm" className="h-8 py-0" onClick={send} disabled={busy || angle === sentAngle?.key}>
        {busy ? '加入中…' : sentAngle ? (angle === sentAngle.key ? '已加入' : '改角度') : '加入訓練資料'}
      </Button>
    </span>
  );
}
