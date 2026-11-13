/** 加入訓練資料: copy this match into a cuts-user-<angle> cut with the App's
 *  corrected analysis as pre-labels (web/app_review/training.py). */

import { useState } from 'react';
import { api, errMsg } from '@/lib/api';
import { fieldCls } from '@/components/form/Field';
import { Button } from '@/components/ui/Button';
import { toast } from '@/components/feedback/toast';
import { cn } from '@/lib/cn';
import type { LibraryMatch } from './types';

const ANGLES = [
  { key: 'baseline', label: '底線' },
  { key: 'corner', label: '角落' },
  { key: 'high', label: '高處' },
  { key: 'center', label: '中線側面' },
] as const;
type Angle = (typeof ANGLES)[number]['key'];

export function SendToTraining({ user, match }: { user: string; match: LibraryMatch }) {
  const own = ANGLES.find((a) => a.key === match.angle)?.key;
  const [angle, setAngle] = useState<Angle>(own ?? 'corner');
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
      setBusy(false);
    }
  };

  return (
    <span className="inline-flex items-center gap-1.5">
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
      <Button size="sm" className="h-8 py-0" onClick={send} disabled={busy}>
        {busy ? '加入中…' : '加入訓練資料'}
      </Button>
    </span>
  );
}
