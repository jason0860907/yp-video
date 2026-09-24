/** What the App shows around the feeds: roster, court-side score per set,
 *  loss reasons — and what this view cannot show. */

import { SectionLabel } from '@/components/ui/SectionLabel';
import type { LibraryMatch, Preview } from './types';
import { SIDE_LABEL, fmtDate, lossReasonLabel, playerLabel } from './types';

export function MatchInfo({ match, preview }: { match: LibraryMatch; preview: Preview }) {
  return (
    <div className="space-y-5 text-sm">
      <div>
        <SectionLabel>比賽</SectionLabel>
        <p className="font-medium">{match.title}</p>
        <p className="text-xs text-text-muted">
          {match.angle} · {fmtDate(match.recorded_at ?? match.updated_at)} · 分析{' '}
          {match.analysis?.status ?? '無'}
        </p>
      </div>
      {preview.warnings.length > 0 && (
        <div className="space-y-1">
          {preview.warnings.map((w) => (
            <p key={w} className="rounded-md bg-amber-500/10 px-2 py-1.5 text-xs text-amber-400">
              {w}
            </p>
          ))}
        </div>
      )}
      <div>
        <SectionLabel>場邊比分（依得分方）</SectionLabel>
        {preview.court_totals.length === 0 && <p className="text-text-muted">沒有回合。</p>}
        {preview.court_totals.map((t) => (
          <p key={t.set}>
            第 {t.set} 局：
            {Object.entries(t.points)
              .map(([side, n]) => `${SIDE_LABEL[side] ?? side} ${n}`)
              .join(' : ')}
            {t.unknown > 0 && <span className="text-text-muted">（未知 {t.unknown}）</span>}
          </p>
        ))}
        <p className="mt-1 text-xs text-text-muted">場邊累計不代表換場後的球隊比分。</p>
      </div>
      <div>
        <SectionLabel>失分原因</SectionLabel>
        {Object.keys(preview.loss_reasons).length === 0 ? (
          <p className="text-text-muted">尚未標記。</p>
        ) : (
          Object.entries(preview.loss_reasons).map(([code, n]) => (
            <p key={code}>
              {lossReasonLabel(code)} · {n}
            </p>
          ))
        )}
      </div>
      <div>
        <SectionLabel>名單</SectionLabel>
        {preview.roster.length === 0 ? (
          <p className="text-text-muted">沒有名單。</p>
        ) : (
          <p>{preview.roster.map(playerLabel).join('、')}</p>
        )}
      </div>
      <p className="text-xs text-text-muted">
        只存在使用者裝置上的狀態看不到：看過標記、收藏內的手動排序、各比賽的取景偏好。
      </p>
    </div>
  );
}
