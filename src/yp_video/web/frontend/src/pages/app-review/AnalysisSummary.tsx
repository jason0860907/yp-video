/** The App's 分析 → 摘要 for one match (MatchAnalysisView /
 *  AnalysisStatsContent, wide layout), from app_review/stats.py. Parts that
 *  need a court calibration — 球路, 球速與高度 and the court findings — are
 *  not shown. 我方 is read from the user's corrections; it is set in the App. */

import { Card } from '@/components/ui/Card';
import { APP, displayName, hms, KIND, KINDS, lossReasonLabel, SIDE } from './appTerms';
import type { LibraryMatch, Preview, Stats } from './types';

const SERIES = [APP.primary, APP.accent, '#64B5FF', '#D27575'];

export function AnalysisSummary({ match, preview }: { match: LibraryMatch; preview: Preview }) {
  const stats = preview.stats;
  const date = new Date((match.recorded_at ?? match.created_at) * 1000).toLocaleDateString('zh-TW');
  const sides = stats.team_sides;
  return (
    <div className="space-y-4">
      <Card>
        <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
          <div>
            <p className="text-lg font-semibold text-text-primary">分析</p>
            <p className="text-xs text-text-secondary">
              {[date, `${stats.rally_count} 個 Rally`, `比賽時間 ${stats.in_rally_seconds > 0 ? hms(stats.in_rally_seconds) : '—'}`].join(' · ')}
            </p>
          </div>
          <div className="text-base font-semibold tabular-nums text-text-primary">
            {stats.team_score ? (
              `我方 ${stats.team_score.us} : 對手 ${stats.team_score.them}`
            ) : (
              preview.court_totals.map((t) => (
                <div key={t.set}>
                  {Object.entries(t.points).map(([s, n]) => `${SIDE[s]} ${n}`).join(' : ')}
                  {t.unknown > 0 && <span className="block text-xs font-normal text-text-secondary">另有 {t.unknown} 球勝方待確認</span>}
                </div>
              ))
            )}
          </div>
          <span className="ml-auto rounded-full bg-ink/5 px-3 py-1 text-xs text-text-secondary">
            {sides ? `我方：${SIDE[sides.start]}起` : '尚未設定我方'}
          </span>
        </div>
      </Card>

      {stats.findings.length > 0 && (
        <Card label="重點">
          {stats.findings.map((f) => (
            <div key={f.kind} className="flex items-center gap-2 text-sm">
              {f.team && <TeamBadge team={f.team} />}
              <span className="font-semibold text-text-primary">失分最多：{lossReasonLabel(f.reason)}</span>
              <span className="text-xs text-text-secondary">
                {f.count}／{f.total} 分
              </span>
            </div>
          ))}
        </Card>
      )}

      <div className="grid gap-4 lg:grid-cols-3">
        {stats.progression.length > 0 && (
          <Card label="得分走勢">
            {!stats.has_teams && <p className="mb-2 text-[11px] text-text-muted">依畫面左右邊計分，不代表隊伍</p>}
            <Progression stats={stats} />
          </Card>
        )}
        {stats.players.length > 0 && (
          <Card label="球員觸球">
            <PlayerTouches stats={stats} />
          </Card>
        )}
        {stats.losses.length > 0 && (
          <Card label="失分原因">
            <p className="mb-2 text-[11px] text-text-muted">含依回合推測的原因</p>
            <Losses stats={stats} />
          </Card>
        )}
      </div>

      <Card label="球員數據">
        {stats.players.length === 0 ? (
          <div className="py-6 text-center text-xs text-text-secondary">
            <p className="text-sm text-text-primary">還沒有球員數據</p>
            <p>完成球員配對後，這裡會顯示每位球員的觸球與進攻參與。</p>
          </div>
        ) : (
          <PlayerTable stats={stats} />
        )}
      </Card>
    </div>
  );
}

function TeamBadge({ team }: { team: string }) {
  return (
    <span
      className="rounded-full px-2 py-0.5 text-[10px] font-bold"
      style={team === 'us' ? { background: APP.primary, color: '#000' } : { background: 'rgb(var(--fg-rgb) / 0.1)' }}
    >
      {SIDE[team]}
    </span>
  );
}

/** Step-end lines, one per side, over rally positions in play order. */
function Progression({ stats }: { stats: Stats }) {
  const order = [...new Set(stats.progression.map((s) => s.side))];
  const rallies = Math.max(...stats.progression.map((s) => s.rally));
  const top = Math.max(1, ...stats.progression.map((s) => s.points));
  const W = 300;
  const H = 150;
  const x = (r: number) => (r / rallies) * W;
  const y = (p: number) => H - (p / top) * H;
  return (
    <div>
      <div className="mb-1 flex gap-3 text-[11px] text-text-secondary">
        {order.map((side, i) => (
          <span key={side} className="inline-flex items-center gap-1">
            <span className="h-2 w-2 rounded-full" style={{ background: SERIES[i % SERIES.length] }} />
            {SIDE[side]}
          </span>
        ))}
      </div>
      <svg viewBox={`-24 -6 ${W + 30} ${H + 22}`} className="w-full">
        <line x1={0} y1={H} x2={W} y2={H} stroke="currentColor" className="text-border" />
        <text x={-6} y={4} textAnchor="end" className="fill-text-muted text-[9px]">{top}</text>
        <text x={-6} y={H} textAnchor="end" className="fill-text-muted text-[9px]">0</text>
        <text x={W} y={H + 14} textAnchor="end" className="fill-text-muted text-[9px]">Rally {rallies}</text>
        {order.map((side, i) => {
          const steps = stats.progression.filter((s) => s.side === side);
          let d = `M0 ${y(0)}`;
          for (const s of steps) d += ` H${x(s.rally)} V${y(s.points)}`;
          return <path key={side} d={d} fill="none" stroke={SERIES[i % SERIES.length]} strokeWidth={2} />;
        })}
      </svg>
    </div>
  );
}

/** One stacked bar per player, a segment per kind, most touches first. */
function PlayerTouches({ stats }: { stats: Stats }) {
  const kinds = KINDS.filter((k) => stats.players.some((p) => (p.touches[k] ?? 0) > 0));
  const top = Math.max(1, ...stats.players.map((p) => kinds.reduce((n, k) => n + (p.touches[k] ?? 0), 0)));
  return (
    <div className="space-y-1.5">
      <div className="flex flex-wrap gap-2 text-[11px] text-text-secondary">
        {kinds.map((k) => (
          <span key={k} className="inline-flex items-center gap-1">
            <span className="h-2 w-2 rounded-sm" style={{ background: KIND[k].color }} />
            {KIND[k].long}
          </span>
        ))}
      </div>
      {stats.players.map((p) => (
        <div key={p.player.number} className="flex items-center gap-2 text-[11px]">
          <span className="w-16 shrink-0 truncate text-text-secondary">{displayName(p.player)}</span>
          <div className="flex h-3 flex-1 overflow-hidden rounded-sm">
            {kinds.map((k) => (
              <div key={k} style={{ width: `${((p.touches[k] ?? 0) / top) * 100}%`, background: KIND[k].color }} title={`${KIND[k].long} ${(p.touches[k] ?? 0)}`} />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function Losses({ stats }: { stats: Stats }) {
  return (
    <div className="space-y-3">
      {stats.losses.map((group) => (
        <div key={group.side} className="space-y-1.5">
          <p className="text-xs font-semibold text-text-primary">{SIDE[group.side]}失分</p>
          {group.reasons.map((r) => (
            <div key={r.reason} className="space-y-0.5 text-[11px]">
              <div className="flex justify-between">
                <span className="text-text-primary">{lossReasonLabel(r.reason)}</span>
                <span className="tabular-nums text-text-secondary">
                  {r.count} · {r.percent}%
                </span>
              </div>
              <div className="h-1.5 rounded-full bg-ink/10">
                <div className="h-full rounded-full" style={{ width: `max(4px, ${(r.count / group.total) * 100}%)`, background: APP.danger }} />
              </div>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

function PlayerTable({ stats }: { stats: Stats }) {
  return (
    <table className="w-full text-left text-[12.5px]">
      <thead>
        <tr className="border-b border-border text-[11px] text-text-muted">
          <th className="pb-2 pr-3 font-normal">球員</th>
          <th className="pb-2 pr-3 text-right font-normal">觸球</th>
          {KINDS.map((k) => (
            <th key={k} className="pb-2 pr-3 text-right font-normal">
              {KIND[k].short}
            </th>
          ))}
          <th className="pb-2 text-right font-normal">決定分</th>
        </tr>
      </thead>
      <tbody>
        {stats.players.map((p) => (
          <tr key={p.player.number} className="border-b border-border-light/60 last:border-0">
            <td className="py-1.5 pr-3 text-text-primary">
              #{p.player.number} {p.player.name}
            </td>
            <td className="py-1.5 pr-3 text-right font-semibold tabular-nums">{KINDS.reduce((n, k) => n + (p.touches[k] ?? 0), 0)}</td>
            {KINDS.map((k) => (
              <td key={k} className="py-1.5 pr-3 text-right tabular-nums text-text-secondary">
                {(p.touches[k] ?? 0)}
              </td>
            ))}
            <td className="py-1.5 text-right tabular-nums">{p.deciding}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
