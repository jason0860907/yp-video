/** The App's 分析 → 摘要 for one match (MatchAnalysisView /
 *  AnalysisStatsContent, wide layout), from app_review/stats.py: the meta
 *  line and score, then 得分走勢, 球員觸球 and 失分原因 as Swift Charts draws
 *  them, and 球員數據. Parts that need a court calibration — 球路, 球速與高度
 *  and the court findings — are not shown. 我方 is read from the user's
 *  corrections; it is set in the App. */

import type { ReactNode } from 'react';
import { APP, displayName, hms, KIND, KINDS, lossReasonLabel, playerColor, SIDE } from './appTerms';
import type { LibraryMatch, Preview, Stats } from './types';

/** Swift Charts' default series colours (dark mode), in series order. */
const SERIES = ['#0A84FF', '#30D158', '#FF9F0A', '#BF5AF2'];
const SIDE_ORDER = ['us', 'them', 'left', 'right', 'near', 'far'];

/** Every side the charts plot, in one order, so a side keeps its colour
 *  from 得分走勢 to 失分原因. */
function seriesOf(stats: Stats): Map<string, string> {
  const sides = new Set([...stats.progression.map((s) => s.side), ...stats.losses.map((l) => l.side)]);
  const order = SIDE_ORDER.filter((s) => sides.has(s));
  return new Map(order.map((s, i) => [s, SERIES[i % SERIES.length] as string]));
}

/** Round axis ticks from 0 past `max`, about four of them. */
function ticks(max: number): number[] {
  const step = [1, 2, 5, 10, 20, 25, 50, 100].find((s) => max / s <= 4) ?? 100;
  const out = [0];
  while (out[out.length - 1]! < max) out.push(out[out.length - 1]! + step);
  return out;
}

export function AnalysisSummary({ match, preview }: { match: LibraryMatch; preview: Preview }) {
  const stats = preview.stats;
  const date = new Date((match.recorded_at ?? match.created_at) * 1000).toLocaleDateString('zh-TW');
  const sides = stats.team_sides;
  const series = seriesOf(stats);
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 px-1">
        <p className="text-xs text-text-secondary">
          {[date, `${stats.rally_count} 個 Rally`, `比賽時間 ${stats.in_rally_seconds > 0 ? hms(stats.in_rally_seconds) : '—'}`].join(' · ')}
        </p>
        <span className="rounded-full bg-ink/[0.08] px-2.5 py-0.5 text-[11px] text-text-secondary">
          {sides ? `我方：${SIDE[sides.start]}起` : '尚未設定我方'}
        </span>
        <span className="flex-1" />
        <div className="text-right text-base font-semibold tabular-nums text-text-primary">
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
      </div>

      {stats.findings.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {stats.findings.map((f) => (
            <div key={f.kind} className="flex items-center gap-2 rounded-xl bg-surface-100 px-3.5 py-2.5 text-sm">
              <span style={{ color: APP.accent }}>⚠</span>
              {f.team && <TeamBadge team={f.team} />}
              <span className="font-semibold text-text-primary">失分最多：{lossReasonLabel(f.reason)}</span>
              <span className="text-xs text-text-secondary">
                {f.count}／{f.total} 分
              </span>
            </div>
          ))}
        </div>
      )}

      <div className="grid gap-4 lg:grid-cols-3">
        {stats.progression.length > 0 && (
          <Panel title="得分走勢" note={stats.has_teams ? undefined : '依畫面左右邊計分，不代表隊伍'}>
            <Legend items={[...series].filter(([s]) => stats.progression.some((p) => p.side === s)).map(([s, c]) => [SIDE[s] ?? s, c])} />
            <Progression stats={stats} series={series} />
          </Panel>
        )}
        {stats.players.length > 0 && (
          <Panel title="球員觸球">
            <PlayerTouches stats={stats} />
          </Panel>
        )}
        {stats.losses.length > 0 && (
          <Panel title="失分原因" note="含依回合推測的原因">
            <Legend items={stats.losses.map((l) => [SIDE[l.side] ?? l.side, series.get(l.side) ?? SERIES[0]!])} />
            <Losses stats={stats} series={series} />
          </Panel>
        )}
      </div>

      <Panel title="球員數據">
        {stats.players.length === 0 ? (
          <div className="py-6 text-center text-xs text-text-secondary">
            <p className="text-sm text-text-primary">還沒有球員數據</p>
            <p>完成球員配對後，這裡會顯示每位球員的觸球與進攻參與。</p>
          </div>
        ) : (
          <PlayerTable stats={stats} />
        )}
      </Panel>
    </div>
  );
}

/** A chart card the App's way: bold title, the caveat muted on the right. */
function Panel({ title, note, children }: { title: string; note?: string; children: ReactNode }) {
  return (
    <section className="rounded-2xl bg-surface-100 p-4">
      <div className="mb-2 flex items-baseline justify-between gap-3">
        <h3 className="font-body text-[13px] font-semibold text-text-primary">{title}</h3>
        {note && <span className="truncate text-[11px] text-text-muted">{note}</span>}
      </div>
      {children}
    </section>
  );
}

function Legend({ items }: { items: [label: string, color: string][] }) {
  return (
    <div className="mb-2 flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-text-secondary">
      {items.map(([label, color]) => (
        <span key={label} className="inline-flex items-center gap-1">
          <span className="h-2 w-2 rounded-full" style={{ background: color }} />
          {label}
        </span>
      ))}
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

/** Step-end lines, one per side, over rally positions in play order; grid
 *  and the value axis on the right, as Swift Charts lays them out. */
function Progression({ stats, series }: { stats: Stats; series: Map<string, string> }) {
  const rallies = Math.max(...stats.progression.map((s) => s.rally));
  const yTicks = ticks(Math.max(1, ...stats.progression.map((s) => s.points)));
  const xTicks = ticks(rallies);
  const yMax = yTicks[yTicks.length - 1]!;
  const xMax = xTicks[xTicks.length - 1]!;
  const W = 300;
  const H = 180;
  const x = (r: number) => (r / xMax) * W;
  const y = (p: number) => H - (p / yMax) * H;
  return (
    <svg viewBox={`-2 -6 ${W + 26} ${H + 22}`} className="w-full text-[9px]">
      {yTicks.map((t) => (
        <g key={`y${t}`}>
          <line x1={0} x2={W} y1={y(t)} y2={y(t)} stroke="var(--line)" />
          <text x={W + 6} y={y(t) + 3} className="fill-text-muted">{t}</text>
        </g>
      ))}
      {xTicks.map((t) => (
        <g key={`x${t}`}>
          <line x1={x(t)} x2={x(t)} y1={0} y2={H} stroke="var(--line)" strokeDasharray="2 3" />
          <text x={x(t)} y={H + 14} textAnchor={t === 0 ? 'start' : 'middle'} className="fill-text-muted">{t}</text>
        </g>
      ))}
      {[...series].map(([side, color]) => {
        const steps = stats.progression.filter((s) => s.side === side);
        if (!steps.length) return null;
        let d = `M0 ${y(0)}`;
        for (const s of steps) d += ` H${x(s.rally)} V${y(s.points)}`;
        return <path key={side} d={d} fill="none" stroke={color} strokeWidth={2} strokeLinejoin="round" />;
      })}
    </svg>
  );
}

/** Horizontal bars behind a dashed value grid, the axis labels under them. */
function BarGrid({ max, children }: { max: number; children: (scale: (n: number) => string) => ReactNode }) {
  const t = ticks(Math.max(1, max));
  const top = t[t.length - 1]!;
  return (
    <div>
      <div className="relative">
        {t.map((n) => (
          <span key={n} className="absolute inset-y-0 border-l border-dashed border-border" style={{ left: `${(n / top) * 100}%` }} />
        ))}
        <div className="relative space-y-2 py-1">{children((n) => `${(n / top) * 100}%`)}</div>
      </div>
      <div className="relative mt-1 h-3 text-[9px] text-text-muted">
        {t.map((n) => (
          <span key={n} className="absolute -translate-x-1/2 first:translate-x-0" style={{ left: `${(n / top) * 100}%` }}>
            {n}
          </span>
        ))}
      </div>
    </div>
  );
}

/** One stacked bar per player, a segment per kind, most touches first. */
function PlayerTouches({ stats }: { stats: Stats }) {
  const kinds = KINDS.filter((k) => stats.players.some((p) => (p.touches[k] ?? 0) > 0));
  const total = (p: Stats['players'][number]) => kinds.reduce((n, k) => n + (p.touches[k] ?? 0), 0);
  return (
    <>
      <Legend items={kinds.map((k) => [KIND[k].long, KIND[k].color])} />
      <BarGrid max={Math.max(...stats.players.map(total))}>
        {(scale) =>
          stats.players.map((p) => (
            <div key={p.player.number}>
              <p className="mb-0.5 truncate text-[10px] text-text-secondary">{displayName(p.player)}</p>
              <div className="flex h-4">
                {kinds.map((k) => (
                  <div key={k} style={{ width: scale(p.touches[k] ?? 0), background: KIND[k].color }} title={`${KIND[k].long} ${p.touches[k] ?? 0}`} />
                ))}
              </div>
            </div>
          ))
        }
      </BarGrid>
    </>
  );
}

/** Each reason, a bar per losing side in the side's colour. */
function Losses({ stats, series }: { stats: Stats; series: Map<string, string> }) {
  const totals = new Map<string, number>();
  for (const g of stats.losses) for (const r of g.reasons) totals.set(r.reason, (totals.get(r.reason) ?? 0) + r.count);
  const reasons = [...totals].sort(([a, x], [b, y]) => y - x || a.localeCompare(b)).map(([r]) => r);
  const count = (side: string, reason: string) =>
    stats.losses.find((g) => g.side === side)?.reasons.find((r) => r.reason === reason)?.count ?? 0;
  return (
    <BarGrid max={Math.max(...stats.losses.flatMap((g) => g.reasons.map((r) => r.count)))}>
      {(scale) =>
        reasons.map((reason) => (
          <div key={reason}>
            <p className="mb-0.5 truncate text-[10px] text-text-secondary">{lossReasonLabel(reason)}</p>
            <div className="space-y-0.5">
              {stats.losses.map((g) => {
                const n = count(g.side, reason);
                return (
                  <div
                    key={g.side}
                    className="h-3"
                    style={{ width: scale(n), background: series.get(g.side) }}
                    title={`${SIDE[g.side] ?? g.side}失分 ${n}`}
                  />
                );
              })}
            </div>
          </div>
        ))
      }
    </BarGrid>
  );
}

function PlayerTable({ stats }: { stats: Stats }) {
  return (
    <div className="overflow-x-auto rounded-xl bg-ink/[0.04] px-2 py-1">
      <table className="w-full min-w-[560px] text-[12.5px]">
        <thead>
          <tr className="text-[11px] font-semibold" style={{ color: APP.primary }}>
            <th className="px-2 py-2 text-left font-semibold">球員</th>
            <th className="px-2 py-2 text-center font-semibold">觸球</th>
            {KINDS.map((k) => (
              <th key={k} className="px-2 py-2 text-center font-semibold">
                {KIND[k].short}
              </th>
            ))}
            <th className="px-2 py-2 text-center font-semibold">決定分</th>
          </tr>
        </thead>
        <tbody>
          {stats.players.map((p) => (
            <tr key={p.player.number} className="border-t border-border hover:bg-ink/[0.04]">
              <td className="px-2 py-1.5">
                <span className="inline-flex items-center gap-2 text-text-primary">
                  <span
                    className="inline-flex h-6 w-6 items-center justify-center rounded-full text-[11px] font-bold text-white"
                    style={{ background: playerColor(p.player) }}
                  >
                    {p.player.number}
                  </span>
                  {displayName(p.player)}
                </span>
              </td>
              <td className="px-2 py-1.5 text-center font-semibold tabular-nums text-text-primary">
                {KINDS.reduce((n, k) => n + (p.touches[k] ?? 0), 0)}
              </td>
              {KINDS.map((k) => (
                <td key={k} className="px-2 py-1.5 text-center tabular-nums text-text-secondary">
                  {p.touches[k] ?? 0}
                </td>
              ))}
              <td className="px-2 py-1.5 text-center tabular-nums text-text-primary">{p.deciding}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
