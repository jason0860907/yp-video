/** The App's 影片 feed beside the player (RallyFeedView in its iPad list
 *  mode): the count line, Rally | Action | Score, the 進階選項 filters and the
 *  scope's list. Same defaults as the App: sorted by time, Action opening on
 *  扣球, Action and Score in Rally only. One filter serves every scope.
 *  Each rally the user corrected carries an Edited chip that opens Review —
 *  Review's addition; the App shows only the new values. */

import { useMemo, useState } from 'react';
import { cn } from '@/lib/cn';
import { Badge } from '@/components/ui/Badge';
import { AppSegmented } from './AppSegmented';
import { ActionCard, CourtScoreText, RallyCard, ScoreCard } from './AppRows';
import { APP, displayName, FRAMING, hms, KIND, KINDS, kindOf, lossReasonLabel, OPPOSITE, playerColor, SIDE } from './appTerms';
import type { Candidate, Clip, Preview, Scope, Tag, WindowMode } from './types';

const SCOPES: { key: Scope; label: string }[] = [
  { key: 'rallies', label: 'Rally' },
  { key: 'actions', label: 'Action' },
  { key: 'scores', label: 'Score' },
];

interface Filter {
  player: number | null;
  kind: string | null;
  inRally: boolean;
  byScore: boolean;
}

export function AppFeed({
  preview,
  totalSeconds,
  tagsOf,
  edits,
  scope,
  mode,
  selected,
  onScope,
  onMode,
  onSelect,
  onOpenEdit,
}: {
  preview: Preview;
  /** The whole video's length (總長). */
  totalSeconds: number;
  /** A clip's live favourite categories. */
  tagsOf: (clip: Clip) => Tag[];
  /** Rally index → the user's corrections inside it. */
  edits: Map<number, Candidate[]>;
  scope: Scope;
  mode: WindowMode;
  selected: Clip | null;
  onScope: (scope: Scope) => void;
  onMode: (mode: WindowMode) => void;
  /** A clip picked; `queue` is what auto-next walks after it. */
  onSelect: (clip: Clip, queue: Clip[]) => void;
  onOpenEdit: (first: Candidate) => void;
}) {
  const [filter, setFilter] = useState<Filter>({ player: null, kind: null, inRally: true, byScore: false });
  const [open, setOpen] = useState(false);

  const kinds = KINDS.filter((k) => preview.actions.some((c) => c.kind === k));
  const clips = useMemo(() => {
    const { player, kind } = filter;
    if (scope === 'rallies') {
      const list = preview.rallies.filter(
        (r) =>
          (player == null && kind == null) ||
          r.touches.some((t) => (kind == null || t.kind === kind) && (player == null || t.player?.number === player)),
      );
      return filter.byScore
        ? [...list].sort((a, b) => (b.score ?? 0) - (a.score ?? 0))
        : [...list].sort((a, b) => (a.set ?? 1) - (b.set ?? 1) || a.start - b.start);
    }
    return preview[scope].filter(
      (c) =>
        (!filter.inRally || c.rally_index != null) &&
        (player == null || c.player?.number === player) &&
        (kind == null || (scope === 'actions' ? c.kind === kind : c.touches.some((t) => t.kind === kind))),
    );
  }, [preview, scope, filter]);

  const edited = (rallyIndex: number | null) => {
    const list = rallyIndex != null ? edits.get(rallyIndex) : undefined;
    const first = list?.[0];
    if (!list || !first) return null;
    return (
      <span
        role="button"
        tabIndex={0}
        title="Open this rally's corrections in Review"
        onClick={(e) => {
          e.stopPropagation();
          onOpenEdit(first);
        }}
      >
        <Badge tone="warning">Edited {list.length}</Badge>
      </span>
    );
  };

  const summary = [
    filter.player != null && `#${filter.player}`,
    filter.kind && kindOf(filter.kind).short,
    scope !== 'rallies' && !filter.inRally && '全部',
    scope === 'actions' && mode !== 'full_play' && FRAMING[mode],
    scope === 'rallies' && filter.byScore && '信心度',
  ].filter(Boolean);
  const active = filter.player != null || filter.kind != null;

  return (
    <div className="flex min-h-0 flex-col gap-2.5">
      <p className="text-xs text-text-secondary">
        <span className="font-semibold text-text-primary">{clips.length}</span>{' '}
        {scope === 'rallies' ? (
          <>
            個 rally · 總長 <span className="font-mono">{totalSeconds > 0 ? hms(totalSeconds) : '—'}</span> · 局內{' '}
            <span className="font-mono">{preview.stats.in_rally_seconds > 0 ? hms(preview.stats.in_rally_seconds) : '—'}</span>
          </>
        ) : scope === 'actions' ? (
          `個${FRAMING[mode]}片段`
        ) : (
          '個得失分片段'
        )}
      </p>
      <AppSegmented
        fill
        options={SCOPES}
        value={scope}
        onChange={(next) => {
          // Switching into Action opens it on 扣球, as the App does.
          if (next === 'actions' && scope !== 'actions' && filter.kind == null) setFilter({ ...filter, kind: 'spike' });
          onScope(next);
        }}
      />

      <div>
        <div className="flex items-center gap-2 px-0.5 text-xs">
          <button type="button" className="flex min-w-0 flex-1 items-center gap-1.5 text-left" onClick={() => setOpen(!open)}>
            <span
              className="inline-flex h-3.5 w-3.5 shrink-0 items-center justify-center rounded-full border text-[8px] leading-none"
              style={{ color: summary.length ? APP.primary : APP.accent, borderColor: 'currentColor' }}
            >
              ≡
            </span>
            <span className="text-text-secondary">進階選項</span>
            {summary.length > 0 && <span className="truncate font-semibold" style={{ color: APP.primary }}>{summary.join(' · ')}</span>}
            <span className="text-[10px] text-text-muted">{open ? '▴' : '▾'}</span>
          </button>
          {active && (
            <button type="button" className="font-semibold" style={{ color: APP.primary }} onClick={() => setFilter({ ...filter, player: null, kind: null })}>
              清除
            </button>
          )}
        </div>
        {open && (
          <div className="mt-2 space-y-2 rounded-xl bg-ink/[0.05] p-2.5">
            {preview.roster.length > 0 && (
              <Axis title="球員">
                <Chip selected={filter.player == null} onClick={() => setFilter({ ...filter, player: null })}>
                  全部球員
                </Chip>
                {preview.roster.map((p) => (
                  <Chip
                    key={p.number}
                    selected={filter.player === p.number}
                    dot={playerColor(p)}
                    glyph={String(p.number)}
                    onClick={() => setFilter({ ...filter, player: filter.player === p.number ? null : p.number })}
                  >
                    {displayName(p)}
                  </Chip>
                ))}
              </Axis>
            )}
            {kinds.length > 1 && (
              <Axis title="動作">
                <Chip selected={filter.kind == null} onClick={() => setFilter({ ...filter, kind: null })}>
                  全部動作
                </Chip>
                {kinds.map((k) => (
                  <Chip
                    key={k}
                    selected={filter.kind === k}
                    dot={KIND[k].color}
                    glyph={KIND[k].short}
                    onClick={() => setFilter({ ...filter, kind: filter.kind === k ? null : k })}
                  >
                    {KIND[k].long}
                  </Chip>
                ))}
              </Axis>
            )}
            {scope !== 'rallies' && (
              <Axis title="範圍">
                <Chip selected={!filter.inRally} onClick={() => setFilter({ ...filter, inRally: false })}>
                  全部
                </Chip>
                <Chip selected={filter.inRally} onClick={() => setFilter({ ...filter, inRally: true })}>
                  In Rally
                </Chip>
              </Axis>
            )}
            {scope === 'actions' && (
              <Axis title="片段取景">
                {(Object.keys(FRAMING) as WindowMode[]).map((m) => (
                  <Chip key={m} selected={mode === m} onClick={() => onMode(m)}>
                    {FRAMING[m]}
                  </Chip>
                ))}
              </Axis>
            )}
            {scope === 'rallies' && (
              <Axis title="排序">
                <Chip selected={filter.byScore} onClick={() => setFilter({ ...filter, byScore: true })}>
                  信心度
                </Chip>
                <Chip selected={!filter.byScore} onClick={() => setFilter({ ...filter, byScore: false })}>
                  時間
                </Chip>
              </Axis>
            )}
          </div>
        )}
      </div>

      <div className="vq-list max-h-[calc(100vh-17rem)] min-h-0 space-y-1.5 overflow-y-auto pr-1">
        {scope === 'scores' && <ScoreSummary preview={preview} scores={clips} />}
        {clips.length === 0 && (
          <p className="py-6 text-center text-xs text-text-muted">
            {scope === 'scores' ? '沒有符合篩選條件的得失分片段' : '沒有符合篩選條件的片段'}
          </p>
        )}
        {clips.map((c) => {
          const props = { selected: c.key === selected?.key, onClick: () => onSelect(c, clips) };
          if (scope === 'rallies') return <RallyCard key={c.key} rally={c} tags={tagsOf(c)} extra={edited(c.rally_index)} {...props} />;
          if (scope === 'actions')
            return <ActionCard key={c.key} clip={c} number={(filter.kind ? c.kind_index : c.index) ?? 0} {...props} />;
          return <ScoreCard key={c.key} clip={c} extra={edited(c.rally_index)} {...props} />;
        })}
      </div>
    </div>
  );
}

function Axis({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1">
      <p className="text-[10px] font-semibold text-text-muted">{title}</p>
      <div className="flex flex-wrap gap-1.5">{children}</div>
    </div>
  );
}

function Chip({
  selected,
  dot,
  glyph,
  onClick,
  children,
}: {
  selected: boolean;
  dot?: string;
  glyph?: string;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px]',
        selected ? 'border-primary/70 bg-primary/15 text-text-primary' : 'border-border text-text-secondary hover:bg-ink/5',
      )}
    >
      {dot && (
        <span className="inline-flex h-4 min-w-4 items-center justify-center rounded-full px-0.5 text-[9px] font-bold text-white" style={{ background: dot }}>
          {glyph}
        </span>
      )}
      {children}
    </button>
  );
}

/** The Score scope's head: each set's camera-side total, then each losing
 *  side's reasons (count · share, unmarked last and grey). The filter
 *  narrows the reasons too, as in the App. */
function ScoreSummary({ preview, scores }: { preview: Preview; scores: Clip[] }) {
  const [open, setOpen] = useState(false);
  const sides = ['left', 'right', 'near', 'far'];
  return (
    <div className="space-y-2 pb-1">
      {preview.rallies.length > 0 && (
        <div className="flex items-baseline justify-between gap-3 text-xs">
          <span className="font-semibold text-text-secondary">辨識比分累計</span>
          <span className="flex flex-col items-end gap-1">
            {preview.court_totals.map((t) => (
              <span key={t.set} className="text-right">
                <CourtScoreText score={t} />
                {t.unknown > 0 && <span className="block text-[11px] text-text-secondary">另有 {t.unknown} 球勝方待確認</span>}
              </span>
            ))}
          </span>
        </div>
      )}
      {scores.length > 0 && (
        <div className="text-xs">
          <button type="button" className="flex w-full items-center font-semibold text-text-secondary" onClick={() => setOpen(!open)}>
            各側失分原因 <span className="ml-auto text-text-muted">{open ? '▴' : '▾'}</span>
          </button>
          {open && (
            <div className="mt-2 space-y-2">
              {sides.map((side) => {
                const losses = scores.filter((c) => c.winner && OPPOSITE[c.winner] === side);
                if (!losses.length) return null;
                const counts = new Map<string | null, number>();
                for (const c of losses) counts.set(c.loss_reason ?? null, (counts.get(c.loss_reason ?? null) ?? 0) + 1);
                const ranked = [...counts].sort(([a, x], [b, y]) => y - x || (a ?? '').localeCompare(b ?? ''));
                return (
                  <div key={side} className="space-y-2 rounded-xl bg-surface-100 p-3">
                    <div className="flex items-baseline justify-between">
                      <span className="font-semibold text-text-primary">{SIDE[side]}失分原因</span>
                      <span className="text-text-secondary">{losses.length} 個失分片段</span>
                    </div>
                    {[...ranked.filter(([r]) => r), ...ranked.filter(([r]) => !r)].map(([reason, n]) => (
                      <ReasonBar key={reason ?? ''} name={reason ? lossReasonLabel(reason) : '未標記'} count={n} total={losses.length} marked={!!reason} />
                    ))}
                  </div>
                );
              })}
              {scores.some((c) => !c.winner) && <p className="text-text-secondary">得分方未確認的回合不納入失分原因統計。</p>}
            </div>
          )}
        </div>
      )}
      <p className="pt-1 text-[10px] font-bold uppercase tracking-wide text-text-secondary">逐球結果 · 場側勝方與比分</p>
    </div>
  );
}

function ReasonBar({ name, count, total, marked }: { name: string; count: number; total: number; marked: boolean }) {
  const share = count / Math.max(1, total);
  return (
    <div className="space-y-1">
      <div className="flex justify-between">
        <span className={marked ? 'text-text-primary' : 'text-text-secondary'}>{name}</span>
        <span className="tabular-nums text-text-secondary">
          {count} 次 · {Math.floor(share * 100)}%
        </span>
      </div>
      <div className="h-[5px] rounded-full bg-ink/10">
        <div className="h-full rounded-full" style={{ width: `${share * 100}%`, background: marked ? APP.danger : 'var(--text-muted)' }} />
      </div>
    </div>
  );
}
