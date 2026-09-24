/** The App's three feeds — Rally, Action, Score — with its filters: player,
 *  touch kind, rally and favorites. */

import { useCallback, useMemo, useState } from 'react';
import { cn } from '@/lib/cn';
import { fieldCls } from '@/components/form/Field';
import { Badge } from '@/components/ui/Badge';
import type { Clip, Preview, Scope, Tag } from './types';
import { KIND_LABEL, SIDE_LABEL, fmtTime, lossReasonLabel, playerLabel } from './types';

const SCOPES: [Scope, string][] = [
  ['rallies', 'Rally'],
  ['actions', 'Action'],
  ['scores', 'Score'],
];

export function ClipList({
  preview,
  tags,
  rallyTags,
  scope,
  selected,
  onScope,
  onSelect,
}: {
  preview: Preview;
  tags: Tag[];
  /** Rally UUID (lowercase) → its favorite tag ids. */
  rallyTags: Map<string, string[]>;
  scope: Scope;
  selected: string;
  onScope: (scope: Scope) => void;
  onSelect: (clip: Clip, visible: Clip[]) => void;
}) {
  const [kind, setKind] = useState('');
  const [player, setPlayer] = useState('');
  const [rally, setRally] = useState('');
  const [tag, setTag] = useState('');
  const [inRally, setInRally] = useState(true);
  const tagName = useMemo(() => new Map(tags.map((t) => [t.id.toLowerCase(), t.name])), [tags]);
  const tagsOf = useCallback(
    (c: Clip) =>
      (c.id ? rallyTags.get(c.id.toLowerCase()) : c.tag_ids)?.map((t) => t.toLowerCase()) ?? [],
    [rallyTags],
  );

  const clips = useMemo(
    () =>
      preview[scope].filter((c) => {
        const byPlayer = (n: number | undefined) => !player || String(n) === player;
        if (tag && !tagsOf(c).includes(tag)) return false;
        if (rally && String(c.rally_index) !== rally) return false;
        if (scope === 'rallies')
          return (
            (!kind && !player) ||
            c.touches.some((t) => (!kind || t.kind === kind) && byPlayer(t.player?.number))
          );
        if (inRally && c.rally_index == null) return false;
        if (!byPlayer(c.player?.number)) return false;
        if (scope === 'scores') return !kind || c.touches.some((t) => t.kind === kind);
        return !kind || c.kind === kind;
      }),
    [preview, scope, player, kind, rally, tag, inRally, tagsOf],
  );

  return (
    <div className="flex min-h-0 flex-col gap-3">
      <div className="flex gap-1 rounded-lg bg-surface-200 p-1">
        {SCOPES.map(([key, label]) => (
          <button
            key={key}
            onClick={() => onScope(key)}
            className={cn(
              'flex-1 rounded-md py-1.5 text-xs font-semibold',
              key === scope ? 'bg-surface-50 text-text-primary shadow-sm' : 'text-text-muted',
            )}
          >
            {label} <span className="font-normal text-text-muted">{preview[key].length}</span>
          </button>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-2">
        <select
          aria-label="球員"
          className={fieldCls}
          value={player}
          onChange={(e) => setPlayer(e.target.value)}
        >
          <option value="">全部球員</option>
          {preview.roster.map((p) => (
            <option key={p.number} value={p.number}>
              {playerLabel(p)}
            </option>
          ))}
        </select>
        <select
          aria-label="動作"
          className={fieldCls}
          value={kind}
          onChange={(e) => setKind(e.target.value)}
        >
          <option value="">全部動作</option>
          {['serve', 'receive', 'set', 'spike', 'block'].map((k) => (
            <option key={k} value={k}>
              {KIND_LABEL[k]}
            </option>
          ))}
        </select>
        <select
          aria-label="回合"
          className={fieldCls}
          value={rally}
          onChange={(e) => setRally(e.target.value)}
        >
          <option value="">全部回合</option>
          {preview.rallies.map((r) => (
            <option key={r.key} value={String(r.rally_index)}>
              Rally {r.rally_index}
            </option>
          ))}
        </select>
        <select
          aria-label="收藏"
          className={fieldCls}
          value={tag}
          onChange={(e) => setTag(e.target.value)}
        >
          <option value="">全部收藏</option>
          {tags
            .filter((t) => t.deleted_at === null)
            .map((t) => (
              <option key={t.id} value={t.id.toLowerCase()}>
                {t.name}
              </option>
            ))}
        </select>
      </div>
      {scope !== 'rallies' && (
        <label className="flex items-center gap-2 text-xs text-text-secondary">
          <input type="checkbox" checked={inRally} onChange={(e) => setInRally(e.target.checked)} />
          只看回合內
        </label>
      )}
      <p className="text-xs text-text-muted">{clips.length} 個片段</p>
      <div className="min-h-0 flex-1 space-y-1 overflow-auto pr-1">
        {clips.map((c) => (
          <button
            key={c.key}
            onClick={() => onSelect(c, clips)}
            className={cn(
              'w-full rounded-lg border px-3 py-2 text-left text-sm',
              c.key === selected
                ? 'border-primary/60 bg-primary/10'
                : 'border-border-light hover:border-border-bright',
            )}
          >
            <span className="flex items-baseline gap-2">
              <span className="font-semibold">
                {scope === 'rallies'
                  ? `Rally ${c.rally_index}`
                  : `${KIND_LABEL[c.kind] ?? c.kind} #${c.index}`}
              </span>
              <span className="font-mono text-xs text-text-muted">
                {fmtTime(c.start)}–{fmtTime(c.end)}
              </span>
            </span>
            <span className="mt-1 flex flex-wrap items-center gap-1.5 text-xs text-text-secondary">
              {scope !== 'rallies' && c.rally_index != null && <span>Rally {c.rally_index}</span>}
              {c.player && <span>{playerLabel(c.player)}</span>}
              {c.winner && <Badge tone="success">{SIDE_LABEL[c.winner] ?? c.winner}得分</Badge>}
              {c.loss_reason && (
                <Badge tone={c.loss_reason_inferred ? 'neutral' : 'warning'}>
                  {lossReasonLabel(c.loss_reason)}
                  {c.loss_reason_inferred && '（推測）'}
                </Badge>
              )}
              {tagsOf(c).map((t) => (
                <Badge key={t} tone="accent">
                  {tagName.get(t) ?? '收藏'}
                </Badge>
              ))}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
