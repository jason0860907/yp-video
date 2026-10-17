/** The App's three feeds — Rally, Action, Score — with its filters (player,
 *  touch kind, favorite), listed the Label way: rally rows that expand.
 *
 *  Rally: each row is a clip; it opens into its touches, and a touch seeks.
 *  Action / Score: rows group the clips by rally, one group open at a time;
 *  clips outside any rally sit in the closed outside group, which is the
 *  App's default "in rally" membership. A rally the user corrected carries
 *  an Edited chip that opens its corrections in Review. */

import { useCallback, useMemo, useState } from 'react';
import { cn } from '@/lib/cn';
import { fieldCls } from '@/components/form/Field';
import { Badge } from '@/components/ui/Badge';
import { Segmented } from '@/components/ui/Segmented';
import { ActionDot, EventRows, OutsideRow, RallyRow } from '@/components/action/RallyRows';
import { formatActionTime } from '@/lib/actionEditorModel';
import type { Candidate, Clip, Preview, Scope, Tag, Touch } from './types';
import { lossReasonLabel, playerLabel, sideLabel } from './types';

const OUTSIDE = '__outside__';
const KINDS = ['serve', 'receive', 'set', 'spike', 'block'];
const SCOPE_LABEL: Record<Scope, string> = { rallies: 'Rally', actions: 'Action', scores: 'Score' };

export function ClipList({
  preview,
  tags,
  rallyTags,
  edits,
  scope,
  selected,
  onScope,
  onSelect,
  onSeek,
  onOpenEdit,
}: {
  preview: Preview;
  tags: Tag[];
  /** Rally UUID (lowercase) → its favorite tag ids. */
  rallyTags: Map<string, string[]>;
  /** Rally index → the user's corrections inside it. */
  edits: Map<number, Candidate[]>;
  scope: Scope;
  selected: Clip | null;
  onScope: (scope: Scope) => void;
  /** A clip picked; `queue` is what auto-next walks after it. */
  onSelect: (clip: Clip, queue: Clip[]) => void;
  onSeek: (time: number) => void;
  onOpenEdit: (first: Candidate) => void;
}) {
  const [kind, setKind] = useState('');
  const [player, setPlayer] = useState('');
  const [tag, setTag] = useState('');
  const [expanded, setExpanded] = useState<string | null>(null);
  const tagName = useMemo(() => new Map(tags.map((t) => [t.id.toLowerCase(), t.name])), [tags]);
  const tagsOf = useCallback(
    (c: Clip) => (c.id ? rallyTags.get(c.id.toLowerCase()) : c.tag_ids)?.map((t) => t.toLowerCase()) ?? [],
    [rallyTags],
  );

  const clips = useMemo(
    () =>
      preview[scope].filter((c) => {
        const byPlayer = (n: number | undefined) => !player || String(n) === player;
        if (tag && !tagsOf(c).includes(tag)) return false;
        if (scope === 'rallies')
          return (!kind && !player) || c.touches.some((t) => (!kind || t.kind === kind) && byPlayer(t.player?.number));
        if (!byPlayer(c.player?.number)) return false;
        if (scope === 'scores') return !kind || c.touches.some((t) => t.kind === kind);
        return !kind || c.kind === kind;
      }),
    [preview, scope, player, kind, tag, tagsOf],
  );

  const groupOf = (c: Clip) => (c.rally_index == null ? OUTSIDE : String(c.rally_index));
  // The selection's group opens with it — also when auto-next walks on.
  const selectedGroup = selected ? groupOf(selected) : null;
  const [openedFor, setOpenedFor] = useState<string | null>(null);
  if (selectedGroup !== openedFor) {
    setOpenedFor(selectedGroup);
    if (selectedGroup) setExpanded(selectedGroup);
  }

  // Auto-next stays on the side of the outside line it started on, so the
  // closed outside group is not played into unasked.
  const select = (c: Clip) =>
    onSelect(c, clips.filter((x) => (x.rally_index == null) === (c.rally_index == null)));
  const rallyClip = useMemo(() => new Map(preview.rallies.map((r) => [String(r.rally_index), r])), [preview.rallies]);
  const rallyBounds = (r: Clip) => ({ rally_id: r.rally_index ?? 0, start: r.start, end: r.end });

  /** The rally's edits, outcome and favorites, on whichever row stands for it. */
  const chips = (r: Clip | undefined) => {
    if (!r) return null;
    const edited = r.rally_index != null ? edits.get(r.rally_index) : undefined;
    const first = edited?.[0];
    return (
      <>
        {edited && first && (
          <button
            type="button"
            title="Open this rally's corrections in Review"
            onClick={(e) => {
              e.stopPropagation();
              onOpenEdit(first);
            }}
          >
            <Badge tone="warning" className="hover:bg-amber-500/25">
              Edited {edited.length}
            </Badge>
          </button>
        )}
        {r.winner && <Badge tone="success">Winner {sideLabel(r.winner)}</Badge>}
        {r.loss_reason && (
          <Badge tone={r.loss_reason_inferred ? 'neutral' : 'warning'}>
            {lossReasonLabel(r.loss_reason)}
            {r.loss_reason_inferred && ' (inferred)'}
          </Badge>
        )}
        {tagsOf(r).map((t) => (
          <Badge key={t} tone="accent">
            {tagName.get(t) ?? 'Favorite'}
          </Badge>
        ))}
      </>
    );
  };

  const touchRows = (r: Clip) => (
    <EventRows<Touch & { id: string }>
      entries={r.touches.map((t) => ({ ...t, id: t.event_id }))}
      empty="No touches in this rally"
      columns="auto 4.5rem minmax(0,1fr) auto"
      selectedId={null}
      isActive={() => false}
      onJump={(t) => onSeek(t.time)}
    >
      {(t) => (
        <>
          <ActionDot label={t.kind} visible />
          <span className="text-xs capitalize text-text-primary">{t.kind}</span>
          <span className="truncate text-xs text-text-secondary">{t.player && playerLabel(t.player)}</span>
          <span className="font-mono text-[11px] tabular-nums text-text-muted">{formatActionTime(t.time)}</span>
        </>
      )}
    </EventRows>
  );

  const clipRows = (entries: Clip[]) => (
    <EventRows<Clip & { id: string }>
      entries={entries.map((c) => ({ ...c, id: c.key }))}
      empty="No clips"
      columns="auto minmax(0,1fr) auto"
      selectedId={selected?.key ?? null}
      isActive={() => false}
      onJump={select}
    >
      {(c) => (
        <>
          <ActionDot label={c.kind} visible />
          <span className="flex min-w-0 flex-wrap items-center gap-1.5 text-xs">
            <span className="capitalize text-text-primary">
              {c.kind} #{c.index}
            </span>
            {c.player && <span className="truncate text-text-secondary">{playerLabel(c.player)}</span>}
            {scope === 'scores' && chips(c)}
          </span>
          <span className="font-mono text-[11px] tabular-nums text-text-muted">
            {formatActionTime(c.start)} → {formatActionTime(c.end)}
          </span>
        </>
      )}
    </EventRows>
  );

  const toggle = (key: string) => setExpanded(expanded === key ? null : key);
  const outside = clips.filter((c) => c.rally_index == null);

  return (
    <div className="flex min-h-0 flex-col gap-3">
      <Segmented
        options={(Object.keys(SCOPE_LABEL) as Scope[]).map((key) => ({
          key,
          label: `${SCOPE_LABEL[key]} ${preview[key].length}`,
        }))}
        value={scope}
        onChange={onScope}
      />
      <div className="grid grid-cols-3 gap-2">
        <select aria-label="Player" className={cn(fieldCls, 'text-xs')} value={player} onChange={(e) => setPlayer(e.target.value)}>
          <option value="">All players</option>
          {preview.roster.map((p) => (
            <option key={p.number} value={p.number}>
              {playerLabel(p)}
            </option>
          ))}
        </select>
        <select aria-label="Kind" className={cn(fieldCls, 'text-xs')} value={kind} onChange={(e) => setKind(e.target.value)}>
          {/* Rally / Score rows hold several touches: the filter keeps those
              containing the kind, so say so. */}
          <option value="">{scope === 'actions' ? 'All kinds' : 'Any touches'}</option>
          {KINDS.map((k) => (
            <option key={k} value={k}>
              {scope === 'actions' ? k.charAt(0).toUpperCase() + k.slice(1) : `Has ${k}`}
            </option>
          ))}
        </select>
        <select aria-label="Favorite" className={cn(fieldCls, 'text-xs')} value={tag} onChange={(e) => setTag(e.target.value)}>
          <option value="">All favorites</option>
          {tags
            .filter((t) => t.deleted_at === null)
            .map((t) => (
              <option key={t.id} value={t.id.toLowerCase()}>
                {t.name}
              </option>
            ))}
        </select>
      </div>
      <div className="h-px bg-border" />
      <div className="vq-list max-h-[calc(100vh-24rem)] min-h-0 space-y-1.5 overflow-y-auto pr-1">
        {clips.length === 0 && <p className="py-6 text-center text-xs text-text-muted">No clips match the filters.</p>}
        {scope === 'rallies'
          ? clips.map((r) => {
              const key = groupOf(r);
              const open = expanded === key;
              return (
                <div key={r.key} className="space-y-1.5">
                  <RallyRow
                    rally={rallyBounds(r)}
                    count={r.touches.length}
                    open={open}
                    selected={r.key === selected?.key}
                    live={false}
                    onSelect={() => select(r)}
                    onToggle={() => (open ? setExpanded(null) : select(r))}
                  >
                    {chips(r)}
                  </RallyRow>
                  {open && touchRows(r)}
                </div>
              );
            })
          : preview.rallies.map((r) => {
              const key = groupOf(r);
              const entries = clips.filter((c) => groupOf(c) === key);
              if (!entries.length) return null;
              return (
                <div key={r.key} className="space-y-1.5">
                  <RallyRow
                    rally={rallyBounds(r)}
                    count={entries.length}
                    open={expanded === key}
                    selected={selectedGroup === key}
                    live={false}
                    onSelect={() => toggle(key)}
                    onToggle={() => toggle(key)}
                  >
                    {scope === 'actions' && chips(rallyClip.get(key))}
                  </RallyRow>
                  {expanded === key && clipRows(entries)}
                </div>
              );
            })}
        {scope !== 'rallies' && outside.length > 0 && (
          <div className="space-y-1.5">
            <OutsideRow rowKey={OUTSIDE} count={outside.length} open={expanded === OUTSIDE} onToggle={() => toggle(OUTSIDE)} />
            {expanded === OUTSIDE && clipRows(outside)}
          </div>
        )}
      </div>
      <p className="text-[11px] text-text-muted">
        {clips.length} of {preview[scope].length} clips
      </p>
    </div>
  );
}
