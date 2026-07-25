/** What is left to review, one chip per action event, grouped by rally.
 *
 *  The page's whole job is "find the events nobody has looked at yet", so the
 *  filter defaults to Unreviewed rather than All: opening a half-done video
 *  should show the work, not the archive. Chips carry the verdict as a glyph
 *  instead of a colour alone — the action colour is already spoken for.
 *
 *  Grouped by rally so the remaining work is legible per rally, using the
 *  same rule the player's sidebar groups with (labeling/shared.rallyOf) — the
 *  two disagreeing would be invisible until a rally confirm hit the wrong
 *  events. Confirming lives on the sidebar, next to the video you are
 *  watching; this list is the overview and the way in.
 */

import { useMemo, useState } from 'react';
import { cn } from '@/lib/cn';
import { actionColor } from '@/lib/actionColors';
import { EmptyState } from '@/components/ui/EmptyState';
import { canConfirm, rallyOf, verdictOf, VERDICT, type ActorVerdict, type Rally } from '@/components/labeling/shared';
import type { ReidRecord } from '@/types/api';

type Filter = 'unreviewed' | ActorVerdict | 'all';

const FILTERS: { key: Filter; label: string }[] = [
  { key: 'unreviewed', label: 'Unreviewed' },
  { key: 'manual', label: 'Manual' },
  { key: 'occluded', label: 'Occluded' },
  { key: 'confirmed_auto', label: 'Confirmed' },
  { key: 'all', label: 'All' },
];

interface Section {
  key: string;
  title: string | null;
  records: ReidRecord[];
}

export interface EventReviewListProps {
  records: ReidRecord[];
  rallies: Rally[];
  fps: number;
  /** The event the player is parked on — highlighted, never auto-scrolled. */
  selectedId: string | null;
  onJump: (record: ReidRecord) => void;
}

export function EventReviewList({
  records,
  rallies,
  fps,
  selectedId,
  onJump,
}: EventReviewListProps) {
  const [filter, setFilter] = useState<Filter>('unreviewed');

  const counts = useMemo(() => {
    const out: Partial<Record<ActorVerdict, number>> = {};
    for (const r of records) {
      const v = verdictOf(r);
      out[v] = (out[v] ?? 0) + 1;
    }
    return out;
  }, [records]);

  const sections = useMemo<Section[]>(() => {
    if (!rallies.length) return [{ key: 'all', title: null, records }];
    const byRally = new Map<number, ReidRecord[]>(rallies.map((r) => [r.rally_id, []]));
    const outside: ReidRecord[] = [];
    for (const record of records) {
      const rally = rallyOf(rallies, record, fps);
      if (rally) byRally.get(rally.rally_id)!.push(record);
      else outside.push(record);
    }
    return [
      ...rallies.map((r) => ({
        key: `r${r.rally_id}`,
        title: `Rally ${r.rally_id}`,
        records: byRally.get(r.rally_id) ?? [],
      })),
      ...(outside.length
        ? [{ key: 'outside', title: 'Outside rallies', records: outside }]
        : []),
    ];
  }, [records, rallies, fps]);

  const visible = sections
    .map((s) => ({
      ...s,
      shown: filter === 'all' ? s.records : s.records.filter((r) => verdictOf(r) === filter),
      confirmable: s.records.filter(canConfirm),
    }))
    .filter((s) => s.shown.length > 0);

  return (
    <>
      <div className="mb-3 flex flex-wrap items-center gap-1.5">
        {FILTERS.map(({ key, label }) => {
          const n = key === 'all' ? records.length : counts[key as ActorVerdict] ?? 0;
          return (
            <button
              key={key}
              type="button"
              onClick={() => setFilter(key)}
              className={cn(
                'rounded-full px-3 py-1 text-xs font-medium transition-colors',
                filter === key
                  ? 'bg-primary text-on-primary'
                  : 'bg-surface-50 text-text-secondary ring-1 ring-border hover:bg-ink/[0.04]',
                !n && filter !== key && 'opacity-40',
              )}
            >
              {label} <span className="font-mono tabular-nums">{n}</span>
            </button>
          );
        })}
      </div>

      {!visible.length ? (
        <EmptyState
          icon={
            <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          }
          title={filter === 'unreviewed' ? 'Every action has a verdict' : 'Nothing in this filter'}
          subtitle={
            filter === 'unreviewed'
              ? 'This video is ready to feed association training.'
              : 'Switch filters to see the rest.'
          }
        />
      ) : (
        <div className="space-y-3">
          {visible.map((section) => (
            <div key={section.key}>
              {section.title && (
                <div className="mb-1.5 flex items-center gap-2">
                  <span className="text-[11px] font-semibold uppercase tracking-wider text-text-secondary">
                    {section.title}
                  </span>
                  <span className="font-mono text-[10px] tabular-nums text-text-muted">
                    {section.shown.length}/{section.records.length}
                  </span>
                  {section.confirmable.length > 0 && (
                    <span
                      className="font-mono text-[10px] tabular-nums text-primary-light"
                      title={`${section.confirmable.length} automatic picks here can be confirmed from this rally's row in the sidebar`}
                    >
                      ✓{section.confirmable.length} confirmable
                    </span>
                  )}
                </div>
              )}
              <div className="flex flex-wrap gap-1.5">
                {section.shown.map((r) => {
                  const v = verdictOf(r);
                  return (
                    <button
                      key={r.id}
                      type="button"
                      onClick={() => onJump(r)}
                      title={`${r.label ?? 'action'} · f${r.frame} — ${VERDICT[v].title}. Click to park the video here.`}
                      className={cn(
                        'inline-flex items-center gap-1.5 rounded-lg px-2 py-1 font-mono text-[11px] tabular-nums ring-1 transition-colors',
                        selectedId === r.id
                          ? 'bg-primary/20 text-primary-light ring-primary/40'
                          : 'bg-surface-50 text-text-secondary ring-border hover:bg-ink/[0.04]',
                      )}
                    >
                      <span
                        className="h-1.5 w-1.5 flex-shrink-0 rounded-full"
                        style={{ background: actionColor(r.label) }}
                      />
                      f{r.frame}
                      <span className={cn('text-[10px]', v === 'unreviewed' && 'text-text-muted')}>
                        {VERDICT[v].glyph}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      )}
    </>
  );
}
