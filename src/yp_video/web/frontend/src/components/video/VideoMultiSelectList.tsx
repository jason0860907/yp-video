import { useEffect, useState } from 'react';
import type { ReactNode } from 'react';
import { cn } from '@/lib/cn';
import { Button } from '@/components/ui/Button';
import { EmptyState } from '@/components/ui/EmptyState';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { KindBadge } from './KindBadge';
import type { CutKind } from '@/types/api';

export interface VideoListItem {
  name: string;
  kind: CutKind;
}

export interface StatusOption<T> {
  value: string;
  label: string;
  /** Row stays visible while this option is active. */
  predicate: (video: T) => boolean;
}

export interface QuickSelect<T> {
  label: string;
  /** Applied to every visible row: true selects it, false deselects it. */
  predicate: (video: T) => boolean;
}

interface VideoMultiSelectListProps<T extends VideoListItem> {
  videos: T[];
  /** Controlled selection, keyed by video name so it survives refetches. */
  selected: Set<string>;
  onSelectedChange: (next: Set<string>) => void;
  title?: string;
  /** Status filter dropdown. The first option is the default; omit to hide. */
  statusOptions?: Array<StatusOption<T>>;
  /** One-click selection presets, shown as buttons next to the filters. */
  quickSelects?: Array<QuickSelect<T>>;
  /** Trailing row content — status badges, counts. */
  renderMeta?: (video: T) => ReactNode;
  maxHeightClass?: string;
  emptyTitle?: string;
  emptySubtitle?: string;
}

type KindFilter = 'all' | CutKind;

const selectCls =
  'w-auto cursor-pointer appearance-none rounded-lg border border-border-light bg-surface-50 px-3 py-1 text-xs text-text-primary focus:border-primary/50 focus:outline-none focus:ring-2 focus:ring-primary/15';

/** Filterable, multi-select video list: kind + status filters, select-visible,
 *  quick-select presets. Domain-specific status logic and row badges come in
 *  through `statusOptions` / `renderMeta`; the list itself knows nothing about
 *  detections or annotations. */
export function VideoMultiSelectList<T extends VideoListItem>({
  videos,
  selected,
  onSelectedChange,
  title = 'Videos',
  statusOptions,
  quickSelects,
  renderMeta,
  maxHeightClass = 'max-h-[56vh]',
  emptyTitle = 'No videos',
  emptySubtitle,
}: VideoMultiSelectListProps<T>) {
  const [kindFilter, setKindFilter] = useState<KindFilter>('all');
  const [status, setStatus] = useState(statusOptions?.[0]?.value ?? '');

  // Prune selections whose video vanished (deleted or refetched away) so the
  // selection count never claims videos that no longer exist.
  useEffect(() => {
    const names = new Set(videos.map((v) => v.name));
    if ([...selected].some((name) => !names.has(name))) {
      onSelectedChange(new Set([...selected].filter((name) => names.has(name))));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [videos]);

  const statusPredicate = statusOptions?.find((o) => o.value === status)?.predicate;
  const visible = videos.filter(
    (v) => (kindFilter === 'all' || v.kind === kindFilter) && (!statusPredicate || statusPredicate(v)),
  );
  const counts = {
    all: videos.length,
    broadcast: videos.filter((v) => v.kind === 'broadcast').length,
    sideline: videos.filter((v) => v.kind === 'sideline').length,
  };

  const toggle = (name: string, on: boolean) => {
    const next = new Set(selected);
    if (on) next.add(name);
    else next.delete(name);
    onSelectedChange(next);
  };
  const allVisibleSelected = visible.length > 0 && visible.every((v) => selected.has(v.name));
  const setVisibleSelection = (predicate: (video: T) => boolean) => {
    const next = new Set(selected);
    visible.forEach((v) => (predicate(v) ? next.add(v.name) : next.delete(v.name)));
    onSelectedChange(next);
  };

  return (
    <>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <SectionLabel className="mb-0">{title}</SectionLabel>
        <div className="flex items-center gap-2">
          {quickSelects?.map((qs) => (
            <Button key={qs.label} size="sm" onClick={() => setVisibleSelection(qs.predicate)}>
              {qs.label}
            </Button>
          ))}
          <select value={kindFilter} onChange={(e) => setKindFilter(e.target.value as KindFilter)} className={selectCls}>
            <option value="all">All kinds ({counts.all})</option>
            <option value="broadcast">Broadcast ({counts.broadcast})</option>
            <option value="sideline">Sideline ({counts.sideline})</option>
          </select>
          {statusOptions && (
            <select value={status} onChange={(e) => setStatus(e.target.value)} className={selectCls}>
              {statusOptions.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
          )}
        </div>
      </div>

      <div className="mb-2 flex items-center justify-between text-xs text-text-muted">
        <label className="inline-flex cursor-pointer items-center gap-2">
          <input
            type="checkbox"
            checked={allVisibleSelected}
            onChange={(e) => setVisibleSelection(() => e.target.checked)}
            className="h-3.5 w-3.5 accent-primary"
          />
          Select visible
        </label>
        <span className="font-mono tabular-nums">
          {selected.size} selected / {visible.length} shown
        </span>
      </div>

      <div className={cn('space-y-1 overflow-auto pr-1', maxHeightClass)}>
        {visible.length === 0 ? (
          <EmptyState
            icon={
              <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                />
              </svg>
            }
            title={emptyTitle}
            subtitle={emptySubtitle}
          />
        ) : (
          visible.map((v) => (
            <label
              key={v.name}
              className="flex w-max min-w-full cursor-pointer items-center gap-3 rounded-lg border border-border bg-surface-50 px-3 py-2 transition-colors hover:border-border-light"
            >
              <input
                type="checkbox"
                checked={selected.has(v.name)}
                onChange={(e) => toggle(v.name, e.target.checked)}
                className="h-3.5 w-3.5 flex-shrink-0 accent-primary"
              />
              <KindBadge kind={v.kind} />
              <span className="flex-1 whitespace-nowrap text-sm text-text-primary">{v.name}</span>
              {renderMeta?.(v)}
            </label>
          ))
        )}
      </div>
    </>
  );
}
