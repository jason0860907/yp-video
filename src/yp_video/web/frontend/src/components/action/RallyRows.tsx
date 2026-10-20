/** The Action Label sidebar's building blocks: rally rows that expand into
 *  their action rows. Every page listing a video's play by rally (Action,
 *  Court, Association) composes these, so the lists look and behave as one thing; each
 *  page supplies only its own columns.
 *
 *  Rows carry `data-rally-row` / `data-action-id` for lib/sidebarScroll. */

import type { ReactNode } from 'react';
import { cn } from '@/lib/cn';
import { actionColor } from '@/lib/actionColors';
import { formatActionTime } from '@/lib/actionEditorModel';

export function RallyRow({
  index,
  rally,
  count,
  open,
  selected,
  live,
  onSelect,
  onToggle,
  children,
}: {
  /** Position in time order, 0-based. */
  index: number;
  rally: { rally_id: number; start: number; end: number };
  /** How many actions it holds. */
  count: number;
  open: boolean;
  selected: boolean;
  /** The playhead is inside it. */
  live: boolean;
  onSelect: () => void;
  /** The actions chip: open or collapse the group. */
  onToggle: () => void;
  /** The page's own chips, after the actions chip. */
  children?: ReactNode;
}) {
  return (
    <div
      data-rally-row={rally.rally_id}
      onClick={onSelect}
      className={cn(
        'flex cursor-pointer flex-wrap items-center gap-x-2.5 gap-y-1 rounded-xl border px-3 py-2.5 transition-colors',
        selected ? 'border-primary/40 bg-primary/[0.1]' : 'border-primary/15 bg-primary/[0.04] hover:bg-primary/[0.08]',
        live && 'ring-1 ring-accent/50',
      )}
    >
      <span className="w-4 shrink-0 select-none text-right font-heading text-[10px] text-text-muted/60">{index + 1}</span>
      <span
        className="w-7 shrink-0 select-none font-mono text-[9px] text-text-muted/40"
        title={`rally_id ${rally.rally_id} — stable id, not the time order`}
      >
        #{rally.rally_id}
      </span>
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          onToggle();
        }}
        className="flex shrink-0 items-center gap-1 whitespace-nowrap rounded-full bg-primary/20 px-2 py-0.5 text-[11px] font-medium text-primary-text ring-1 ring-primary/25"
      >
        <span className={cn('transition-transform', open && 'rotate-90')}>▸</span> actions <span className="opacity-70">{count}</span>
      </button>
      {children}
      {/* One unit, so a crowded row moves the time to its own line instead
          of breaking it mid-range. */}
      <span className="ml-auto flex shrink-0 items-center gap-2.5">
        <span className="whitespace-nowrap font-mono text-[11px] tabular-nums text-text-muted">
          {formatActionTime(rally.start)} → {formatActionTime(rally.end)}
        </span>
        <span className="rounded bg-surface-200/40 px-1.5 py-0.5 font-mono text-[10px] tabular-nums text-text-muted">
          {Math.max(0, rally.end - rally.start).toFixed(1)}s
        </span>
      </span>
    </div>
  );
}

/** The group of actions that fall in no rally. */
export function OutsideRow({
  rowKey,
  count,
  open,
  onToggle,
  children,
}: {
  /** Its `data-rally-row`, for scrolling to it. */
  rowKey: string;
  count: number;
  open: boolean;
  onToggle: () => void;
  /** The page's own chips, after the outside chip. */
  children?: ReactNode;
}) {
  return (
    <div
      data-rally-row={rowKey}
      onClick={onToggle}
      className="flex cursor-pointer items-center gap-2.5 rounded-xl border border-amber-500/20 bg-amber-500/[0.04] px-3 py-2.5 hover:bg-amber-500/[0.08]"
    >
      <span className="w-4 select-none text-right font-heading text-[10px] text-text-muted/60">out</span>
      <span className="w-7 select-none" />
      <span className="flex items-center gap-1 rounded-full bg-amber-500/15 px-2 py-0.5 text-[11px] font-medium text-amber-300 ring-1 ring-amber-500/25">
        <span className={cn('transition-transform', open && 'rotate-90')}>▸</span> outside <span className="opacity-70">{count}</span>
      </span>
      {children}
      <span className="ml-auto font-heading text-[11px] text-text-muted">outside rally</span>
    </div>
  );
}

/** An expanded group's action rows: the running number, then the page's own
 *  cells laid out on `columns` (a grid-cols template for everything after
 *  the number). */
export function EventRows<T extends { id: string }>({
  entries,
  empty,
  columns,
  selectedId,
  isActive,
  onJump,
  rowClassName,
  children,
}: {
  entries: T[];
  empty: string;
  columns: string;
  selectedId: string | null;
  /** Near the playhead — the row lights up. */
  isActive: (entry: T) => boolean;
  onJump: (entry: T) => void;
  rowClassName?: (entry: T) => string | false | undefined;
  children: (entry: T) => ReactNode;
}) {
  if (!entries.length) {
    return (
      <div className="ml-6 rounded-xl border border-border bg-surface-100 px-3 py-2 text-xs text-text-muted">
        {empty}
      </div>
    );
  }
  return (
    <div className="ml-6 space-y-1.5 rounded-xl border border-border bg-surface-100 p-2">
      {entries.map((e, row) => (
        <div
          key={e.id}
          data-action-id={e.id}
          onClick={() => onJump(e)}
          style={{ gridTemplateColumns: `1rem ${columns}` }}
          className={cn(
            'grid cursor-pointer items-center gap-1.5 rounded-lg border px-2 py-1.5 transition-colors',
            e.id === selectedId
              ? 'border-primary/35 bg-primary/10'
              : 'border-border bg-surface-50 hover:bg-surface-200/40',
            isActive(e) && 'ring-1 ring-accent/50',
            rowClassName?.(e),
          )}
        >
          <span className="text-right font-heading text-[10px] text-text-muted/70">{row + 1}</span>
          {children(e)}
        </div>
      ))}
    </div>
  );
}

/** The action's colour dot — hollow when the ball is not visible. A button
 *  when `onToggle` is given. */
export function ActionDot({
  label,
  visible,
  onToggle,
}: {
  label: string | null | undefined;
  visible: boolean;
  onToggle?: () => void;
}) {
  const color = actionColor(label);
  const props = {
    className: cn('h-2.5 w-2.5 flex-shrink-0 rounded-full', !visible && 'border'),
    style: visible ? { background: color } : { borderColor: color },
  };
  if (!onToggle) return <span {...props} title={visible ? undefined : 'Non-visible action'} />;
  return (
    <button
      type="button"
      onClick={onToggle}
      title={visible ? 'Visible — click to hide' : 'Non-visible — click to show'}
      {...props}
    />
  );
}
