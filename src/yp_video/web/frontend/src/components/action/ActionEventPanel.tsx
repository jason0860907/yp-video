import { useState } from 'react';
import { cn } from '@/lib/cn';
import { actionColor } from '@/lib/actionColors';
import { formatActionTime } from '@/lib/actionEditorModel';
import type { ActionEvent } from '@/types/api';

/** Inline frame editor. Typing only moves a local draft — Enter or leaving the
 *  cell applies it, Escape reverts.
 *
 *  Committing per keystroke cannot work here: every commit re-derives the
 *  event's rally_id from the new frame, so a half-typed number re-homes the
 *  event to another rally, unmounting the very row being typed in. */
function FrameCell({ frame, onCommit }: { frame: number; onCommit: (frame: number) => void }) {
  const [draft, setDraft] = useState<string | null>(null);
  return (
    <input
      value={draft ?? String(frame)}
      onClick={(event) => event.stopPropagation()}
      onFocus={(event) => setDraft(event.target.value)}
      onChange={(event) => setDraft(event.target.value)}
      onBlur={() => {
        const parsed = draft !== null && draft.trim() !== '' ? Number(draft) : NaN;
        const next = Math.max(0, Math.round(parsed));
        // Tabbing through a cell, or typing the value back to what it was,
        // must not mark the editor dirty and re-sort the list for nothing.
        if (Number.isFinite(parsed) && next !== frame) onCommit(next);
        setDraft(null);
      }}
      onKeyDown={(event) => {
        // Escape deliberately does NOT blur: setDraft is async, so blurring in
        // the same tick would let onBlur read the stale draft and commit it
        // anyway. Clearing the draft alone puts the true frame back on screen,
        // and the later blur becomes a no-op.
        if (event.key === 'Enter') event.currentTarget.blur();
        else if (event.key === 'Escape') setDraft(null);
      }}
      title="frame 號碼 — Enter 或離開欄位才套用，Esc 還原"
      className="w-full border-0 border-b border-white/10 bg-transparent text-center font-heading text-[11px] tabular-nums text-text-primary focus:border-primary-light focus:outline-none"
    />
  );
}

interface ActionEventPanelProps {
  entries: ActionEvent[];
  empty: string;
  labels: string[];
  selectedId: string | null;
  fps: number;
  /** Current playhead frame — rows within ±½ s light up. */
  frame: number;
  onEdit: (id: string, patch: Partial<ActionEvent>) => void;
  onDelete: (id: string) => void;
  onJump: (id: string) => void;
}

export function ActionEventPanel({
  entries,
  empty,
  labels,
  selectedId,
  fps,
  frame,
  onEdit,
  onDelete,
  onJump,
}: ActionEventPanelProps) {
  if (!entries.length) {
    return (
      <div className="ml-6 rounded-xl border border-border bg-surface-100 px-3 py-2 text-xs text-text-muted">
        {empty}
      </div>
    );
  }
  const windowFrames = Math.max(1, Math.round((fps || 30) / 2));
  return (
    <div className="ml-6 space-y-1.5 rounded-xl border border-border bg-surface-100 p-2">
      {entries.map((e, row) => {
        const color = actionColor(e.label);
        const active = Math.abs(e.frame - frame) <= windowFrames;
        return (
          <div
            key={e.id}
            onClick={() => onJump(e.id)}
            className={cn(
              'grid cursor-pointer grid-cols-[1rem_minmax(5rem,1fr)_3.6rem_2.6rem_2.4rem] items-center gap-1.5 rounded-lg border px-2 py-1.5 transition-colors',
              e.id === selectedId
                ? 'border-primary/35 bg-primary/10'
                : 'border-border bg-surface-50 hover:bg-surface-200/40',
              active && 'ring-1 ring-accent/50',
            )}
          >
            <span className="text-right font-heading text-[10px] text-text-muted/70">
              {row + 1}
            </span>
            <span
              className="flex min-w-0 items-center gap-1.5"
              onClick={(event) => event.stopPropagation()}
            >
              <button
                type="button"
                onClick={() => onEdit(e.id, { visible: !e.visible })}
                className={cn(
                  'h-2.5 w-2.5 flex-shrink-0 rounded-full',
                  !e.visible && 'border',
                )}
                style={e.visible ? { background: color } : { borderColor: color }}
                title={
                  e.visible
                    ? 'Visible — click to hide'
                    : 'Non-visible — click to show'
                }
              />
              <select
                value={e.label}
                onChange={(event) => onEdit(e.id, { label: event.target.value })}
                className="w-full min-w-0 rounded-lg border border-border bg-surface-100 px-1.5 py-1 text-xs text-text-primary"
              >
                {labels.map((label) => (
                  <option key={label} value={label}>
                    {label}
                  </option>
                ))}
              </select>
            </span>
            <FrameCell frame={e.frame} onCommit={(f) => onEdit(e.id, { frame: f })} />
            <span className="text-center font-heading text-[10px] tabular-nums text-text-muted">
              {formatActionTime(e.frame / (fps || 30))}
            </span>
            <span
              className="flex items-center justify-end gap-1"
              onClick={(event) => event.stopPropagation()}
            >
              <button
                type="button"
                onClick={() => onJump(e.id)}
                className="text-primary-light hover:text-text-primary"
                title="Jump to event"
              >
                →
              </button>
              <button
                type="button"
                onClick={() => onDelete(e.id)}
                className="text-red-400/60 hover:text-red-400"
                title="Delete"
              >
                ✕
              </button>
            </span>
          </div>
        );
      })}
    </div>
  );
}
