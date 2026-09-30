import { useState } from 'react';
import { formatActionTime } from '@/lib/actionEditorModel';
import type { ActionEvent } from '@/types/api';
import { ActionDot, EventRows } from './RallyRows';

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
  const windowFrames = Math.max(1, Math.round((fps || 30) / 2));
  return (
    <EventRows
      entries={entries}
      empty={empty}
      columns="minmax(5rem,1fr) 3.6rem 2.6rem 2.4rem"
      selectedId={selectedId}
      isActive={(e) => Math.abs(e.frame - frame) <= windowFrames}
      onJump={(e) => onJump(e.id)}
    >
      {(e) => (
        <>
          <span
            className="flex min-w-0 items-center gap-1.5"
            onClick={(event) => event.stopPropagation()}
          >
            <ActionDot
              label={e.label}
              visible={e.visible}
              onToggle={() => onEdit(e.id, { visible: !e.visible })}
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
        </>
      )}
    </EventRows>
  );
}
