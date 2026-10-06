import { useState } from 'react';
import { formatActionTime } from '@/lib/actionEditorModel';
import { cn } from '@/lib/cn';
import { COURT_SIDES, SIDE_DISPLAY } from '@/lib/courtSide';
import type { ActionAttributeDefaults, ActionEvent, CourtSide } from '@/types/api';
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

const JUMP_DISPLAY = { true: '跳', false: '站' } as const;
/** Hand-set jump cycles unset (the default applies) → jumped → grounded. */
const nextJump = (jump: boolean | undefined) => (jump === undefined ? true : jump ? false : undefined);

/** The actor's court side and whether they jumped. A hand-set value shows
 *  bright; otherwise the muted value is what training assumes. */
function AttributeCell({
  event,
  defaults,
  onEdit,
}: {
  event: ActionEvent;
  defaults: ActionAttributeDefaults | undefined;
  onEdit: (patch: Partial<ActionEvent>) => void;
}) {
  const derivedSide = defaults?.side;
  const jump = event.jump ?? defaults?.jump ?? null;
  return (
    <span className="flex items-center gap-1" onClick={(e) => e.stopPropagation()}>
      <select
        value={event.side ?? ''}
        onChange={(e) => onEdit({ side: (e.target.value || undefined) as CourtSide | undefined })}
        title="Actor 在哪一側 — 「自」= 交給規則推導"
        className={cn(
          'w-full min-w-0 rounded-lg border border-border bg-surface-100 px-0.5 py-1 text-xs',
          event.side ? 'text-text-primary' : 'text-text-muted',
        )}
      >
        <option value="">{derivedSide ? `自${SIDE_DISPLAY[derivedSide]}` : '自'}</option>
        {COURT_SIDES.map((side) => (
          <option key={side} value={side}>
            {SIDE_DISPLAY[side]}
          </option>
        ))}
      </select>
      <button
        type="button"
        onClick={() => onEdit({ jump: nextJump(event.jump) })}
        title="有沒有跳 — 點擊切換：預設 → 跳 → 站"
        className={cn(
          'w-5 shrink-0 text-xs',
          event.jump === undefined ? 'text-text-muted' : 'font-bold text-primary-light',
        )}
      >
        {jump === null ? '–' : JUMP_DISPLAY[`${jump}`]}
      </button>
    </span>
  );
}

interface ActionEventPanelProps {
  entries: ActionEvent[];
  empty: string;
  labels: string[];
  /** Per event id: what training assumes when nothing is stored. */
  attributeDefaults: Record<string, ActionAttributeDefaults>;
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
  attributeDefaults,
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
      columns="minmax(5rem,1fr) 3.6rem 3.6rem 2.6rem 2.4rem"
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
          <AttributeCell
            event={e}
            defaults={attributeDefaults[e.id]}
            onEdit={(patch) => onEdit(e.id, patch)}
          />
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
