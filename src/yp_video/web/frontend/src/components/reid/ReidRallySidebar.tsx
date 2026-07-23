/** The ReID player's rally list — same interaction as the Rally Label /
 *  Action Label sidebars: rallies expand into their action rows, clicking a
 *  row parks the video there.
 *
 *  Split out of ReidVideoPlayer and memoized for one reason: the player's
 *  frame clock sets state on EVERY presented frame (30–60/s), and this list
 *  is its largest subtree (~40 rally rows + an expanded action panel, several
 *  hundred elements on a real video). Almost none of it is frame-dependent —
 *  the rally highlight changes once per rally, the row highlight once per
 *  action. So the frame-derived inputs arrive pre-reduced (``activeRallyId``,
 *  ``activeActionIds``) and this component re-renders when they actually
 *  change, not 60 times a second.
 *
 *  That memo only holds while every prop keeps a stable identity — callers
 *  must useCallback the handlers and keep ``activeActionIds`` identity-stable
 *  across frames where the membership doesn't move.
 */

import { memo, type RefObject } from 'react';
import { cn } from '@/lib/cn';
import { actionColor } from '@/lib/actionColors';
import { Card } from '@/components/ui/Card';
import { SectionLabel } from '@/components/ui/SectionLabel';
import type { ReidPlayers } from '@/types/api';
import { fmtTime, type Rally, type SidebarAction } from './shared';

/** Sentinel key for the "outside any rally" section. */
export const OUTSIDE = '__outside__';

interface ReidEventPanelProps {
  entries: SidebarAction[];
  empty: string;
  matches: ReidPlayers['matches'];
  /** Events marked occluded (no player to identify) — shown as a verdict pill. */
  occludedIds: Set<string>;
  selectedEventId: string | null;
  fps: number;
  /** Actions within ±½ s of the playhead — those rows light up. */
  activeActionIds: Set<string>;
  onJump: (a: SidebarAction) => void;
  onJumpToCrop: (eventId: string) => void;
}

/** Read-only twin of the Action Label event panel: action dot + label,
 *  matched player, frame and time — click a row to park the video there. */
function ReidEventPanel({ entries, empty, matches, occludedIds, selectedEventId, fps, activeActionIds, onJump, onJumpToCrop }: ReidEventPanelProps) {
  if (!entries.length) return <div className="ml-6 rounded-xl border border-border bg-surface-100 px-3 py-2 text-xs text-text-muted">{empty}</div>;
  return (
    <div className="ml-6 space-y-1.5 rounded-xl border border-border bg-surface-100 p-2">
      {entries.map((a, row) => {
        const m = matches[a.id];
        const color = actionColor(a.label);
        const active = activeActionIds.has(a.id);
        return (
          <div
            key={a.id}
            data-action-id={a.id}
            onClick={() => onJump(a)}
            title="Click to jump the video to this action"
            className={cn(
              'grid cursor-pointer grid-cols-[1rem_minmax(4.5rem,1fr)_minmax(3rem,6.5rem)_3.6rem_2.6rem] items-center gap-1.5 rounded-lg border px-2 py-1.5 transition-colors',
              a.id === selectedEventId ? 'border-primary/35 bg-primary/10' : 'border-border bg-surface-50 hover:bg-surface-200/40',
              active && 'ring-1 ring-accent/50',
            )}
          >
            <span className="text-right font-heading text-[10px] text-text-muted/70">{row + 1}</span>
            <span className="flex min-w-0 items-center gap-1.5">
              <span
                className={cn('h-2.5 w-2.5 flex-shrink-0 rounded-full', !a.visible && 'border')}
                style={a.visible ? { background: color } : { borderColor: color }}
                title={a.visible ? undefined : 'Non-visible action'}
              />
              <span className="truncate text-xs text-text-primary">{a.label ?? '—'}</span>
            </span>
            {m?.player ? (
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  onJumpToCrop(a.id);
                }}
                title={
                  m.assigned
                    ? "Jump to this event's crop in the identities board below"
                    : `Not assigned — nearest match is ${m.player}. Click to jump to the crop.`
                }
                className={cn(
                  'max-w-full justify-self-end truncate rounded-full px-2 py-0.5 text-[11px] ring-1 transition-colors',
                  m.assigned
                    ? 'bg-primary/15 text-primary-light ring-primary/30 hover:bg-primary/30'
                    : 'bg-surface-200/40 text-text-muted ring-border hover:bg-surface-200/80 hover:text-text-secondary',
                )}
              >
                {m.assigned ? m.player : `~${m.player}`}
              </button>
            ) : occludedIds.has(a.id) ? (
              <span
                className="max-w-full justify-self-end truncate rounded-full bg-surface-200/40 px-2 py-0.5 text-[11px] italic text-text-muted ring-1 ring-border"
                title="Marked occluded — no player to identify"
              >
                Occluded
              </span>
            ) : (
              <span />
            )}
            <span className="text-center font-heading text-[11px] tabular-nums text-text-primary">f{a.frame}</span>
            <span className="text-center font-heading text-[10px] tabular-nums text-text-muted">{fmtTime(a.time != null ? a.time : a.frame / (fps || 30))}</span>
          </div>
        );
      })}
    </div>
  );
}

export interface ReidRallySidebarProps {
  rallies: Rally[];
  byRally: Map<number, SidebarAction[]>;
  outside: SidebarAction[];
  totalActions: number;
  fps: number;
  matches: ReidPlayers['matches'];
  occludedIds: Set<string>;
  /** The rally under the playhead — the list's only frame-derived scalar. */
  activeRallyId: number | null;
  /** Actions within ±½ s of the playhead; identity-stable between changes. */
  activeActionIds: Set<string>;
  expanded: string | null;
  selectedRally: number | 'all';
  selectedEventId: string | null;
  /** The scroll container — the player pins/centres rows through it. */
  listRef: RefObject<HTMLDivElement>;
  onSelectAll: () => void;
  onJumpRally: (rally: Rally) => void;
  onSetExpanded: (key: string | null) => void;
  onJumpEvent: (a: SidebarAction) => void;
  onJumpToCrop: (eventId: string) => void;
}

export const ReidRallySidebar = memo(function ReidRallySidebar({
  rallies, byRally, outside, totalActions, fps, matches, occludedIds,
  activeRallyId, activeActionIds, expanded, selectedRally, selectedEventId,
  listRef, onSelectAll, onJumpRally, onSetExpanded, onJumpEvent, onJumpToCrop,
}: ReidRallySidebarProps) {
  return (
    <div className="min-w-0 lg:w-[420px] lg:flex-shrink-0">
      <Card>
        <SectionLabel>
          Rallies ({rallies.length} rally · {totalActions} action)
        </SectionLabel>
        <div ref={listRef} className="vq-list max-h-[calc(45vh+2.25rem)] space-y-1.5 overflow-y-auto pr-1">
          <div
            onClick={onSelectAll}
            className={cn(
              'ae-row flex cursor-pointer items-center gap-1.5 rounded-xl border px-3 py-2.5 transition-colors',
              selectedRally === 'all'
                ? 'border-primary/45 bg-primary/[0.12]'
                : 'border-primary/20 bg-primary/[0.05] hover:bg-primary/[0.10]',
            )}
          >
            <span className="text-xs font-medium text-text-primary">All rallies</span>
            <span className="ml-auto font-mono text-[10px] tabular-nums text-text-muted">{totalActions} action</span>
          </div>
          {rallies.map((rally, i) => {
            const entries = byRally.get(rally.rally_id) ?? [];
            const isOpen = expanded === String(rally.rally_id);
            const active = rally.rally_id === activeRallyId;
            const selected = selectedRally === rally.rally_id;
            return (
              <div key={rally.rally_id} className="space-y-1.5">
                <div
                  data-rally-row={rally.rally_id}
                  onClick={() => onJumpRally(rally)}
                  className={cn(
                    'ae-row flex cursor-pointer items-center gap-2.5 rounded-xl border px-3 py-2.5 transition-colors',
                    selected ? 'border-primary/45 bg-primary/[0.12]' : 'border-primary/20 bg-primary/[0.05] hover:bg-primary/[0.10]',
                    active && 'ring-1 ring-accent/50',
                  )}
                >
                  <span className="w-4 select-none text-right font-heading text-[10px] text-text-muted/60">{i + 1}</span>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      // Collapse if open; otherwise select + expand + seek to the rally start.
                      if (isOpen) onSetExpanded(null);
                      else onJumpRally(rally);
                    }}
                    className="flex items-center gap-1 rounded-full bg-primary/20 px-2 py-0.5 text-[11px] font-medium text-primary-text ring-1 ring-primary/25"
                  >
                    <span className={cn('transition-transform', isOpen && 'rotate-90')}>▸</span> actions <span className="opacity-70">{entries.length}</span>
                  </button>
                  <span className="ml-auto font-mono text-[11px] tabular-nums text-text-muted">
                    {fmtTime(rally.start)} → {fmtTime(rally.end)}
                  </span>
                  <span className="rounded bg-surface-200/40 px-1.5 py-0.5 font-mono text-[10px] tabular-nums text-text-muted">
                    {Math.max(0, rally.end - rally.start).toFixed(1)}s
                  </span>
                </div>
                {isOpen && (
                  <ReidEventPanel
                    entries={entries}
                    empty="No actions in this rally"
                    matches={matches}
                    occludedIds={occludedIds}
                    selectedEventId={selectedEventId}
                    fps={fps}
                    activeActionIds={activeActionIds}
                    onJump={onJumpEvent}
                    onJumpToCrop={onJumpToCrop}
                  />
                )}
              </div>
            );
          })}
          {outside.length > 0 && (
            <div className="space-y-1.5">
              <div
                data-rally-row={OUTSIDE}
                onClick={() => onSetExpanded(OUTSIDE)}
                className="flex cursor-pointer items-center gap-2.5 rounded-xl border border-amber-500/20 bg-amber-500/[0.04] px-3 py-2.5 hover:bg-amber-500/[0.08]"
              >
                <span className="w-4 select-none text-right font-heading text-[10px] text-text-muted/60">out</span>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    onSetExpanded(expanded === OUTSIDE ? null : OUTSIDE);
                  }}
                  className="flex items-center gap-1 rounded-full bg-amber-500/15 px-2 py-0.5 text-[11px] font-medium text-amber-300 ring-1 ring-amber-500/25"
                >
                  <span className={cn('transition-transform', expanded === OUTSIDE && 'rotate-90')}>▸</span> outside <span className="opacity-70">{outside.length}</span>
                </button>
                <span className="ml-auto font-heading text-[11px] text-text-muted">outside rally</span>
              </div>
              {expanded === OUTSIDE && (
                <ReidEventPanel
                  entries={outside}
                  empty="No outside actions"
                  matches={matches}
                  occludedIds={occludedIds}
                  selectedEventId={selectedEventId}
                  fps={fps}
                  activeActionIds={activeActionIds}
                  onJump={onJumpEvent}
                  onJumpToCrop={onJumpToCrop}
                />
              )}
            </div>
          )}
        </div>
      </Card>
    </div>
  );
});
