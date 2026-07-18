/** The identities board: group rows, the locked-groups dock, crop tiles with
 *  drag/multi-select/marquee, and the video↔board jump target. All state here
 *  is view-only (selection, drag hover, flashes) — every actual edit goes
 *  through the useGroupBoard actions passed in via ``board``. */

import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
  type DragEvent as ReactDragEvent,
  type MouseEvent as ReactMouseEvent,
  type PointerEvent as ReactPointerEvent,
} from 'react';
import { API, apiUrl } from '@/lib/api';
import { cn } from '@/lib/cn';
import { actionColor } from '@/lib/actionColors';
import { Badge } from '@/components/ui/Badge';
import { CropImage } from '@/components/video/CropImage';
import { toast } from '@/components/feedback/toast';
import type { ReidRecord } from '@/types/api';
import { trackColor, trackKeyOf, type TrackData } from './shared';
import { MIN_CLUSTER_SIZE, type useGroupBoard } from './useGroupBoard';

const STATUS_DOT: Record<ReidRecord['status'], string> = {
  ok: 'bg-primary-light',
  multi: 'bg-amber-400',
  miss: 'bg-red-400',
};

export interface BoardHandle {
  /** Scroll the event's crop into view and pulse it; when the tile isn't
   *  rendered (docked groups show only 3 tiles, the status filter hides
   *  some) the group card holding it pulses instead. */
  jumpToCrop: (eventId: string) => void;
}

export interface GroupBoardProps {
  picked: string;
  records: ReidRecord[];
  recordById: Map<string, ReidRecord>;
  board: ReturnType<typeof useGroupBoard>;
  /** Where locked groups live: pinned on top as full rows, or docked in a
   *  sticky right rail showing just 3 crops per group. */
  lockedDock: 'top' | 'right';
  statusFilter: 'all' | ReidRecord['status'];
  showSkeleton: boolean;
  /** Show the background-suppressed crops the masked embedders saw. */
  showMasked: boolean;
  trackLinks: TrackData['links'];
  onSeekToEvent: (r: ReidRecord) => void;
}

/** Name field that keeps edits local until committed (Enter or blur).
 *  Committing renames + locks the group, which can relocate the row (dock,
 *  autosave rebuild) and unmount the input — so it must not run per
 *  keystroke. Esc reverts. */
function NameInput({ value, onCommit, placeholder, className }: { value: string; onCommit: (name: string) => void; placeholder?: string; className?: string }) {
  const [draft, setDraft] = useState(value);
  useEffect(() => setDraft(value), [value]);
  return (
    <input
      value={draft}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={() => {
        if (draft !== value) onCommit(draft);
      }}
      onKeyDown={(e) => {
        e.stopPropagation(); // keep Space/Esc from the page-level shortcuts
        if (e.key === 'Enter') e.currentTarget.blur();
        else if (e.key === 'Escape') setDraft(value);
      }}
      placeholder={placeholder}
      className={className}
    />
  );
}

export const GroupBoard = forwardRef<BoardHandle, GroupBoardProps>(function GroupBoard(
  { picked, records, recordById, board, lockedDock, statusFilter, showSkeleton, showMasked, trackLinks, onSeekToEvent },
  ref,
) {
  const { groups } = board;
  // What's under the cursor during a drag. mode: 'merge' drops INTO the row,
  // 'before'/'after' reorder around it (group drags only, via edge bands).
  const [dragOver, setDragOver] = useState<{ key: string; mode: 'merge' | 'before' | 'after' } | null>(null);
  // Payload kind of the in-flight drag (dataTransfer is unreadable during
  // dragover, so remember it at dragstart).
  const dragKind = useRef<'group' | 'events' | null>(null);
  // Multi-select for bulk drag: Ctrl/Cmd/Shift-click crops to toggle, plain
  // click still seeks the video. Esc clears.
  const [selectedCrops, setSelectedCrops] = useState<Set<string>>(new Set());
  // Hovered crop's tracklet — same-track crops light up across the board.
  const [hoverTrack, setHoverTrack] = useState<string | null>(null);
  // Right-click with a selection → tiny context menu (new group / clear).
  const [ctxMenu, setCtxMenu] = useState<{ x: number; y: number } | null>(null);

  // Switching video invalidates every view-only state at once.
  useEffect(() => {
    setSelectedCrops(new Set());
    setCtxMenu(null);
    setDragOver(null);
    setHoverTrack(null);
  }, [picked]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setSelectedCrops(new Set());
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, []);

  /** User-driven moves clear the selection (it just went somewhere). */
  const moveSelection = (eventIds: string[], toKey: string) => {
    board.moveEvents(eventIds, toKey);
    setSelectedCrops(new Set());
  };

  // Reverse jump (video → board): scroll the event's crop into view and pulse
  // it; fall back to the group card when the tile isn't rendered.
  const [flashCrop, setFlashCrop] = useState<string | null>(null);
  const [flashGroup, setFlashGroup] = useState<string | null>(null);
  const flashTimer = useRef(0);
  useEffect(() => () => window.clearTimeout(flashTimer.current), []);
  const pulse = (el: Element, kind: 'crop' | 'group', id: string) => {
    el.scrollIntoView({ block: 'center', behavior: 'smooth' });
    window.clearTimeout(flashTimer.current);
    setFlashCrop(kind === 'crop' ? id : null);
    setFlashGroup(kind === 'group' ? id : null);
    flashTimer.current = window.setTimeout(() => {
      setFlashCrop(null);
      setFlashGroup(null);
    }, 1600);
  };
  useImperativeHandle(ref, () => ({
    jumpToCrop: (eventId: string) => {
      const record = recordById.get(eventId);
      if (!record?.crop) {
        toast.warning('This event has no crop (score / non-visible action)');
        return;
      }
      const tile = document.querySelector(`[data-event-id="${CSS.escape(eventId)}"]`);
      if (tile) {
        pulse(tile, 'crop', eventId);
        return;
      }
      const group = groups.find((g) => g.eventIds.includes(eventId));
      const card = group && document.querySelector(`[data-group-key="${CSS.escape(group.key)}"]`);
      if (card) {
        pulse(card, 'group', group.key);
        return;
      }
      toast.warning(
        statusFilter !== 'all' && record.status !== statusFilter
          ? `Crop is hidden by the status filter (its status is "${record.status}")`
          : 'Crop is not on the board yet',
      );
    },
  }));

  // ── Marquee (rubber-band) selection over the group board ──
  // Starts on empty board space only; crops keep their native drag. Ctrl/⌘
  // makes it additive. Viewport coordinates throughout, so scrolling during
  // the drag stays correct.
  const boardRef = useRef<HTMLDivElement>(null);
  const [marquee, setMarquee] = useState<{ x0: number; y0: number; x1: number; y1: number } | null>(null);
  // Unmounting mid-drag (video switch, filter change) would otherwise leave
  // the document listeners attached to a dead component.
  const marqueeAbort = useRef<AbortController | null>(null);
  useEffect(() => () => marqueeAbort.current?.abort(), []);
  const startMarquee = (e: ReactPointerEvent) => {
    if (e.button !== 0) return;
    const target = e.target as HTMLElement;
    if (target.closest('[data-event-id],button,input,select,[draggable="true"]')) return;
    e.preventDefault(); // stop text selection
    const sx = e.clientX;
    const sy = e.clientY;
    const base = e.ctrlKey || e.metaKey || e.shiftKey ? new Set(selectedCrops) : new Set<string>();
    let active = false;
    const ac = new AbortController();
    marqueeAbort.current = ac;
    const onMove = (ev: PointerEvent) => {
      if (!active && Math.hypot(ev.clientX - sx, ev.clientY - sy) < 5) return;
      active = true;
      const rect = {
        x0: Math.min(sx, ev.clientX),
        y0: Math.min(sy, ev.clientY),
        x1: Math.max(sx, ev.clientX),
        y1: Math.max(sy, ev.clientY),
      };
      setMarquee(rect);
      const hits = new Set(base);
      boardRef.current?.querySelectorAll('[data-event-id]').forEach((el) => {
        const b = el.getBoundingClientRect();
        if (b.left < rect.x1 && b.right > rect.x0 && b.top < rect.y1 && b.bottom > rect.y0) {
          hits.add((el as HTMLElement).dataset.eventId!);
        }
      });
      setSelectedCrops(hits);
    };
    const onUp = () => {
      ac.abort();
      marqueeAbort.current = null;
      setMarquee(null);
      if (!active) setSelectedCrops(base); // plain background click clears (or keeps ctrl-base)
    };
    document.addEventListener('pointermove', onMove, { signal: ac.signal });
    document.addEventListener('pointerup', onUp, { signal: ac.signal });
    document.addEventListener('pointercancel', onUp, { signal: ac.signal });
  };

  /** Drag ghost for multi-crop drags: a fanned stack of thumbnails + count. */
  const setMultiDragImage = (e: ReactDragEvent, ids: string[]) => {
    if (ids.length < 2) return;
    const ghost = document.createElement('div');
    ghost.style.cssText = 'position:fixed;top:-600px;left:-600px;display:flex;align-items:center;padding:8px;';
    ids.slice(0, 4).forEach((eid, i) => {
      const img = boardRef.current?.querySelector(`[data-event-id="${CSS.escape(eid)}"] img`) as HTMLImageElement | null;
      if (!img) return;
      const c = img.cloneNode() as HTMLImageElement;
      c.style.cssText = `height:76px;width:auto;border-radius:6px;border:2px solid #fff;box-shadow:0 2px 10px rgba(0,0,0,.55);margin-left:${i ? '-44px' : '0'};transform:rotate(${(i - 1.5) * 5}deg);background:#000`;
      ghost.appendChild(c);
    });
    const badge = document.createElement('div');
    badge.textContent = String(ids.length);
    badge.style.cssText =
      'position:relative;z-index:1;margin-left:-16px;min-width:24px;height:24px;padding:0 7px;border-radius:999px;background:#e8b93c;color:#111;display:flex;align-items:center;justify-content:center;font:700 12px/1 ui-sans-serif,system-ui;box-shadow:0 1px 5px rgba(0,0,0,.45)';
    ghost.appendChild(badge);
    document.body.appendChild(ghost);
    e.dataTransfer.setDragImage(ghost, 46, 44);
    setTimeout(() => ghost.remove(), 0);
  };

  const boardContextMenu = (e: ReactMouseEvent) => {
    if (!selectedCrops.size) return; // native menu when nothing is selected
    e.preventDefault();
    setCtxMenu({ x: e.clientX, y: e.clientY });
  };
  const selectionToNewGroup = () => {
    const ids = [...selectedCrops];
    moveSelection(ids, board.newGroupBelow(ids[0]));
    setCtxMenu(null);
  };

  const onDropTo = (toKey: string) => (e: ReactDragEvent) => {
    e.preventDefault();
    const mode = dragOver?.key === toKey ? dragOver.mode : 'merge';
    setDragOver(null);
    dragKind.current = null;
    const parts = e.dataTransfer.getData('text/plain').split('\n');
    if (parts[0] === 'group') {
      const fromKey = parts[1];
      if (!fromKey || toKey === '__new__') return;
      if (mode === 'merge') board.mergeGroups(fromKey, toKey);
      else board.reorderGroup(fromKey, toKey, mode);
      return;
    }
    const [kind, idList] = parts;
    const eventIds = (idList ?? '').split(',').filter(Boolean);
    if (kind !== 'events' || !eventIds.length) return;
    moveSelection(eventIds, toKey === '__new__' ? board.newGroupBelow(eventIds[0]) : toKey);
  };

  /** Shared drag/select/jump behavior for a crop tile (image or placeholder). */
  const cropTileProps = (id: string, r: ReidRecord) => ({
    draggable: true,
    onDragStart: (e: ReactDragEvent<HTMLDivElement>) => {
      // Dragging a selected crop carries the whole selection; an unselected
      // one moves alone.
      dragKind.current = 'events';
      const ids = selectedCrops.has(id) ? [...selectedCrops] : [id];
      e.dataTransfer.setData('text/plain', `events\n${ids.join(',')}`);
      setMultiDragImage(e, ids);
    },
    onClick: (e: ReactMouseEvent<HTMLDivElement>) => {
      // Plain click = this crop only; ⌘/Ctrl adds to the selection. (A
      // double-click fires this twice — idempotent either way, then jumps.)
      setSelectedCrops((prev) => {
        if (!(e.ctrlKey || e.metaKey || e.shiftKey)) return new Set([id]);
        const next = new Set(prev);
        if (next.has(id)) next.delete(id);
        else next.add(id);
        return next;
      });
    },
    onDoubleClick: () => onSeekToEvent(r),
    // Hovering a tracked crop lights up its whole tracklet across the board.
    onMouseEnter: () => setHoverTrack(trackKeyOf(trackLinks, id)),
    onMouseLeave: () => setHoverTrack(null),
  });

  const tileSelectCls = (id: string) =>
    cn(
      selectedCrops.has(id) ? 'border-accent ring-2 ring-accent/70' : 'border-border',
      flashCrop === id && 'animate-pulse border-accent ring-2 ring-accent',
    );

  /** One draggable/selectable crop thumbnail — shared by the board rows and
   *  the locked-groups dock (which renders them smaller). Events without a
   *  crop (miss / no-actor) render as a placeholder tile instead. */
  const renderCrop = (id: string, heightCls = 'h-28') => {
    const r = recordById.get(id);
    if (!r) return null;
    const trackKey = trackKeyOf(trackLinks, id);
    const sameTrackHover = trackKey != null && hoverTrack === trackKey;
    if (!r.crop) {
      return (
        <div
          key={id}
          data-event-id={id}
          {...cropTileProps(id, r)}
          title={`${r.label} f${r.frame} — no crop (${r.status}): double-click to jump there, then Pick actor to fix`}
          className={cn(
            heightCls,
            'flex aspect-[1/2] cursor-grab flex-col items-center justify-center gap-1 rounded-md border border-dashed bg-surface-100 active:cursor-grabbing',
            tileSelectCls(id),
          )}
        >
          <span className={cn('h-2 w-2 rounded-full', STATUS_DOT[r.status])} />
          <span className="px-1 text-center text-[9px] leading-tight text-text-muted">
            {r.label}
            <br />f{r.frame}
          </span>
        </div>
      );
    }
    return (
      <CropImage
        key={id}
        src={apiUrl(API.reid.crop(picked, r.crop, showMasked))}
        keypoints={r.keypoints}
        skeleton={showSkeleton}
        alt={id}
        dataId={id}
        {...cropTileProps(id, r)}
        title={`${r.label} f${r.frame} — click to select, ⌘/Ctrl-click to multi-select, double-click to jump the video there`}
        className={cn(heightCls, 'w-auto cursor-grab rounded-md border active:cursor-grabbing', tileSelectCls(id))}
      >
        <span
          className={cn(
            'pointer-events-none absolute right-1 top-1 h-2 w-2 rounded-full',
            // Dot = the action's hue (same palette as the video overlay);
            // an amber ring singles out ambiguous (multi) associations.
            r.status === 'multi' ? 'ring-2 ring-amber-400' : 'ring-1 ring-black/50',
          )}
          style={{ background: actionColor(r.label) }}
          title={`${r.label} · ${r.status} · ${r.candidates} candidate(s)`}
        />
        {trackKey && (
          <span
            className="pointer-events-none absolute bottom-0.5 left-0.5 rounded px-1 font-mono text-[8px] font-bold leading-tight text-black/80"
            style={{ background: trackColor(trackKey) }}
            title={`Tracklet ${trackKey} — hover highlights every crop on it`}
          >
            t{trackLinks[id]?.track_id}
          </span>
        )}
        {sameTrackHover && (
          <span
            className="pointer-events-none absolute inset-0 rounded-md"
            style={{ boxShadow: `inset 0 0 0 2.5px ${trackColor(trackKey)}` }}
          />
        )}
      </CropImage>
    );
  };

  // Split the board by dock mode: docked-right locked groups leave the main
  // column; docked-top ones just sort first (stable within each half).
  const dockGroups = lockedDock === 'right' ? groups.filter((g) => g.locked) : [];
  const boardGroups =
    lockedDock === 'right'
      ? groups.filter((g) => !g.locked)
      : [...groups].sort((a, b) => Number(b.locked) - Number(a.locked));
  // Crop-less events (miss / no-actor) have no embedding, so clustering never
  // places them in a group — surface them in their own board section. Ones
  // dragged into a group render there instead.
  const groupedIds = new Set(groups.flatMap((g) => g.eventIds));
  const missIds = records.filter((r) => !r.crop && !groupedIds.has(r.id)).map((r) => r.id);
  // The status filter applies to both views; on the board it hides tiles but
  // keeps every row visible as a drop target.
  const statusPass = (id: string) => statusFilter === 'all' || recordById.get(id)?.status === statusFilter;

  return (
    <div className="flex items-start gap-3">
      <div ref={boardRef} onPointerDown={startMarquee} onContextMenu={boardContextMenu} className={cn('relative min-w-0 flex-1 space-y-2', marquee && 'select-none')}>
        {ctxMenu && (
          <>
            <div className="fixed inset-0 z-40" onClick={() => setCtxMenu(null)} onContextMenu={(e) => { e.preventDefault(); setCtxMenu(null); }} />
            <div className="fixed z-50 min-w-44 rounded-lg border border-border bg-surface-100 p-1 shadow-lg" style={{ left: ctxMenu.x, top: ctxMenu.y }}>
              <button
                type="button"
                onClick={selectionToNewGroup}
                className="block w-full rounded-md px-3 py-1.5 text-left text-xs text-text-primary transition-colors hover:bg-primary/15"
              >
                Move {selectedCrops.size} crop(s) to a new group
              </button>
              <button
                type="button"
                onClick={() => { setSelectedCrops(new Set()); setCtxMenu(null); }}
                className="block w-full rounded-md px-3 py-1.5 text-left text-xs text-text-muted transition-colors hover:bg-ink/[0.06]"
              >
                Clear selection
              </button>
            </div>
          </>
        )}
        {marquee && (
          <div
            className="pointer-events-none fixed z-50 rounded-sm border border-accent/70 bg-accent/10"
            style={{ left: marquee.x0, top: marquee.y0, width: marquee.x1 - marquee.x0, height: marquee.y1 - marquee.y0 }}
          />
        )}
        {boardGroups.map((g, i) => {
          // The shared pool of tiny clusters (< MIN_CLUSTER_SIZE) is
          // noise awaiting triage, not a player — render it as
          // explicitly group-less instead of "just another group".
          const isPool = g.key.startsWith('pool:');
          const shownIds = statusFilter === 'all' ? g.eventIds : g.eventIds.filter(statusPass);
          return (
          <div
            key={g.key}
            data-group-key={g.key}
            onDragOver={(e) => {
              e.preventDefault();
              let mode: 'merge' | 'before' | 'after' = 'merge';
              if (dragKind.current === 'group') {
                // Top/bottom quarter of the row = reorder around it.
                const r = e.currentTarget.getBoundingClientRect();
                const frac = (e.clientY - r.top) / Math.max(r.height, 1);
                if (frac < 0.25) mode = 'before';
                else if (frac > 0.75) mode = 'after';
              }
              setDragOver((prev) => (prev?.key === g.key && prev.mode === mode ? prev : { key: g.key, mode }));
            }}
            onDragLeave={() => setDragOver((prev) => (prev?.key === g.key ? null : prev))}
            onDrop={onDropTo(g.key)}
            className={cn(
              // content-visibility keeps far-offscreen rows unrendered
              // (and their lazy crop images unfetched) — without it a
              // video switch fires hundreds of image requests at once.
              'rounded-xl border bg-surface-50 p-2.5 transition-colors [contain-intrinsic-size:auto_12rem] [content-visibility:auto]',
              dragOver?.key === g.key && dragOver.mode === 'merge'
                ? 'border-primary/60 bg-primary/5'
                : isPool
                  ? 'border-dashed border-amber-500/40'
                  : 'border-border',
              flashGroup === g.key && 'animate-pulse border-accent ring-2 ring-accent/70',
            )}
            style={
              dragOver?.key === g.key
                ? dragOver.mode === 'before'
                  ? { boxShadow: '0 -3px 0 0 #fbbf24' }
                  : dragOver.mode === 'after'
                    ? { boxShadow: '0 3px 0 0 #fbbf24' }
                    : dragKind.current === 'group'
                      // Merge = both edges lit, completing the visual
                      // language: top line, bottom line, or both.
                      ? { boxShadow: '0 -3px 0 0 #fbbf24, 0 3px 0 0 #fbbf24' }
                      : undefined
                : undefined
            }
          >
            <div className="mb-2 flex items-center gap-3">
              <span
                draggable
                onDragStart={(e) => {
                  dragKind.current = 'group';
                  e.dataTransfer.setData('text/plain', `group\n${g.key}`);
                }}
                onDragEnd={() => {
                  dragKind.current = null;
                  setDragOver(null);
                }}
                title="Drag onto a row's middle to merge into it, or to its top/bottom edge to reorder"
                className={cn(
                  'flex h-6 min-w-8 cursor-grab items-center justify-center rounded-md px-1.5 font-mono text-[11px] font-bold tabular-nums active:cursor-grabbing',
                  g.locked ? 'bg-primary/15 text-primary-light ring-1 ring-primary/25' : 'bg-ink/5 text-text-secondary ring-1 ring-ink/10',
                )}
              >
                {i + 1}
              </span>
              <button
                type="button"
                onClick={() => board.toggleLock(g.key)}
                title={g.locked ? 'Locked — survives threshold/model changes. Click to unlock.' : 'Unlocked — re-clusters on threshold/model change. Click to lock.'}
                className={cn('flex-shrink-0 transition-colors', g.locked ? 'text-primary-light' : 'text-text-muted hover:text-text-secondary')}
              >
                {g.locked ? (
                  <svg className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M16.5 10.5V6.75a4.5 4.5 0 10-9 0v3.75m-.75 11.25h10.5a2.25 2.25 0 002.25-2.25v-6.75a2.25 2.25 0 00-2.25-2.25H6.75a2.25 2.25 0 00-2.25 2.25v6.75a2.25 2.25 0 002.25 2.25z" />
                  </svg>
                ) : (
                  <svg className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 10.5V6.75a4.5 4.5 0 119 0v3.75M3.75 21.75h10.5a2.25 2.25 0 002.25-2.25v-6.75a2.25 2.25 0 00-2.25-2.25H3.75a2.25 2.25 0 00-2.25 2.25v6.75a2.25 2.25 0 002.25 2.25z" />
                  </svg>
                )}
              </button>
              {isPool ? (
                <span
                  className="text-xs font-medium text-amber-300/90"
                  title={`Auto clusters smaller than ${MIN_CLUSTER_SIZE} events land here — noise, not a player. Drag crops onto a player row to assign them.`}
                >
                  No group — tiny clusters (&lt; {MIN_CLUSTER_SIZE})
                </span>
              ) : (
                <NameInput
                  value={g.name}
                  onCommit={(name) => board.renameGroup(g.key, name)}
                  placeholder="Player name…"
                  className="w-44 rounded-lg border border-border-light bg-surface-100 px-2.5 py-1 text-xs text-text-primary focus:border-primary/50 focus:outline-none"
                />
              )}
              <span className="font-mono text-[11px] tabular-nums text-text-muted">
                {statusFilter === 'all' ? `${g.eventIds.length}` : `${shownIds.length}/${g.eventIds.length}`} events
              </span>
              {isPool ? <Badge tone="warning">no group</Badge> : !g.name.trim() && <Badge tone="neutral">unassigned</Badge>}
            </div>
            <div className="flex min-h-[7rem] flex-wrap items-start gap-1.5">
              {shownIds.map((id) => renderCrop(id))}
            </div>
          </div>
          );
        })}
        {missIds.length > 0 && (statusFilter === 'all' || statusFilter === 'miss') && (
          <div className="rounded-xl border border-dashed border-red-500/30 bg-red-500/[0.03] p-2.5">
            <div className="mb-2 flex items-center gap-3">
              <span
                className="text-xs font-medium text-red-300/90"
                title="Events with no actor picked — no crop, no embedding, so clustering never sees them. Double-click one to jump there, then use Pick actor."
              >
                Miss — no actor
              </span>
              <span className="font-mono text-[11px] tabular-nums text-text-muted">{missIds.length} events</span>
            </div>
            <div className="flex flex-wrap items-start gap-1.5">{missIds.map((id) => renderCrop(id))}</div>
          </div>
        )}
        {/* Drop here to split a crop into a brand-new group */}
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver({ key: '__new__', mode: 'merge' });
          }}
          onDragLeave={() => setDragOver((prev) => (prev?.key === '__new__' ? null : prev))}
          onDrop={onDropTo('__new__')}
          className={cn(
            'flex items-center justify-center rounded-xl border border-dashed py-6 text-xs transition-colors',
            dragOver?.key === '__new__' ? 'border-primary/60 bg-primary/5 text-primary-light' : 'border-border text-text-muted',
          )}
        >
          Drop a crop here to start a new group
        </div>
      </div>
      {/* Locked-groups dock: compact, sticky drop targets while the
          main board scrolls — 3 crops per group, count for the rest. */}
      {dockGroups.length > 0 && (
        <aside className="sticky top-3 max-h-[calc(100vh-1.5rem)] w-60 flex-shrink-0 space-y-2 overflow-y-auto pr-0.5">
          {dockGroups.map((g) => {
            const shownIds = g.eventIds.filter(statusPass);
            return (
            <div
              key={g.key}
              data-group-key={g.key}
              onDragOver={(e) => {
                e.preventDefault();
                setDragOver((prev) => (prev?.key === g.key && prev.mode === 'merge' ? prev : { key: g.key, mode: 'merge' }));
              }}
              onDragLeave={() => setDragOver((prev) => (prev?.key === g.key ? null : prev))}
              onDrop={onDropTo(g.key)}
              className={cn(
                'rounded-xl border bg-surface-50 p-2 transition-colors',
                dragOver?.key === g.key ? 'border-primary/60 bg-primary/5' : 'border-border',
                flashGroup === g.key && 'animate-pulse border-accent ring-2 ring-accent/70',
              )}
            >
              <div className="mb-1.5 flex items-center gap-1.5">
                <button
                  type="button"
                  onClick={() => board.toggleLock(g.key)}
                  title="Locked — click to unlock (moves it back onto the board)"
                  className="flex-shrink-0 text-primary-light"
                >
                  <svg className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M16.5 10.5V6.75a4.5 4.5 0 10-9 0v3.75m-.75 11.25h10.5a2.25 2.25 0 002.25-2.25v-6.75a2.25 2.25 0 00-2.25-2.25H6.75a2.25 2.25 0 00-2.25 2.25v6.75a2.25 2.25 0 002.25 2.25z" />
                  </svg>
                </button>
                <NameInput
                  value={g.name}
                  onCommit={(name) => board.renameGroup(g.key, name)}
                  placeholder="Player name…"
                  className="w-full min-w-0 rounded-md border border-border-light bg-surface-100 px-2 py-0.5 text-xs text-text-primary focus:border-primary/50 focus:outline-none"
                />
                <span className="flex-shrink-0 font-mono text-[10px] tabular-nums text-text-muted">{g.eventIds.length}</span>
              </div>
              <div className="flex items-start gap-1">
                {shownIds.slice(0, 3).map((id) => renderCrop(id, 'h-20'))}
                {shownIds.length > 3 && (
                  <span className="self-center font-mono text-[10px] text-text-muted">+{shownIds.length - 3}</span>
                )}
              </div>
            </div>
            );
          })}
        </aside>
      )}
    </div>
  );
});
