import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
  type DragEvent,
  type DragEvent as ReactDragEvent,
  type MouseEvent as ReactMouseEvent,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
} from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { API, ApiError, apiFetch, apiUrl } from '@/lib/api';
import { cn } from '@/lib/cn';
import { actionColor } from '@/lib/actionColors';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { EmptyState } from '@/components/ui/EmptyState';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { CropImage } from '@/components/video/CropImage';
import { KindBadge } from '@/components/video/KindBadge';
import { VideoCombobox } from '@/components/video/VideoCombobox';
import { RallyTimeline } from '@/components/editor/RallyTimeline';
import type { EditorAnnotation } from '@/components/editor/AnnotationEditor';
import { toast } from '@/components/feedback/toast';
import { confirm } from '@/components/feedback/confirm';
import type { ActionAnnotationData, ReidCluster, ReidPlayers, ReidRecord, ReidVideo } from '@/types/api';

// Appearance embedders available server-side (see reid/embedder.py registry).
const EMBEDDERS = ['clip-reid', 'kpr'] as const;
type Embedder = (typeof EMBEDDERS)[number];
// Cosine-distance scales differ per model: CLIP-ReID's ViT features sit in a
// tight cone (hence tiny cutoffs); KPR's foreground embeddings spread wide.
// Ranges calibrated on real match footage (~12 people on court).
const EMBEDDER_THRESHOLDS: Record<Embedder, { min: number; max: number; def: number }> = {
  'clip-reid': { min: 0.08, max: 0.3, def: 0.15 },
  kpr: { min: 0.3, max: 0.8, def: 0.55 },
};
// Auto clusters this small are noise, not players — they pool into one
// shared "unsorted" row instead of each getting its own.
const MIN_CLUSTER_SIZE = 3;

const STATUS_DOT: Record<ReidRecord['status'], string> = {
  ok: 'bg-primary-light',
  multi: 'bg-amber-400',
  miss: 'bg-red-400',
};

const selectCls =
  'w-auto cursor-pointer appearance-none rounded-lg border border-border-light bg-surface-50 px-3 py-1 text-xs text-text-primary focus:border-primary/50 focus:outline-none';

const fieldCls = 'rounded-lg border border-border-light bg-surface-50 px-3 py-2 text-sm text-text-primary focus:border-primary/50 focus:outline-none';

const errMsg = (e: unknown) => (e instanceof ApiError ? e.body : e instanceof Error ? e.message : String(e));

/** One editable identity group: named = a player, unnamed = an auto cluster.
 *  Locked groups survive re-clustering (threshold/model changes); any group
 *  the user edits locks itself automatically. */
interface Group {
  key: string;
  name: string;
  eventIds: string[];
  locked: boolean;
}

interface PlayerHandle {
  /** Park the video on an event's frame, select + expand its rally, and pin
   *  that rally to the top of the sidebar list. */
  jumpToEvent: (a: { id: string; frame: number; time: number | null }) => void;
}

interface Rally {
  rally_id: number;
  start: number;
  end: number;
}

/** {box} = manual pick, {none} = nobody is the actor, {} = revert to auto. */
type ActorFix = { box?: [number, number, number, number]; none?: boolean };

interface ReidVideoPlayerProps {
  src: string;
  fps: number;
  frameSize: [number, number];
  records: ReidRecord[];
  /** Full action annotation — includes score / non-visible events that the
   *  ReID extraction skips, so the sidebar can still list their times. */
  actionEvents: SidebarAction[];
  matches: ReidPlayers['matches'];
  rallies: Rally[];
  selectedRally: number | 'all';
  onSelectRally: (rally: number | 'all') => void;
  onFixActor: (eventId: string, fix: ActorFix) => void;
  onJumpToCrop: (eventId: string) => void;
}

/** One sidebar row: an action event's time, whether or not it has a ReID
 *  record (score / non-visible events have none — no box, just the time). */
interface SidebarAction {
  id: string;
  frame: number;
  time: number | null;
  label?: string;
  visible: boolean;
}

function FieldLabel({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="block min-w-0 space-y-1.5">
      <span className="block text-[10px] font-semibold uppercase tracking-widest text-text-muted">{label}</span>
      {children}
    </label>
  );
}

const fmtTime = (s: number) => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(Math.floor(s % 60)).padStart(2, '0')}`;

const OUTSIDE = '__outside__';

interface ReidEventPanelProps {
  entries: SidebarAction[];
  empty: string;
  matches: ReidPlayers['matches'];
  selectedEventId: string | null;
  fps: number;
  /** Current playhead frame — rows within ±½ s light up (Rally Label rule). */
  playheadFrame: number;
  onJump: (a: SidebarAction) => void;
  onJumpToCrop: (eventId: string) => void;
}

/** Read-only twin of the Action Label event panel: action dot + label,
 *  matched player, frame and time — click a row to park the video there. */
function ReidEventPanel({ entries, empty, matches, selectedEventId, fps, playheadFrame, onJump, onJumpToCrop }: ReidEventPanelProps) {
  if (!entries.length) return <div className="ml-6 rounded-xl border border-border bg-surface-100 px-3 py-2 text-xs text-text-muted">{empty}</div>;
  const windowFrames = Math.max(1, Math.round(fps / 2));
  return (
    <div className="ml-6 space-y-1.5 rounded-xl border border-border bg-surface-100 p-2">
      {entries.map((a, row) => {
        const m = matches[a.id];
        const color = actionColor(a.label);
        const active = Math.abs(a.frame - playheadFrame) <= windowFrames;
        return (
          <div
            key={a.id}
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
                title="Jump to this event's crop in the identities board below"
                className={cn(
                  'max-w-full justify-self-end truncate rounded-full px-2 py-0.5 text-[11px] ring-1 transition-colors',
                  m.assigned
                    ? 'bg-primary/15 text-primary-light ring-primary/30 hover:bg-primary/30'
                    : 'bg-surface-200/40 text-text-muted ring-border hover:bg-surface-200/80 hover:text-text-secondary',
                )}
              >
                {m.player}
              </button>
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

/** Video player whose overlay mirrors the ReID results: every event whose
 *  frame is within ±½ s of the playhead shows its player box + identity,
 *  sharpening as playback crosses the exact annotated frame. */
const ReidVideoPlayer = forwardRef<PlayerHandle, ReidVideoPlayerProps>(function ReidVideoPlayer(
  { src, fps, frameSize, records, actionEvents, matches, rallies, selectedRally, onSelectRally, onFixActor, onJumpToCrop },
  ref,
) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [frame, setFrame] = useState(0);
  const [duration, setDuration] = useState(0);
  const [playing, setPlaying] = useState(false);
  // Expanded rally (or OUTSIDE) in the sidebar + last event jumped to — same
  // interaction as the Action Label rally list.
  const [expanded, setExpanded] = useState<string | null>(null);
  const [selectedEventId, setSelectedEventId] = useState<string | null>(null);
  // Actor-picker mode: park on an event's frame, then click the right person.
  const [pickMode, setPickMode] = useState(false);
  // Picker display floor: extraction stores every detection ≥ 0.1, this
  // slider decides how deep into the low-confidence pile to show.
  const [minDetScore, setMinDetScore] = useState(0.25);
  useEffect(() => {
    setExpanded(null);
    setSelectedEventId(null);
    setPickMode(false);
  }, [src]);
  // Read inside the frame-clock callback without re-arming it.
  const rallyRef = useRef<Rally | null>(null);
  rallyRef.current = selectedRally === 'all' ? null : rallies.find((r) => r.rally_id === selectedRally) ?? null;

  const togglePlay = () => {
    const el = videoRef.current;
    if (!el) return;
    if (el.paused) {
      // Parked at (or past) the selected rally's end the frame clock would
      // pause again on the very first frame — play there replays the rally.
      const r = rallyRef.current;
      if (r && el.currentTime >= r.end - 1 / fps) el.currentTime = r.start + 0.5 / fps;
      void el.play();
    } else {
      el.pause();
    }
  };

  useImperativeHandle(ref, () => ({
    jumpToEvent: (a: { id: string; frame: number; time: number | null }) => {
      const rally = seekEvent(a);
      scrollRallyTop(rally ? rally.rally_id : OUTSIDE);
      videoRef.current?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    },
  }));

  // Frame clock via requestVideoFrameCallback — same approach as the Action
  // Label editor, re-armed per presented frame.
  const prevTimeRef = useRef(0);
  useEffect(() => {
    const el = videoRef.current;
    if (!el) return;
    let alive = true;
    let id = 0;
    const tick = (_now: number, meta: { mediaTime: number }) => {
      if (!alive) return;
      setFrame(Math.round(meta.mediaTime * fps));
      // With a rally selected, playback stops at its end (Action Label rule) —
      // but only when crossing it from inside; a playhead parked beyond the
      // rally (event jumps, scrubs) must never trip it.
      const r = rallyRef.current;
      const prev = prevTimeRef.current;
      prevTimeRef.current = meta.mediaTime;
      if (r && !el.paused && prev >= r.start && prev < r.end && meta.mediaTime >= r.end) el.pause();
      id = el.requestVideoFrameCallback(tick);
    };
    id = el.requestVideoFrameCallback(tick);
    const onSeeked = () => {
      prevTimeRef.current = el.currentTime;
      setFrame(Math.round(el.currentTime * fps));
    };
    el.addEventListener('seeked', onSeeked);
    return () => {
      alive = false;
      el.cancelVideoFrameCallback(id);
      el.removeEventListener('seeked', onSeeked);
    };
  }, [fps, src]);

  // Space = play/pause everywhere on the page (same contract as Action
  // Label): text fields keep the key for typing, but the seek slider and
  // <select>s hand it back so scrubbing → space "just works".
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== ' ') return;
      const target = e.target as HTMLElement | null;
      const tag = target?.tagName;
      if (tag === 'TEXTAREA') return;
      if (tag === 'INPUT' && (target as HTMLInputElement).type !== 'range') return;
      e.preventDefault();
      togglePlay();
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, []);

  const [w, h] = frameSize;
  const windowFrames = Math.max(1, Math.round(fps / 2));
  const visible = records.filter((r) => r.box && Math.abs(r.frame - frame) <= windowFrames);
  const time = frame / fps;
  // The event whose actor is being picked: the record closest to the playhead.
  const pickTarget = useMemo(() => {
    if (!pickMode) return null;
    const near = records.filter((r) => Math.abs(r.frame - frame) <= 2);
    if (!near.length) return null;
    return near.reduce((a, b) => (Math.abs(a.frame - frame) <= Math.abs(b.frame - frame) ? a : b));
  }, [pickMode, records, frame]);
  // Sidebar rows come from the full action annotation, so score / non-visible
  // events keep their time even though extraction skipped them (no box to
  // draw, nothing to re-identify). Falls back to the extraction records when
  // no annotation is loaded (yet).
  const sidebarActions = useMemo<SidebarAction[]>(() => {
    const rows: SidebarAction[] = actionEvents.length
      ? actionEvents
      : records.map((r) => ({ id: r.id, frame: r.frame, time: r.time ?? null, label: r.label, visible: true }));
    return [...rows].sort((a, b) => a.frame - b.frame);
  }, [actionEvents, records]);

  // Actions grouped per rally, plus the ones outside any rally — mirrors the
  // Action Label sidebar.
  const { byRally, outside } = useMemo(() => {
    const map = new Map<number, SidebarAction[]>(rallies.map((r) => [r.rally_id, []]));
    const out: SidebarAction[] = [];
    for (const a of sidebarActions) {
      const t = a.time != null ? a.time : a.frame / fps;
      const rally = rallies.find((x) => t >= x.start && t <= x.end);
      if (rally) map.get(rally.rally_id)!.push(a);
      else out.push(a);
    }
    return { byRally: map, outside: out };
  }, [rallies, sidebarActions, fps]);

  const jumpToRally = (rally: Rally) => {
    onSelectRally(rally.rally_id);
    setExpanded(String(rally.rally_id));
    const el = videoRef.current;
    if (el) el.currentTime = rally.start + 0.5 / fps;
  };

  const listRef = useRef<HTMLDivElement>(null);
  /** Pin a rally row (or the outside header) to the top of the sidebar list.
   *  Measured a frame later, so the expand/collapse this jump caused has
   *  already re-laid-out the list. */
  const scrollRallyTop = (key: number | string) => {
    requestAnimationFrame(() => {
      const list = listRef.current;
      const row = list?.querySelector<HTMLElement>(`[data-rally-row="${CSS.escape(String(key))}"]`);
      if (!list || !row) return;
      list.scrollTo({
        top: row.getBoundingClientRect().top - list.getBoundingClientRect().top + list.scrollTop,
        behavior: 'smooth',
      });
    });
  };

  const seekEvent = (a: { id: string; frame: number; time: number | null }) => {
    setSelectedEventId(a.id);
    // Keep the rally selection in sync with the jump (Action Label contract) —
    // a stale selection would strand the playhead outside the "selected" rally.
    const t = a.time != null ? a.time : a.frame / fps;
    const rally = rallies.find((x) => t >= x.start && t <= x.end);
    onSelectRally(rally ? rally.rally_id : 'all');
    setExpanded(rally ? String(rally.rally_id) : OUTSIDE);
    const el = videoRef.current;
    if (el) {
      el.pause();
      el.currentTime = (a.frame + 0.5) / fps;
    }
    return rally;
  };

  const timelineAnnotations = useMemo<EditorAnnotation[]>(
    () => rallies.map((r) => ({ rally_id: r.rally_id, start: r.start, end: r.end, label: 'rally' })),
    [rallies],
  );

  const aspect = w / h;
  return (
    <div className="flex flex-col gap-5 lg:flex-row lg:items-start">
      {/* Player — same console styling as the Rally Label / Action Label editors */}
      <div className="min-w-0 flex-1">
        <Card>
          <div className="overflow-hidden rounded-2xl bg-black shadow-lg shadow-black/40 ring-1 ring-white/[0.06]">
            <div className="relative mx-auto" style={{ aspectRatio: `${aspect}`, maxWidth: `calc(var(--video-max-h, 45vh) * ${aspect})` }}>
              <video
                ref={videoRef}
                src={src}
                preload="metadata"
                onClick={pickMode ? undefined : togglePlay}
                onPlay={() => setPlaying(true)}
                onPause={() => setPlaying(false)}
                onEnded={() => setPlaying(false)}
                onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)}
                className="block h-full w-full cursor-pointer bg-black object-contain"
              />
              <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className="pointer-events-none absolute left-0 top-0 h-full w-full">
              {visible.map((r) => {
                const [x0, y0, x1, y1] = r.box!;
                const m = matches[r.id];
                // Same hue per action as the Action Label editor.
                const color = actionColor(r.label);
                const exact = Math.abs(r.frame - frame) <= 2;
                const label = m ? m.player : r.label ?? '';
                return (
                  <g key={r.id} opacity={exact ? 1 : 0.45}>
                    <rect
                      x={x0}
                      y={y0}
                      width={x1 - x0}
                      height={y1 - y0}
                      fill="none"
                      stroke={color}
                      strokeWidth={exact ? 2.5 : 1.5}
                      vectorEffect="non-scaling-stroke"
                    />
                    <text
                      x={x0 + 4}
                      y={Math.max(y0 - 8, 22)}
                      fill={color}
                      stroke="#000"
                      strokeWidth={4}
                      paintOrder="stroke"
                      fontSize={Math.round(h / 42)}
                      fontFamily="ui-monospace, SF Mono, Menlo"
                    >
                      {label}
                      {r.box_source === 'manual' ? ' ✎' : ''} · f{r.frame}
                    </text>
                  </g>
                );
              })}
              {/* Actor picker: every detected person above the score slider as a clickable outline */}
              {pickMode &&
                pickTarget?.detections?.filter((d) => d.score >= minDetScore).map((d, i) => {
                  const [x0, y0, x1, y1] = d.box;
                  return (
                    <rect
                      key={`det-${i}`}
                      x={x0}
                      y={y0}
                      width={x1 - x0}
                      height={y1 - y0}
                      fill="transparent"
                      stroke="#fff"
                      strokeOpacity={d.score >= 0.5 ? 0.9 : 0.45}
                      strokeWidth={1.5}
                      strokeDasharray={d.score >= 0.5 ? undefined : '3 5'}
                      vectorEffect="non-scaling-stroke"
                      className="pointer-events-auto cursor-pointer hover:fill-white/20"
                      onClick={(e) => {
                        e.stopPropagation();
                        onFixActor(pickTarget.id, { box: d.box });
                      }}
                    >
                      <title>{`person · score ${d.score.toFixed(2)} — click to set as the actor`}</title>
                    </rect>
                  );
                })}
              </svg>
              <div className="pointer-events-none absolute left-2 top-2 rounded-md bg-black/60 px-2 py-0.5 font-mono text-[10.5px] tabular-nums text-white">
                f{frame} · {visible.length} box(es)
              </div>
            </div>
          </div>
          {rallies.length > 0 && (
            <div className="mt-3">
              <RallyTimeline
                videoRef={videoRef}
                annotations={timelineAnnotations}
                duration={duration}
                markStart={null}
                onSeek={(t) => {
                  const el = videoRef.current;
                  if (el) el.currentTime = t;
                }}
              />
            </div>
          )}
          <div className="mt-2 flex items-center gap-3">
            <button
              type="button"
              onClick={togglePlay}
              aria-label={playing ? 'Pause' : 'Play'}
              className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-lg bg-primary text-on-primary transition-colors hover:brightness-110"
            >
              {playing ? (
                <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 24 24">
                  <rect x="6" y="5" width="4" height="14" rx="1" />
                  <rect x="14" y="5" width="4" height="14" rx="1" />
                </svg>
              ) : (
                <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M8 5.14v13.72a1 1 0 001.54.84l10.7-6.86a1 1 0 000-1.68L9.54 4.3A1 1 0 008 5.14z" />
                </svg>
              )}
            </button>
            <input
              type="range"
              min={0}
              max={duration || 0}
              step={1 / fps}
              value={Math.min(time, duration || 0)}
              onChange={(e) => {
                const el = videoRef.current;
                if (el) el.currentTime = Number(e.target.value);
              }}
              onPointerUp={(e) => e.currentTarget.blur()}
              className="h-1.5 min-w-0 flex-1 cursor-pointer accent-primary"
            />
            <span className="flex-shrink-0 rounded-lg border border-border bg-surface-200/50 px-2.5 py-1 font-mono text-sm tabular-nums text-text-primary">
              {fmtTime(time)} / {fmtTime(duration)}
            </span>
            <Button
              size="sm"
              intent={pickMode ? 'primary' : 'default'}
              onClick={() => setPickMode((m) => !m)}
              title="Pick actor: park on an action's frame, then click the person who performed it"
            >
              Pick actor
            </Button>
          </div>
          {pickMode && (
            <div className="mt-3 flex flex-wrap items-center gap-2.5 rounded-xl border border-primary/20 bg-primary/10 p-3 text-xs">
              {pickTarget ? (
                <>
                  <span className="h-2 w-2 flex-shrink-0 rounded-full animate-pulse-dot" style={{ background: actionColor(pickTarget.label) }} />
                  <span className="text-primary-light">
                    Picking actor for <strong>{pickTarget.label}</strong> f{pickTarget.frame}
                    {pickTarget.detections?.length
                      ? ' — click the correct person in the video'
                      : ' — no stored detections (re-run extraction for this video)'}
                  </span>
                  <span className="ml-auto flex items-center gap-3">
                    <label className="flex items-center gap-1.5 text-[11px] text-text-secondary" title="Hide detections below this score — drag left to reveal weaker boxes (extraction keeps everything ≥ 0.1)">
                      <span className="whitespace-nowrap">
                        score ≥ <span className="font-mono tabular-nums">{minDetScore.toFixed(2)}</span>
                      </span>
                      <input
                        type="range"
                        min={0.1}
                        max={1}
                        step={0.05}
                        value={minDetScore}
                        onChange={(e) => setMinDetScore(Number(e.target.value))}
                        onPointerUp={(e) => e.currentTarget.blur()}
                        className="h-1 w-24 cursor-pointer accent-primary"
                      />
                      <span className="font-mono text-[10px] tabular-nums text-text-muted">
                        {(pickTarget.detections ?? []).filter((d) => d.score >= minDetScore).length}/{(pickTarget.detections ?? []).length}
                      </span>
                    </label>
                    <Button size="sm" onClick={() => onFixActor(pickTarget.id, { none: true })} title="Nobody in frame is the actor — clears this event's crop">
                      No actor
                    </Button>
                    {pickTarget.box_source === 'manual' && (
                      <Button size="sm" onClick={() => onFixActor(pickTarget.id, {})} title="Discard the manual fix and re-run the automatic pick">
                        Revert to auto
                      </Button>
                    )}
                  </span>
                </>
              ) : (
                <span className="text-text-muted">Jump to an action first (double-click a crop or click a sidebar row), then click the right person.</span>
              )}
            </div>
          )}
        </Card>
      </div>

      {/* Rally list — same sidebar as the Rally Label / Action Label editors */}
      {rallies.length > 0 && (
        <div className="min-w-0 lg:w-[420px] lg:flex-shrink-0">
          <Card>
            <SectionLabel>
              Rallies ({rallies.length} rally · {sidebarActions.length} action)
            </SectionLabel>
            <div ref={listRef} className="vq-list max-h-[calc(45vh+2.25rem)] space-y-1.5 overflow-y-auto pr-1">
              <div
                onClick={() => onSelectRally('all')}
                className={cn(
                  'ae-row flex cursor-pointer items-center gap-1.5 rounded-xl border px-3 py-2.5 transition-colors',
                  selectedRally === 'all'
                    ? 'border-primary/45 bg-primary/[0.12]'
                    : 'border-primary/20 bg-primary/[0.05] hover:bg-primary/[0.10]',
                )}
              >
                <span className="text-xs font-medium text-text-primary">All rallies</span>
                <span className="ml-auto font-mono text-[10px] tabular-nums text-text-muted">{sidebarActions.length} action</span>
              </div>
              {rallies.map((rally, i) => {
                const entries = byRally.get(rally.rally_id) ?? [];
                const isOpen = expanded === String(rally.rally_id);
                const active = time >= rally.start && time <= rally.end;
                const selected = selectedRally === rally.rally_id;
                return (
                  <div key={rally.rally_id} className="space-y-1.5">
                    <div
                      data-rally-row={rally.rally_id}
                      onClick={() => jumpToRally(rally)}
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
                          if (isOpen) setExpanded(null);
                          else jumpToRally(rally);
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
                      <ReidEventPanel entries={entries} empty="No actions in this rally" matches={matches} selectedEventId={selectedEventId} fps={fps} playheadFrame={frame} onJump={seekEvent} onJumpToCrop={onJumpToCrop} />
                    )}
                  </div>
                );
              })}
              {outside.length > 0 && (
                <div className="space-y-1.5">
                  <div
                    data-rally-row={OUTSIDE}
                    onClick={() => setExpanded(OUTSIDE)}
                    className="flex cursor-pointer items-center gap-2.5 rounded-xl border border-amber-500/20 bg-amber-500/[0.04] px-3 py-2.5 hover:bg-amber-500/[0.08]"
                  >
                    <span className="w-4 select-none text-right font-heading text-[10px] text-text-muted/60">out</span>
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        setExpanded(expanded === OUTSIDE ? null : OUTSIDE);
                      }}
                      className="flex items-center gap-1 rounded-full bg-amber-500/15 px-2 py-0.5 text-[11px] font-medium text-amber-300 ring-1 ring-amber-500/25"
                    >
                      <span className={cn('transition-transform', expanded === OUTSIDE && 'rotate-90')}>▸</span> outside <span className="opacity-70">{outside.length}</span>
                    </button>
                    <span className="ml-auto font-heading text-[11px] text-text-muted">outside rally</span>
                  </div>
                  {expanded === OUTSIDE && (
                    <ReidEventPanel entries={outside} empty="No outside actions" matches={matches} selectedEventId={selectedEventId} fps={fps} playheadFrame={frame} onJump={seekEvent} onJumpToCrop={onJumpToCrop} />
                  )}
                </div>
              )}
            </div>
          </Card>
        </div>
      )}
    </div>
  );
});

export function ReidLabelPage() {
  const qc = useQueryClient();
  const [picked, setPicked] = useState('');
  const [kindFilter, setKindFilter] = useState<'all' | 'broadcast' | 'sideline'>('all');
  const [pickStatus, setPickStatus] = useState<'all' | 'unlabeled' | 'labeled'>('all');
  const [selectedRally, setSelectedRally] = useState<number | 'all'>('all');
  // Where locked groups live on the groups board: pinned on top as full rows,
  // or docked in a sticky right rail showing just 3 crops per group.
  const [lockedDock, setLockedDock] = useState<'top' | 'right'>('right');
  const [embedder, setEmbedder] = useState<Embedder>('clip-reid');
  // Draft follows the slider live; the applied value (= clusters query key)
  // trails it by a debounce so dragging doesn't fire a re-cluster per pixel.
  const [thresholdDraft, setThresholdDraft] = useState<number>(EMBEDDER_THRESHOLDS['clip-reid'].def);
  const [threshold, setThreshold] = useState<number>(EMBEDDER_THRESHOLDS['clip-reid'].def);
  useEffect(() => {
    const t = setTimeout(() => setThreshold(thresholdDraft), 350);
    return () => clearTimeout(t);
  }, [thresholdDraft]);
  const [showSkeleton, setShowSkeleton] = useState(false);
  const [showVideo, setShowVideo] = useState(true);
  const playerRef = useRef<PlayerHandle>(null);
  const [statusFilter, setStatusFilter] = useState<'all' | ReidRecord['status']>('all');
  const [groups, setGroups] = useState<Group[]>([]);
  const [dirty, setDirty] = useState(false);
  // What's under the cursor during a drag. mode: 'merge' drops INTO the row,
  // 'before'/'after' reorder around it (group drags only, via edge bands).
  const [dragOver, setDragOver] = useState<{ key: string; mode: 'merge' | 'before' | 'after' } | null>(null);
  // Payload kind of the in-flight drag (dataTransfer is unreadable during
  // dragover, so remember it at dragstart).
  const dragKind = useRef<'group' | 'events' | null>(null);
  // Multi-select for bulk drag: Ctrl/Cmd/Shift-click crops to toggle, plain
  // click still seeks the video. Esc clears.
  const [selectedCrops, setSelectedCrops] = useState<Set<string>>(new Set());
  const newGroupSeq = useRef(0);

  const videosQuery = useQuery({
    queryKey: ['reid-videos'],
    queryFn: () => apiFetch<ReidVideo[]>(API.reid.videos),
  });
  const extracted = (videosQuery.data ?? []).filter((v) => v.has_reid);
  // Picker filters, mirroring the Action Label / Rally Label pickers.
  const pickable = extracted.filter((v) => {
    if (kindFilter !== 'all' && v.kind !== kindFilter) return false;
    if (pickStatus === 'unlabeled' && (v.player_count ?? 0) > 0) return false;
    if (pickStatus === 'labeled' && (v.player_count ?? 0) === 0) return false;
    return true;
  });

  const resultsQuery = useQuery({
    queryKey: ['reid-results', picked],
    queryFn: () => apiFetch<{ meta: Record<string, unknown>; records: ReidRecord[] }>(API.reid.results(picked)),
    enabled: Boolean(picked),
  });
  const records = useMemo(() => resultsQuery.data?.records ?? [], [resultsQuery.data]);
  const recordById = useMemo(() => new Map(records.map((r) => [r.id, r])), [records]);
  const meta = (resultsQuery.data?.meta ?? {}) as {
    fps?: number;
    frame_size?: [number, number];
    rallies?: Rally[];
  };
  const seekToEvent = (r?: ReidRecord) => {
    // The player owns the whole jump: rally selection, panel expansion,
    // sidebar pinning and the actual seek.
    if (r) playerRef.current?.jumpToEvent({ id: r.id, frame: r.frame, time: r.time ?? null });
  };

  // Reverse jump (video → board): scroll the event's crop into view and pulse
  // it. Works in both views — crops carry data-event-id either way.
  const [flashCrop, setFlashCrop] = useState<string | null>(null);
  const flashTimer = useRef(0);
  useEffect(() => () => window.clearTimeout(flashTimer.current), []);
  const jumpToCrop = (eventId: string) => {
    if (!recordById.get(eventId)?.crop) {
      toast.warning('This event has no crop (score / non-visible action)');
      return;
    }
    const el = document.querySelector(`[data-event-id="${CSS.escape(eventId)}"]`);
    if (!el) {
      toast.warning('Crop is hidden by the current status/rally filter');
      return;
    }
    el.scrollIntoView({ block: 'center', behavior: 'smooth' });
    window.clearTimeout(flashTimer.current);
    setFlashCrop(eventId);
    flashTimer.current = window.setTimeout(() => setFlashCrop(null), 1600);
  };

  // Full action annotation — the sidebar lists every action's time, including
  // score / non-visible events the ReID extraction skipped.
  const actionsQuery = useQuery({
    queryKey: ['reid-action-events', picked],
    queryFn: () => apiFetch<ActionAnnotationData>(API.actionAnnotate.annotation(picked)),
    enabled: Boolean(picked),
  });
  const actionEvents = useMemo<SidebarAction[]>(
    () =>
      (actionsQuery.data?.events ?? []).flatMap((raw) => {
        const x = raw as Record<string, unknown>;
        if (x.frame == null) return [];
        const frame = Math.max(0, Math.round(Number(x.frame) || 0));
        return [
          {
            // Same id fallback as the extraction pipeline, so matches line up.
            id: typeof x.id === 'string' && x.id ? x.id : `f${frame}`,
            frame,
            time: typeof x.time === 'number' ? x.time : null,
            label: typeof x.label === 'string' ? x.label : undefined,
            visible: x.visible !== false,
          },
        ];
      }),
    [actionsQuery.data],
  );

  const clustersQuery = useQuery({
    queryKey: ['reid-clusters', picked, threshold, embedder],
    queryFn: () => apiFetch<{ clusters: ReidCluster[] }>(API.reid.clusters(picked, threshold, embedder)),
    enabled: Boolean(picked),
  });
  const clusters = useMemo(() => clustersQuery.data?.clusters ?? [], [clustersQuery.data]);

  const playersQuery = useQuery({
    queryKey: ['reid-players', picked, embedder],
    queryFn: () => apiFetch<ReidPlayers>(API.reid.players(picked, embedder)),
    enabled: Boolean(picked),
  });
  const assignments = useMemo(() => playersQuery.data?.assignments ?? {}, [playersQuery.data]);
  const matches = playersQuery.data?.matches ?? {};

  // (Re)build the board whenever the clustering or saved players change:
  // locked rows carry over untouched, saved players fill in what locked rows
  // don't already hold, fresh clusters cover the rest. Unlocked rows are
  // disposable by design — edits lock their row automatically.
  useEffect(() => {
    if (!picked) return;
    setGroups((prev) => {
      const out: Group[] = prev.filter((g) => g.locked).map((g) => ({ ...g, eventIds: [...g.eventIds] }));
      const covered = new Set(out.flatMap((g) => g.eventIds));

      const byPlayer = new Map<string, string[]>();
      for (const [id, name] of Object.entries(assignments)) {
        if (!covered.has(id) && recordById.has(id)) byPlayer.set(name, [...(byPlayer.get(name) ?? []), id]);
      }
      for (const [name, ids] of [...byPlayer.entries()].sort((a, b) => a[0].localeCompare(b[0]))) {
        const lockedSame = out.find((g) => g.name.trim() === name);
        if (lockedSame) {
          lockedSame.eventIds.push(...ids);
        } else {
          // Saved players are confirmed human work — they come back
          // locked, so a reload looks identical to the pre-save board
          // and re-clustering can never dissolve them.
          out.push({ key: `p:${name}`, name, eventIds: ids, locked: true });
        }
        ids.forEach((id) => covered.add(id));
      }

      const tiny: string[] = [];
      for (const c of clusters) {
        const rest = c.event_ids.filter((id) => !covered.has(id));
        if (!rest.length) continue;
        if (rest.length < MIN_CLUSTER_SIZE) tiny.push(...rest);
        else out.push({ key: `c:${embedder}:${threshold}:${c.id}`, name: '', eventIds: rest, locked: false });
      }
      if (tiny.length) out.push({ key: `pool:${embedder}:${threshold}`, name: '', eventIds: tiny, locked: false });
      return out;
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [picked, assignments, clusters, recordById]);

  /** Move one or many events into ``toKey``, wherever they currently live —
   *  the selection may span multiple source groups. */
  const moveEvents = (eventIds: string[], toKey: string) => {
    const moving = new Set(eventIds);
    if (!moving.size) return;
    setGroups((prev) =>
      prev
        .map((g) => {
          // Receiving an edit locks the row so re-clustering can't undo it.
          if (g.key === toKey) return { ...g, eventIds: [...g.eventIds.filter((i) => !moving.has(i)), ...eventIds], locked: true };
          return { ...g, eventIds: g.eventIds.filter((i) => !moving.has(i)) };
        })
        // An emptied auto-cluster group is noise; an emptied named player stays.
        .filter((g) => g.eventIds.length > 0 || g.name.trim()),
    );
    setSelectedCrops(new Set());
    setDirty(true);
  };

  /** Merge every event of one row into another; the target's name wins,
   *  falling back to the source's if the target is unnamed. */
  const mergeGroups = (fromKey: string, toKey: string) => {
    if (fromKey === toKey) return;
    setGroups((prev) => {
      const from = prev.find((g) => g.key === fromKey);
      if (!from) return prev;
      return prev
        .filter((g) => g.key !== fromKey)
        .map((g) =>
          g.key === toKey
            ? { ...g, name: g.name.trim() || from.name, eventIds: [...g.eventIds, ...from.eventIds], locked: true }
            : g,
        );
    });
    setDirty(true);
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setSelectedCrops(new Set());
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, []);

  // ── Marquee (rubber-band) selection over the group board ──
  // Starts on empty board space only; crops keep their native drag. Ctrl/⌘
  // makes it additive. Viewport coordinates throughout, so scrolling during
  // the drag stays correct.
  const boardRef = useRef<HTMLDivElement>(null);
  const [marquee, setMarquee] = useState<{ x0: number; y0: number; x1: number; y1: number } | null>(null);
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

  // Right-click with a selection → tiny context menu (new group / clear).
  const [ctxMenu, setCtxMenu] = useState<{ x: number; y: number } | null>(null);
  const boardContextMenu = (e: ReactMouseEvent) => {
    if (!selectedCrops.size) return; // native menu when nothing is selected
    e.preventDefault();
    setCtxMenu({ x: e.clientX, y: e.clientY });
  };
  /** New empty group inserted right below the group holding anchorId —
   *  keeps the new row next to where the crops came from for comparison. */
  const newGroupBelow = (anchorId: string | undefined) => {
    const key = `n:${newGroupSeq.current++}`;
    setGroups((prev) => {
      const at = anchorId ? prev.findIndex((g) => g.eventIds.includes(anchorId)) : -1;
      const out = [...prev];
      out.splice(at >= 0 ? at + 1 : out.length, 0, { key, name: '', eventIds: [], locked: true });
      return out;
    });
    return key;
  };

  const selectionToNewGroup = () => {
    const ids = [...selectedCrops];
    moveEvents(ids, newGroupBelow(ids[0]));
    setCtxMenu(null);
  };

  const toggleLock = (key: string) =>
    setGroups((prev) => prev.map((g) => (g.key === key ? { ...g, locked: !g.locked } : g)));

  /** Move a whole row so it sits before/after targetKey. View-only —
   *  ordering isn't persisted, but locked rows keep it for the session. */
  const reorderGroup = (fromKey: string, targetKey: string, mode: 'before' | 'after') => {
    if (fromKey === targetKey) return;
    setGroups((prev) => {
      const from = prev.find((g) => g.key === fromKey);
      if (!from) return prev;
      const rest = prev.filter((g) => g.key !== fromKey);
      const at = rest.findIndex((g) => g.key === targetKey);
      if (at < 0) return prev;
      rest.splice(mode === 'before' ? at : at + 1, 0, from);
      return rest;
    });
  };

  const onDropTo = (toKey: string) => (e: DragEvent) => {
    e.preventDefault();
    const mode = dragOver?.key === toKey ? dragOver.mode : 'merge';
    setDragOver(null);
    dragKind.current = null;
    const parts = e.dataTransfer.getData('text/plain').split('\n');
    if (parts[0] === 'group') {
      const fromKey = parts[1];
      if (!fromKey || toKey === '__new__') return;
      if (mode === 'merge') mergeGroups(fromKey, toKey);
      else reorderGroup(fromKey, toKey, mode);
      return;
    }
    const [kind, idList] = parts;
    const eventIds = (idList ?? '').split(',').filter(Boolean);
    if (kind !== 'events' || !eventIds.length) return;
    moveEvents(eventIds, toKey === '__new__' ? newGroupBelow(eventIds[0]) : toKey);
  };

  const savingRef = useRef(false);
  const save = async (auto = false) => {
    if (savingRef.current) return;
    savingRef.current = true;
    // Locked-but-unnamed rows are curated work — persist them under a
    // placeholder identity (P1, P2, …) the user can rename any time.
    // Untouched auto-clusters stay ephemeral by design.
    const used = new Set(groups.map((g) => g.name.trim()).filter(Boolean));
    let seq = 1;
    const nextPlaceholder = () => {
      while (used.has(`P${seq}`)) seq += 1;
      const name = `P${seq}`;
      used.add(name);
      return name;
    };
    const named = groups.map((g) => (g.locked && !g.name.trim() ? { ...g, name: nextPlaceholder() } : g));

    const next: Record<string, string> = {};
    for (const g of named) {
      const name = g.name.trim();
      if (!name) continue;
      for (const id of g.eventIds) next[id] = name;
    }
    try {
      await apiFetch(API.reid.players(picked, embedder), { method: 'PUT', body: { assignments: next } });
      setGroups(named); // show the placeholders the save just minted
      setDirty(false);
      await qc.invalidateQueries({ queryKey: ['reid-players', picked] });
      if (!auto) toast.success(`Saved ${new Set(Object.values(next)).size} player(s), ${Object.keys(next).length} events`);
    } catch (e) {
      toast.error(`Save failed: ${errMsg(e)}`);
    } finally {
      savingRef.current = false;
    }
  };

  // Auto-save: group edits persist ~1.5 s after the last change; failures
  // leave dirty=true so the next edit (or the Save button) retries.
  useEffect(() => {
    if (!dirty || !picked) return;
    const t = setTimeout(() => void save(true), 1500);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dirty, groups, picked]);

  // Actor fix: persists into the players file server-side, re-crops and
  // re-embeds the event, then refreshes everything derived from embeddings.
  const fixActor = async (eventId: string, fix: ActorFix) => {
    try {
      await apiFetch(API.reid.actorFix(picked), { method: 'POST', body: { event_id: eventId, ...fix } });
      toast.success(fix.none ? 'Marked as no actor' : fix.box ? 'Actor updated' : 'Reverted to the auto pick');
      await Promise.all([
        qc.invalidateQueries({ queryKey: ['reid-results', picked] }),
        qc.invalidateQueries({ queryKey: ['reid-clusters', picked] }),
        qc.invalidateQueries({ queryKey: ['reid-players', picked] }),
      ]);
    } catch (e) {
      toast.error(`Actor fix failed: ${errMsg(e)}`);
    }
  };

  /** Seeded regroup: every locked/named row anchors a player; the backend
   *  assigns all remaining events to the nearest seed centroid (within the
   *  current threshold) and clusters what's left over into fresh pools. */
  const seedRegroup = async () => {
    const anchors = groups.filter((g) => g.locked || g.name.trim());
    const seeds = anchors.filter((g) => g.eventIds.length > 0);
    if (!seeds.length) {
      toast.warning('Lock or name at least one non-empty group to use as a seed');
      return;
    }
    try {
      const res = await apiFetch<{ groups: Record<string, string[]>; leftover_clusters: string[][] }>(
        API.reid.seedCluster(picked),
        { method: 'POST', body: { seeds: Object.fromEntries(seeds.map((g) => [g.key, g.eventIds])), threshold, model: embedder } },
      );
      const assigned = Object.values(res.groups).reduce((s, a) => s + a.length, 0);
      setGroups(() => {
        // Anchors absorb their assignments (locked so nothing re-clusters
        // them away); leftovers become fresh unlocked pools.
        const out = anchors.map((g) => ({ ...g, locked: true, eventIds: [...g.eventIds, ...(res.groups[g.key] ?? [])] }));
        const tiny: string[] = [];
        res.leftover_clusters.forEach((ids, i) => {
          if (ids.length < MIN_CLUSTER_SIZE) tiny.push(...ids);
          else out.push({ key: `seed:${embedder}:${threshold}:${i}`, name: '', eventIds: ids, locked: false });
        });
        if (tiny.length) out.push({ key: `pool:seed:${embedder}:${threshold}`, name: '', eventIds: tiny, locked: false });
        return out;
      });
      setDirty(true);
      toast.success(`Assigned ${assigned} event(s) to ${seeds.length} seeded group(s) · ${res.leftover_clusters.length} leftover pool(s)`);
    } catch (e) {
      toast.error(`Seed regroup failed: ${errMsg(e)}`);
    }
  };

  const rebuildFromSaved = () => {
    setGroups([]); // dropping every lock lets the effect rebuild from saved state
    setDirty(false);
    void qc.invalidateQueries({ queryKey: ['reid-clusters', picked] });
  };

  const reset = async () => {
    if (dirty) {
      const ok = await confirm({
        title: 'Discard unsaved changes?',
        body: 'Group edits since the last save (including locks) will be lost.',
        confirmText: 'Discard',
        variant: 'danger',
      });
      if (!ok) return;
    }
    rebuildFromSaved();
  };

  const pickVideo = async (name: string) => {
    if (dirty && name !== picked) {
      const ok = await confirm({
        title: 'Discard unsaved changes?',
        body: 'The current group edits have not been saved.',
        confirmText: 'Discard',
        variant: 'danger',
      });
      if (!ok) return;
    }
    setGroups([]);
    setDirty(false);
    setSelectedCrops(new Set());
    setSelectedRally('all');
    setPicked(name);
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
    onDoubleClick: () => seekToEvent(r),
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
        src={apiUrl(API.reid.crop(picked, r.crop))}
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

  // Actions somebody actually performs on camera: score marks where the ball
  // lands and non-visible events happen off-frame — neither has anyone to
  // identify. Falls back to the extraction records (which already skip both)
  // when no annotation is loaded.
  const actionableCount = actionEvents.length
    ? actionEvents.filter((a) => a.visible && a.label !== 'score').length
    : records.length;
  // Events that already carry an identity: member of a named group on the
  // live board (unsaved edits count — the board is the source of truth).
  const assignedCount = new Set(groups.filter((g) => g.name.trim()).flatMap((g) => g.eventIds)).size;


  return (
    <div className="mx-auto max-w-screen-2xl space-y-5">
      {/* Picker — same shape as the Action Label / Rally Label pickers */}
      <Card>
        <div className="grid grid-cols-1 items-end gap-3 lg:grid-cols-[8.5rem_8.5rem_minmax(18rem,1fr)]">
          <FieldLabel label="Kind">
            <select value={kindFilter} onChange={(e) => setKindFilter(e.target.value as typeof kindFilter)} className={cn(fieldCls, 'h-9 w-full py-0')}>
              <option value="all">All kinds</option>
              <option value="broadcast">Broadcast</option>
              <option value="sideline">Sideline</option>
            </select>
          </FieldLabel>
          <FieldLabel label="Status">
            <select value={pickStatus} onChange={(e) => setPickStatus(e.target.value as typeof pickStatus)} className={cn(fieldCls, 'h-9 w-full py-0')}>
              <option value="all">All</option>
              <option value="unlabeled">Unlabeled</option>
              <option value="labeled">Labeled</option>
            </select>
          </FieldLabel>
          <FieldLabel label="Video">
            <VideoCombobox
              items={pickable}
              value={picked}
              onChange={pickVideo}
              placeholder={`Search ${pickable.length} extracted videos…`}
              renderItem={(v) => (
                <>
                  <KindBadge kind={v.kind} />
                  <span className="min-w-0 flex-1 break-all font-mono">{v.name}</span>
                  <span className="shrink-0 font-mono text-[10px] tabular-nums text-text-muted">{v.event_count}ev</span>
                  {(v.player_count ?? 0) > 0 && <Badge tone="brand">{v.player_count}P</Badge>}
                </>
              )}
            />
          </FieldLabel>
        </div>
      </Card>

      {picked && showVideo && meta.fps && meta.frame_size && (
        <ReidVideoPlayer
          ref={playerRef}
          src={apiUrl(API.actionAnnotate.video(picked))}
          fps={meta.fps}
          frameSize={meta.frame_size}
          records={records}
          actionEvents={actionEvents}
          matches={matches}
          rallies={meta.rallies ?? []}
          selectedRally={selectedRally}
          onSelectRally={setSelectedRally}
          onFixActor={fixActor}
          onJumpToCrop={jumpToCrop}
        />
      )}

      <Card>
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-3">
            <SectionLabel className="mb-0">Identities</SectionLabel>
            {picked && (
              <span
                className="font-mono text-[11px] tabular-nums text-text-muted"
                title="Assigned to a player / actions with someone to identify (score and non-visible events excluded)"
              >
                <span className={assignedCount >= actionableCount ? 'text-primary-light' : undefined}>{assignedCount}</span>/{actionableCount} actions
              </span>
            )}
            <div
              className="inline-flex items-center gap-1.5"
              title="Where locked groups dock: pinned on top as full rows, or in a compact right rail (3 crops each)"
            >
              <svg className="h-3.5 w-3.5 text-text-muted" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M16.5 10.5V6.75a4.5 4.5 0 10-9 0v3.75m-.75 11.25h10.5a2.25 2.25 0 002.25-2.25v-6.75a2.25 2.25 0 00-2.25-2.25H6.75a2.25 2.25 0 00-2.25 2.25v6.75a2.25 2.25 0 002.25 2.25z" />
              </svg>
              <div className="inline-flex rounded-lg border border-border bg-surface-50 p-0.5">
                {(['top', 'right'] as const).map((pos) => (
                  <button
                    key={pos}
                    type="button"
                    onClick={() => setLockedDock(pos)}
                    className={cn(
                      'rounded-md px-3 py-1 text-xs font-medium capitalize transition-colors',
                      lockedDock === pos ? 'bg-primary text-on-primary' : 'text-text-secondary hover:bg-ink/[0.04]',
                    )}
                  >
                    {pos}
                  </button>
                ))}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <label className="inline-flex cursor-pointer items-center gap-1.5 text-xs text-text-secondary">
              <input
                type="checkbox"
                checked={showVideo}
                onChange={(e) => setShowVideo(e.target.checked)}
                className="h-3.5 w-3.5 accent-primary"
              />
              Video
            </label>
            <label className="inline-flex cursor-pointer items-center gap-1.5 text-xs text-text-secondary">
              <input
                type="checkbox"
                checked={showSkeleton}
                onChange={(e) => setShowSkeleton(e.target.checked)}
                className="h-3.5 w-3.5 accent-primary"
              />
              Skeleton
            </label>
            <select
              value={embedder}
              onChange={(e) => {
                const m = e.target.value as Embedder;
                setEmbedder(m);
                // Distance scales differ per model — jump to its default.
                setThresholdDraft(EMBEDDER_THRESHOLDS[m].def);
                setThreshold(EMBEDDER_THRESHOLDS[m].def);
              }}
              className={selectCls}
              title="Appearance embedding model — compare how each one groups the players"
            >
              {EMBEDDERS.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
            <label
              className="inline-flex items-center gap-1.5 text-xs text-text-secondary"
              title="Cluster threshold for unassigned events — lower splits, higher merges. Locked rows are unaffected."
            >
              <span className="whitespace-nowrap">
                threshold <span className="font-mono tabular-nums">{thresholdDraft.toFixed(2)}</span>
              </span>
              <input
                type="range"
                min={EMBEDDER_THRESHOLDS[embedder].min}
                max={EMBEDDER_THRESHOLDS[embedder].max}
                step={0.01}
                value={thresholdDraft}
                onChange={(e) => setThresholdDraft(Number(e.target.value))}
                onPointerUp={(e) => e.currentTarget.blur()}
                className="h-1 w-28 cursor-pointer accent-primary"
              />
            </label>
            <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value as typeof statusFilter)} className={selectCls}>
              <option value="all">All statuses</option>
              <option value="ok">ok</option>
              <option value="multi">multi</option>
              <option value="miss">miss</option>
            </select>
            <Button
              size="sm"
              onClick={seedRegroup}
              disabled={!picked || !groups.some((g) => (g.locked || g.name.trim()) && g.eventIds.length > 0)}
              title="Use every locked/named group as a player anchor: all other events join the nearest anchor (within the threshold); the rest re-cluster into leftover pools"
            >
              Seed regroup
            </Button>
            <Button size="sm" onClick={reset} disabled={!dirty}>
              Reset
            </Button>
            <Button size="sm" intent="primary" onClick={() => void save()} disabled={!picked}>
              {dirty ? 'Save •' : 'Save'}
            </Button>
          </div>
        </div>

        {!picked ? (
          <EmptyState
            icon={
              <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.5 20.25a8.25 8.25 0 0115 0" />
              </svg>
            }
            title="Pick an extracted video"
            subtitle="Drag crops between players to fix identities"
          />
        ) : clustersQuery.isPending || resultsQuery.isPending ? (
          <div className="py-8 text-center text-xs text-text-muted">Clustering…</div>
        ) : (
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
                        onClick={() => toggleLock(g.key)}
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
                        <input
                          value={g.name}
                          onChange={(e) => {
                            const name = e.target.value;
                            // Renaming is an edit too — lock so the name sticks.
                            setGroups((prev) => prev.map((x) => (x.key === g.key ? { ...x, name, locked: true } : x)));
                            setDirty(true);
                          }}
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
                      onDragOver={(e) => {
                        e.preventDefault();
                        setDragOver((prev) => (prev?.key === g.key && prev.mode === 'merge' ? prev : { key: g.key, mode: 'merge' }));
                      }}
                      onDragLeave={() => setDragOver((prev) => (prev?.key === g.key ? null : prev))}
                      onDrop={onDropTo(g.key)}
                      className={cn(
                        'rounded-xl border bg-surface-50 p-2 transition-colors',
                        dragOver?.key === g.key ? 'border-primary/60 bg-primary/5' : 'border-border',
                      )}
                    >
                      <div className="mb-1.5 flex items-center gap-1.5">
                        <button
                          type="button"
                          onClick={() => toggleLock(g.key)}
                          title="Locked — click to unlock (moves it back onto the board)"
                          className="flex-shrink-0 text-primary-light"
                        >
                          <svg className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" d="M16.5 10.5V6.75a4.5 4.5 0 10-9 0v3.75m-.75 11.25h10.5a2.25 2.25 0 002.25-2.25v-6.75a2.25 2.25 0 00-2.25-2.25H6.75a2.25 2.25 0 00-2.25 2.25v6.75a2.25 2.25 0 002.25 2.25z" />
                          </svg>
                        </button>
                        <input
                          value={g.name}
                          onChange={(e) => {
                            const name = e.target.value;
                            setGroups((prev) => prev.map((x) => (x.key === g.key ? { ...x, name, locked: true } : x)));
                            setDirty(true);
                          }}
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
        )}
        {/* Association stats */}
        {picked && records.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2 text-[11px]">
            <Badge tone="success">ok {records.filter((r) => r.status === 'ok').length}</Badge>
            <Badge tone="warning">multi {records.filter((r) => r.status === 'multi').length}</Badge>
            <Badge tone="danger">miss {records.filter((r) => r.status === 'miss').length}</Badge>
            {(playersQuery.data?.players ?? []).map((p) => (
              <Badge key={p} tone="brand">
                {p} {Object.values(matches).filter((m) => m.player === p).length}
              </Badge>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}
