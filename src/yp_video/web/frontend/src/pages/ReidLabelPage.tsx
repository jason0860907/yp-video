import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
  type DragEvent,
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

// CLIP-ReID's ViT features sit in a tight cosine cone, hence the small values.
const THRESHOLDS = [0.12, 0.15, 0.18, 0.21, 0.24];
const DEFAULT_THRESHOLD = 0.15;
// Below this cosine similarity a predicted identity is rendered as doubtful.
const LOW_SIM = 0.6;

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
  seek: (frame: number) => void;
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
}

/** Read-only twin of the Action Label event panel: action dot + label,
 *  matched player, frame and time — click a row to park the video there. */
function ReidEventPanel({ entries, empty, matches, selectedEventId, fps, playheadFrame, onJump }: ReidEventPanelProps) {
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
            <span className={cn('truncate text-right text-[11px]', m?.assigned ? 'text-primary-light' : 'text-text-muted')}>{m?.player ?? ''}</span>
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
  { src, fps, frameSize, records, actionEvents, matches, rallies, selectedRally, onSelectRally, onFixActor },
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
  useEffect(() => {
    setExpanded(null);
    setSelectedEventId(null);
    setPickMode(false);
  }, [src]);
  // Read inside the frame-clock callback without re-arming it.
  const rallyEndRef = useRef<number | null>(null);
  rallyEndRef.current =
    selectedRally === 'all' ? null : rallies.find((r) => r.rally_id === selectedRally)?.end ?? null;

  const togglePlay = () => {
    const el = videoRef.current;
    if (!el) return;
    if (el.paused) void el.play();
    else el.pause();
  };

  useImperativeHandle(
    ref,
    () => ({
      seek: (f: number) => {
        const el = videoRef.current;
        if (!el) return;
        el.pause();
        el.currentTime = (f + 0.5) / fps;
        el.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      },
    }),
    [fps],
  );

  // Frame clock via requestVideoFrameCallback — same approach as the Action
  // Label editor, re-armed per presented frame.
  useEffect(() => {
    const el = videoRef.current;
    if (!el) return;
    let alive = true;
    let id = 0;
    const tick = (_now: number, meta: { mediaTime: number }) => {
      if (!alive) return;
      setFrame(Math.round(meta.mediaTime * fps));
      // With a rally selected, playback stops at its end (Action Label rule).
      if (rallyEndRef.current != null && meta.mediaTime >= rallyEndRef.current && !el.paused) el.pause();
      id = el.requestVideoFrameCallback(tick);
    };
    id = el.requestVideoFrameCallback(tick);
    const onSeeked = () => setFrame(Math.round(el.currentTime * fps));
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

  const seekEvent = (a: SidebarAction) => {
    setSelectedEventId(a.id);
    const el = videoRef.current;
    if (!el) return;
    el.pause();
    el.currentTime = (a.frame + 0.5) / fps;
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
              {/* Actor picker: every detected person as a clickable outline */}
              {pickMode &&
                pickTarget?.detections?.map((d, i) => {
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
                      strokeOpacity={0.9}
                      strokeWidth={1.5}
                      strokeDasharray="6 4"
                      vectorEffect="non-scaling-stroke"
                      className="pointer-events-auto cursor-pointer hover:fill-white/20"
                      onClick={(e) => {
                        e.stopPropagation();
                        onFixActor(pickTarget.id, { box: d.box });
                      }}
                    >
                      <title>{`person ${(d.score * 100).toFixed(0)}% — click to set as the actor`}</title>
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
                  <span className="ml-auto flex items-center gap-2">
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
                <span className="text-text-muted">Jump to an action first (click a crop or a sidebar row), then click the right person.</span>
              )}
            </div>
          )}
          <p className="mt-3 text-[11px] text-text-muted">
            Boxes appear within ±½ s of each event and sharpen on the exact frame — click any crop below to jump there.
            Wrong person boxed? Toggle <span className="text-text-secondary">Pick actor</span> and click the right one (✎ marks fixed events).
          </p>
        </Card>
      </div>

      {/* Rally list — same sidebar as the Rally Label / Action Label editors */}
      {rallies.length > 0 && (
        <div className="min-w-0 lg:w-[420px] lg:flex-shrink-0">
          <Card>
            <SectionLabel>
              Rallies ({rallies.length} rally · {sidebarActions.length} action)
            </SectionLabel>
            <div className="vq-list max-h-[calc(45vh+2.25rem)] space-y-1.5 overflow-y-auto pr-1">
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
                      <ReidEventPanel entries={entries} empty="No actions in this rally" matches={matches} selectedEventId={selectedEventId} fps={fps} playheadFrame={frame} onJump={seekEvent} />
                    )}
                  </div>
                );
              })}
              {outside.length > 0 && (
                <div className="space-y-1.5">
                  <div
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
                    <ReidEventPanel entries={outside} empty="No outside actions" matches={matches} selectedEventId={selectedEventId} fps={fps} playheadFrame={frame} onJump={seekEvent} />
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
  const [view, setView] = useState<'groups' | 'crops'>('groups');
  const [threshold, setThreshold] = useState<number>(DEFAULT_THRESHOLD);
  const [showSkeleton, setShowSkeleton] = useState(true);
  const [showVideo, setShowVideo] = useState(true);
  const playerRef = useRef<PlayerHandle>(null);
  const [statusFilter, setStatusFilter] = useState<'all' | ReidRecord['status']>('all');
  const [groups, setGroups] = useState<Group[]>([]);
  const [dirty, setDirty] = useState(false);
  const [dragOver, setDragOver] = useState<string | null>(null);
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
    if (r) playerRef.current?.seek(r.frame);
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
    queryKey: ['reid-clusters', picked, threshold],
    queryFn: () => apiFetch<{ clusters: ReidCluster[] }>(API.reid.clusters(picked, threshold)),
    enabled: Boolean(picked),
  });
  const clusters = useMemo(() => clustersQuery.data?.clusters ?? [], [clustersQuery.data]);

  const playersQuery = useQuery({
    queryKey: ['reid-players', picked],
    queryFn: () => apiFetch<ReidPlayers>(API.reid.players(picked)),
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
          out.push({ key: `p:${name}`, name, eventIds: ids, locked: false });
        }
        ids.forEach((id) => covered.add(id));
      }

      for (const c of clusters) {
        const rest = c.event_ids.filter((id) => !covered.has(id));
        if (rest.length) out.push({ key: `c:${threshold}:${c.id}`, name: '', eventIds: rest, locked: false });
      }
      return out;
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [picked, assignments, clusters, recordById]);

  const moveEvent = (eventId: string, fromKey: string, toKey: string) => {
    if (fromKey === toKey) return;
    setGroups((prev) =>
      prev
        .map((g) => {
          if (g.key === fromKey) return { ...g, eventIds: g.eventIds.filter((i) => i !== eventId) };
          // Receiving an edit locks the row so re-clustering can't undo it.
          if (g.key === toKey) return { ...g, eventIds: [...g.eventIds, eventId], locked: true };
          return g;
        })
        // An emptied auto-cluster group is noise; an emptied named player stays.
        .filter((g) => g.eventIds.length > 0 || g.name.trim()),
    );
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

  const toggleLock = (key: string) =>
    setGroups((prev) => prev.map((g) => (g.key === key ? { ...g, locked: !g.locked } : g)));

  const onDropTo = (toKey: string) => (e: DragEvent) => {
    e.preventDefault();
    setDragOver(null);
    const parts = e.dataTransfer.getData('text/plain').split('\n');
    if (parts[0] === 'group') {
      const fromKey = parts[1];
      if (fromKey && toKey !== '__new__') mergeGroups(fromKey, toKey);
      return;
    }
    const [kind, eventId, fromKey] = parts;
    if (kind !== 'event' || !eventId || !fromKey) return;
    if (toKey === '__new__') {
      const key = `n:${newGroupSeq.current++}`;
      setGroups((prev) => [...prev, { key, name: '', eventIds: [], locked: true }]);
      moveEvent(eventId, fromKey, key);
    } else {
      moveEvent(eventId, fromKey, toKey);
    }
  };

  const save = async () => {
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
      await apiFetch(API.reid.players(picked), { method: 'PUT', body: { assignments: next } });
      setGroups(named); // show the placeholders the save just minted
      setDirty(false);
      await qc.invalidateQueries({ queryKey: ['reid-players', picked] });
      toast.success(`Saved ${new Set(Object.values(next)).size} player(s), ${Object.keys(next).length} events`);
    } catch (e) {
      toast.error(`Save failed: ${errMsg(e)}`);
    }
  };

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
    setSelectedRally('all');
    setPicked(name);
  };

  const playerCaption = (r: ReidRecord) => {
    const m = matches[r.id];
    if (!m) return null;
    return (
      <span
        className={cn(
          'truncate font-medium',
          m.assigned ? 'text-primary-light' : m.sim >= LOW_SIM ? 'text-text-secondary' : 'text-text-muted line-through',
        )}
        title={m.assigned ? `${m.player} (labeled)` : `${m.player} · sim ${m.sim.toFixed(2)}`}
      >
        {m.player}
      </span>
    );
  };

  const rallySpan = selectedRally === 'all' ? null : (meta.rallies ?? []).find((r) => r.rally_id === selectedRally);
  const shown = records.filter((r) => {
    if (statusFilter !== 'all' && r.status !== statusFilter) return false;
    if (rallySpan) {
      const t = r.time != null ? r.time : meta.fps ? r.frame / meta.fps : null;
      if (t == null || t < rallySpan.start || t > rallySpan.end) return false;
    }
    return true;
  });
  const namedCount = groups.filter((g) => g.name.trim()).length;

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
        />
      )}

      <Card>
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-3">
            <SectionLabel className="mb-0">Identities</SectionLabel>
            <div className="inline-flex rounded-lg border border-border bg-surface-50 p-0.5">
              {(['groups', 'crops'] as const).map((v) => (
                <button
                  key={v}
                  type="button"
                  onClick={() => setView(v)}
                  className={cn(
                    'rounded-md px-3 py-1 text-xs font-medium capitalize transition-colors',
                    view === v ? 'bg-primary text-on-primary' : 'text-text-secondary hover:bg-ink/[0.04]',
                  )}
                >
                  {v}
                </button>
              ))}
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
            {view === 'groups' ? (
              <select value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} className={selectCls} title="Cluster threshold for unassigned events">
                {THRESHOLDS.map((t) => (
                  <option key={t} value={t}>
                    threshold {t}
                  </option>
                ))}
              </select>
            ) : (
              <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value as typeof statusFilter)} className={selectCls}>
                <option value="all">All statuses</option>
                <option value="ok">ok</option>
                <option value="multi">multi</option>
                <option value="miss">miss</option>
              </select>
            )}
            <Button size="sm" onClick={reset} disabled={!dirty}>
              Reset
            </Button>
            <Button size="sm" intent="primary" onClick={save} disabled={!picked}>
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
            subtitle="Groups: drag crops between players to fix identities. Crops: review every event's match"
          />
        ) : view === 'groups' ? (
          clustersQuery.isPending || resultsQuery.isPending ? (
            <div className="py-8 text-center text-xs text-text-muted">Clustering…</div>
          ) : (
            <div className="space-y-3">
              <p className="text-xs text-text-muted">
                {namedCount} named player(s) · drag a crop onto another group to reassign it, or drag a row's
                number handle onto another row to merge them (target name wins). Edited rows lock themselves
                (🔒) and survive threshold changes; unlocked rows re-cluster freely. Save persists named rows
                — locked rows without a name get a placeholder (P1, P2, …) you can rename later.
              </p>
              <div className="max-h-[70vh] space-y-2 overflow-auto pr-1">
                {groups.map((g, i) => (
                  <div
                    key={g.key}
                    onDragOver={(e) => {
                      e.preventDefault();
                      setDragOver(g.key);
                    }}
                    onDragLeave={() => setDragOver((k) => (k === g.key ? null : k))}
                    onDrop={onDropTo(g.key)}
                    className={cn(
                      'rounded-xl border bg-surface-50 p-2.5 transition-colors',
                      dragOver === g.key ? 'border-primary/60 bg-primary/5' : 'border-border',
                    )}
                  >
                    <div className="mb-2 flex items-center gap-3">
                      <span
                        draggable
                        onDragStart={(e) => e.dataTransfer.setData('text/plain', `group\n${g.key}`)}
                        title="Drag onto another row to merge this whole group into it"
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
                      <span className="font-mono text-[11px] tabular-nums text-text-muted">{g.eventIds.length} events</span>
                      {!g.name.trim() && <Badge tone="neutral">unassigned</Badge>}
                    </div>
                    <div className="flex min-h-[7rem] flex-wrap items-start gap-1.5">
                      {g.eventIds.map((id) => {
                        const r = recordById.get(id);
                        if (!r?.crop) return null;
                        return (
                          <CropImage
                            key={id}
                            src={apiUrl(API.reid.crop(picked, r.crop))}
                            keypoints={r.keypoints}
                            skeleton={showSkeleton}
                            alt={id}
                            draggable
                            onDragStart={(e) => e.dataTransfer.setData('text/plain', `event\n${id}\n${g.key}`)}
                            onClick={() => seekToEvent(r)}
                            title={`${r.label} f${r.frame} — drag to move, click to jump the video there`}
                            className="h-28 w-auto cursor-grab rounded-md border border-border active:cursor-grabbing"
                          />
                        );
                      })}
                    </div>
                  </div>
                ))}
                {/* Drop here to split a crop into a brand-new group */}
                <div
                  onDragOver={(e) => {
                    e.preventDefault();
                    setDragOver('__new__');
                  }}
                  onDragLeave={() => setDragOver((k) => (k === '__new__' ? null : k))}
                  onDrop={onDropTo('__new__')}
                  className={cn(
                    'flex items-center justify-center rounded-xl border border-dashed py-6 text-xs transition-colors',
                    dragOver === '__new__' ? 'border-primary/60 bg-primary/5 text-primary-light' : 'border-border text-text-muted',
                  )}
                >
                  Drop a crop here to start a new group
                </div>
              </div>
            </div>
          )
        ) : resultsQuery.isPending ? (
          <div className="py-8 text-center text-xs text-text-muted">Loading…</div>
        ) : (
          <>
            <div className="grid max-h-[70vh] grid-cols-3 gap-2 overflow-auto pr-1 sm:grid-cols-5 lg:grid-cols-8">
              {shown.map((r) => (
                <figure
                  key={r.id}
                  onClick={() => seekToEvent(r)}
                  title="Click to jump the video to this event"
                  className="cursor-pointer overflow-hidden rounded-lg border border-border bg-surface-50"
                >
                  {r.crop ? (
                    <CropImage
                      src={apiUrl(API.reid.crop(picked, r.crop))}
                      keypoints={r.keypoints}
                      skeleton={showSkeleton}
                      alt={`${r.label} f${r.frame}`}
                      className="w-full"
                    />
                  ) : (
                    <div className="flex aspect-[1/2] w-full items-center justify-center text-[10px] text-text-muted">no crop</div>
                  )}
                  <figcaption className="space-y-0.5 px-1.5 py-1 text-[9.5px]">
                    <div className="flex items-center justify-between gap-1">
                      <span className="truncate font-mono text-text-secondary">
                        {r.label} f{r.frame}
                      </span>
                      <span className={cn('h-1.5 w-1.5 flex-shrink-0 rounded-full', STATUS_DOT[r.status])} title={`${r.status} · ${r.candidates} candidate(s)`} />
                    </div>
                    {playerCaption(r)}
                  </figcaption>
                </figure>
              ))}
              {shown.length === 0 && <div className="col-span-full py-8 text-center text-xs text-text-muted">No events match</div>}
            </div>
            {records.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-2 text-[11px]">
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
          </>
        )}
      </Card>
    </div>
  );
}
