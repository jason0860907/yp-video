import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { API, apiFetch, apiPostBlob, apiUrl, errMsg } from '@/lib/api';
import { cn } from '@/lib/cn';
import { formatTime, formatTimePrecise, parseTime } from '@/lib/format';
import { copyText, downloadBlob } from '@/lib/download';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { EmptyState } from '@/components/ui/EmptyState';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { toast } from '@/components/feedback/toast';
import { useVideoRecovery } from '@/lib/useVideoRecovery';
import { DownloadClipsModal } from './DownloadClipsModal';
import { RallyTimeline } from './RallyTimeline';

/** One keyboard key, rendered as a keycap so it reads apart from prose. */
function Kbd({ children }: { children: ReactNode }) {
  return (
    <kbd className="inline-block min-w-[1.1rem] rounded border border-border bg-surface-200/60 px-1 py-px text-center font-mono text-[10px] not-italic text-text-secondary">
      {children}
    </kbd>
  );
}

//: The boundary nudge steps offered per field, smallest useful to largest.
const NUDGE_STEPS: { label: string; delta: number }[] = [
  { label: '−1s', delta: -1 },
  { label: '−0.1s', delta: -0.1 },
  { label: '+0.1s', delta: 0.1 },
  { label: '+1s', delta: 1 },
];
//: [minus key, plus key] per field, for the button tooltips.
const NUDGE_KEYS = { start: ['a', 's'], end: ['d', 'f'] } as const;

/** A time input that commits on blur/Enter (Escape reverts).
 *
 *  Editing holds a draft string: a controlled input that re-parses and
 *  re-formats every keystroke eats partial values — "1:" becomes 01:00
 *  before the seconds can be typed. Rally boundaries live on a 0.1 s grid
 *  (the nudges step by 0.1), so display and draft both use one decimal. */
function TimeField({ seconds, onCommit }: { seconds: number; onCommit: (value: number) => void }) {
  const [draft, setDraft] = useState<string | null>(null);
  return (
    <input
      value={draft ?? formatTimePrecise(seconds, 1)}
      onFocus={() => setDraft(formatTimePrecise(seconds, 1))}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={() => {
        if (draft !== null && draft.trim() !== '') onCommit(parseTime(draft));
        setDraft(null);
      }}
      onKeyDown={(e) => {
        if (e.key === 'Enter') e.currentTarget.blur();
        else if (e.key === 'Escape') setDraft(null);
      }}
      title="mm:ss(.s) or bare seconds — Enter/blur applies, Escape reverts"
      className="w-[4.2rem] border-0 border-b border-ink/10 bg-transparent text-center font-heading text-[11px] tabular-nums text-text-primary focus:border-primary-light focus:outline-none focus:ring-0"
    />
  );
}

export type RallySide = 'left' | 'right' | 'near' | 'far';
const RALLY_SIDES: RallySide[] = ['left', 'right', 'near', 'far'];
const SIDE_DISPLAY: Record<RallySide, string> = { left: '左', right: '右', near: '近', far: '遠' };

/** Which pair of sides this video's camera angle offers. One axis per video —
 *  picking it once up top keeps each row to two buttons instead of four. */
type SideAxis = 'lr' | 'nf';
const AXIS_SIDES: Record<SideAxis, RallySide[]> = { lr: ['left', 'right'], nf: ['near', 'far'] };
const AXIS_DISPLAY: Record<SideAxis, string> = { lr: '左/右', nf: '遠/近' };

export interface EditorAnnotation {
  rally_id: number | null;
  start: number;
  end: number;
  label: string;
  /** Court side that won the rally (camera-frame); null = not annotated. */
  side: RallySide | null;
  score?: number | null;
}
export interface EditorData {
  video?: string;
  /** Which store the file was loaded from (e.g. rally-spot-pre-annotations). */
  source?: string;
  source_video?: string;
  metadata?: { video?: string };
  results?: Array<Record<string, unknown>>;
}

interface AnnotationEditorProps {
  data: EditorData | null;
  saveEndpoint: string;
  videoStreamPath: (videoPath: string) => string;
  previewBackoff?: number;
  /** Runs after an EXPLICIT save only. Autosave fires every couple of
   *  seconds while editing, and this hook is where the page pushes the whole
   *  match to the app — per keystroke that hammered a slow tunnel with full
   *  publishes (and surfaced as Cloudflare 524s mid-labeling). */
  onSaved?: (videoName: string) => Promise<void> | void;
  /** Where to start playback once metadata loads (null/0 = from the top).
   *  Read at load time, so the parent can hand over a position captured
   *  elsewhere (e.g. another tab's player). */
  initialTime?: () => number | null;
  /** Fired with the playhead position on every timeupdate (plays and seeks). */
  onTimeChange?: (t: number) => void;
  /** The Download-Clips modal, controlled by the page: the button that opens
   *  it lives up in the picker row (next to Done), but the rally list the
   *  modal needs lives here. */
  clipsOpen: boolean;
  onClipsClose: () => void;
}

const normalizeRallyId = (v: unknown): number | null => {
  const n = Number(v);
  return Number.isInteger(n) && n > 0 ? n : null;
};
const normalizeSide = (v: unknown): RallySide | null =>
  RALLY_SIDES.includes(v as RallySide) ? (v as RallySide) : null;
const num = (...vals: unknown[]): number => {
  for (const v of vals) if (typeof v === 'number' && Number.isFinite(v)) return v;
  return 0;
};

// No localStorage drafts, on purpose: a restored draft paints stale (or
// merely unsaved) state over what the server holds and makes the screen lie
// about what is on disk — that is precisely how saved labels once "reverted".
// The server is the only store; the 2 s autosave plus the pagehide flush
// below keep the window between screen and disk negligible.
const AUTOSAVE_MS = 2000;

export function AnnotationEditor({ data, saveEndpoint, videoStreamPath, previewBackoff = 3, onSaved, initialTime, onTimeChange, clipsOpen, onClipsClose }: AnnotationEditorProps) {
  const videoRef = useRef<HTMLVideoElement>(null);

  const [annotations, setAnnotations] = useState<EditorAnnotation[]>([]);
  const [videoName, setVideoName] = useState('');
  const [duration, setDuration] = useState(0);
  const [markStart, setMarkStart] = useState<number | null>(null);
  const [selectedIdx, setSelectedIdx] = useState(-1);
  const [dirty, setDirty] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [shift, setShift] = useState('0');
  const [saving, setSaving] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [sideAxis, setSideAxis] = useState<SideAxis>('lr');

  const togglePlay = () => {
    const el = videoRef.current;
    if (!el) return;
    if (el.paused) void el.play();
    else el.pause();
  };

  // Presigned video URLs expire and range requests can hang; reload the src
  // (which fetches a fresh URL) and seek back to where the user was.
  useVideoRecovery(videoRef, {
    src: () => (videoName ? videoStreamPath(videoName) : ''),
    onRecover: () => toast.info('影片串流中斷，已自動重新載入'),
    onGiveUp: () => toast.error('影片重載後仍卡在同一處，已停止自動重試 — 請把 DevTools Console 的 [video-recovery] 記錄回報'),
  });

  // ── Load a file ──
  useEffect(() => {
    if (!data) return;
    const path = data.video || data.source_video || data.metadata?.video || '';
    setVideoName(path);
    const fromServer: EditorAnnotation[] = (data.results ?? [])
      .map((r) => ({
        rally_id: normalizeRallyId(r.rally_id),
        start: num(r.start, r.start_time, (r.segment as number[] | undefined)?.[0]),
        end: num(r.end, r.end_time, (r.segment as number[] | undefined)?.[1]),
        label: 'rally',
        side: normalizeSide(r.side),
        score: (r.confidence ?? r.score ?? null) as number | null,
      }))
      .sort((a, b) => a.start - b.start);
    setAnnotations(fromServer);
    const firstSide = fromServer.find((a) => a.side)?.side;
    if (firstSide) setSideAxis(AXIS_SIDES.nf.includes(firstSide) ? 'nf' : 'lr');
    setDirty(false);
    setSelectedIdx(-1);
    setMarkStart(null);
    const el = videoRef.current;
    if (path && el) {
      el.pause();
      el.removeAttribute('src');
      el.load();
      el.src = videoStreamPath(path);
      el.load();
    }
  }, [data, videoStreamPath, saveEndpoint]);

  const addAnnotation = useCallback(() => {
    const el = videoRef.current;
    setMarkStart((ms) => {
      if (ms == null) {
        toast.warning('Mark start first with [');
        return ms;
      }
      const end = el?.currentTime ?? 0;
      if (end <= ms) {
        toast.warning('End must be after start');
        return ms;
      }
      setAnnotations((prev) => [...prev, { rally_id: null, start: ms, end, label: 'rally', side: null }].sort((a, b) => a.start - b.start));
      setDirty(true);
      return null;
    });
  }, []);

  const doMarkStart = useCallback(() => {
    const el = videoRef.current;
    if (!el?.src) return;
    setMarkStart(el.currentTime);
  }, []);

  const seekTo = (t: number, play = true) => {
    const el = videoRef.current;
    if (!el) return;
    el.currentTime = t;
    if (play) void el.play();
  };

  // Nudge the selected rally's boundary and park the video on it, paused —
  // the frame on screen IS the boundary being placed.
  const nudge = useCallback((field: 'start' | 'end', delta: number) => {
    setSelectedIdx((idx) => {
      if (idx < 0) {
        toast.warning('Select a rally first');
        return idx;
      }
      setAnnotations((prev) => {
        const a = prev[idx];
        if (!a) return prev;
        const value = Math.max(0, Math.round((a[field] + delta) * 1000) / 1000);
        if (field === 'start' ? value >= a.end : value <= a.start) {
          toast.warning('Start must stay before end');
          return prev;
        }
        const el = videoRef.current;
        if (el) {
          el.pause();
          el.currentTime = value;
        }
        setDirty(true);
        return prev.map((row, i) => (i === idx ? { ...row, [field]: value } : row));
      });
      return idx;
    });
  }, []);

  // ── Keyboard ──
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
      const el = videoRef.current;
      if (!el) return;
      switch (e.key) {
        case ' ':
          e.preventDefault();
          if (el.paused) void el.play();
          else el.pause();
          break;
        case '[':
          e.preventDefault();
          doMarkStart();
          break;
        case ']':
        case 'Enter':
          e.preventDefault();
          addAnnotation();
          break;
        case 'ArrowLeft':
          e.preventDefault();
          el.currentTime = Math.max(0, el.currentTime - 5);
          break;
        case 'ArrowRight':
          e.preventDefault();
          el.currentTime += 5;
          break;
        case 'Delete':
        case 'Backspace':
          setSelectedIdx((idx) => {
            if (idx >= 0) {
              setAnnotations((prev) => prev.filter((_, i) => i !== idx));
              setDirty(true);
              return -1;
            }
            return idx;
          });
          break;
        // Boundary nudges for the selected rally, on the left home row:
        // a/s move start, d/f move end — 0.1 s steps, Shift makes them 1 s.
        case 'a':
        case 'A':
          e.preventDefault();
          nudge('start', e.shiftKey ? -1 : -0.1);
          break;
        case 's':
        case 'S':
          e.preventDefault();
          nudge('start', e.shiftKey ? 1 : 0.1);
          break;
        case 'd':
        case 'D':
          e.preventDefault();
          nudge('end', e.shiftKey ? -1 : -0.1);
          break;
        case 'f':
        case 'F':
          e.preventDefault();
          nudge('end', e.shiftKey ? 1 : 0.1);
          break;
      }
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [addAnnotation, doMarkStart, nudge]);

  const onTimeUpdate = () => {
    const el = videoRef.current;
    if (!el) return;
    setCurrentTime(el.currentTime);
    onTimeChange?.(el.currentTime);
    if (selectedIdx >= 0 && selectedIdx < annotations.length) {
      const a = annotations[selectedIdx]!;
      if (!el.paused && el.currentTime >= a.end) {
        el.pause();
        el.currentTime = a.end;
        setSelectedIdx(-1);
      }
    }
  };

  // Mirror of the annotations state, for the freshness check in save():
  // every edit creates a new array, so object identity says whether the
  // response still describes what is on screen.
  const latestAnnotations = useRef(annotations);
  useEffect(() => {
    latestAnnotations.current = annotations;
  }, [annotations]);

  const save = useCallback(
    async ({ silent = false }: { silent?: boolean } = {}) => {
      if (!videoName) {
        if (!silent) toast.warning('No video loaded');
        return;
      }
      setSaving(true);
      const sent = annotations;
      try {
        // Only the contract's fields: rows loaded from a pre-annotation carry
        // extras (`score` — model confidence), and the server forbids unknown
        // fields. A human save is a verdict; the model's confidence in its
        // own guess is not part of it.
        const body = {
          video: videoName,
          duration,
          annotations: sent.map(({ rally_id, start, end, label, side }) => ({ rally_id, start, end, label, side })),
        };
        const saved = await apiFetch<{ saved: string; count: number; annotations: EditorAnnotation[] }>(
          saveEndpoint,
          { method: 'POST', body },
        );
        // Adopt the rows as written — the server assigns ids to new rallies,
        // and without adopting them every autosave would mint fresh ones.
        // Only if nothing changed while the request was in flight; otherwise
        // the newer edit stays and the next save reconciles.
        if (latestAnnotations.current === sent) {
          // Server rows omit `side` when null — normalize back to the editor shape.
          setAnnotations(saved.annotations.map((r) => ({ ...r, side: normalizeSide(r.side) })));
          setDirty(false);
        }
        if (!silent) {
          toast.success('Annotations saved!');
          if (onSaved) await onSaved(videoName);
        }
      } catch (e) {
        // Dirty stays true so autosave retries on the next edit.
        toast.error(`Save failed: ${errMsg(e)}`);
      } finally {
        setSaving(false);
      }
    },
    [videoName, duration, annotations, saveEndpoint, onSaved],
  );

  // ── Flush unsaved work when the page goes away ──
  // beforeunload only warns (it fires before the leave dialog is answered);
  // pagehide fires once leaving is settled, and keepalive lets the flush
  // request outlive the tab. Same pattern as the Action panel.
  const flushRef = useRef({ dirty, videoName, duration });
  flushRef.current = { dirty, videoName, duration };
  useEffect(() => {
    const warn = (e: BeforeUnloadEvent) => {
      const cur = flushRef.current;
      if (cur.dirty && cur.videoName) e.preventDefault();
    };
    const flush = () => {
      const cur = flushRef.current;
      if (!cur.dirty || !cur.videoName) return;
      void fetch(apiUrl(saveEndpoint), {
        method: 'POST',
        keepalive: true,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video: cur.videoName,
          duration: cur.duration,
          annotations: latestAnnotations.current.map(({ rally_id, start, end, label, side }) => ({ rally_id, start, end, label, side })),
        }),
      });
    };
    window.addEventListener('beforeunload', warn);
    window.addEventListener('pagehide', flush);
    return () => {
      window.removeEventListener('beforeunload', warn);
      window.removeEventListener('pagehide', flush);
    };
  }, [saveEndpoint]);

  // ── Debounced autosave: push to the server AUTOSAVE_MS after editing stops ──
  const saveRef = useRef(save);
  saveRef.current = save;
  useEffect(() => {
    if (!dirty || !videoName) return;
    const t = setTimeout(() => void saveRef.current({ silent: true }), AUTOSAVE_MS);
    return () => clearTimeout(t);
  }, [annotations, dirty, videoName]);

  const copyTimestamps = async () => {
    if (annotations.length === 0) return toast.warning('No rallies to copy');
    const sh = parseTime(shift);
    const maxStart = Math.max(...annotations.map((a) => a.start)) + sh;
    const useHours = maxStart >= 3600;
    const fmt = (s: number) => {
      s = Math.max(0, Math.floor(s));
      const h = Math.floor(s / 3600);
      const m = Math.floor((s % 3600) / 60);
      const sec = s % 60;
      const pad = (n: number) => String(n).padStart(2, '0');
      return useHours ? `${h}:${pad(m)}:${pad(sec)}` : `${pad(m)}:${pad(sec)}`;
    };
    const text = annotations.map((a, i) => `${fmt(a.start + sh)} Rally ${i + 1}`).join('\n');
    try {
      await copyText(text);
      toast.success(`Copied ${annotations.length} timestamp(s)`);
    } catch (e) {
      toast.error(`Copy failed: ${errMsg(e)}`);
    }
  };

  const downloadClip = async (a: EditorAnnotation) => {
    if (!videoName) return toast.warning('No video loaded');
    try {
      const blob = await apiPostBlob(API.annotate.clip, { video: videoName, segment: { start: a.start, end: a.end, label: 'rally' } });
      downloadBlob(blob, `rally_${Math.round(a.start)}-${Math.round(a.end)}.mp4`);
      toast.success('Clip downloaded');
    } catch (e) {
      toast.error(`Download failed: ${errMsg(e)}`);
    }
  };

  const updateField = (idx: number, field: 'start' | 'end', value: number) => {
    setAnnotations((prev) => prev.map((a, i) => (i === idx ? { ...a, [field]: value } : a)));
    setDirty(true);
  };

  // Click the active side again to clear it back to unannotated.
  const updateSide = (idx: number, side: RallySide) => {
    setAnnotations((prev) => prev.map((a, i) => (i === idx ? { ...a, side: a.side === side ? null : side } : a)));
    setDirty(true);
  };

  const totalDuration = useMemo(() => annotations.reduce((s, a) => s + (a.end - a.start), 0), [annotations]);

  // Playhead-relative highlight: the row currently under the playhead.
  const playingIdx = annotations.findIndex((a) => currentTime >= a.start && currentTime < a.end);

  const selectedAnnotation = selectedIdx >= 0 ? annotations[selectedIdx] : undefined;

  return (
    <div className="flex flex-col gap-5 lg:flex-row lg:items-start">
      {/* Player + timeline */}
      <div className="min-w-0 flex-1 space-y-4">
        <Card>
          <div className="overflow-hidden rounded-2xl bg-black shadow-lg shadow-black/40 ring-1 ring-white/[0.06]">
            <video
              ref={videoRef}
              className="vq-video max-h-[45vh] w-full cursor-pointer"
              onClick={togglePlay}
              onPlay={() => setPlaying(true)}
              onPause={() => setPlaying(false)}
              onLoadedMetadata={(e) => {
                const el = e.currentTarget;
                setDuration(el.duration);
                const t = initialTime?.();
                if (t) el.currentTime = Math.min(t, el.duration || t);
              }}
              onTimeUpdate={onTimeUpdate}
            />
          </div>
          <div className="mt-3">
            <RallyTimeline videoRef={videoRef} annotations={annotations} duration={duration} markStart={markStart} onSeek={(t) => seekTo(t, false)} />
          </div>
          <div className="mt-2 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={togglePlay}
                aria-label={playing ? 'Pause' : 'Play'}
                className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary text-on-primary transition-colors hover:brightness-110"
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
              <span className="rounded-lg border border-border bg-surface-200/50 px-2.5 py-1 font-mono text-sm tabular-nums text-text-primary">{formatTime(currentTime)}</span>
            </div>
            <div className="flex items-center gap-2">
              <Button size="sm" intent="primary" onClick={doMarkStart}>
                Start [
              </Button>
              <Button size="sm" intent="primary" onClick={addAnnotation}>
                End ]
              </Button>
            </div>
          </div>
          {/* Boundary nudges for the selected rally, next to the frame they
              move — selecting a row in the list arms this bar. */}
          <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 rounded-xl border border-border bg-surface-100 px-3 py-2">
            {selectedAnnotation ? (
              <>
                <span className="w-6 text-center font-heading text-[11px] text-primary-light">
                  R{selectedAnnotation.rally_id ?? '·'}
                </span>
                {(['start', 'end'] as const).map((field) => (
                  <div key={field} className="flex items-center gap-1">
                    <span className="text-[10px] uppercase tracking-wider text-text-muted">{field}</span>
                    {NUDGE_STEPS.map(({ label, delta }) => (
                      <button
                        key={label}
                        type="button"
                        onClick={() => nudge(field, delta)}
                        title={`${field} ${label} — key: ${Math.abs(delta) === 1 ? 'Shift+' : ''}${NUDGE_KEYS[field][delta > 0 ? 1 : 0]}`}
                        className="rounded border border-border bg-surface-50 px-1.5 py-0.5 font-mono text-[10px] tabular-nums text-text-secondary transition-colors hover:border-primary/40 hover:text-text-primary"
                      >
                        {label}
                      </button>
                    ))}
                    <span className="font-heading text-[10px] tabular-nums text-text-muted/70">
                      {formatTimePrecise(selectedAnnotation[field], 1)}
                    </span>
                  </div>
                ))}
                <span className="ml-auto flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] text-text-muted/80">
                  <span>
                    <Kbd>a</Kbd> <Kbd>s</Kbd> start −/＋0.1s
                  </span>
                  <span>
                    <Kbd>d</Kbd> <Kbd>f</Kbd> end −/＋0.1s
                  </span>
                  <span>
                    hold <Kbd>Shift</Kbd> for 1s steps
                  </span>
                </span>
              </>
            ) : (
              <span className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-text-muted">
                <span>
                  <Kbd>Space</Kbd> play/pause
                </span>
                <span>
                  <Kbd>←</Kbd> <Kbd>→</Kbd> seek ±5s
                </span>
                <span>
                  <Kbd>[</Kbd> mark start
                </span>
                <span>
                  <Kbd>]</Kbd> add rally
                </span>
                <span>
                  <Kbd>Del</Kbd> delete selected
                </span>
                <span className="text-text-muted/60">— select a rally to nudge: <Kbd>a</Kbd><Kbd>s</Kbd> start, <Kbd>d</Kbd><Kbd>f</Kbd> end</span>
              </span>
            )}
          </div>
          {markStart != null && (
            <div className="mt-3 flex items-center gap-2.5 rounded-xl border border-primary/20 bg-primary/10 p-3">
              <span className="h-2 w-2 rounded-full bg-primary-light animate-pulse-dot" />
              <span className="text-xs text-primary-light">
                Start marked at <strong className="font-mono">{formatTime(markStart)}</strong> — press ] to set end
              </span>
            </div>
          )}
        </Card>
      </div>

      {/* Annotation list */}
      <div className="lg:w-[460px] lg:flex-shrink-0">
        <Card>
          <div className="mb-3 flex items-center justify-between gap-2">
            <SectionLabel className="mb-0">
              Annotations ({annotations.length} rally){totalDuration > 0 ? ` · ${formatTime(totalDuration)} played` : ''}
            </SectionLabel>
            <div className="flex items-center gap-2">
              <div className="flex overflow-hidden rounded-lg border border-border" title="得分側方向（畫面視角）">
                {(['lr', 'nf'] as const).map((axis) => (
                  <button
                    key={axis}
                    type="button"
                    onClick={() => setSideAxis(axis)}
                    className={cn(
                      'px-2 py-1 text-[11px] leading-none transition-colors',
                      sideAxis === axis
                        ? 'bg-primary/20 text-primary-text'
                        : 'text-text-muted/60 hover:bg-surface-200/40 hover:text-text-primary',
                    )}
                  >
                    {AXIS_DISPLAY[axis]}
                  </button>
                ))}
              </div>
              <Button size="sm" intent="primary" onClick={() => void save()} disabled={saving}>
                {dirty ? 'Save •' : 'Save'}
              </Button>
            </div>
          </div>

          <div className="vq-list max-h-[calc(45vh+2.25rem)] space-y-1.5 overflow-y-auto pr-1">
            {annotations.length === 0 ? (
              <EmptyState
                icon={
                  <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v12m6-6H6" />
                  </svg>
                }
                title="No annotations"
                subtitle="Use [ ] to mark segments"
              />
            ) : (
              annotations.map((a, i) => {
                const selected = selectedIdx === i;
                const playing = playingIdx === i;
                return (
                  <div
                    key={i}
                    onClick={() => {
                      setSelectedIdx(i);
                      seekTo(a.start);
                    }}
                    className={cn(
                      'ae-row group flex cursor-pointer items-center gap-1.5 rounded-xl border px-3 py-2.5 transition-colors',
                      selected ? 'border-primary/45 bg-primary/[0.12]' : 'border-primary/20 bg-primary/[0.05] hover:bg-primary/[0.10]',
                      playing && 'ring-1 ring-accent/50',
                    )}
                  >
                    {/* Every cell has a fixed width so the columns line up
                        across rows and the row never outgrows the card. */}
                    <span className="w-4 shrink-0 select-none text-right font-heading text-[10px] text-text-muted/60">{i + 1}</span>
                    <span
                      className="w-7 shrink-0 select-none font-mono text-[9px] text-text-muted/40"
                      title={a.rally_id === null ? 'New rally — gets its id on save' : `rally_id ${a.rally_id} — stable id, not the time order`}
                    >
                      {a.rally_id === null ? 'new' : `#${a.rally_id}`}
                    </span>
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        void downloadClip(a);
                      }}
                      className="shrink-0 rounded-full bg-primary/20 p-1 text-primary-text ring-1 ring-primary/30 transition-colors hover:bg-primary/30"
                      title="Download this rally clip"
                    >
                      <svg className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M7 11l5 5 5-5M12 16V4" />
                      </svg>
                    </button>
                    <div className="ml-auto flex shrink-0 items-center gap-1" onClick={(e) => e.stopPropagation()}>
                      <TimeField seconds={a.start} onCommit={(value) => updateField(i, 'start', value)} />
                      <span className="text-[10px] text-text-muted/40">→</span>
                      <TimeField seconds={a.end} onCommit={(value) => updateField(i, 'end', value)} />
                    </div>
                    <span className="w-11 shrink-0 rounded bg-surface-200/40 px-1 py-0.5 text-right font-mono text-[10px] tabular-nums text-text-muted">{(a.end - a.start).toFixed(1)}s</span>
                    <div
                      className="flex w-11 shrink-0 items-center justify-center gap-0.5"
                      onClick={(e) => e.stopPropagation()}
                      title="得分側（畫面視角）— 再點一次取消"
                    >
                      {a.side && !AXIS_SIDES[sideAxis].includes(a.side) ? (
                        // A stored side from the other axis: show it alone so the
                        // label never disappears silently — click clears it, then
                        // the current axis's pair takes the cell back.
                        <button
                          type="button"
                          onClick={() => updateSide(i, a.side!)}
                          title="與目前方向不同的舊標記 — 點一下清除"
                          className="w-5 rounded py-0.5 text-center text-[10px] leading-none bg-accent/30 text-text-primary ring-1 ring-accent/60"
                        >
                          {SIDE_DISPLAY[a.side]}
                        </button>
                      ) : (
                        AXIS_SIDES[sideAxis].map((side) => (
                          <button
                            key={side}
                            type="button"
                            onClick={() => updateSide(i, side)}
                            className={cn(
                              'w-5 rounded py-0.5 text-center text-[10px] leading-none transition-colors',
                              a.side === side
                                ? 'bg-accent/30 text-text-primary ring-1 ring-accent/60'
                                : 'text-text-muted/50 hover:bg-surface-200/40 hover:text-text-muted',
                            )}
                          >
                            {SIDE_DISPLAY[side]}
                          </button>
                        ))
                      )}
                    </div>
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        setSelectedIdx(i);
                        seekTo(Math.max(a.start, a.end - previewBackoff));
                      }}
                      className="shrink-0 text-primary-light transition-colors hover:text-text-primary"
                      title="Jump to end"
                    >
                      <svg className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M13 5l7 7-7 7M5 5l7 7-7 7" />
                      </svg>
                    </button>
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        setAnnotations((prev) => prev.filter((_, j) => j !== i));
                        setDirty(true);
                        if (selectedIdx === i) setSelectedIdx(-1);
                      }}
                      className="shrink-0 text-red-400/60 opacity-0 transition-all hover:text-red-400 group-hover:opacity-100"
                      title="Delete"
                    >
                      <svg className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  </div>
                );
              })
            )}
          </div>

          <div className="mt-3 flex items-center gap-2 border-t border-border pt-3">
            <label className="whitespace-nowrap text-[11px] text-text-muted">YT shift</label>
            <input
              value={shift}
              onChange={(e) => setShift(e.target.value)}
              placeholder="0 or 1:23"
              className="min-w-0 flex-1 rounded-lg border border-border-light bg-surface-50 px-2.5 py-1.5 font-mono text-xs tabular-nums text-text-primary focus:border-primary/50 focus:outline-none focus:ring-2 focus:ring-primary/15"
            />
            <Button size="sm" intent="primary" onClick={copyTimestamps}>
              Copy YT timestamps
            </Button>
          </div>
        </Card>
      </div>

      {clipsOpen && <DownloadClipsModal video={videoName} segments={annotations} onClose={onClipsClose} />}
    </div>
  );
}
