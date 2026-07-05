import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { API, ApiError, apiFetch, apiPostBlob } from '@/lib/api';
import { cn } from '@/lib/cn';
import { formatTime, parseTime } from '@/lib/format';
import { copyText, downloadBlob } from '@/lib/download';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { EmptyState } from '@/components/ui/EmptyState';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { toast } from '@/components/feedback/toast';
import { useVideoRecovery } from '@/lib/useVideoRecovery';
import { DownloadClipsModal } from './DownloadClipsModal';
import { RallyTimeline } from './RallyTimeline';

export interface EditorAnnotation {
  rally_id: number | null;
  start: number;
  end: number;
  label: string;
  score?: number | null;
}
export interface EditorData {
  video?: string;
  source_video?: string;
  metadata?: { video?: string };
  results?: Array<Record<string, unknown>>;
}

interface AnnotationEditorProps {
  data: EditorData | null;
  saveEndpoint: string;
  videoStreamPath: (videoPath: string) => string;
  rowExtras?: (a: EditorAnnotation) => ReactNode;
  previewBackoff?: number;
  onSaved?: (videoName: string) => Promise<void> | void;
}

const errMsg = (e: unknown) => (e instanceof ApiError ? e.body : e instanceof Error ? e.message : String(e));
const normalizeRallyId = (v: unknown): number | null => {
  const n = Number(v);
  return Number.isInteger(n) && n > 0 ? n : null;
};
const num = (...vals: unknown[]): number => {
  for (const v of vals) if (typeof v === 'number' && Number.isFinite(v)) return v;
  return 0;
};

// ── Local draft persistence ──
// The network is not trustworthy (flaky tunnel, dropped connections), so
// unsaved work is mirrored to localStorage on every edit. A draft exists
// *only* while there is unsaved work: it is cleared on a successful save.
// On load, a leftover draft means the page died before saving — restore it.
const AUTOSAVE_MS = 2000;
const DRAFT_PREFIX = 'vq:annot-draft';

interface AnnotationDraft {
  video: string;
  duration: number;
  annotations: EditorAnnotation[];
}

const draftKey = (endpoint: string, video: string) => `${DRAFT_PREFIX}:${endpoint}:${video}`;

const readDraft = (endpoint: string, video: string): AnnotationDraft | null => {
  try {
    const raw = localStorage.getItem(draftKey(endpoint, video));
    if (!raw) return null;
    const d = JSON.parse(raw) as AnnotationDraft;
    return Array.isArray(d.annotations) ? d : null;
  } catch {
    return null; // corrupt JSON / privacy mode — drafts are best-effort
  }
};

const writeDraft = (endpoint: string, draft: AnnotationDraft): void => {
  try {
    localStorage.setItem(draftKey(endpoint, draft.video), JSON.stringify(draft));
  } catch {
    /* quota exceeded / privacy mode — nothing we can do, skip */
  }
};

const clearDraft = (endpoint: string, video: string): void => {
  try {
    localStorage.removeItem(draftKey(endpoint, video));
  } catch {
    /* ignore */
  }
};

export function AnnotationEditor({ data, saveEndpoint, videoStreamPath, rowExtras, previewBackoff = 3, onSaved }: AnnotationEditorProps) {
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
  const [modalOpen, setModalOpen] = useState(false);
  const [playing, setPlaying] = useState(false);

  const togglePlay = () => {
    const el = videoRef.current;
    if (!el) return;
    el.paused ? void el.play() : el.pause();
  };

  // Presigned video URLs expire and range requests can hang; reload the src
  // (which fetches a fresh URL) and seek back to where the user was.
  useVideoRecovery(videoRef, {
    src: () => (videoName ? videoStreamPath(videoName) : ''),
    onRecover: () => toast.info('影片串流中斷，已自動重新載入'),
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
        score: (r.confidence ?? r.score ?? null) as number | null,
      }))
      .sort((a, b) => a.start - b.start);
    // A leftover draft is unsaved work from a previous session — prefer it
    // so a crash/reload never loses annotations.
    const draft = path ? readDraft(saveEndpoint, path) : null;
    if (draft) {
      setAnnotations(draft.annotations);
      setDirty(true);
      toast.info('已還原上次未儲存的標註草稿');
    } else {
      setAnnotations(fromServer);
      setDirty(false);
    }
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
      setAnnotations((prev) => [...prev, { rally_id: null, start: ms, end, label: 'rally' }].sort((a, b) => a.start - b.start));
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
          el.paused ? void el.play() : el.pause();
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
      }
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [addAnnotation, doMarkStart]);

  const onTimeUpdate = () => {
    const el = videoRef.current;
    if (!el) return;
    setCurrentTime(el.currentTime);
    if (selectedIdx >= 0 && selectedIdx < annotations.length) {
      const a = annotations[selectedIdx]!;
      if (!el.paused && el.currentTime >= a.end) {
        el.pause();
        el.currentTime = a.end;
        setSelectedIdx(-1);
      }
    }
  };

  const save = useCallback(
    async ({ silent = false }: { silent?: boolean } = {}) => {
      if (!videoName) {
        if (!silent) toast.warning('No video loaded');
        return;
      }
      setSaving(true);
      try {
        await apiFetch(saveEndpoint, { method: 'POST', body: { video: videoName, duration, annotations } });
        setDirty(false);
        clearDraft(saveEndpoint, videoName); // server now holds the truth; drop the local backup
        if (!silent) toast.success('Annotations saved!');
        if (onSaved) await onSaved(videoName);
      } catch (e) {
        // Keep dirty=true and the draft intact so the work survives — autosave
        // will retry on the next edit, and the draft survives a reload.
        toast.error(`Save failed: ${errMsg(e)}`);
      } finally {
        setSaving(false);
      }
    },
    [videoName, duration, annotations, saveEndpoint, onSaved],
  );

  // ── Mirror unsaved work to localStorage on every edit ──
  useEffect(() => {
    if (!dirty || !videoName) return;
    writeDraft(saveEndpoint, { video: videoName, duration, annotations });
  }, [annotations, duration, dirty, videoName, saveEndpoint]);

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
      const blob = await apiPostBlob(API.review.clip, { video: videoName, segment: { start: a.start, end: a.end, label: 'rally' } });
      downloadBlob(blob, `rally_${Math.round(a.start)}-${Math.round(a.end)}.mp4`);
      toast.success('Clip downloaded');
    } catch (e) {
      toast.error(`Download failed: ${errMsg(e)}`);
    }
  };

  const updateField = (idx: number, field: 'start' | 'end', value: string) => {
    setAnnotations((prev) => prev.map((a, i) => (i === idx ? { ...a, [field]: parseTime(value) } : a)));
    setDirty(true);
  };

  const totalDuration = useMemo(() => annotations.reduce((s, a) => s + (a.end - a.start), 0), [annotations]);

  // Playhead-relative highlight: the row currently under the playhead.
  const playingIdx = annotations.findIndex((a) => currentTime >= a.start && currentTime < a.end);

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
              onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)}
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
      <div className="lg:w-[420px] lg:flex-shrink-0">
        <Card>
          <div className="mb-3 flex items-center justify-between gap-2">
            <SectionLabel className="mb-0">
              Annotations ({annotations.length} rally){totalDuration > 0 ? ` · ${formatTime(totalDuration)} played` : ''}
            </SectionLabel>
            <div className="flex items-center gap-2">
              <Button size="sm" onClick={() => setModalOpen(true)}>
                Clips
              </Button>
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
                    <span className="w-4 select-none text-right font-heading text-[10px] text-text-muted/60">{i + 1}</span>
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        void downloadClip(a);
                      }}
                      className="flex items-center gap-1 rounded-full bg-primary/20 px-2 py-0.5 text-[11px] font-medium text-primary-text ring-1 ring-primary/30 transition-colors hover:bg-primary/30"
                      title="Download this rally clip"
                    >
                      <svg className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M7 11l5 5 5-5M12 16V4" />
                      </svg>
                      rally
                    </button>
                    <div className="ml-auto flex items-center gap-1.5" onClick={(e) => e.stopPropagation()}>
                      <input
                        value={formatTime(a.start)}
                        onChange={(e) => updateField(i, 'start', e.target.value)}
                        className="w-11 border-0 border-b border-ink/10 bg-transparent text-center font-heading text-[11px] tabular-nums text-text-primary focus:border-primary-light focus:outline-none focus:ring-0"
                      />
                      <span className="text-[10px] text-text-muted/40">→</span>
                      <input
                        value={formatTime(a.end)}
                        onChange={(e) => updateField(i, 'end', e.target.value)}
                        className="w-11 border-0 border-b border-ink/10 bg-transparent text-center font-heading text-[11px] tabular-nums text-text-primary focus:border-primary-light focus:outline-none focus:ring-0"
                      />
                    </div>
                    <span className="rounded bg-surface-200/40 px-1.5 py-0.5 font-mono text-[10px] tabular-nums text-text-muted">{(a.end - a.start).toFixed(1)}s</span>
                    {rowExtras?.(a)}
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        setSelectedIdx(i);
                        seekTo(Math.max(a.start, a.end - previewBackoff));
                      }}
                      className="text-primary-light transition-colors hover:text-text-primary"
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
                      className="text-red-400/60 opacity-0 transition-all hover:text-red-400 group-hover:opacity-100"
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

      {modalOpen && <DownloadClipsModal video={videoName} segments={annotations} onClose={() => setModalOpen(false)} />}
    </div>
  );
}
