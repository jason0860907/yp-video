import { useCallback, useEffect, useMemo, useRef, useState, type MouseEvent as ReactMouseEvent, type PointerEvent as ReactPointerEvent, type ReactNode, type SyntheticEvent } from 'react';
import { useQuery } from '@tanstack/react-query';
import { API, ApiError, apiFetch, apiUrl } from '@/lib/api';
import { cn } from '@/lib/cn';
import { copyText } from '@/lib/download';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { EmptyState } from '@/components/ui/EmptyState';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { toast } from '@/components/feedback/toast';
import { confirm } from '@/components/feedback/confirm';
import { ActionTimeline } from '@/components/editor/ActionTimeline';
import { KindBadge } from '@/components/video/KindBadge';
import { VideoCombobox } from '@/components/video/VideoCombobox';
import { useVideoRecovery } from '@/lib/useVideoRecovery';
import { ACTION_COLORS, actionColor } from '@/lib/actionColors';
import type { ActionAnnotationData, ActionEvent, ActionRally, ActionVideo, WaveformData } from '@/types/api';

const DEFAULT_LABELS = ['serve', 'receive', 'set', 'spike', 'block', 'score'];
const OUTSIDE = '__outside__';

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const round4 = (v: number) => Math.round(v * 1e4) / 1e4;
const errMsg = (e: unknown) => (e instanceof ApiError ? e.body : e instanceof Error ? e.message : String(e));
const fmt = (s: number) => {
  if (!Number.isFinite(s)) return '00:00';
  const m = Math.floor(s / 60);
  return `${String(m).padStart(2, '0')}:${String(Math.floor(s - m * 60)).padStart(2, '0')}`;
};
const makeId = () => `act_${(crypto.randomUUID?.() ?? `${Date.now().toString(36)}${Math.random().toString(36).slice(2)}`).replace(/-/g, '').slice(0, 16)}`;
const normRallyId = (v: unknown): number | null => {
  const n = Number(v);
  return Number.isInteger(n) && n > 0 ? n : null;
};

interface Editor {
  video: string;
  duration: number;
  fps: number;
  numFrames: number;
  rallies: ActionRally[];
  events: ActionEvent[];
  dirty: boolean;
}
const EMPTY: Editor = { video: '', duration: 0, fps: 30, numFrames: 0, rallies: [], events: [], dirty: false };

// ── Local draft persistence ──
// The network is not trustworthy (flaky tunnel, dropped connections), so
// unsaved work is mirrored to localStorage on every edit. A draft exists
// *only* while there is unsaved work: it is cleared on a successful save or
// an explicit discard. On load, a leftover draft means the page died before
// saving — restore the user's events over the server snapshot.
const AUTOSAVE_MS = 2000;
const DRAFT_DEBOUNCE_MS = 300; // coalesce rapid edits (e.g. point dragging)
const ACTION_DRAFT_PREFIX = 'vq:action-draft';
const actionDraftKey = (video: string) => `${ACTION_DRAFT_PREFIX}:${video}`;

const readActionDraft = (video: string): Editor | null => {
  try {
    const raw = localStorage.getItem(actionDraftKey(video));
    if (!raw) return null;
    const d = JSON.parse(raw) as Editor;
    return Array.isArray(d.events) ? d : null;
  } catch {
    return null; // corrupt JSON / privacy mode — drafts are best-effort
  }
};

const writeActionDraft = (ed: Editor): void => {
  try {
    localStorage.setItem(actionDraftKey(ed.video), JSON.stringify(ed));
  } catch {
    /* quota exceeded / privacy mode — nothing we can do, skip */
  }
};

const clearActionDraft = (video: string): void => {
  try {
    localStorage.removeItem(actionDraftKey(video));
  } catch {
    /* ignore */
  }
};

// Waveform request density (points scale with duration), ported from the legacy UI.
const WAVEFORM_POINTS_PER_SECOND = 120;
const WAVEFORM_MIN_POINTS = 2400;
const WAVEFORM_MAX_POINTS = 96000;
const EMPTY_WAVE: WaveformData = { video: '', loading: false, error: '', hasAudio: false, duration: 0, peaks: [], rms: [] };
const waveformPointCount = (durationSeconds: number) =>
  clamp(Math.ceil(Math.max(0, durationSeconds) * WAVEFORM_POINTS_PER_SECOND) || WAVEFORM_MIN_POINTS, WAVEFORM_MIN_POINTS, WAVEFORM_MAX_POINTS);

const findRally = (frame: number, ed: Editor): ActionRally | null => {
  const t = frame / (ed.fps || 30);
  return ed.rallies.find((r) => t >= r.start && t < r.end) ?? null;
};
const withRally = (e: ActionEvent, ed: Editor): ActionEvent => {
  const frame = Math.max(0, Math.round(e.frame || 0));
  const t = frame / (ed.fps || 30);
  const rally = findRally(frame, ed);
  return { ...e, frame, time: round4(t), rally_id: rally?.rally_id ?? null, relative_frame: rally ? Math.max(0, Math.round((t - rally.start) * (ed.fps || 30))) : null };
};
const sortEvents = (evs: ActionEvent[]) => [...evs].sort((a, b) => a.frame - b.frame || a.label.localeCompare(b.label) || a.id.localeCompare(b.id));

function normalize(data: ActionAnnotationData, labels: string[]): Editor {
  const fps = Number(data.fps) || 30;
  const rallies: ActionRally[] = (data.rallies ?? [])
    .map((r, i) => ({ rally_id: normRallyId(r.rally_id) ?? i + 1, start: Number(r.start) || 0, end: Number(r.end) || 0, label: r.label || 'rally' }))
    .sort((a, b) => a.start - b.start || a.end - b.end || a.rally_id - b.rally_id);
  const ed: Editor = { video: data.source_video || data.video || '', duration: Number(data.duration) || 0, fps, numFrames: Number(data.num_frames) || 0, rallies, events: [], dirty: false };
  ed.events = sortEvents(
    (data.events ?? []).map((e) => {
      const x = e as Record<string, unknown>;
      const xy = (x.xy as number[] | undefined) ?? [Number(x.x ?? 0.5), Number(x.y ?? 0.5)];
      return withRally(
        {
          id: (x.id as string) || makeId(),
          rally_id: null,
          frame: Math.max(0, Math.round(Number(x.frame) || 0)),
          time: Number(x.time) || null,
          relative_frame: Number.isInteger(x.relative_frame) ? (x.relative_frame as number) : null,
          label: labels.includes(x.label as string) ? (x.label as string) : labels[0]!,
          xy: [clamp(Number(xy[0] ?? 0.5), 0, 1), clamp(Number(xy[1] ?? 0.5), 0, 1)],
          visible: x.visible !== false,
        },
        ed,
      );
    }),
  );
  return ed;
}

const hasActive = (v: ActionVideo) => Boolean(v.has_action_annotation || v.has_action_final_annotation || v.has_action_pre_annotation);
const isReviewed = (v: ActionVideo) => Boolean(v.action_reviewed);


export function ActionAnnotatePage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [pointMode, setPointMode] = useState(false);
  const [aspect, setAspect] = useState(16 / 9);
  const [waveform, setWaveform] = useState<WaveformData>(EMPTY_WAVE);
  const waveformReq = useRef(0);
  const drag = useRef<{ id: string; moved: boolean } | null>(null);
  const suppressClick = useRef(false);
  const [selectedLabel, setSelectedLabel] = useState('serve');
  const [kindFilter, setKindFilter] = useState<'all' | 'broadcast' | 'sideline'>('all');
  const [progressFilter, setProgressFilter] = useState<'all' | 'unlabeled' | 'pre-labeled' | 'labeled'>('all');
  const [picked, setPicked] = useState('');
  const [loading, setLoading] = useState(false);

  const [ed, setEd] = useState<Editor>(EMPTY);
  const [selectedIdx, setSelectedIdx] = useState(-1);
  const [selectedRallyId, setSelectedRallyId] = useState<number | 'all'>('all');
  const [expanded, setExpanded] = useState<string | null>(null);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);

  // Frame-clock refs (read inside the requestVideoFrameCallback loop).
  const lockedFrame = useRef<number | null>(null);
  const presented = useRef<number | null>(null);
  const cbId = useRef<number | null>(null);
  const gen = useRef(0);
  const edRef = useRef(ed);
  edRef.current = ed;
  const selRallyRef = useRef(selectedRallyId);
  selRallyRef.current = selectedRallyId;

  const videosQuery = useQuery({ queryKey: ['action-videos'], queryFn: () => apiFetch<ActionVideo[]>(API.actionAnnotate.videos) });
  const labelsQuery = useQuery({ queryKey: ['action-labels'], queryFn: () => apiFetch<{ labels?: string[] }>(API.actionAnnotate.labels) });
  const videos = videosQuery.data ?? [];
  const labels = labelsQuery.data?.labels ?? DEFAULT_LABELS;

  // ── Frame clock ──
  const computeFrame = () => {
    const e = edRef.current;
    if (lockedFrame.current !== null) return clamp(lockedFrame.current, 0, Math.max(0, e.numFrames - 1));
    const el = videoRef.current;
    const t = presented.current != null && Number.isFinite(presented.current) ? presented.current : el?.currentTime || 0;
    return clamp(Math.round(t * (e.fps || 30)), 0, Math.max(0, e.numFrames - 1));
  };
  const refreshPlayhead = () => {
    const f = computeFrame();
    setFrame(f);
    // Auto-pause at the end of the selected rally during playback, so a rally
    // doesn't run on into the next one.
    const el = videoRef.current;
    const e = edRef.current;
    const rid = selRallyRef.current;
    if (!el || el.paused || rid === 'all' || !e.fps) return;
    const rally = e.rallies.find((r) => r.rally_id === rid);
    if (!rally) return;
    const endFrame = Math.max(0, Math.ceil(rally.end * e.fps) - 1);
    if (f >= endFrame) {
      el.pause();
      seekFrame(endFrame);
    }
  };

  useEffect(() => {
    let alive = true;
    const tick = () => {
      const el = videoRef.current;
      if (!el?.requestVideoFrameCallback) return;
      const myGen = gen.current;
      cbId.current = el.requestVideoFrameCallback((_n, meta) => {
        if (!alive) return;
        // A seek (or load) bumped the generation — restart the clock with the
        // new generation instead of letting the loop die.
        if (myGen !== gen.current) {
          tick();
          return;
        }
        if (!el.paused) lockedFrame.current = null;
        if (Number.isFinite(meta?.mediaTime) && (lockedFrame.current === null || !el.paused)) presented.current = meta.mediaTime;
        refreshPlayhead();
        tick();
      });
    };
    tick();
    const poll = setInterval(refreshPlayhead, 120);
    return () => {
      alive = false;
      clearInterval(poll);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Presigned video URLs expire and range requests can hang; reload the src
  // (which fetches a fresh URL) and seek back to where the user was.
  useVideoRecovery(videoRef, {
    src: () => (edRef.current.video ? apiUrl(API.actionAnnotate.video(edRef.current.video)) : ''),
    onRecover: () => toast.info('影片串流中斷，已自動重新載入'),
    onGiveUp: () => toast.error('影片重載後仍卡在同一處，已停止自動重試 — 請把 DevTools Console 的 [video-recovery] 記錄回報'),
  });

  // Track play/pause so the timeline only follows the playhead during playback.
  useEffect(() => {
    const el = videoRef.current;
    if (!el) return;
    const onPlay = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    el.addEventListener('play', onPlay);
    el.addEventListener('pause', onPause);
    el.addEventListener('ended', onPause);
    return () => {
      el.removeEventListener('play', onPlay);
      el.removeEventListener('pause', onPause);
      el.removeEventListener('ended', onPause);
    };
  }, []);

  const seekFrame = (f: number) => {
    const el = videoRef.current;
    const e = edRef.current;
    if (!el || !e.fps) return;
    const target = clamp(f, 0, Math.max(0, e.numFrames - 1));
    lockedFrame.current = target;
    presented.current = target / e.fps;
    gen.current += 1;
    el.currentTime = e.duration > 0 ? clamp((target + 0.5) / e.fps, 0, e.duration) : Math.max(0, (target + 0.5) / e.fps);
    setFrame(target);
  };
  const stepFrame = (d: number) => {
    videoRef.current?.pause();
    seekFrame((lockedFrame.current ?? computeFrame()) + d);
  };
  const togglePlay = () => {
    const el = videoRef.current;
    if (!el?.src) return;
    if (el.paused) {
      // Release any seek lock so the playhead tracks playback from frame one.
      lockedFrame.current = null;
      void el.play().catch((e) => toast.error(`Play failed: ${errMsg(e)}`));
    } else {
      el.pause();
    }
  };

  // ── On-video overlay ──
  // The wrap div carries the video's exact aspect ratio, so the video fills it
  // with no letterbox and normalized point coords map 1:1 onto the div — any
  // layout change (sidebar toggle, window resize) repositions dots via CSS alone.
  const clientToPoint = (cx: number, cy: number): [number, number] | null => {
    const r = wrapRef.current?.getBoundingClientRect();
    if (!r || !r.width || !r.height) return null;
    return [round4(clamp((cx - r.left) / r.width, 0, 1)), round4(clamp((cy - r.top) / r.height, 0, 1))];
  };
  const onVideoMetadata = (e: SyntheticEvent<HTMLVideoElement>) => {
    const el = e.currentTarget;
    if (el.videoWidth && el.videoHeight) setAspect(el.videoWidth / el.videoHeight);
  };

  const onVideoClick = (e: ReactMouseEvent) => {
    if (!edRef.current.video || !pointMode) return;
    const p = clientToPoint(e.clientX, e.clientY);
    if (p) addEvent(p[0], p[1]);
  };
  const onVideoContextMenu = (e: ReactMouseEvent) => {
    e.preventDefault();
    if (!edRef.current.video) return;
    const p = clientToPoint(e.clientX, e.clientY);
    if (p) addEvent(p[0], p[1], false);
  };

  const startDrag = (e: ReactPointerEvent, evt: ActionEvent, idx: number) => {
    e.preventDefault();
    e.stopPropagation();
    setSelectedIdx(idx);
    videoRef.current?.pause();
    drag.current = { id: evt.id, moved: false };
    (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
    const onMove = (ev: PointerEvent) => {
      if (!drag.current) return;
      if (!drag.current.moved) {
        // Re-stamp the event's frame to the playhead only once an actual drag
        // begins — a bare click must never touch the time, or xy and frame
        // drift apart (clicking a point while parked on another frame used to
        // silently move the event there).
        drag.current.moved = true;
        const f = computeFrame();
        mutate((prev) => ({ ...prev, events: prev.events.map((x) => (x.id === drag.current!.id ? withRally({ ...x, frame: f }, prev) : x)) }));
      }
      const p = clientToPoint(ev.clientX, ev.clientY);
      if (!p) return;
      setEd((prev) => ({ ...prev, dirty: true, events: prev.events.map((x) => (x.id === drag.current!.id ? { ...x, xy: p } : x)) }));
    };
    // pointercancel (touch gesture, browser takeover) must run the same
    // cleanup as pointerup, or the document-level move listener leaks and
    // every later mouse move keeps dragging the point.
    const ac = new AbortController();
    const onUp = () => {
      ac.abort();
      if (drag.current?.moved) {
        suppressClick.current = true;
        setTimeout(() => {
          suppressClick.current = false;
        }, 0);
      }
      drag.current = null;
      setEd((prev) => ({ ...prev, events: sortEvents(prev.events) }));
    };
    document.addEventListener('pointermove', onMove, { signal: ac.signal });
    document.addEventListener('pointerup', onUp, { signal: ac.signal });
    document.addEventListener('pointercancel', onUp, { signal: ac.signal });
  };

  // ── Video list filtering (text search happens inside the combobox) ──
  const filtered = useMemo(
    () =>
      videos.filter((v) => {
        if (kindFilter !== 'all' && v.kind !== kindFilter) return false;
        if (progressFilter === 'unlabeled' && hasActive(v)) return false;
        if (progressFilter === 'pre-labeled' && !(hasActive(v) && !isReviewed(v))) return false;
        if (progressFilter === 'labeled' && !isReviewed(v)) return false;
        return true;
      }),
    [videos, kindFilter, progressFilter],
  );

  const loadWaveform = useCallback(async (name: string, duration: number) => {
    const reqId = ++waveformReq.current;
    setWaveform({ ...EMPTY_WAVE, video: name, loading: true });
    try {
      const pts = waveformPointCount(duration);
      const data = await apiFetch<{ has_audio?: boolean; duration?: number; peaks?: number[]; rms?: number[] }>(`${API.actionAnnotate.waveform(name)}?points=${pts}`);
      if (reqId !== waveformReq.current) return;
      const peaks = Array.isArray(data.peaks) ? data.peaks.map((v) => clamp(Number(v) || 0, 0, 1)) : [];
      const rms = Array.isArray(data.rms) && data.rms.length === peaks.length ? data.rms.map((v) => clamp(Number(v) || 0, 0, 1)) : peaks;
      setWaveform({ video: name, loading: false, error: '', hasAudio: Boolean(data.has_audio), duration: Number(data.duration) || duration || 0, peaks, rms });
    } catch (e) {
      if (reqId !== waveformReq.current) return;
      setWaveform({ ...EMPTY_WAVE, video: name, error: errMsg(e) });
    }
  }, []);

  const load = async (name: string) => {
    if (!name) return;
    if (ed.dirty && name !== ed.video) {
      const ok = await confirm({ title: 'Discard unsaved changes?', body: 'The current action labels have not been saved.', confirmText: 'Discard', variant: 'danger' });
      if (!ok) return;
      clearActionDraft(ed.video); // user explicitly abandoned this video's edits
    }
    setPicked(name);
    setLoading(true);
    try {
      const data = await apiFetch<ActionAnnotationData>(API.actionAnnotate.annotation(name));
      let next = normalize(data, labels);
      // A leftover draft is unsaved work from a previous session: restore the
      // user's events (server stays authoritative for rally/fps structure).
      const draft = readActionDraft(next.video);
      if (draft) {
        next = { ...next, events: sortEvents(draft.events.map((e) => withRally(e, next))), dirty: true };
      }
      setEd(next);
      edRef.current = next;
      setSelectedIdx(-1);
      setSelectedRallyId(next.rallies[0]?.rally_id ?? 'all');
      setExpanded(next.rallies[0] ? String(next.rallies[0].rally_id) : null);
      lockedFrame.current = null;
      presented.current = 0;
      gen.current += 1;
      const el = videoRef.current;
      if (el) {
        el.pause();
        el.src = apiUrl(API.actionAnnotate.video(next.video));
        el.load();
      }
      loadWaveform(next.video, next.duration);
      setFrame(0);
      if (next.dirty) toast.info(`已還原上次未儲存的草稿（${next.events.length} 個動作）`);
      else toast.success(`Loaded ${next.events.length} event(s)`);
    } catch (e) {
      toast.error(`Load failed: ${errMsg(e)}`);
    } finally {
      setLoading(false);
    }
  };

  const mutate = (fn: (ed: Editor) => Editor) => setEd((prev) => ({ ...fn(prev), dirty: true }));

  const addEvent = (x = 0.5, y = 0.5, visible = true) => {
    if (!ed.video) return toast.warning('Load a video first');
    const f = clampToRally(computeFrame(), ed, selectedRallyId);
    if (f !== computeFrame()) seekFrame(f);
    mutate((prev) => {
      const evt = withRally({ id: makeId(), rally_id: null, frame: f, time: null, relative_frame: null, label: selectedLabel, xy: [round4(x), round4(y)], visible }, prev);
      const events = sortEvents([...prev.events, evt]);
      setSelectedIdx(events.indexOf(evt));
      if (evt.rally_id) {
        setSelectedRallyId(evt.rally_id);
        setExpanded(String(evt.rally_id));
      } else setExpanded(OUTSIDE);
      return { ...prev, events };
    });
  };

  const editEvent = (idx: number, patch: Partial<ActionEvent>) =>
    mutate((prev) => {
      const events = prev.events.map((e, i) => (i === idx ? withRally({ ...e, ...patch }, prev) : e));
      return { ...prev, events: patch.frame !== undefined ? sortEvents(events) : events };
    });
  const deleteEvent = (idx: number) => {
    setSelectedIdx(-1);
    mutate((prev) => ({ ...prev, events: prev.events.filter((_, i) => i !== idx) }));
  };

  const save = async (silent = false) => {
    if (!ed.video) {
      if (!silent) toast.warning('No video loaded');
      return;
    }
    try {
      await apiFetch(API.actionAnnotate.annotations, { method: 'POST', body: { video: ed.video, fps: ed.fps, num_frames: ed.numFrames, events: ed.events } });
      setEd((prev) => ({ ...prev, dirty: false }));
      clearActionDraft(ed.video); // server now holds the truth; drop the local backup
      if (!silent) {
        void videosQuery.refetch();
        toast.success('Action annotations saved');
      }
    } catch (e) {
      // Keep dirty=true and the draft intact so the work survives — autosave
      // retries on the next edit, and the draft survives a reload.
      toast.error(`Save failed: ${errMsg(e)}`);
    }
  };

  // ── Mirror unsaved work to localStorage (debounced to coalesce drags) ──
  useEffect(() => {
    if (!ed.dirty || !ed.video) return;
    const t = setTimeout(() => writeActionDraft(ed), DRAFT_DEBOUNCE_MS);
    return () => clearTimeout(t);
  }, [ed]);

  // ── Debounced autosave: push to the server AUTOSAVE_MS after editing stops ──
  const saveRef = useRef(save);
  saveRef.current = save;
  useEffect(() => {
    if (!ed.dirty || !ed.video) return;
    const t = setTimeout(() => void saveRef.current(true), AUTOSAVE_MS);
    return () => clearTimeout(t);
  }, [ed]);
  const copyVideoName = async () => {
    const name = ed.video || picked;
    if (!name) return toast.warning('No video loaded');
    try {
      await copyText(name);
      toast.success(`Copied ${name}`);
    } catch {
      toast.error('Copy failed');
    }
  };

  const jumpToEvent = (idx: number) => {
    const evt = ed.events[idx];
    if (!evt) return;
    setSelectedIdx(idx);
    if (evt.rally_id) {
      setSelectedRallyId(evt.rally_id);
      setExpanded(String(evt.rally_id));
    } else {
      setSelectedRallyId('all');
      setExpanded(OUTSIDE);
    }
    videoRef.current?.pause();
    seekFrame(evt.frame);
  };
  const selectRally = (id: number | 'all', seek = true) => {
    setSelectedRallyId(id);
    setExpanded(id === 'all' ? null : String(id));
    if (seek && id !== 'all') {
      const r = ed.rallies.find((x) => x.rally_id === id);
      if (r) seekFrame(Math.round(r.start * ed.fps));
    }
  };
  const stepRally = (d: number) => {
    if (!ed.rallies.length) return;
    const i = ed.rallies.findIndex((r) => r.rally_id === selectedRallyId);
    const ni = i < 0 ? (d > 0 ? 0 : ed.rallies.length - 1) : clamp(i + d, 0, ed.rallies.length - 1);
    selectRally(ed.rallies[ni]!.rally_id);
  };

  // ── Keyboard ──
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // Ctrl/Cmd+S saves from anywhere — checked before the input guard so it
      // works mid-typing, and preventDefault blocks the browser's save dialog.
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') {
        e.preventDefault();
        void save();
        return;
      }
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA') return;
      // On a focused <select> (e.g. the per-action label dropdown) space would
      // open the native menu — but space must always be play/pause. Hijack just
      // space here and leave every other key to the native select.
      if (tag === 'SELECT') {
        if (e.key === ' ') {
          e.preventDefault();
          togglePlay();
        }
        return;
      }
      if (e.key >= '1' && e.key <= '6') {
        const l = labels[Number(e.key) - 1];
        if (l) setSelectedLabel(l);
        return;
      }
      if (e.key === ' ') {
        e.preventDefault();
        togglePlay();
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        stepFrame(e.shiftKey ? -10 : -1);
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        stepFrame(e.shiftKey ? 10 : 1);
      } else if (e.key === 'Enter') {
        e.preventDefault();
        addEvent(0.5, 0.5);
      } else if (e.key.toLowerCase() === 'p') {
        e.preventDefault();
        setPointMode((m) => !m);
      } else if ((e.key === 'Delete' || e.key === 'Backspace') && selectedIdx >= 0) {
        deleteEvent(selectedIdx);
      }
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  });

  const eventsByRally = (rid: number) => ed.events.map((e, idx) => ({ e, idx })).filter(({ e }) => e.rally_id === rid);
  const outside = ed.events.map((e, idx) => ({ e, idx })).filter(({ e }) => !e.rally_id);

  return (
    <div className="mx-auto max-w-screen-2xl space-y-5">
      {/* Picker */}
      <Card>
        <div className="grid grid-cols-1 items-end gap-3 lg:grid-cols-[8.5rem_8.5rem_minmax(18rem,1fr)_auto]">
          <FieldLabel label="Kind">
            <select value={kindFilter} onChange={(e) => setKindFilter(e.target.value as typeof kindFilter)} className={cn(fieldCls, 'h-9 w-full py-0')}>
              <option value="all">All kinds</option>
              <option value="broadcast">Broadcast</option>
              <option value="sideline">Sideline</option>
            </select>
          </FieldLabel>
          <FieldLabel label="Status">
            <select value={progressFilter} onChange={(e) => setProgressFilter(e.target.value as typeof progressFilter)} className={cn(fieldCls, 'h-9 w-full py-0')}>
              <option value="all">All</option>
              <option value="unlabeled">Unlabeled</option>
              <option value="pre-labeled">Pre-labeled</option>
              <option value="labeled">Labeled</option>
            </select>
          </FieldLabel>
          <FieldLabel label="Video">
            <VideoCombobox
              items={filtered}
              value={picked}
              onChange={setPicked}
              placeholder="Type to search filename…"
              renderItem={(v) => (
                <>
                  <KindBadge kind={v.kind} />
                  <span className={cn('shrink-0 text-[11px]', isReviewed(v) ? 'text-primary-light' : hasActive(v) ? 'text-amber-300' : 'text-text-muted')}>{isReviewed(v) ? '✓' : hasActive(v) ? 'P' : '○'}</span>
                  <span className="min-w-0 flex-1 break-all font-mono">{v.name}</span>
                  <span className="shrink-0 text-[10px] text-text-muted">{isReviewed(v) ? `${v.event_count || 0} labeled` : hasActive(v) ? `${v.event_count || 0} pre` : ''}</span>
                </>
              )}
            />
          </FieldLabel>
          <div className="flex items-stretch gap-2">
            <Button intent="primary" className="h-9 py-0" onClick={() => load(picked)} disabled={loading || !picked}>
              {loading ? 'Loading…' : 'Load'}
            </Button>
            <Button className="h-9 py-0" onClick={copyVideoName}>
              Copy Filename
            </Button>
          </div>
        </div>
      </Card>

      <div className="flex flex-col gap-5 lg:flex-row">
        {/* Player */}
        <div className="min-w-0 flex-1 space-y-4">
          <Card>
            <div className="mb-3 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
              <h3 className="font-heading text-sm font-semibold text-text-primary">Action Labels</h3>
              <div className="grid grid-cols-3 gap-2 sm:grid-cols-6">
                {labels.map((l, i) => {
                  const active = l === selectedLabel;
                  const color = actionColor(l);
                  return (
                    <button
                      key={l}
                      type="button"
                      onClick={() => setSelectedLabel(l)}
                      className={cn('rounded-lg border px-3 py-2 font-heading text-xs font-semibold capitalize transition-colors', active ? 'text-white' : 'text-text-secondary hover:text-text-primary')}
                      style={{ borderColor: active ? color : 'var(--line)', background: active ? `${color}33` : 'transparent' }}
                    >
                      <span className="opacity-60">{i + 1}</span> {l}
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="overflow-hidden rounded-2xl bg-black ring-1 ring-white/[0.06]">
              <div
                ref={wrapRef}
                className="relative mx-auto"
                style={{ aspectRatio: `${aspect}`, maxWidth: `calc(var(--video-max-h, 45vh) * ${aspect})` }}
              >
                <video
                  ref={videoRef}
                  className={cn('block h-full w-full bg-black object-contain', pointMode && ed.video && 'cursor-crosshair')}
                  playsInline
                  preload="metadata"
                  onClick={onVideoClick}
                  onContextMenu={onVideoContextMenu}
                  onLoadedMetadata={onVideoMetadata}
                />
                <div className="pointer-events-none absolute inset-0">
                  {ed.events
                    .map((e, idx) => ({ e, idx }))
                    .filter(({ e }) => e.visible && (selectedRallyId === 'all' || e.rally_id === selectedRallyId) && Math.abs(e.frame - frame) <= 2)
                    .map(({ e, idx }) => {
                      const color = actionColor(e.label);
                      return (
                        <button
                          key={e.id}
                          type="button"
                          onPointerDown={(ev) => startDrag(ev, e, idx)}
                          onClick={(ev) => {
                            ev.stopPropagation();
                            if (!suppressClick.current) jumpToEvent(idx);
                          }}
                          className="pointer-events-auto absolute -ml-3 -mt-3 h-6 w-6 cursor-grab touch-none active:cursor-grabbing"
                          style={{ left: `${e.xy[0] * 100}%`, top: `${e.xy[1] * 100}%` }}
                          title={`${e.label} frame ${e.frame}`}
                        >
                          {e.frame === frame && (
                            <span className="absolute left-1/2 top-1/2 h-5 w-5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white/90" style={{ boxShadow: `0 0 0 1px ${color}88` }} />
                          )}
                          <span className="absolute left-1/2 top-1/2 h-2 w-2 -translate-x-1/2 -translate-y-1/2 rounded-full border border-white/85" style={{ background: color, boxShadow: `0 0 0 1px ${color}55` }} />
                        </button>
                      );
                    })}
                </div>
              </div>
            </div>

            {/* Zoomable timeline + waveform (All / 10m / 5m / 3m, rally bands R1, R2 …) */}
            <div className="mt-3">
              <ActionTimeline
                duration={ed.duration}
                fps={ed.fps}
                numFrames={ed.numFrames}
                frame={frame}
                rallies={ed.rallies}
                events={ed.events}
                selectedRallyId={selectedRallyId}
                selectedIdx={selectedIdx}
                playing={playing}
                waveform={waveform}
                colors={ACTION_COLORS}
                onSeekFrame={seekFrame}
                onJumpEvent={jumpToEvent}
              />
            </div>
            <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
              <span className="rounded-lg border border-border bg-surface-200/50 px-2.5 py-1 font-mono text-sm tabular-nums text-text-primary">
                {fmt(frame / (ed.fps || 30))} / f{frame}
              </span>
              <div className="flex items-center gap-2">
                <Button size="sm" onClick={togglePlay}>
                  Play
                </Button>
                <Button size="sm" onClick={() => stepFrame(-1)}>
                  ◂
                </Button>
                <Button size="sm" onClick={() => stepFrame(1)}>
                  ▸
                </Button>
                <Button size="sm" intent={pointMode ? 'primary' : 'default'} onClick={() => setPointMode((m) => !m)} title="Point mode: click the video to drop the selected action">
                  {pointMode ? 'Point mode' : 'Review mode'}
                </Button>
                <Button size="sm" intent="primary" onClick={() => addEvent(0.5, 0.5)}>
                  Add center
                </Button>
              </div>
            </div>
            <div className="mt-2 font-mono text-[11px] tabular-nums text-text-muted">{ed.video ? `${ed.fps.toFixed(3)} fps · ${ed.numFrames} frames` : ''}</div>
          </Card>
          <p className="px-1 text-[11px] text-text-muted">
            <kbd className="rounded bg-surface-200 px-1.5 py-0.5 font-mono text-[10px] text-text-secondary">1-6</kbd> label ·{' '}
            <kbd className="rounded bg-surface-200 px-1.5 py-0.5 font-mono text-[10px] text-text-secondary">← →</kbd> frame ·{' '}
            <kbd className="rounded bg-surface-200 px-1.5 py-0.5 font-mono text-[10px] text-text-secondary">Enter</kbd> add ·{' '}
            <kbd className="rounded bg-surface-200 px-1.5 py-0.5 font-mono text-[10px] text-text-secondary">P</kbd> point mode ·{' '}
            <kbd className="rounded bg-surface-200 px-1.5 py-0.5 font-mono text-[10px] text-text-secondary">Del</kbd> remove
          </p>
        </div>

        {/* Rallies + events */}
        <div className="min-w-0 lg:w-[420px] lg:flex-shrink-0">
          <Card>
            <div className="mb-3 flex items-center justify-between gap-2">
              <SectionLabel className="mb-0">
                Rallies ({ed.rallies.length} rally · {ed.events.length} action){ed.dirty ? ' ·' : ''}
              </SectionLabel>
              <div className="flex items-center gap-2">
                <Button size="sm" intent="primary" onClick={() => void save()}>
                  {ed.dirty ? 'Save •' : 'Save'}
                </Button>
              </div>
            </div>
            <div className="mb-2 flex items-center gap-2">
              <select value={selectedRallyId} onChange={(e) => selectRally(e.target.value === 'all' ? 'all' : Number(e.target.value))} className={cn(fieldCls, 'flex-1 text-xs')}>
                <option value="all">All rallies ({ed.events.length})</option>
                {ed.rallies.map((r, i) => (
                  <option key={r.rally_id} value={r.rally_id}>
                    R{i + 1} · {fmt(r.start)}-{fmt(r.end)} · {eventsByRally(r.rally_id).length}
                  </option>
                ))}
              </select>
              <Button size="sm" onClick={() => stepRally(-1)} disabled={!ed.rallies.length}>
                Prev
              </Button>
              <Button size="sm" onClick={() => stepRally(1)} disabled={!ed.rallies.length}>
                Next
              </Button>
            </div>
            <div className="h-px bg-border" />

            <div className="mt-2 max-h-[calc(100vh-18rem)] space-y-1.5 overflow-y-auto pr-1">
              {!ed.video ? (
                <EmptyState icon={<DotIcon />} title="No video loaded" />
              ) : ed.rallies.length === 0 && outside.length === 0 ? (
                <EmptyState icon={<DotIcon />} title="No rally annotations" />
              ) : (
                <>
                  {ed.rallies.map((rally, i) => {
                    const entries = eventsByRally(rally.rally_id);
                    const isOpen = expanded === String(rally.rally_id);
                    const sel = selectedRallyId === rally.rally_id;
                    const t = frame / (ed.fps || 30);
                    const live = t >= rally.start && t < rally.end;
                    return (
                      <div key={rally.rally_id} className="space-y-1.5">
                        <div
                          onClick={() => selectRally(rally.rally_id)}
                          className={cn(
                            'flex cursor-pointer items-center gap-2.5 rounded-xl border px-3 py-2.5 transition-colors',
                            sel ? 'border-primary/40 bg-primary/[0.1]' : 'border-primary/15 bg-primary/[0.04] hover:bg-primary/[0.08]',
                            live && 'ring-1 ring-accent/50',
                          )}
                        >
                          <span className="w-4 select-none text-right font-heading text-[10px] text-text-muted/60">{i + 1}</span>
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              // Collapse if open; otherwise select + expand + seek to the rally start.
                              if (isOpen) setExpanded(null);
                              else selectRally(rally.rally_id);
                            }}
                            className="flex items-center gap-1 rounded-full bg-primary/20 px-2 py-0.5 text-[11px] font-medium text-primary-text ring-1 ring-primary/25"
                          >
                            <span className={cn('transition-transform', isOpen && 'rotate-90')}>▸</span> actions <span className="opacity-70">{entries.length}</span>
                          </button>
                          <span className="ml-auto font-mono text-[11px] tabular-nums text-text-muted">
                            {fmt(rally.start)} → {fmt(rally.end)}
                          </span>
                          <span className="rounded bg-surface-200/40 px-1.5 py-0.5 font-mono text-[10px] tabular-nums text-text-muted">{Math.max(0, rally.end - rally.start).toFixed(1)}s</span>
                        </div>
                        {isOpen && <EventPanel entries={entries} empty="No actions in this rally" {...{ labels, selectedIdx, fps: ed.fps, frame, onEdit: editEvent, onDelete: deleteEvent, onJump: jumpToEvent }} />}
                      </div>
                    );
                  })}
                  {outside.length > 0 && (
                    <div className="space-y-1.5">
                      <div onClick={() => setExpanded(OUTSIDE)} className="flex cursor-pointer items-center gap-2.5 rounded-xl border border-amber-500/20 bg-amber-500/[0.04] px-3 py-2.5 hover:bg-amber-500/[0.08]">
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
                      {expanded === OUTSIDE && <EventPanel entries={outside} empty="No outside actions" {...{ labels, selectedIdx, fps: ed.fps, frame, onEdit: editEvent, onDelete: deleteEvent, onJump: jumpToEvent }} />}
                    </div>
                  )}
                </>
              )}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}

const fieldCls = 'rounded-lg border border-border-light bg-surface-50 px-3 py-2 text-sm text-text-primary focus:border-primary/50 focus:outline-none';
const clampToRally = (frame: number, ed: Editor, rid: number | 'all') => {
  if (rid === 'all') return frame;
  const r = ed.rallies.find((x) => x.rally_id === rid);
  if (!r) return frame;
  const sf = Math.max(0, Math.round(r.start * ed.fps));
  const ef = Math.max(sf, Math.ceil(r.end * ed.fps) - 1);
  return clamp(frame, sf, Math.min(ef, Math.max(0, ed.numFrames - 1)));
};

function FieldLabel({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="block min-w-0 space-y-1.5">
      <span className="block text-[10px] font-semibold uppercase tracking-widest text-text-muted">{label}</span>
      {children}
    </label>
  );
}

function DotIcon() {
  return (
    <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v12m6-6H6" />
    </svg>
  );
}

interface EventPanelProps {
  entries: Array<{ e: ActionEvent; idx: number }>;
  empty: string;
  labels: string[];
  selectedIdx: number;
  fps: number;
  /** Current playhead frame — rows within ±½ s light up (Rally Label rule). */
  frame: number;
  onEdit: (idx: number, patch: Partial<ActionEvent>) => void;
  onDelete: (idx: number) => void;
  onJump: (idx: number) => void;
}
function EventPanel({ entries, empty, labels, selectedIdx, fps, frame, onEdit, onDelete, onJump }: EventPanelProps) {
  if (!entries.length) return <div className="ml-6 rounded-xl border border-border bg-surface-100 px-3 py-2 text-xs text-text-muted">{empty}</div>;
  const windowFrames = Math.max(1, Math.round((fps || 30) / 2));
  return (
    <div className="ml-6 space-y-1.5 rounded-xl border border-border bg-surface-100 p-2">
      {entries.map(({ e, idx }, row) => {
        const color = actionColor(e.label);
        const active = Math.abs(e.frame - frame) <= windowFrames;
        return (
          <div
            key={e.id}
            onClick={() => onJump(idx)}
            className={cn(
              'grid cursor-pointer grid-cols-[1rem_minmax(5rem,1fr)_3.6rem_2.6rem_2.4rem] items-center gap-1.5 rounded-lg border px-2 py-1.5 transition-colors',
              idx === selectedIdx ? 'border-primary/35 bg-primary/10' : 'border-border bg-surface-50 hover:bg-surface-200/40',
              active && 'ring-1 ring-accent/50',
            )}
          >
            <span className="text-right font-heading text-[10px] text-text-muted/70">{row + 1}</span>
            <span className="flex min-w-0 items-center gap-1.5" onClick={(ev) => ev.stopPropagation()}>
              <button
                type="button"
                onClick={() => onEdit(idx, { visible: !e.visible })}
                className={cn('h-2.5 w-2.5 flex-shrink-0 rounded-full', !e.visible && 'border')}
                style={e.visible ? { background: color } : { borderColor: color }}
                title={e.visible ? 'Visible — click to hide' : 'Non-visible — click to show'}
              />
              <select value={e.label} onChange={(ev) => onEdit(idx, { label: ev.target.value })} className="w-full min-w-0 rounded-lg border border-border bg-surface-100 px-1.5 py-1 text-xs text-text-primary">
                {labels.map((l) => (
                  <option key={l} value={l}>
                    {l}
                  </option>
                ))}
              </select>
            </span>
            <input
              value={e.frame}
              onClick={(ev) => ev.stopPropagation()}
              onChange={(ev) => onEdit(idx, { frame: Math.max(0, Math.round(Number(ev.target.value) || 0)) })}
              className="w-full border-0 border-b border-white/10 bg-transparent text-center font-heading text-[11px] tabular-nums text-text-primary focus:border-primary-light focus:outline-none"
            />
            <span className="text-center font-heading text-[10px] tabular-nums text-text-muted">{fmt(e.frame / (fps || 30))}</span>
            <span className="flex items-center justify-end gap-1" onClick={(ev) => ev.stopPropagation()}>
              <button type="button" onClick={() => onJump(idx)} className="text-primary-light hover:text-text-primary" title="Jump to event">
                →
              </button>
              <button type="button" onClick={() => onDelete(idx)} className="text-red-400/60 hover:text-red-400" title="Delete">
                ✕
              </button>
            </span>
          </div>
        );
      })}
    </div>
  );
}
