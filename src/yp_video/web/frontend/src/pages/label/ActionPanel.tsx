/** Action Label panel: place per-frame action events on one video.
 *
 *  Extracted from the Action Label page — the picker moved to the parent,
 *  everything else (frame clock, on-video points, timeline, drafts,
 *  autosave, keyboard) is unchanged. The `video` prop drives loading; the
 *  old pick-time "discard unsaved changes?" confirm is now the registered
 *  dirty guard, so the parent asks BEFORE changing the video or mode.
 */

import { useEffect, useMemo, useRef, useState, type MouseEvent as ReactMouseEvent, type PointerEvent as ReactPointerEvent, type SyntheticEvent } from 'react';
import { useQuery } from '@tanstack/react-query';
import { API, apiFetch, apiUrl, errMsg } from '@/lib/api';
import { fieldCls } from '@/components/form/Field';
import { cn } from '@/lib/cn';
import { hasRealTime, usePlayheadHandover } from '@/lib/playheadHandover';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { EmptyState } from '@/components/ui/EmptyState';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { toast } from '@/components/feedback/toast';
import { confirm } from '@/components/feedback/confirm';
import { ActionTimeline } from '@/components/editor/ActionTimeline';
import { ActionEventPanel } from '@/components/action/ActionEventPanel';
import { useVideoRecovery } from '@/lib/useVideoRecovery';
import { useSerializedSave } from '@/lib/useSerializedSave';
import {
  DEFAULT_ACTION_LABELS,
  EMPTY_ACTION_EDITOR,
  OUTSIDE_RALLY_KEY,
  clamp,
  findRallyAtTime,
  formatActionTime,
  hasActiveActionAnnotation,
  makeActionId,
  normalizeActionEditor,
  round4,
  sortActionEvents,
  withActionRally,
  type ActionEditor,
} from '@/lib/actionEditorModel';
import { useActionWaveform } from '@/lib/useActionWaveform';
import { scrollActionIntoView, scrollRallyTop } from '@/lib/sidebarScroll';
import { ACTION_COLORS, actionColor } from '@/lib/actionColors';
import type { ActionAnnotationData, ActionEvent, ActionVideo } from '@/types/api';
import { actionStatus } from '@/lib/labelStatus';
import { STATUS_OPTIONS, type LabelSource, type LoadedSource, type ModeDescriptor, type PlaybackClock, type RegisterGuard } from './mode';

const ACTION_AUTOSAVE_MS = 2000;

export const ACTION_MODE: ModeDescriptor = {
  key: 'action',
  label: 'Action',
  statusOptions: STATUS_OPTIONS,
  status: actionStatus,
  matches: (row, status) => status === 'all' || actionStatus(row) === status,
  available: (row) => Boolean(row.action),
  hint: () => 'No cut video is listed for this annotation — Rally tab only',
  rowExtras: (row) => {
    const v = row.action;
    if (!v || !hasActiveActionAnnotation(v)) return null;
    return <span className="shrink-0 font-mono text-[10px] tabular-nums text-text-muted">{v.event_count || 0}ev</span>;
  },
  doneApi: (video) => API.actionAnnotate.done(video),
  listKey: 'action-videos',
  hasSources: true,
};

export function ActionPanel({ video, source = 'annotation', onLoaded, onSaved, registerGuard, clock }: { video: string; source?: LabelSource; onLoaded?: (s: LoadedSource) => void; onSaved?: () => void; registerGuard?: RegisterGuard; clock?: PlaybackClock }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  // The rally list's scroll box — the panel pins rows to its top through it.
  const listRef = useRef<HTMLDivElement>(null);
  const [pointMode, setPointMode] = useState(false);
  const [aspect, setAspect] = useState(16 / 9);
  const { waveform, loadWaveform } = useActionWaveform();
  const drag = useRef<{ id: string; moved: boolean } | null>(null);
  const suppressClick = useRef(false);
  const [selectedLabel, setSelectedLabel] = useState('serve');

  const [ed, setEd] = useState<ActionEditor>(EMPTY_ACTION_EDITOR);
  // Every persisted editor mutation advances this counter. Saves capture it
  // before sending so an older response can never mark newer work clean.
  const editRevision = useRef(0);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selectedRallyId, setSelectedRallyId] = useState<number | 'all'>('all');
  const [expanded, setExpanded] = useState<string | null>(null);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);

  const takeHandover = usePlayheadHandover(
    clock ? () => clock.read(video) : undefined,
    video,
  );

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
  const labels = labelsQuery.data?.labels ?? DEFAULT_ACTION_LABELS;

  // ── Frame clock ──
  const computeFrame = () => {
    const e = edRef.current;
    if (lockedFrame.current !== null) return clamp(lockedFrame.current, 0, Math.max(0, e.numFrames - 1));
    const el = videoRef.current;
    const t = presented.current != null && Number.isFinite(presented.current) ? presented.current : el?.currentTime || 0;
    return clamp(Math.round(t * (e.fps || 30)), 0, Math.max(0, e.numFrames - 1));
  };
  const prevPlayFrame = useRef(0);
  const refreshPlayhead = () => {
    const f = computeFrame();
    setFrame(f);
    const prev = prevPlayFrame.current;
    prevPlayFrame.current = f;
    // Auto-pause at the end of the selected rally during playback, so a rally
    // doesn't run on into the next one — but only when crossing the end from
    // inside the rally; a playhead parked beyond it must never trip this.
    const el = videoRef.current;
    const e = edRef.current;
    const rid = selRallyRef.current;
    if (!el || el.paused || rid === 'all' || !e.fps) return;
    const rally = e.rallies.find((r) => r.rally_id === rid);
    if (!rally) return;
    const startFrame = Math.round(rally.start * e.fps);
    const endFrame = Math.max(0, Math.ceil(rally.end * e.fps) - 1);
    if (prev >= startFrame && prev < endFrame && f >= endFrame) {
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
      // Parked at the selected rally's end — refreshPlayhead would pause again
      // on the very next tick, so play there means "replay the rally".
      const rid = selRallyRef.current;
      const { fps, rallies } = edRef.current;
      const rally = rid === 'all' || !fps ? undefined : rallies.find((r) => r.rally_id === rid);
      if (rally && computeFrame() >= Math.max(0, Math.ceil(rally.end * fps) - 1)) {
        seekFrame(Math.round(rally.start * fps));
      }
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
    // Resume at the position handed over from another tab's player — through
    // seekFrame so the frame clock stays in step with the seek. Deferred until
    // the element can actually seek: at loadedmetadata `seekable` may still be
    // empty (the src is a 302 to a presigned URL) and the seek is dropped
    // without error. seekFrame would still have locked the frame counter, so
    // the failure used to show up as a correct-looking readout over a video
    // sitting at 0.
    const t = takeHandover();
    if (t == null) return;
    const arriveAt = () => {
      const f = Math.round(t * (edRef.current.fps || 30));
      seekFrame(f);
      // Arriving from another tab means arriving at a POSITION, not at rally 1.
      // load() had to guess before it knew where the playhead would land; now
      // that it is known, point the sidebar at the rally that holds it.
      //
      // Looked up by the handed-over time rather than the rounded frame: at a
      // rally's first frame the rounding can land a hair before `start`. And
      // spans are half-open everywhere else, but the Rally tab parks the
      // playhead exactly ON a rally's end when it plays one through — arriving
      // there means arriving in that rally, so fall back to the rally holding
      // the frame before.
      const step = 1 / (edRef.current.fps || 30);
      const rally = findRallyAtTime(t, edRef.current) ?? findRallyAtTime(t - step, edRef.current);
      setSelectedRallyId(rally?.rally_id ?? 'all');
      setExpanded(rally ? String(rally.rally_id) : null);
      if (!rally) return;
      scrollRallyTop(listRef.current, rally.rally_id);
      // What waits on this tab is an action, not the rally that holds it: land
      // on the one nearest the arrival time within that rally. Selection only —
      // the playhead stays where the Rally tab left it, so the arrival position
      // is still what is on screen. A rally with nothing labeled yet leaves the
      // selection as the load left it (null).
      const near = edRef.current.events.reduce<ActionEvent | null>(
        (best, e) => (e.rally_id !== rally.rally_id || (best !== null && Math.abs(best.frame - f) <= Math.abs(e.frame - f)) ? best : e),
        null,
      );
      if (near) setSelectedId(near.id);
    };
    if (el.seekable.length > 0) arriveAt();
    else el.addEventListener('canplay', arriveAt, { once: true });
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

  const startDrag = (e: ReactPointerEvent, evt: ActionEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setSelectedId(evt.id);
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
        mutate((prev) => ({ ...prev, events: prev.events.map((x) => (x.id === drag.current!.id ? withActionRally({ ...x, frame: f }, prev) : x)) }));
      }
      const p = clientToPoint(ev.clientX, ev.clientY);
      if (!p) return;
      editRevision.current += 1;
      const current = edRef.current;
      const next = {
        ...current,
        dirty: true,
        events: current.events.map((x) =>
          x.id === drag.current!.id ? { ...x, xy: p } : x,
        ),
      };
      edRef.current = next;
      setEd(next);
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
      const current = edRef.current;
      const next = { ...current, events: sortActionEvents(current.events) };
      edRef.current = next;
      setEd(next);
    };
    document.addEventListener('pointermove', onMove, { signal: ac.signal });
    document.addEventListener('pointerup', onUp, { signal: ac.signal });
    document.addEventListener('pointercancel', onUp, { signal: ac.signal });
  };

  const load = async (name: string) => {
    if (!name) return;
    try {
      const data = await apiFetch<ActionAnnotationData>(
        API.actionAnnotate.annotation(name, { source }),
      );
      onLoaded?.(data.loaded_source ?? 'none');
      const next = normalizeActionEditor(data, labels);
      // Never reuse a revision: a response from the previous video must not
      // be able to compare equal and mark this editor clean.
      editRevision.current += 1;
      setEd(next);
      edRef.current = next;
      setSelectedId(null);
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
      toast.success(`Loaded ${next.events.length} event(s)`);
    } catch (e) {
      onLoaded?.('none');
      toast.error(`Load failed: ${errMsg(e)}`);
    }
  };

  // The parent owns picking. The dirty confirm already ran (registered guard
  // below), so a changed prop is a settled decision to leave. A Source
  // switch re-reads the same video from the newly chosen store.
  useEffect(() => {
    void load(video);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [video, source]);

  // Leaving with unsaved work: same confirm the old pick flow ran.
  useEffect(() => {
    if (!registerGuard) return;
    registerGuard(async () => {
      const cur = edRef.current;
      if (!cur.dirty) return true;
      return confirm({ title: 'Discard unsaved changes?', body: 'The current action labels have not been saved.', confirmText: 'Discard', variant: 'danger' });
    });
    return () => registerGuard(null);
  }, [registerGuard]);

  const mutate = (fn: (ed: ActionEditor) => ActionEditor) => {
    editRevision.current += 1;
    const next = { ...fn(edRef.current), dirty: true };
    edRef.current = next;
    setEd(next);
  };

  const addEvent = (x = 0.5, y = 0.5, visible = true) => {
    if (!ed.video) return toast.warning('Load a video first');
    const f = clampToRally(computeFrame(), ed, selectedRallyId);
    if (f !== computeFrame()) seekFrame(f);
    mutate((prev) => {
      const evt = withActionRally({ id: makeActionId(), rally_id: null, frame: f, time: null, relative_frame: null, label: selectedLabel, xy: [round4(x), round4(y)], visible }, prev);
      const events = sortActionEvents([...prev.events, evt]);
      setSelectedId(evt.id);
      if (evt.rally_id) {
        setSelectedRallyId(evt.rally_id);
        setExpanded(String(evt.rally_id));
      } else setExpanded(OUTSIDE_RALLY_KEY);
      return { ...prev, events };
    });
  };

  const editEvent = (id: string, patch: Partial<ActionEvent>) =>
    mutate((prev) => {
      const events = prev.events.map((e) => (e.id === id ? withActionRally({ ...e, ...patch }, prev) : e));
      return { ...prev, events: patch.frame !== undefined ? sortActionEvents(events) : events };
    });
  const deleteEvent = (id: string) => {
    setSelectedId(null);
    mutate((prev) => ({ ...prev, events: prev.events.filter((e) => e.id !== id) }));
  };

  const save = useSerializedSave({
    revision: editRevision,
    save: async ({ revision, silent }) => {
      // The ref is updated synchronously with every mutation. A queued save
      // can start before React's next render and still capture the new data.
      const snapshot = edRef.current;
      if (!snapshot.video) {
        if (!silent) toast.warning('No video loaded');
        return;
      }
      const video = snapshot.video;
      await apiFetch(API.actionAnnotate.annotations, {
        method: 'POST',
        body: {
          video,
          fps: snapshot.fps,
          num_frames: snapshot.numFrames,
          events: snapshot.events,
        },
      });
      // A request only owns the revision it sent. Mid-request edits retain
      // dirty=true; useSerializedSave queues the latest.
      if (editRevision.current === revision) {
        const next = { ...edRef.current, dirty: false };
        edRef.current = next;
        setEd(next);
      }
      // The save wrote the annotation store — that is what's on screen now.
      onLoaded?.('annotation');
      onSaved?.();
      if (!silent) {
        void videosQuery.refetch();
        toast.success('Action annotations saved');
      }
    },
    onError: (e) => {
      // Dirty state deliberately survives a failed request.
      toast.error(`Save failed: ${errMsg(e)}`);
    },
  });

  // ── Flush unsaved work when the page goes away ──
  // beforeunload only warns (it fires before the leave dialog is answered);
  // pagehide fires once leaving is settled, and keepalive lets the flush
  // request outlive the tab.
  useEffect(() => {
    const warn = (e: BeforeUnloadEvent) => {
      const cur = edRef.current;
      if (cur.dirty && cur.video) e.preventDefault();
    };
    const flush = () => {
      const cur = edRef.current;
      if (!cur.dirty || !cur.video) return;
      void fetch(apiUrl(API.actionAnnotate.annotations), {
        method: 'POST',
        keepalive: true,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video: cur.video,
          fps: cur.fps,
          num_frames: cur.numFrames,
          events: cur.events,
        }),
      });
    };
    window.addEventListener('beforeunload', warn);
    window.addEventListener('pagehide', flush);
    return () => {
      window.removeEventListener('beforeunload', warn);
      window.removeEventListener('pagehide', flush);
    };
  }, []);

  // ── Debounced autosave ──
  useEffect(() => {
    if (!ed.dirty || !ed.video) return;
    const t = setTimeout(() => void save(true), ACTION_AUTOSAVE_MS);
    return () => clearTimeout(t);
  }, [ed, save]);
  // ── Playback follows along in the sidebar ──
  // The rally under the playhead, and the nearest action inside it. Both are
  // ids, so the effects below fire on CHANGE, not on every frame tick.
  const currentRallyId = useMemo(() => {
    const t = frame / (ed.fps || 30);
    return ed.rallies.find((r) => t >= r.start && t < r.end)?.rally_id ?? null;
  }, [ed.rallies, ed.fps, frame]);
  const currentActionId = useMemo(() => {
    if (currentRallyId == null) return null;
    let best: ActionEvent | null = null;
    for (const e of ed.events) {
      if (e.rally_id !== currentRallyId) continue;
      if (!best || Math.abs(e.frame - frame) < Math.abs(best.frame - frame)) best = e;
    }
    return best?.id ?? null;
  }, [ed.events, currentRallyId, frame]);
  // Entering a rally opens its group. Only on rally change while playing, so
  // a manual collapse mid-rally sticks. Selection is left alone: the
  // selected rally is what auto-pauses playback at its end.
  useEffect(() => {
    if (!playing || currentRallyId == null) return;
    setExpanded(String(currentRallyId));
  }, [playing, currentRallyId]);
  // Keep the action being played on screen; paused, the list is the user's.
  useEffect(() => {
    if (!playing || !currentActionId) return;
    scrollActionIntoView(listRef.current, currentActionId);
  }, [playing, currentActionId]);

  // Point the sidebar at the rally holding this event and open that group.
  // A frame edit re-derives rally_id, so anything that moves an event has to
  // re-point the sidebar too — otherwise the row it just touched is left
  // sitting inside a collapsed group somewhere else.
  const revealEvent = (evt: ActionEvent) => {
    if (evt.rally_id) {
      setSelectedRallyId(evt.rally_id);
      setExpanded(String(evt.rally_id));
    } else {
      setSelectedRallyId('all');
      setExpanded(OUTSIDE_RALLY_KEY);
    }
  };
  const jumpToEvent = (id: string) => {
    const evt = ed.events.find((e) => e.id === id);
    if (!evt) return;
    setSelectedId(id);
    revealEvent(evt);
    scrollActionIntoView(listRef.current, id);
    videoRef.current?.pause();
    seekFrame(evt.frame);
  };
  // Nudge the selected action by whole frames and carry the playhead with it,
  // so the next nudge is judged against what is actually on screen. The seek
  // goes through seekFrame: writing currentTime directly desyncs the frame
  // clock (see onVideoMetadata).
  const nudgeEvent = (d: number) => {
    if (!selectedId) {
      toast.warning('先選一個動作');
      return;
    }
    const before = edRef.current.events.find((e) => e.id === selectedId);
    if (!before) return;
    const f = clamp(before.frame + d, 0, Math.max(0, edRef.current.numFrames - 1));
    if (f === before.frame) return;
    editEvent(selectedId, { frame: f });
    // mutate() writes edRef synchronously, so the re-homed event is readable
    // here, before React has re-rendered. Follow it only when the nudge pushed
    // it into a different rally — otherwise every keypress would yank the list
    // back and fight the user's own scrolling.
    const after = edRef.current.events.find((e) => e.id === selectedId);
    if (after && after.rally_id !== before.rally_id) {
      revealEvent(after);
      scrollRallyTop(listRef.current, after.rally_id ?? OUTSIDE_RALLY_KEY);
    }
    videoRef.current?.pause();
    seekFrame(f);
  };
  const selectRally = (id: number | 'all', seek = true) => {
    setSelectedRallyId(id);
    setExpanded(id === 'all' ? null : String(id));
    // The pick can come from far outside the visible list (Prev/Next, the
    // dropdown), and even a clicked row wants its freshly expanded actions
    // on screen rather than pushed below the fold.
    if (id !== 'all') scrollRallyTop(listRef.current, id);
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
      } else if ((e.key === 'a' || e.key === 'A') && !e.ctrlKey && !e.metaKey) {
        // Frame nudges for the selected action, on the left home row: the
        // arrows move the playhead, a/s move the event. Shift makes them 10.
        // Ctrl/Cmd+S returned long before this, and the modifier check leaves
        // Ctrl/Cmd+A (select all) to the browser.
        e.preventDefault();
        nudgeEvent(e.shiftKey ? -10 : -1);
      } else if ((e.key === 's' || e.key === 'S') && !e.ctrlKey && !e.metaKey) {
        e.preventDefault();
        nudgeEvent(e.shiftKey ? 10 : 1);
      } else if ((e.key === 'Delete' || e.key === 'Backspace') && selectedId) {
        deleteEvent(selectedId);
      }
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  });

  const eventsByRally = (rid: number) => ed.events.filter((e) => e.rally_id === rid);
  const outside = ed.events.filter((e) => !e.rally_id);

  return (
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
                onTimeUpdate={(e) => {
                  if (ed.video && hasRealTime(e.currentTarget)) {
                    clock?.write(video, e.currentTarget.currentTime);
                  }
                }}
              />
              <div className="pointer-events-none absolute inset-0">
                {ed.events
                  .filter((e) => e.visible && (selectedRallyId === 'all' || e.rally_id === selectedRallyId) && Math.abs(e.frame - frame) <= 2)
                  .map((e) => {
                    const color = actionColor(e.label);
                    return (
                      <button
                        key={e.id}
                        type="button"
                        onPointerDown={(ev) => startDrag(ev, e)}
                        onClick={(ev) => {
                          ev.stopPropagation();
                          if (!suppressClick.current) jumpToEvent(e.id);
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
              selectedId={selectedId}
              playing={playing}
              waveform={waveform}
              colors={ACTION_COLORS}
              onSeekFrame={seekFrame}
              onJumpEvent={jumpToEvent}
            />
          </div>
          <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
            <span className="rounded-lg border border-border bg-surface-200/50 px-2.5 py-1 font-mono text-sm tabular-nums text-text-primary">
              {formatActionTime(frame / (ed.fps || 30))} / f{frame}
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
          <kbd className="rounded bg-surface-200 px-1.5 py-0.5 font-mono text-[10px] text-text-secondary">A S</kbd> nudge frame ·{' '}
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
                  R{i + 1} · {formatActionTime(r.start)}-{formatActionTime(r.end)} · {eventsByRally(r.rally_id).length} · #{r.rally_id}
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

          <div ref={listRef} className="mt-2 max-h-[calc(100vh-18rem)] space-y-1.5 overflow-y-auto pr-1">
            {!ed.video ? (
              <EmptyState icon={<DotIcon />} title="No video loaded" />
            ) : ed.rallies.length === 0 && outside.length === 0 ? (
              <EmptyState icon={<DotIcon />} title="No rally annotations" />
            ) : (
              <>
                {ed.rallies.map((rally, ri) => {
                  const entries = eventsByRally(rally.rally_id);
                  const isOpen = expanded === String(rally.rally_id);
                  const sel = selectedRallyId === rally.rally_id;
                  const t = frame / (ed.fps || 30);
                  const live = t >= rally.start && t < rally.end;
                  return (
                    <div key={rally.rally_id} className="space-y-1.5">
                      <div
                        data-rally-row={rally.rally_id}
                        onClick={() => selectRally(rally.rally_id)}
                        className={cn(
                          'flex cursor-pointer items-center gap-2.5 rounded-xl border px-3 py-2.5 transition-colors',
                          sel ? 'border-primary/40 bg-primary/[0.1]' : 'border-primary/15 bg-primary/[0.04] hover:bg-primary/[0.08]',
                          live && 'ring-1 ring-accent/50',
                        )}
                      >
                        <span className="w-4 select-none text-right font-heading text-[10px] text-text-muted/60">{ri + 1}</span>
                        <span
                          className="w-7 select-none font-mono text-[9px] text-text-muted/40"
                          title={`rally_id ${rally.rally_id} — stable id, not the time order`}
                        >
                          #{rally.rally_id}
                        </span>
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
                          {formatActionTime(rally.start)} → {formatActionTime(rally.end)}
                        </span>
                        <span className="rounded bg-surface-200/40 px-1.5 py-0.5 font-mono text-[10px] tabular-nums text-text-muted">{Math.max(0, rally.end - rally.start).toFixed(1)}s</span>
                      </div>
                      {isOpen && <ActionEventPanel entries={entries} empty="No actions in this rally" {...{ labels, selectedId, fps: ed.fps, frame, onEdit: editEvent, onDelete: deleteEvent, onJump: jumpToEvent }} />}
                    </div>
                  );
                })}
                {outside.length > 0 && (
                  <div className="space-y-1.5">
                    <div data-rally-row={OUTSIDE_RALLY_KEY} onClick={() => setExpanded(OUTSIDE_RALLY_KEY)} className="flex cursor-pointer items-center gap-2.5 rounded-xl border border-amber-500/20 bg-amber-500/[0.04] px-3 py-2.5 hover:bg-amber-500/[0.08]">
                      <span className="w-4 select-none text-right font-heading text-[10px] text-text-muted/60">out</span>
                      <span className="w-7 select-none" />
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          setExpanded(expanded === OUTSIDE_RALLY_KEY ? null : OUTSIDE_RALLY_KEY);
                        }}
                        className="flex items-center gap-1 rounded-full bg-amber-500/15 px-2 py-0.5 text-[11px] font-medium text-amber-300 ring-1 ring-amber-500/25"
                      >
                        <span className={cn('transition-transform', expanded === OUTSIDE_RALLY_KEY && 'rotate-90')}>▸</span> outside <span className="opacity-70">{outside.length}</span>
                      </button>
                      <span className="ml-auto font-heading text-[11px] text-text-muted">outside rally</span>
                    </div>
                    {expanded === OUTSIDE_RALLY_KEY && <ActionEventPanel entries={outside} empty="No outside actions" {...{ labels, selectedId, fps: ed.fps, frame, onEdit: editEvent, onDelete: deleteEvent, onJump: jumpToEvent }} />}
                  </div>
                )}
              </>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}

const clampToRally = (frame: number, ed: ActionEditor, rid: number | 'all') => {
  if (rid === 'all') return frame;
  const r = ed.rallies.find((x) => x.rally_id === rid);
  if (!r) return frame;
  const sf = Math.max(0, Math.round(r.start * ed.fps));
  const ef = Math.max(sf, Math.ceil(r.end * ed.fps) - 1);
  return clamp(frame, sf, Math.min(ef, Math.max(0, ed.numFrames - 1)));
};


function DotIcon() {
  return (
    <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v12m6-6H6" />
    </svg>
  );
}
