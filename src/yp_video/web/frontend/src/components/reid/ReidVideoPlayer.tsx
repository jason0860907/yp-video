/** Video player whose overlay mirrors the ReID results: the event box draws
 *  on its exact annotated frame, tracklets follow the previous/next action,
 *  and pick mode turns every stored detection into a clickable actor choice.
 *  Ships with the rally sidebar (same interaction as the Action Label list). */

import { forwardRef, useEffect, useImperativeHandle, useMemo, useRef, useState, type MouseEvent as ReactMouseEvent } from 'react';
import { useQuery } from '@tanstack/react-query';
import { API, apiFetch } from '@/lib/api';
import { cn } from '@/lib/cn';
import { actionColor } from '@/lib/actionColors';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { RallyTimeline } from '@/components/editor/RallyTimeline';
import type { EditorAnnotation } from '@/components/editor/AnnotationEditor';
import type { ReidPlayers, ReidRecord } from '@/types/api';
import { fmtTime, trackColor, type ActorFix, type Rally, type SidebarAction, type TrackData, type TrackMasks } from './shared';

// Detections below this score never win automatic association — they exist
// only as manual-picker choices (mirrors reid/detector.AUTO_PICK_MIN_SCORE).
const AUTO_PICK_MIN_SCORE = 0.5;

const OUTSIDE = '__outside__';

export interface PlayerHandle {
  /** Park the video on an event's frame, select + expand its rally, and pin
   *  that rally to the top of the sidebar list. */
  jumpToEvent: (a: { id: string; frame: number; time: number | null }) => void;
}

export interface ReidVideoPlayerProps {
  src: string;
  /** The picked video's name — track-mask lookups key on it. */
  videoName: string;
  /** Raw tracklets (frames arrays align mask rows to the playhead). */
  tracklets: TrackData['tracklets'];
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
  /** An actor fix is in flight (re-crop + re-embed server-side) — the picker
   *  dims and ignores clicks so it can't fire twice. */
  fixing?: boolean;
  onJumpToCrop: (eventId: string) => void;
  /** frame → ByteTrack boxes, for the tracklet overlay (empty = no tracking). */
  trackBoxes: Map<number, { key: string; trackId: number; box: [number, number, number, number] }[]>;
  /** ALL action events, frame-sorted; ``key`` is the event's tracklet (null =
   *  not linked). Only the previous and next action's tracklets draw
   *  persistently — an unlinked slot draws nothing rather than letting a
   *  later action's tracklet take its place. */
  trackEventTimeline: { frame: number; key: string | null; label?: string; player?: string }[];
}

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

export const ReidVideoPlayer = forwardRef<PlayerHandle, ReidVideoPlayerProps>(function ReidVideoPlayer(
  { src, videoName, tracklets, fps, frameSize, records, actionEvents, matches, rallies, selectedRally, onSelectRally, onFixActor, fixing = false, onJumpToCrop, trackBoxes, trackEventTimeline },
  ref,
) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [frame, setFrame] = useState(0);
  const [duration, setDuration] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [showTracks, setShowTracks] = useState(false);
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
      // floor, not round: seeks park mid-frame ((f + 0.5) / fps), and the
      // frame under a timestamp is floor(t·fps) — round would land on f+1.
      setFrame(Math.floor(el.currentTime * fps));
    };
    el.addEventListener('seeked', onSeeked);
    return () => {
      alive = false;
      el.cancelVideoFrameCallback(id);
      el.removeEventListener('seeked', onSeeked);
    };
  }, [fps, src]);

  // Space = play/pause, ←/→ = step one frame (Shift: ten) — the same
  // contract as Action Label: text fields keep their keys for typing, the
  // seek slider hands space back so scrubbing → space "just works", and
  // range inputs keep the arrows for their native nudge.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const tag = target?.tagName;
      if (e.key === ' ') {
        if (tag === 'TEXTAREA') return;
        if (tag === 'INPUT' && (target as HTMLInputElement).type !== 'range') return;
        e.preventDefault();
        togglePlay();
        return;
      }
      if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
        if (tag === 'TEXTAREA' || tag === 'INPUT' || tag === 'SELECT') return;
        const el = videoRef.current;
        if (!el) return;
        e.preventDefault();
        el.pause();
        const step = (e.key === 'ArrowLeft' ? -1 : 1) * (e.shiftKey ? 10 : 1);
        const f = Math.max(0, Math.floor(el.currentTime * fps) + step);
        // Park mid-frame, mirroring seekEvent — floor(t·fps) lands back on f.
        el.currentTime = (f + 0.5) / fps;
      }
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [fps]);

  const [w, h] = frameSize;
  // The event box (person ∪ keypoints ∪ ball, +4% margin) belongs to ONE
  // frame — the action frame — so it only draws when the playhead is on it.
  const visible = records.filter((r) => r.box && r.frame === frame);
  const time = frame / fps;
  // The event whose actor is being picked: always the record NEAREST the
  // playhead (actions split the timeline at their midpoints), on every
  // frame — parking just before an action aims at that action, never the
  // previous one. The banner names the target and switches live as the
  // playhead crosses a midpoint.
  const pickTarget = useMemo(() => {
    if (!pickMode || !records.length) return null;
    return records.reduce((a, b) => (Math.abs(a.frame - frame) <= Math.abs(b.frame - frame) ? a : b));
  }, [pickMode, records, frame]);
  // Near the action frame the record's stored detections are ALSO offered
  // (they cover people without a tracklet, down to the score slider).
  const nearEvent = pickTarget != null && Math.abs(pickTarget.frame - frame) <= 2;

  // The rally under the playhead — masks are fetched per rally, whole
  // tracklets at once, so silhouettes render continuously like the boxes.
  const currentRallyId = useMemo(() => {
    const r = rallies.find((r) => frame >= r.start * fps && frame <= r.end * fps + 1);
    return r ? r.rally_id : null;
  }, [rallies, frame, fps]);

  // Playback follows along in the sidebar: entering a rally expands its
  // action list (whose rows already light up within ±½ s of the playhead)
  // and pins it to the top. Fires only on rally CHANGE while playing, so a
  // manual collapse mid-rally sticks.
  useEffect(() => {
    if (!playing || currentRallyId == null) return;
    setExpanded(String(currentRallyId));
    scrollRallyTop(currentRallyId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, currentRallyId]);

  const masksQuery = useQuery({
    queryKey: ['reid-track-masks', videoName, currentRallyId],
    queryFn: () => apiFetch<TrackMasks>(API.reid.trackMasks(videoName, currentRallyId!)),
    enabled: currentRallyId != null && trackBoxes.size > 0,
    staleTime: Infinity, // immutable per tracking run; jobs invalidate the key
    retry: false, // 404 = video tracked before masks existed → box fallback
  });

  // key → concatenated packed rows; row i ↔ the tracklet's i-th frame.
  const maskData = useMemo(() => {
    const d = masksQuery.data;
    if (!d) return null;
    const byKey = new Map<string, Uint8Array>();
    for (const [key, b64] of Object.entries(d.tracks)) {
      byKey.set(key, Uint8Array.from(atob(b64), (c) => c.charCodeAt(0)));
    }
    return { mh: d.mask_hw[0], mw: d.mask_hw[1], byKey };
  }, [masksQuery.data]);

  const frameRows = useMemo(() => {
    const m = new Map<string, Map<number, number>>();
    for (const t of tracklets) {
      const rows = new Map<number, number>();
      t.frames.forEach((f, i) => rows.set(f, i));
      m.set(`${t.rally_id}:${t.track_id}`, rows);
    }
    return m;
  }, [tracklets]);

  // The previous and next action's tracklets — the boxes AND silhouettes
  // both color by the action; everything else stays neutral.
  const activeTracks = useMemo(() => {
    let prevEv: (typeof trackEventTimeline)[number] | null = null;
    let nextEv: (typeof trackEventTimeline)[number] | null = null;
    for (const e of trackEventTimeline) {
      if (e.frame <= frame) prevEv = e;
      else {
        nextEv = e;
        break;
      }
    }
    const active = new Map<string, { label?: string; player?: string }>();
    // Unlinked slots (key null) still occupy prev/next — they just
    // contribute no box.
    if (nextEv?.key) active.set(nextEv.key, nextEv);
    if (prevEv?.key) active.set(prevEv.key, prevEv); // same track twice → the just-done action's color wins
    return active;
  }, [trackEventTimeline, frame]);

  // Every silhouette on the current frame: bits for hit testing + a tinted
  // data-URL — the action's color for the prev/next action's tracklets,
  // plain white for everyone else. Regenerated per presented frame — a
  // handful of 48×96 canvases, well under a millisecond.
  const frameSilhouettes = useMemo(() => {
    if (!maskData) return [];
    const list = trackBoxes.get(frame) ?? trackBoxes.get(frame - 1) ?? trackBoxes.get(frame + 1);
    if (!list) return [];
    const { mh, mw, byKey } = maskData;
    const rowBytes = (mh * mw) >> 3;
    const out = [];
    for (const t of list) {
      const rows = frameRows.get(t.key);
      const row = rows?.get(frame) ?? rows?.get(frame - 1) ?? rows?.get(frame + 1);
      const all = byKey.get(t.key);
      if (row === undefined || !all) continue;
      const bits = all.subarray(row * rowBytes, (row + 1) * rowBytes);
      const alpha = document.createElement('canvas');
      alpha.width = mw;
      alpha.height = mh;
      const alphaCtx = alpha.getContext('2d')!;
      const img = alphaCtx.createImageData(mw, mh);
      for (let i = 0; i < mw * mh; i++) {
        if ((bits[i >> 3]! >> (7 - (i & 7))) & 1) img.data[i * 4 + 3] = 255;
      }
      alphaCtx.putImageData(img, 0, 0);
      const tinted = document.createElement('canvas');
      tinted.width = mw;
      tinted.height = mh;
      const ctx = tinted.getContext('2d')!;
      const ev = activeTracks.get(t.key);
      ctx.fillStyle = ev ? actionColor(ev.label) : '#ffffff';
      ctx.fillRect(0, 0, mw, mh);
      ctx.globalCompositeOperation = 'destination-in';
      ctx.drawImage(alpha, 0, 0);
      out.push({ key: t.key, box: t.box, bits, mw, mh, url: tinted.toDataURL() });
    }
    return out;
  }, [maskData, trackBoxes, frame, frameRows, activeTracks]);

  const boxIou = (a: [number, number, number, number], b: [number, number, number, number]) => {
    const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
    const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
    const inter = ix * iy;
    const union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    return union > 0 ? inter / union : 0;
  };

  /** Resolve a clicked player into an actor fix for the pinned event.
   *
   *  Track reaches the event frame (±3) → the fix uses the box THERE,
   *  snapped onto the best-matching stored detection so keypoints carry
   *  over. Track doesn't reach it (the actor went undetected around the
   *  action) → cross-frame fix: the clicked frame's box goes through with
   *  its frame number, and the backend crops the pixels that actually
   *  contain the player. */
  const pickFromTrack = (key: string, clickedBox: [number, number, number, number]) => {
    const target = pickTarget;
    if (!target) return;
    let boxAtEvent: [number, number, number, number] | null = null;
    for (let d = 0; d <= 3 && !boxAtEvent; d++) {
      const hit = (trackBoxes.get(target.frame - d) ?? []).concat(trackBoxes.get(target.frame + d) ?? []).find((t) => t.key === key);
      if (hit) boxAtEvent = hit.box;
    }
    if (!boxAtEvent) {
      onFixActor(target.id, { box: clickedBox, frame });
      return;
    }
    let box = boxAtEvent;
    let best = 0.3;
    for (const d of target.detections ?? []) {
      const overlap = boxIou(d.box, boxAtEvent);
      if (overlap >= best) {
        best = overlap;
        box = d.box;
      }
    }
    onFixActor(target.id, { box });
  };

  // Every tracked player at the playhead is a pick target on EVERY frame in
  // pick mode: the whole box is clickable (silhouette bits only arbitrate
  // overlaps), so a click near a player always does something.
  const pickables = useMemo(() => {
    if (!pickMode) return [];
    const list = trackBoxes.get(frame) ?? trackBoxes.get(frame - 1) ?? trackBoxes.get(frame + 1) ?? [];
    return list.map((t) => ({ ...t, sil: frameSilhouettes.find((s) => s.key === t.key) ?? null }));
  }, [pickMode, trackBoxes, frame, frameSilhouettes]);

  const inMaskBits = (s: (typeof frameSilhouettes)[number], px: number, py: number): boolean => {
    const [x0, y0, x1, y1] = s.box;
    const gx = Math.floor(((px - x0) / (x1 - x0)) * s.mw);
    const gy = Math.floor(((py - y0) / (y1 - y0)) * s.mh);
    const i = gy * s.mw + gx;
    return Boolean((s.bits[i >> 3]! >> (7 - (i & 7))) & 1);
  };

  /** Who owns a pointer position: boxes containing it, silhouette hits
   *  first (they resolve overlaps), smallest box wins ties. */
  const pickableAt = (e: ReactMouseEvent<SVGElement>): string | null => {
    const svg = e.currentTarget.ownerSVGElement;
    if (!svg) return null;
    const pt = svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const p = pt.matrixTransform(svg.getScreenCTM()!.inverse());
    const inBox = pickables.filter((t) => p.x >= t.box[0] && p.x < t.box[2] && p.y >= t.box[1] && p.y < t.box[3]);
    if (!inBox.length) return null;
    const inMask = inBox.filter((t) => t.sil && inMaskBits(t.sil, p.x, p.y));
    const pool = inMask.length ? inMask : inBox;
    pool.sort(
      (a, b) => (a.box[2] - a.box[0]) * (a.box[3] - a.box[1]) - (b.box[2] - b.box[0]) * (b.box[3] - b.box[1]),
    );
    return pool[0]!.key;
  };

  // The player under the cursor — highlighted so it's obvious who a click
  // would pick.
  const [hoverKey, setHoverKey] = useState<string | null>(null);
  const pickClick = (e: ReactMouseEvent<SVGElement>) => {
    e.stopPropagation();
    const key = pickableAt(e);
    if (!key) return;
    const t = pickables.find((x) => x.key === key);
    if (t) pickFromTrack(key, t.box);
  };
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
              {/* ByteTrack tracklets at the playhead — under the event boxes.
                  Exact frame first; ±1 covers stride-decoded tracks. Only the
                  PREVIOUS and NEXT action's tracklets show persistently (solid,
                  the action's color) — at most two boxes at a time; the rest
                  (dashed, hashed hue) hide behind the Tracks toggle. */}
              {(() => {
                return (trackBoxes.get(frame) ?? trackBoxes.get(frame - 1) ?? trackBoxes.get(frame + 1))?.map((t) => {
                const ev = activeTracks.get(t.key);
                if (!ev && !showTracks) return null;
                const color = ev ? actionColor(ev.label) : trackColor(t.key);
                const [x0, y0, x1, y1] = t.box;
                const sil = frameSilhouettes.find((s) => s.key === t.key);
                return (
                  <g key={t.key} opacity={ev ? 0.95 : 0.85}>
                    {sil && !pickMode && (
                      // The player's instance mask, riding the box every
                      // frame — same lifetime as the box itself. (Pick mode
                      // renders its own clickable silhouettes.)
                      <image
                        href={sil.url}
                        x={x0}
                        y={y0}
                        width={x1 - x0}
                        height={y1 - y0}
                        preserveAspectRatio="none"
                        opacity={0.45}
                      />
                    )}
                    <rect
                      x={x0}
                      y={y0}
                      width={x1 - x0}
                      height={y1 - y0}
                      fill="none"
                      stroke={color}
                      strokeWidth={ev ? 3 : 2}
                      strokeDasharray={ev ? undefined : '5 4'}
                      vectorEffect="non-scaling-stroke"
                    />
                    <text
                      x={x0 + 3}
                      y={y1 - 5}
                      fill={color}
                      stroke="#000"
                      strokeWidth={4}
                      paintOrder="stroke"
                      fontSize={Math.round(h / (ev ? 44 : 52))}
                      fontWeight="bold"
                      fontFamily="ui-monospace, SF Mono, Menlo"
                    >
                      {ev ? `${ev.player ?? ev.label ?? ''} · t${t.trackId}` : `t${t.trackId}`}
                    </text>
                  </g>
                );
                });
              })()}
              {/* Pick surface — identical on every frame: each tracked
                  player's whole box is clickable, silhouettes render on top
                  when stored, hover brightens the resolved player. The pick
                  follows the clicked track back to the target event. */}
              {pickables.map((t) => (
                <g
                  key={`pick-${t.key}`}
                  className={
                    fixing
                      ? 'pointer-events-none opacity-40'
                      : cn('pointer-events-auto', hoverKey === t.key ? 'cursor-pointer' : 'cursor-default')
                  }
                  onClick={pickClick}
                  onMouseMove={(e) => setHoverKey(pickableAt(e))}
                  onMouseLeave={() => setHoverKey(null)}
                >
                  <rect
                    x={t.box[0]}
                    y={t.box[1]}
                    width={t.box[2] - t.box[0]}
                    height={t.box[3] - t.box[1]}
                    fill="transparent"
                    stroke="#fff"
                    strokeOpacity={hoverKey === t.key ? 0.95 : 0.4}
                    strokeWidth={hoverKey === t.key ? 2.5 : 1.5}
                    strokeDasharray={t.sil ? undefined : '3 4'}
                    vectorEffect="non-scaling-stroke"
                  />
                  {t.sil && (
                    <image
                      href={t.sil.url}
                      x={t.sil.box[0]}
                      y={t.sil.box[1]}
                      width={t.sil.box[2] - t.sil.box[0]}
                      height={t.sil.box[3] - t.sil.box[1]}
                      preserveAspectRatio="none"
                      opacity={hoverKey === t.key ? 0.9 : 0.5}
                    />
                  )}
                  <title>Click to set this player as the actor — the pick follows their track back to the action</title>
                </g>
              ))}
              {visible.map((r) => {
                const [x0, y0, x1, y1] = r.box!;
                const m = matches[r.id];
                // Same hue per action as the Action Label editor.
                const color = actionColor(r.label);
                const label = m ? m.player : r.label ?? '';
                return (
                  <g key={r.id}>
                    <rect
                      x={x0}
                      y={y0}
                      width={x1 - x0}
                      height={y1 - y0}
                      fill="none"
                      stroke={color}
                      strokeWidth={2.5}
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
              {/* Actor picker: near the action frame, every stored detection
                  above the score slider is ALSO clickable (covers people
                  without a tracklet). */}
              {nearEvent &&
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
                      strokeOpacity={d.score >= AUTO_PICK_MIN_SCORE ? 0.9 : 0.45}
                      strokeWidth={1.5}
                      strokeDasharray={d.score >= AUTO_PICK_MIN_SCORE ? undefined : '3 5'}
                      vectorEffect="non-scaling-stroke"
                      className={fixing ? 'pointer-events-none opacity-40' : 'pointer-events-auto cursor-pointer hover:fill-white/20'}
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
                {(() => {
                  // The action under the playhead (±½ s, same rule as the
                  // sidebar rows), named in its color.
                  const window = Math.max(1, Math.round(fps / 2));
                  const near = sidebarActions.filter((a) => Math.abs(a.frame - frame) <= window);
                  if (!near.length) return null;
                  const a = near.reduce((x, y) => (Math.abs(x.frame - frame) <= Math.abs(y.frame - frame) ? x : y));
                  return <span style={{ color: actionColor(a.label) }}> · {a.label}</span>;
                })()}
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
            {trackBoxes.size > 0 && (
              <label
                className="inline-flex flex-shrink-0 cursor-pointer items-center gap-1.5 text-xs text-text-secondary"
                title="Also show tracklets without an action event (dashed, one hue per track) — action tracklets always show in their action's color"
              >
                <input
                  type="checkbox"
                  checked={showTracks}
                  onChange={(e) => setShowTracks(e.target.checked)}
                  className="h-3.5 w-3.5 accent-primary"
                />
                Tracks
              </label>
            )}
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
                    {fixing
                      ? ' — applying…'
                      : nearEvent
                        ? ' — click the right player (tracked boxes anywhere, stored detections on this frame)'
                        : ' — click the right player; the pick follows their track back to the action'}
                  </span>
                  <span className="ml-auto flex items-center gap-3">
                    {nearEvent && (
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
                    )}
                    <Button size="sm" disabled={fixing} onClick={() => onFixActor(pickTarget.id, { none: true })} title="Nobody in frame is the actor — clears this event's crop">
                      No actor
                    </Button>
                    {pickTarget.box_source === 'manual' && (
                      <Button size="sm" disabled={fixing} onClick={() => onFixActor(pickTarget.id, {})} title="Discard the manual fix and re-run the automatic pick">
                        Revert to auto
                      </Button>
                    )}
                  </span>
                </>
              ) : (
                <span className="text-text-muted">No extracted actions to pick for — run ReID on this video first.</span>
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
