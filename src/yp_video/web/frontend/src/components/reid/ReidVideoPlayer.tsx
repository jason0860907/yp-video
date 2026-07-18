/** Video player whose overlay mirrors the ReID results: the event box draws
 *  on its exact annotated frame, tracklets follow the previous/next action,
 *  and pick mode turns every stored detection into a clickable actor choice.
 *  Ships with the rally sidebar (same interaction as the Action Label list). */

import { forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState, type MouseEvent as ReactMouseEvent } from 'react';
import { useQuery } from '@tanstack/react-query';
import { API, apiFetch } from '@/lib/api';
import { cn } from '@/lib/cn';
import { actionColor } from '@/lib/actionColors';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { RallyTimeline } from '@/components/editor/RallyTimeline';
import type { EditorAnnotation } from '@/components/editor/AnnotationEditor';
import type { ReidPlayers, ReidRecord } from '@/types/api';
import { OUTSIDE, ReidRallySidebar } from './ReidRallySidebar';
import { fmtTime, trackColor, trackKeyOf, type ActorFix, type Rally, type SidebarAction, type TrackData, type TrackMasks } from './shared';
import {
  buildFrameRows,
  buildFrameSilhouettes,
  buildTrackBoxes,
  maskRowNear,
  nearestFrame,
  pickableAt as resolvePickable,
  resolveActorFix,
  SilhouetteRenderer,
  trackBoxNearEvent,
  decodeMaskData,
  type Box,
  type Silhouette,
} from './masks';

// Detections below this score never win automatic association — they exist
// only as manual-picker choices (mirrors reid/detector.AUTO_PICK_MIN_SCORE).
const AUTO_PICK_MIN_SCORE = 0.5;

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
  /** Which tracklet each event's actor sits on (empty = no tracking run). */
  trackLinks: TrackData['links'];
}

export const ReidVideoPlayer = forwardRef<PlayerHandle, ReidVideoPlayerProps>(function ReidVideoPlayer(
  { src, videoName, tracklets, fps, frameSize, records, actionEvents, matches, rallies, selectedRally, onSelectRally, onFixActor, fixing = false, onJumpToCrop, trackLinks },
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

  // frame → ByteTrack boxes for the overlay and the picker (measured ~286k
  // boxes on a real cut, ~30 ms to build — rebuilt only per tracking run).
  const trackBoxes = useMemo(() => buildTrackBoxes(tracklets), [tracklets]);

  const [w, h] = frameSize;
  // The event box (person ∪ keypoints ∪ ball, +4% margin) belongs to its
  // action frame, and draws only there — an event's box on any other frame
  // is a stale rectangle over unrelated footage.
  const visible = useMemo(
    () => records.filter((r) => r.box && r.frame === frame),
    [records, frame],
  );
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
  // action list (whose rows light up within ±½ s of the playhead). Fires
  // only on rally CHANGE while playing, so a manual collapse mid-rally
  // sticks; the action-scroll effect below then tracks the playhead within.
  useEffect(() => {
    if (!playing || currentRallyId == null) return;
    setExpanded(String(currentRallyId));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, currentRallyId]);

  const masksQuery = useQuery({
    queryKey: ['reid-track-masks', videoName, currentRallyId],
    queryFn: () => apiFetch<TrackMasks>(API.reid.trackMasks(videoName, currentRallyId!)),
    enabled: currentRallyId != null && trackBoxes.size > 0,
    staleTime: Infinity, // immutable per tracking run; jobs invalidate the key
    // A rally's masks are ~4 MB of base64. Never re-fetched while mounted
    // (staleTime), but retiring them promptly once the playhead has moved on
    // keeps a scrub across many rallies from stacking tens of MB in the query
    // cache — re-fetching one is cheap next to holding all of them.
    gcTime: 60_000,
    retry: false, // 404 = video tracked before masks existed → box fallback
  });

  // Tinted silhouettes are cached across frames; a new payload retires them
  // (the URLs were tinted for the previous rally's tracklets).
  const renderer = useRef(new SilhouetteRenderer()).current;
  const maskData = useMemo(() => {
    renderer.clear();
    return decodeMaskData(masksQuery.data);
  }, [masksQuery.data, renderer]);

  const frameRows = useMemo(() => buildFrameRows(tracklets), [tracklets]);

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

  // The same actions carrying their tracklet (null = not linked). EVERY
  // action occupies a slot, so an unlinked one means "no box right now"
  // rather than letting a later action's tracklet take its place.
  const trackEventTimeline = useMemo(
    () =>
      sidebarActions.map((a) => ({
        frame: a.frame,
        key: trackKeyOf(trackLinks, a.id),
        label: a.label,
        // Assigned only — see the event-box label below.
        player: matches[a.id]?.assigned ? matches[a.id]?.player : undefined,
      })),
    [sidebarActions, trackLinks, matches],
  );

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
  // plain white for everyone else. The URLs come from the renderer's cache,
  // so a presented frame re-encodes only silhouettes it hasn't seen before.
  const frameSilhouettes = useMemo(
    () =>
      buildFrameSilhouettes(maskData, frameRows, trackBoxes, frame, (key) => {
        const ev = activeTracks.get(key);
        return ev ? actionColor(ev.label) : '#ffffff';
      }, renderer),
    [maskData, frameRows, trackBoxes, frame, activeTracks, renderer],
  );

  /** Resolve a clicked player into an actor fix for the pinned event, then
   *  apply it (see masks.resolveActorFix for the arbitration). */
  const pickFromTrack = (key: string, clickedBox: Box) => {
    const target = pickTarget;
    if (!target) return;
    // The clicked tracklet's box AND mask at (or next to) the event frame —
    // the mask's pixels belong to the clicked player even under heavy
    // overlap, so it arbitrates which stored detection is that player.
    const at = trackBoxNearEvent(trackBoxes, key, target.frame);
    const row = at && maskData ? maskRowNear(maskData, frameRows, key, at.frame) : null;
    const silhouette: Silhouette | null =
      at && row && maskData
        ? { key, box: at.box, bits: row.bits, mw: maskData.mw, mh: maskData.mh, row: row.row }
        : null;
    onFixActor(
      target.id,
      resolveActorFix({
        detections: target.detections ?? [],
        trackBox: at?.box ?? null,
        silhouette,
        clickedBox,
        clickedFrame: frame,
      }),
    );
  };

  // Every tracked player at the playhead is a pick target on EVERY frame in
  // pick mode: the whole box is clickable (silhouette bits only arbitrate
  // overlaps), so a click near a player always does something.
  const pickables = useMemo(() => {
    if (!pickMode) return [];
    const list = nearestFrame(trackBoxes, frame) ?? [];
    return list.map((t) => ({ ...t, sil: frameSilhouettes.find((s) => s.key === t.key) ?? null }));
  }, [pickMode, trackBoxes, frame, frameSilhouettes]);

  // Pick targets are the segmentation boxes, one per player. Stored
  // detections stand in ONLY where this frame has no segmentation box at all
  // (outside a rally, or a tracklet too short to survive ByteTrack — measured
  // 3.3% of events, up to 10% on some videos). Showing both would put two
  // near-identical white rectangles on every player: the seg box runs ~1.4x
  // wider than the detector's, so they never line up.
  const detectionFallback = pickMode && !pickables.length;

  /** Who a pointer event would pick, in frame coordinates. */
  const pickableAt = (e: ReactMouseEvent<SVGElement>): string | null => {
    const svg = e.currentTarget.ownerSVGElement;
    if (!svg) return null;
    const pt = svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const p = pt.matrixTransform(svg.getScreenCTM()!.inverse());
    return resolvePickable(pickables, p.x, p.y);
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
  // The action under the playhead (nearest by frame) — drives the sidebar
  // auto-scroll during playback.
  const currentActionId = useMemo(() => {
    if (!sidebarActions.length) return null;
    return sidebarActions.reduce((a, b) => (Math.abs(a.frame - frame) <= Math.abs(b.frame - frame) ? a : b)).id;
  }, [sidebarActions, frame]);

  // Actions within ±½ s of the playhead (the Rally Label highlight rule),
  // reduced to a Set whose IDENTITY only moves when the membership does.
  // The sidebar memo hangs off this: recomputing the ids every frame is
  // O(actions) and trivial, but handing the list a fresh Set 60 times a
  // second would re-render several hundred rows for nothing.
  const activeIdsRef = useRef<Set<string>>(new Set());
  const activeActionIds = useMemo(() => {
    const window = Math.max(1, Math.round(fps / 2));
    const ids = sidebarActions.filter((a) => Math.abs(a.frame - frame) <= window).map((a) => a.id);
    const prev = activeIdsRef.current;
    // Same membership as last frame -> hand back the SAME Set so the memo
    // downstream holds. (Comparing ids beats joining them into a key: an
    // action id may contain any character, a separator may not.)
    if (ids.length === prev.size && ids.every((id) => prev.has(id))) return prev;
    return (activeIdsRef.current = new Set(ids));
  }, [sidebarActions, frame, fps]);

  // Playback scrolls the sidebar to keep the current action in view (its
  // rally is already expanded). Only while playing, so manual scrolling and
  // clicks aren't yanked around.
  useEffect(() => {
    if (!playing || !currentActionId) return;
    scrollActionIntoView(currentActionId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, currentActionId]);

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

  // The sidebar is memoized against the frame clock, so its handlers must
  // keep a stable identity or the memo never holds.
  const jumpToRally = useCallback(
    (rally: Rally) => {
      onSelectRally(rally.rally_id);
      setExpanded(String(rally.rally_id));
      const el = videoRef.current;
      if (el) el.currentTime = rally.start + 0.5 / fps;
    },
    [onSelectRally, fps],
  );
  const selectAllRallies = useCallback(() => onSelectRally('all'), [onSelectRally]);

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
  /** Scroll an action row into the list's view, centering it, but only when
   *  it's actually off-screen — avoids constant re-centering jitter as the
   *  playhead crosses each action during playback. */
  const scrollActionIntoView = (id: string) => {
    requestAnimationFrame(() => {
      const list = listRef.current;
      const row = list?.querySelector<HTMLElement>(`[data-action-id="${CSS.escape(id)}"]`);
      if (!list || !row) return;
      const lr = list.getBoundingClientRect();
      const rr = row.getBoundingClientRect();
      if (rr.top < lr.top || rr.bottom > lr.bottom) {
        list.scrollTo({
          top: rr.top - lr.top + list.scrollTop - lr.height / 2 + rr.height / 2,
          behavior: 'smooth',
        });
      }
    });
  };

  const seekEvent = useCallback(
    (a: { id: string; frame: number; time: number | null }) => {
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
    },
    [rallies, fps, onSelectRally],
  );

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
              {nearestFrame(trackBoxes, frame)?.map((t) => {
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
              })}
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
                // Only an ASSIGNED match is this event's identity. match()
                // also hands every unassigned event its nearest centroid —
                // a suggestion, and drawing it on the box reads as fact.
                // After a re-pick the assignment is gone, so the box must
                // go back to naming the action, not a player.
                const label = m?.assigned ? m.player : r.label ?? '';
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
              {/* Fallback picker: this frame has no segmentation box, so the
                  extraction's stored detections are the only way to point at
                  anybody. Score slider decides how deep to show. */}
              {detectionFallback && nearEvent &&
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
                      : detectionFallback
                        ? ' — no tracking on this frame; click one of the stored detections'
                        : ' — click the right player; the pick follows their track back to the action'}
                  </span>
                  <span className="ml-auto flex items-center gap-3">
                    {detectionFallback && nearEvent && (
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

      {/* Rally list — same sidebar as the Rally Label / Action Label editors.
          Memoized against the frame clock: it takes the playhead pre-reduced
          to activeRallyId / activeActionIds, not the raw frame. */}
      {rallies.length > 0 && (
        <ReidRallySidebar
          rallies={rallies}
          byRally={byRally}
          outside={outside}
          totalActions={sidebarActions.length}
          fps={fps}
          matches={matches}
          activeRallyId={currentRallyId}
          activeActionIds={activeActionIds}
          expanded={expanded}
          selectedRally={selectedRally}
          selectedEventId={selectedEventId}
          listRef={listRef}
          onSelectAll={selectAllRallies}
          onJumpRally={jumpToRally}
          onSetExpanded={setExpanded}
          onJumpEvent={seekEvent}
          onJumpToCrop={onJumpToCrop}
        />
      )}
    </div>
  );
});
