/** Video player whose overlay mirrors the extraction records: the event box
 *  draws on its exact annotated frame, tracklets follow the previous/next
 *  action, and pick mode parks on an event's frame and turns that frame's
 *  2XLarge dense boxes into the actor choices. Ships with the rally sidebar
 *  (same interaction as Action Label).
 *
 *  Shared by both labeling pages, and the ONLY difference between them is
 *  ``onFixActor``: Association Label passes it and gets the picker, ReID
 *  Label omits it and gets a read-only view. Leaving it out removes the
 *  entry point entirely rather than disabling a visible control, so there is
 *  no path from that page to an actor write. */

import { forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { API, apiFetch } from '@/lib/api';
import { cn } from '@/lib/cn';
import { actionColor } from '@/lib/actionColors';
import { hasRealTime, seekWhenSeekable, usePlayheadHandover } from '@/lib/playheadHandover';
import { scrollActionIntoView, scrollRallyTop } from '@/lib/sidebarScroll';
import type { PlaybackClock } from '@/pages/label/mode';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { RallyTimeline } from '@/components/editor/RallyTimeline';
import type { EditorAnnotation } from '@/components/editor/AnnotationEditor';
import { useVideoKeys } from './useVideoKeys';
import { useFrameClock } from './useFrameClock';
import type { BoxCheckEntry, ReidPlayers, ReidRecord } from '@/types/api';
import { OUTSIDE, RallySidebar } from './RallySidebar';
import { canConfirm, fmtTime, rallyOf, trackColor, trackKeyOf, verdictOf, VERDICT, type ActorFix, type ActorVerdict, type Rally, type SidebarAction, type TrackData, type TrackLinks, type TrackMasks } from './shared';
import { buildFrameRows, buildFrameSilhouettes, buildTrackBoxes, nearestFrame, SilhouetteRenderer, decodeMaskData } from './masks';

// The pickable 2XLarge boxes: the person head's label cut (actor/person_labels
// PERSON_LABEL_MIN_SCORE) — weaker boxes draw dashed.
const DENSE_LABEL_SCORE = 0.4;
const DENSE_BOX_COLOR = '#67e8f9';
const LABEL_BOX_COLOR = '#fbbf24';

/** One keycap. Action Label spells these inline; naming it here keeps the
 *  keys in this bar from drifting apart. */
function Key({ children }: { children: React.ReactNode }) {
  return (
    <kbd className="rounded bg-surface-200 px-1.5 py-0.5 font-mono text-[10px] text-text-secondary">
      {children}
    </kbd>
  );
}

const NO_EVENT_BOXES: ReadonlyMap<string, BoxCheckEntry> = new Map();
/** Why an event is in the 2XLarge box-check queue (actor/box_style.py). */
const BOX_CHECK_HINT: Record<NonNullable<BoxCheckEntry['status']>, string> = {
  snapped: 'The label box is a 2XLarge box.',
  contested: "Another person's 2XLarge box overlaps the label box about as well as the best one.",
  unmatched: 'No 2XLarge box overlaps the label box by IoU ≥ 0.5.',
  not_covered: 'The dense 2XLarge pass did not cover this frame — nothing to pick; mark occluded or revert.',
  unresolved: 'The action was moved after the pick, and the box could not be followed to its new frame.',
};
/** Whether an event's label box needs a look: it is not a 2XLarge box. */
const needsBoxCheck = (e: BoxCheckEntry) => e.status != null && e.status !== 'snapped';

export interface PlayerHandle {
  /** Park the video on an event's frame, select + expand its rally, and pin
   *  that rally to the top of the sidebar list. */
  jumpToEvent: (a: { id: string; frame: number; time: number | null }) => void;
}

export interface EventVideoPlayerProps {
  src: string;
  /** The picked video's name — track-mask lookups key on it. */
  videoName: string;
  /** Shared playhead, so switching Label tabs resumes where you were.
   *  See lib/playheadHandover.ts for why this is not just currentTime. */
  clock?: PlaybackClock;
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
  /** Omit for a read-only player: no Pick Player button, no picker at all. */
  onFixActor?: (eventId: string, fix: ActorFix) => void;
  /** Endorse the automatic pick for the parked event. Omitted alongside
   *  onFixActor; when present the button stays visible and goes disabled
   *  once there is nothing left to endorse. */
  onConfirmActor?: (eventId: string) => void;
  /** Events still open to endorsement, and the whole-rally action for them —
   *  rendered as a per-rally button in the sidebar so a rally can be signed
   *  off without leaving the video. */
  confirmableIds?: ReadonlySet<string>;
  onConfirmRally?: (eventIds: string[]) => void;
  /** An actor fix is in flight (re-crop + re-embed server-side) — the picker
   *  dims and ignores clicks so it can't fire twice. */
  fixing?: boolean;
  /** Jump an event to its crop on the identities board; omitted when
   *  the page has no board (Association Label). */
  onJumpToCrop?: (eventId: string) => void;
  /** Which tracklet each event's actor sits on (empty = no tracking run). */
  trackLinks: TrackLinks;
  /** Every event's 2XLarge boxes and label box check, by id — pick mode's
   *  choices, and (status other than `snapped`) the box-check queue. */
  eventBoxes?: ReadonlyMap<string, BoxCheckEntry>;
}

export const EventVideoPlayer = forwardRef<PlayerHandle, EventVideoPlayerProps>(function EventVideoPlayer(
  { src, videoName, clock, tracklets, fps, frameSize, records, actionEvents, matches, rallies, selectedRally, onSelectRally, onFixActor, onConfirmActor, confirmableIds, onConfirmRally, fixing = false, onJumpToCrop, trackLinks, eventBoxes = NO_EVENT_BOXES },
  ref,
) {
  const takeHandover = usePlayheadHandover(
    clock ? () => clock.read(videoName) : undefined,
    videoName,
  );
  const [duration, setDuration] = useState(0);
  // The Action Label frame clock: with a rally selected, playback stops at
  // its end, and play from there replays it.
  const { videoRef, bindVideo, frame, playing, step, togglePlay } = useFrameClock({
    fps,
    numFrames: Math.round(duration * fps),
    rally: selectedRally === 'all' ? null : rallies.find((r) => r.rally_id === selectedRally),
  });
  const [showTracks, setShowTracks] = useState(false);
  // 2XLarge boxes below DENSE_LABEL_SCORE (dashed): often the occluded actor,
  // often clutter — the labeler decides whether they are on screen.
  const [showWeakBoxes, setShowWeakBoxes] = useState(true);
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
  useImperativeHandle(ref, () => ({
    jumpToEvent: (a: { id: string; frame: number; time: number | null }) => {
      const rally = seekEvent(a);
      scrollRallyTop(listRef.current, rally ? rally.rally_id : OUTSIDE);
      videoRef.current?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    },
  }));

  // Association Label passes onFixActor and gets the picker; ReID Label omits
  // it and gets a read-only player. Derived before the key handler so P can
  // stay inert on the read-only side.
  const canFix = Boolean(onFixActor);

  // Space / ←→ are the Label panels' shared keys (useVideoKeys) — the same
  // contract as Action Label. P = the picking mode, inert on the read-only
  // side; text fields keep it for typing.
  useVideoKeys(togglePlay, step);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (e.key !== 'p' && e.key !== 'P') return;
      if (tag === 'TEXTAREA' || tag === 'INPUT' || tag === 'SELECT') return;
      if (!canFix) return;
      e.preventDefault();
      setPickMode((m) => !m);
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [canFix]);

  // frame → ByteTrack boxes for the overlay and the picker (measured ~286k
  // boxes on a real cut, ~30 ms to build — rebuilt only per tracking run).
  const trackBoxes = useMemo(() => buildTrackBoxes(tracklets), [tracklets]);

  const [w, h] = frameSize;
  // The event box (segmentation person box ∪ ball, +4% margin) belongs to its
  // action frame, and draws only there — an event's box on any other frame
  // is a stale rectangle over unrelated footage.
  const visible = useMemo(
    () => records.filter((r) => r.box && r.frame === frame),
    [records, frame],
  );
  const time = frame / fps;
  // Read-only when the page did not hand us a way to write.
  // The event whose actor is being picked: always the record NEAREST the
  // playhead (actions split the timeline at their midpoints), on every
  // frame — parking just before an action aims at that action, never the
  // previous one. The banner names the target and switches live as the
  // playhead crosses a midpoint.
  const pickTarget = useMemo(() => {
    if (!canFix || !pickMode || !records.length) return null;
    return records.reduce((a, b) => (Math.abs(a.frame - frame) <= Math.abs(b.frame - frame) ? a : b));
  }, [canFix, pickMode, records, frame]);
  // A label is a box on the event's own frame, so picking happens there and
  // nowhere else: the target's 2XLarge boxes are clickable only on it.
  const onEventFrame = pickTarget != null && pickTarget.frame === frame;
  // The target's entry: its event frame's 2XLarge boxes, smallest on top so
  // a player standing in front of another stays clickable.
  const targetBoxes = pickTarget ? eventBoxes.get(pickTarget.id) : undefined;
  const pickBoxes = useMemo(
    () =>
      (targetBoxes?.boxes ?? [])
        .filter((d) => showWeakBoxes || d.score >= DENSE_LABEL_SCORE)
        .sort(
          (a, b) => (b.box[2] - b.box[0]) * (b.box[3] - b.box[1]) - (a.box[2] - a.box[0]) * (a.box[3] - a.box[1]),
        ),
    [targetBoxes, showWeakBoxes],
  );
  // Parked, pick mode lands on the target's frame — entering it, or the
  // playhead crossing to the next action, seeks there.
  const pickTargetId = pickTarget?.id;
  const pickTargetFrame = pickTarget?.frame;
  useEffect(() => {
    const el = videoRef.current;
    if (pickTargetFrame == null || playing || !el) return;
    el.currentTime = (pickTargetFrame + 0.5) / fps;
  }, [pickTargetId, pickTargetFrame, playing, fps, videoRef]);
  // The box-check queue: events whose label box is not a 2XLarge box.
  const boxCheckQueue = useMemo(
    () => new Map([...eventBoxes].filter(([, e]) => needsBoxCheck(e))),
    [eventBoxes],
  );

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
  }, [playing, currentRallyId]);

  const masksQuery = useQuery({
    queryKey: ['tracklet-masks', videoName, currentRallyId],
    queryFn: () => apiFetch<TrackMasks>(API.tracklets.masks(videoName, currentRallyId!)),
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

  // Every event's verdict — the sidebar shows it where a player name would
  // sit, so a reviewed event reads as resolved rather than forgotten.
  const verdicts = useMemo<ReadonlyMap<string, ActorVerdict>>(
    () => new Map(records.map((r) => [r.id, verdictOf(r)])),
    [records],
  );

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
    scrollActionIntoView(listRef.current, currentActionId);
  }, [playing, currentActionId]);

  // Actions grouped per rally, plus the ones outside any rally — mirrors the
  // Action Label sidebar.
  const { byRally, outside } = useMemo(() => {
    const map = new Map<number, SidebarAction[]>(rallies.map((r) => [r.rally_id, []]));
    const out: SidebarAction[] = [];
    for (const a of sidebarActions) {
      const rally = rallyOf(rallies, a, fps);
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
    [onSelectRally, fps, videoRef],
  );
  const selectAllRallies = useCallback(() => onSelectRally('all'), [onSelectRally]);

  const listRef = useRef<HTMLDivElement>(null);
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
    [rallies, fps, onSelectRally, videoRef],
  );

  // Step through the 2XLarge box-check queue in frame order from the
  // playhead (wrapping), straight into pick mode where its boxes draw. The
  // playhead is read through a ref so the memoized sidebar keeps holding.
  const frameRef = useRef(frame);
  useEffect(() => {
    frameRef.current = frame;
  }, [frame]);
  const stepBoxCheck = useCallback(
    (dir: 1 | -1) => {
      const queue = sidebarActions.filter((a) => boxCheckQueue.has(a.id));
      if (dir < 0) queue.reverse();
      const at = frameRef.current;
      const target = queue.find((a) => (dir > 0 ? a.frame > at : a.frame < at)) ?? queue[0];
      if (!target) return;
      const rally = seekEvent(target);
      scrollRallyTop(listRef.current, rally ? rally.rally_id : OUTSIDE);
      if (canFix) setPickMode(true);
    },
    [sidebarActions, boxCheckQueue, seekEvent, canFix],
  );

  const timelineAnnotations = useMemo<EditorAnnotation[]>(
    () => rallies.map((r) => ({ rally_id: r.rally_id, start: r.start, end: r.end, label: 'rally', winner: null })),
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
                ref={bindVideo}
                src={src}
                preload="metadata"
                onClick={pickMode ? undefined : togglePlay}
                onTimeUpdate={(e) => {
                  if (hasRealTime(e.currentTarget)) {
                    clock?.write(videoName, e.currentTarget.currentTime);
                  }
                }}
                onLoadedMetadata={(e) => {
                  const el = e.currentTarget;
                  setDuration(el.duration);
                  const t = takeHandover();
                  if (t != null) seekWhenSeekable(el, t);
                }}
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
                    {sil && (
                      // The player's instance mask, riding the box every
                      // frame — same lifetime as the box itself.
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
                      {r.resolution === 'manual' ? ' ✎' : ''} · f{r.frame}
                    </text>
                  </g>
                );
              })}
              {/* Pick surface: on the target's own frame, its label box as it
                  stands (amber) under the frame's 2XLarge boxes (cyan) —
                  clicking one stores exactly that box. Tracklets stay
                  context only. */}
              {onEventFrame && targetBoxes?.label_box && (
                <rect
                  className="pointer-events-none"
                  x={targetBoxes.label_box[0]}
                  y={targetBoxes.label_box[1]}
                  width={targetBoxes.label_box[2] - targetBoxes.label_box[0]}
                  height={targetBoxes.label_box[3] - targetBoxes.label_box[1]}
                  fill="none"
                  stroke={LABEL_BOX_COLOR}
                  strokeWidth={2}
                  strokeDasharray="8 4"
                  vectorEffect="non-scaling-stroke"
                />
              )}
              {onEventFrame &&
                pickBoxes.map((d, i) => {
                  const [x0, y0, x1, y1] = d.box;
                  const strong = d.score >= DENSE_LABEL_SCORE;
                  return (
                    <rect
                      key={`2xl-${i}`}
                      x={x0}
                      y={y0}
                      width={x1 - x0}
                      height={y1 - y0}
                      fill="transparent"
                      stroke={DENSE_BOX_COLOR}
                      strokeOpacity={strong ? 0.95 : 0.6}
                      strokeWidth={strong ? 2 : 1.5}
                      strokeDasharray={strong ? undefined : '4 4'}
                      vectorEffect="non-scaling-stroke"
                      className={fixing ? 'pointer-events-none opacity-40' : 'pointer-events-auto cursor-pointer hover:fill-cyan-300/20'}
                      onClick={(e) => {
                        e.stopPropagation();
                        onFixActor?.(pickTarget.id, { mode: 'pick', box: d.box });
                      }}
                    >
                      <title>{`2XLarge person · score ${d.score.toFixed(2)} — click to set as the actor`}</title>
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
            {canFix && (
              <label
                className="inline-flex flex-shrink-0 cursor-pointer items-center gap-1.5 text-xs text-text-secondary"
                title={`Also show 2XLarge boxes scoring below ${DENSE_LABEL_SCORE} (dashed cyan) — often an occluded player, often clutter`}
              >
                <input
                  type="checkbox"
                  checked={showWeakBoxes}
                  onChange={(e) => setShowWeakBoxes(e.target.checked)}
                  className="h-3.5 w-3.5 accent-primary"
                />
                Weak boxes
              </label>
            )}
            {canFix && (
              <Button
                size="sm"
                intent={pickMode ? 'primary' : 'default'}
                onClick={() => setPickMode((m) => !m)}
                title={
                  pickMode
                    ? 'Pick mode: park on an action, then click who performed it. Press P to go back to reviewing.'
                    : 'Review mode: the video plays and nothing you click changes a verdict. Press P to start picking.'
                }
              >
                {pickMode ? 'Pick mode' : 'Review mode'}
              </Button>
            )}
          </div>
          {pickMode && (
            <div className="mt-3 flex flex-wrap items-center gap-2.5 rounded-xl border border-primary/20 bg-primary/10 p-3 text-xs">
              {pickTarget ? (
                <>
                  <span className="h-2 w-2 flex-shrink-0 rounded-full animate-pulse-dot" style={{ background: actionColor(pickTarget.label) }} />
                  <span className="text-primary-light">
                    Picking player for <strong>{pickTarget.label}</strong> f{pickTarget.frame}
                    {fixing
                      ? ' — applying…'
                      : !onEventFrame
                        ? ` — boxes are pickable on f${pickTarget.frame} only`
                        : pickBoxes.length
                          ? " — click the right person's 2XLarge box"
                          : ' — no 2XLarge boxes on this frame; mark occluded or revert'}
                  </span>
                  {/* What this event already says, so the buttons below read
                      as a change of state rather than a guess. */}
                  <span
                    title={VERDICT[verdictOf(pickTarget)].title}
                    className={cn(
                      'flex-shrink-0 rounded-full px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider ring-1',
                      verdictOf(pickTarget) === 'unreviewed'
                        ? 'bg-surface-200/40 text-text-muted ring-border'
                        : 'bg-primary/15 text-primary-light ring-primary/30',
                    )}
                  >
                    {VERDICT[verdictOf(pickTarget)].glyph} {VERDICT[verdictOf(pickTarget)].label}
                  </span>
                  {targetBoxes && needsBoxCheck(targetBoxes) && (
                    <span
                      title={`${BOX_CHECK_HINT[targetBoxes.status!]} Amber = the label box, cyan = 2XLarge boxes (dashed below ${DENSE_LABEL_SCORE}) — click the right cyan box.`}
                      className="flex-shrink-0 rounded-full bg-cyan-300/10 px-2 py-0.5 font-mono text-[10px] text-cyan-300 ring-1 ring-cyan-300/30"
                    >
                      2XL: {targetBoxes.status!.replaceAll('_', ' ')}
                    </span>
                  )}
                  <span className="ml-auto flex items-center gap-3">
                    {/* Stays put once confirmed, disabled with the reason —
                        a button that vanishes leaves you wondering whether
                        the action exists at all. */}
                    {onConfirmActor && (
                      <Button
                        size="sm"
                        intent="primary"
                        disabled={fixing || !canConfirm(pickTarget)}
                        onClick={() => onConfirmActor(pickTarget.id)}
                        title={
                          verdictOf(pickTarget) === 'confirmed_auto'
                            ? 'Already confirmed'
                            : pickTarget.resolution !== 'auto'
                              ? 'No automatic pick to confirm — this event already has a human verdict'
                              : 'The automatic pick is the right person — record that verdict and move on'
                        }
                      >
                        Confirm
                      </Button>
                    )}
                    <Button size="sm" disabled={fixing} onClick={() => onFixActor?.(pickTarget.id, { mode: 'occluded' })} title="The player is occluded or otherwise unidentifiable — clears this event's crop and drops it from the labeling count">
                      Occluded
                    </Button>
                    {(pickTarget.resolution === 'manual' || pickTarget.resolution === 'occluded') && (
                      <Button size="sm" disabled={fixing} onClick={() => onFixActor?.(pickTarget.id, { mode: 'auto' })} title="Discard the manual fix and re-run the automatic pick">
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
        {/* Same place, same vocabulary as Action Label: a reviewer moving
            between the two pages should not have to relearn the keys. */}
        <p className="px-1 text-[11px] text-text-muted">
          <Key>Space</Key> play · <Key>← →</Key> frame · <Key>Shift ← →</Key> 10 frames
          {canFix && (
            <>
              {' '}· <Key>P</Key> pick mode
            </>
          )}
        </p>
      </div>

      {/* Rally list — same sidebar as the Rally Label / Action Label editors.
          Memoized against the frame clock: it takes the playhead pre-reduced
          to activeRallyId / activeActionIds, not the raw frame. */}
      {rallies.length > 0 && (
        <RallySidebar
          rallies={rallies}
          byRally={byRally}
          outside={outside}
          totalActions={sidebarActions.length}
          fps={fps}
          matches={matches}
          verdicts={verdicts}
          boxChecks={boxCheckQueue}
          onStepBoxCheck={stepBoxCheck}
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
          confirmableIds={confirmableIds}
          onConfirmRally={onConfirmRally}
        />
      )}
    </div>
  );
});
