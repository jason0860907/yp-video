/** ReID Label: name the players behind extracted action events.
 *
 *  This page is orchestration only — queries, top-level controls and the
 *  wiring between its two halves: the video player (components/reid/
 *  ReidVideoPlayer) and the identities board (components/reid/GroupBoard,
 *  state machine in useGroupBoard). The two halves jump into each other
 *  through imperative handles: sidebar → board via jumpToCrop, crop →
 *  video via jumpToEvent.
 */

import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { API, ApiError, apiFetch, apiUrl } from '@/lib/api';
import { cn } from '@/lib/cn';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { EmptyState } from '@/components/ui/EmptyState';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { KindBadge } from '@/components/video/KindBadge';
import { VideoCombobox } from '@/components/video/VideoCombobox';
import { toast } from '@/components/feedback/toast';
import { confirm } from '@/components/feedback/confirm';
import { GroupBoard, type BoardHandle } from '@/components/reid/GroupBoard';
import { ReidVideoPlayer, type PlayerHandle } from '@/components/reid/ReidVideoPlayer';
import { useGroupBoard } from '@/components/reid/useGroupBoard';
import { errMsg, type ActorFix, type Rally, type SidebarAction, type TrackData } from '@/components/reid/shared';
import { LiveJob } from '@/components/job/LiveJob';
import type { ActionAnnotationData, Job, ReidCluster, ReidOptions, ReidPlayers, ReidRecord, ReidVideo } from '@/types/api';

// Embedders and their threshold-slider calibration both come from
// /reid/options (types/api.ts ReidOptions) — cosine-distance scales differ
// wildly per model and the backend registry is the single source of truth.
// Fallback covers only the pre-fetch instant.
const FALLBACK_THRESHOLD = { min: 0.05, max: 0.95, default: 0.3, step: 0.01 };
// Show enough decimals to tell adjacent slider stops apart.
const fmtThreshold = (v: number, step: number) => v.toFixed(step < 0.01 ? 3 : 2);

const selectCls =
  'w-auto cursor-pointer appearance-none rounded-lg border border-border-light bg-surface-50 px-3 py-1 text-xs text-text-primary focus:border-primary/50 focus:outline-none';

const fieldCls = 'rounded-lg border border-border-light bg-surface-50 px-3 py-2 text-sm text-text-primary focus:border-primary/50 focus:outline-none';

function FieldLabel({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="block min-w-0 space-y-1.5">
      <span className="block text-[10px] font-semibold uppercase tracking-widest text-text-muted">{label}</span>
      {children}
    </label>
  );
}

export function ReidLabelPage() {
  const qc = useQueryClient();
  const [picked, setPicked] = useState('');
  const [kindFilter, setKindFilter] = useState<'all' | 'broadcast' | 'sideline'>('all');
  const [pickStatus, setPickStatus] = useState<'all' | 'unlabeled' | 'labeled' | 'done'>('all');
  const [selectedRally, setSelectedRally] = useState<number | 'all'>('all');
  // Where locked groups live on the groups board: pinned on top as full rows,
  // or docked in a sticky right rail showing just 3 crops per group.
  const [lockedDock, setLockedDock] = useState<'top' | 'right'>('top');
  // Embedder + threshold snap to the server's default the moment
  // /reid/options lands (see effect below); queries are gated on `picked`,
  // so nothing fires against the empty pre-fetch value.
  const [embedder, setEmbedder] = useState('');
  // Draft follows the slider live; the applied value (= clusters query key)
  // trails it by a debounce so dragging doesn't fire a re-cluster per pixel.
  const [thresholdDraft, setThresholdDraft] = useState<number>(FALLBACK_THRESHOLD.default);
  const [threshold, setThreshold] = useState<number>(FALLBACK_THRESHOLD.default);
  useEffect(() => {
    const t = setTimeout(() => setThreshold(thresholdDraft), 350);
    return () => clearTimeout(t);
  }, [thresholdDraft]);
  const [showSkeleton, setShowSkeleton] = useState(false);
  const [showMasked, setShowMasked] = useState(false);
  const [showVideo, setShowVideo] = useState(true);
  const [statusFilter, setStatusFilter] = useState<'all' | ReidRecord['status']>('all');
  const playerRef = useRef<PlayerHandle>(null);
  const boardRef = useRef<BoardHandle>(null);

  const videosQuery = useQuery({
    queryKey: ['reid-videos'],
    queryFn: () => apiFetch<ReidVideo[]>(API.reid.videos),
  });
  // Embedder choices AND their threshold calibration come from the server
  // registry — a model only shows up when its weights actually exist there.
  const optionsQuery = useQuery({
    queryKey: ['reid-options'],
    queryFn: () => apiFetch<ReidOptions>(API.reid.options),
    staleTime: Infinity, // static per server run
  });
  const embedderOptions = useMemo(() => optionsQuery.data?.embedders ?? [], [optionsQuery.data]);
  const thresholdsFor = (m: string) => embedderOptions.find((e) => e.name === m)?.threshold ?? FALLBACK_THRESHOLD;
  const isMasked = (m: string) => embedderOptions.find((e) => e.name === m)?.masked ?? false;
  // Snap to the server's default embedder and its calibrated threshold once
  // the registry arrives (exactly once — staleTime is Infinity).
  useEffect(() => {
    if (!optionsQuery.data) return;
    const model = optionsQuery.data.default_embedder;
    setEmbedder(model);
    setThresholdDraft(thresholdsFor(model).default);
    setThreshold(thresholdsFor(model).default);
    setShowMasked(optionsQuery.data.embedders.find((e) => e.name === model)?.masked ?? false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [optionsQuery.data]);
  const extracted = (videosQuery.data ?? []).filter((v) => v.has_reid);
  // Picker filters, mirroring the Action Label / Rally Label pickers.
  const pickable = extracted.filter((v) => {
    if (kindFilter !== 'all' && v.kind !== kindFilter) return false;
    if (pickStatus === 'unlabeled' && (v.player_count ?? 0) > 0) return false;
    if (pickStatus === 'labeled' && ((v.player_count ?? 0) === 0 || v.done)) return false;
    if (pickStatus === 'done' && !v.done) return false;
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

  // Tracklets + event→tracklet links; null = no tracking run yet (404).
  const tracksQuery = useQuery({
    queryKey: ['reid-tracks', picked],
    queryFn: async (): Promise<TrackData | null> => {
      try {
        return await apiFetch<TrackData>(API.reid.tracks(picked));
      } catch (e) {
        if (e instanceof ApiError && e.status === 404) return null;
        throw e;
      }
    },
    enabled: Boolean(picked),
    staleTime: 60_000,
  });
  const trackLinks = useMemo(() => tracksQuery.data?.links ?? {}, [tracksQuery.data]);

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

  // A clusters 404 for a model the video list confirms is missing means "the
  // matrix was never computed" — recoverable right here with a backfill job,
  // no trip to the ReID Predict page. Any other error renders as-is.
  const pickedVideo = extracted.find((v) => v.name === picked);
  const matrixMissing =
    clustersQuery.error instanceof ApiError &&
    clustersQuery.error.status === 404 &&
    !!pickedVideo &&
    !pickedVideo.embedded_models.includes(embedder);
  const [backfillJob, setBackfillJob] = useState<Job | null>(null);
  useEffect(() => setBackfillJob(null), [picked, embedder]);
  const startBackfill = async () => {
    try {
      const job = await apiFetch<Job>(API.reid.embed, {
        method: 'POST',
        body: { videos: [picked], models: [embedder] },
      });
      setBackfillJob(job);
    } catch (e) {
      toast.error(`Backfill start failed: ${errMsg(e)}`);
    }
  };

  const board = useGroupBoard({
    picked,
    embedder,
    threshold,
    records,
    recordById,
    clusters,
    assignments,
    tracks: tracksQuery.data,
  });

  const seekToEvent = (r?: ReidRecord) => {
    // The player owns the whole jump: rally selection, panel expansion,
    // sidebar pinning and the actual seek.
    if (r) playerRef.current?.jumpToEvent({ id: r.id, frame: r.frame, time: r.time ?? null });
  };

  // Actor fix: persists into the players file server-side, re-crops and
  // re-embeds the event, then refreshes everything derived from embeddings.
  // The re-embed takes seconds — fixingEvent gates the picker so a double
  // click can't fire two overlapping fixes.
  const [fixingEvent, setFixingEvent] = useState<string | null>(null);
  const fixActor = async (eventId: string, fix: ActorFix) => {
    if (fixingEvent) return;
    setFixingEvent(eventId);
    try {
      await apiFetch(API.reid.actorFix(picked), { method: 'POST', body: { event_id: eventId, ...fix } });
      // The crop is a different person now — the server dropped the event's
      // assignment; mirror it locally so a locked row's auto-save can't
      // resurrect the stale identity.
      board.removeEvent(eventId);
      toast.success(fix.none ? 'Marked as occluded' : fix.box ? 'Player updated' : 'Reverted to the auto pick');
      await Promise.all([
        qc.invalidateQueries({ queryKey: ['reid-results', picked] }),
        qc.invalidateQueries({ queryKey: ['reid-clusters', picked] }),
        qc.invalidateQueries({ queryKey: ['reid-players', picked] }),
        // The event's box changed, so its event→tracklet link did too — a
        // "no actor" event must drop off the overlay immediately.
        qc.invalidateQueries({ queryKey: ['reid-tracks', picked] }),
      ]);
    } catch (e) {
      toast.error(`Actor fix failed: ${errMsg(e)}`);
    } finally {
      setFixingEvent(null);
    }
  };

  const pickVideo = async (name: string) => {
    if (board.dirty && name !== picked) {
      const ok = await confirm({
        title: 'Discard unsaved changes?',
        body: 'The current group edits have not been saved.',
        confirmText: 'Discard',
        variant: 'danger',
      });
      if (!ok) return;
    }
    board.clearBoard();
    setSelectedRally('all');
    setPicked(name);
  };

  // The denominator is deliberately simple: every action except score (the
  // ball-landing marker — nobody performs it). Off-frame and occluded events
  // stay in the count, so the ratio understates rather than surprises; the
  // Done confirm is a confirm, not a gate.
  const actionableCount = actionEvents.length
    ? actionEvents.filter((a) => a.label !== 'score').length
    : records.length;
  // Events that already carry an identity: member of a named group on the
  // live board (unsaved edits count — the board is the source of truth).
  const assignedCount = new Set(board.groups.filter((g) => g.name.trim()).flatMap((g) => g.eventIds)).size;
  // Occluded verdicts count as handled — the user looked and decided. They
  // are crop-less, so they never overlap the assigned (crop-bearing) set.
  const occludedCount = records.filter((r) => r.box_source === 'manual' && !r.crop).length;
  const resolvedCount = assignedCount + occludedCount;

  const isDone = Boolean(extracted.find((v) => v.name === picked)?.done);

  // Save, then persist the human "this video is finished" verdict (toggles
  // off when pressed on an already-done video). Warns when actions are still
  // unassigned — done should mean done, but partial is the user's call.
  const markDone = async () => {
    if (!picked) return;
    if (!isDone && resolvedCount < actionableCount) {
      const ok = await confirm({
        title: 'Mark as done?',
        body: `${actionableCount - resolvedCount} of ${actionableCount} actions have no player assigned (or occluded verdict) yet.`,
        confirmText: 'Mark done',
      });
      if (!ok) return;
    }
    if (board.dirty && !(await board.save())) return;
    try {
      await apiFetch(API.reid.done(picked), { method: 'PUT', body: { done: !isDone } });
      toast.success(isDone ? 'Done mark removed' : 'Marked done');
      void qc.invalidateQueries({ queryKey: ['reid-videos'] });
    } catch (e) {
      toast.error(`Done failed: ${errMsg(e)}`);
    }
  };

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
              <option value="labeled">In progress</option>
              <option value="done">Done</option>
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
                  {v.done && <Badge tone="success">✓</Badge>}
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
          videoName={picked}
          tracklets={tracksQuery.data?.tracklets ?? []}
          onFixActor={fixActor}
          fixing={Boolean(fixingEvent)}
          onJumpToCrop={(id) => boardRef.current?.jumpToCrop(id)}
          trackLinks={trackLinks}
        />
      )}

      <Card>
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-3">
            <SectionLabel className="mb-0 leading-none">Identities</SectionLabel>
            {picked && (
              <span
                className="font-mono text-[11px] leading-none tabular-nums text-text-muted"
                title="Assigned to a player or marked occluded / all actions except score (off-frame events included, so 100% is not always reachable)"
              >
                <span className={resolvedCount >= actionableCount ? 'text-primary-light' : undefined}>{resolvedCount}</span>/{actionableCount} actions
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
            <label
              className="inline-flex cursor-pointer items-center gap-1.5 text-xs text-text-secondary"
              title="Show the background-suppressed crops the masked embedders embed (original shown where a video's masked embed hasn't run yet)"
            >
              <input
                type="checkbox"
                checked={showMasked}
                onChange={(e) => setShowMasked(e.target.checked)}
                className="h-3.5 w-3.5 accent-primary"
              />
              Masked crops
            </label>
            <select
              value={embedder}
              onChange={(e) => {
                const m = e.target.value;
                setEmbedder(m);
                // Distance scales differ per model — jump to its default.
                setThresholdDraft(thresholdsFor(m).default);
                setThreshold(thresholdsFor(m).default);
                // Show what the selected model actually embeds; still a free toggle.
                setShowMasked(isMasked(m));
              }}
              className={selectCls}
              title="Appearance embedding model — compare how each one groups the players"
            >
              {embedderOptions.map((e) => (
                <option key={e.name} value={e.name}>
                  {e.name}
                </option>
              ))}
            </select>
            <label
              className="inline-flex items-center gap-1.5 text-xs text-text-secondary"
              title="Cluster threshold for unassigned events — lower splits, higher merges. Locked rows are unaffected."
            >
              <span className="whitespace-nowrap">
                threshold <span className="font-mono tabular-nums">{fmtThreshold(thresholdDraft, thresholdsFor(embedder).step)}</span>
              </span>
              <input
                type="range"
                min={thresholdsFor(embedder).min}
                max={thresholdsFor(embedder).max}
                step={thresholdsFor(embedder).step}
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
              onClick={board.seedRegroup}
              disabled={!picked || !board.groups.some((g) => (g.locked || g.name.trim()) && g.eventIds.length > 0)}
              title="Use every locked/named group as a player anchor: all other events join the nearest anchor (within the threshold); the rest re-cluster into leftover pools"
            >
              Seed regroup
            </Button>
            <Button
              size="sm"
              onClick={board.trackPropagate}
              disabled={!picked || !board.groups.some((g) => (g.locked || g.name.trim()) && g.eventIds.length > 0)}
              title="Within a rally, events on the same ByteTrack tracklet are the same player — unassigned events follow the locked/named group their tracklet belongs to (needs Rally Tracking, see ReID Predict)"
            >
              Track propagate
            </Button>
            <Button size="sm" onClick={board.reset} disabled={!board.dirty}>
              Reset
            </Button>
            <Button size="sm" intent="primary" onClick={() => void board.save()} disabled={!picked}>
              {board.dirty ? 'Save •' : 'Save'}
            </Button>
            <Button
              size="sm"
              intent={isDone ? 'default' : 'primary'}
              onClick={() => void markDone()}
              disabled={!picked}
              title={isDone ? 'Labeling marked finished — click to unmark' : 'Save, then mark this video’s labeling as finished'}
            >
              {isDone ? 'Done ✓' : 'Done'}
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
        ) : clustersQuery.isError ? (
          <div className="flex flex-col items-center justify-center gap-3 py-12 text-center">
            {matrixMissing ? (
              <>
                <p className="text-sm font-medium text-text-secondary">
                  No {embedder} embeddings for this video
                </p>
                <p className="max-w-sm text-xs text-text-muted">
                  This video was extracted before {embedder} was registered. Backfill computes its
                  embeddings from the saved crops — no re-extraction needed.
                </p>
                {backfillJob ? (
                  <div className="w-full max-w-md text-left">
                    <LiveJob job={backfillJob} onUpdate={setBackfillJob} />
                  </div>
                ) : (
                  <Button size="sm" intent="primary" onClick={() => void startBackfill()}>
                    Backfill Embeddings
                  </Button>
                )}
              </>
            ) : (
              <>
                <p className="text-sm font-medium text-red-400">Clustering unavailable</p>
                <p className="max-w-sm text-xs text-text-muted">{errMsg(clustersQuery.error)}</p>
              </>
            )}
          </div>
        ) : clustersQuery.isPending || resultsQuery.isPending ? (
          <div className="py-8 text-center text-xs text-text-muted">Clustering…</div>
        ) : (
          <>
          {playersQuery.isError && (
            <p className="mb-2 rounded-lg border border-amber-500/20 bg-amber-500/10 px-3 py-1.5 text-[11px] text-amber-400">
              Player matches unavailable: {errMsg(playersQuery.error)}
            </p>
          )}
          <GroupBoard
            ref={boardRef}
            picked={picked}
            records={records}
            recordById={recordById}
            board={board}
            lockedDock={lockedDock}
            statusFilter={statusFilter}
            showSkeleton={showSkeleton}
            showMasked={showMasked}
            trackLinks={trackLinks}
            onSeekToEvent={seekToEvent}
          />
          </>
        )}
        {/* Association stats */}
        {picked && records.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2 text-[11px]">
            <Badge tone="success">ok {records.filter((r) => r.status === 'ok').length}</Badge>
            <Badge tone="warning">multi {records.filter((r) => r.status === 'multi').length}</Badge>
            <Badge tone="danger">miss {records.filter((r) => r.status === 'miss').length}</Badge>
            {(playersQuery.data?.players ?? []).map((p) => (
              <Badge key={p} tone="brand">
                {p} {Object.values(matches).filter((m) => m.assigned && m.player === p).length}
              </Badge>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}
