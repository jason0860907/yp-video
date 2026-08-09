/** ReID Label panel: name the players behind extracted action events.
 *
 *  Strictly "who is this" — WHICH person performed each action is settled on
 *  the Association panel, so the video player here is read-only (no
 *  onFixActor) and nothing here can write an actor verdict.
 *
 *  Orchestration only — queries, top-level controls and the wiring between
 *  its two halves: the video player (components/labeling/EventVideoPlayer)
 *  and the identities board (components/reid/GroupBoard, state machine in
 *  useGroupBoard). The two halves jump into each other through imperative
 *  handles: sidebar → board via jumpToCrop, crop → video via jumpToEvent.
 *
 *  The picker moved to the parent; the old pick-time "discard unsaved
 *  changes?" confirm is the registered dirty guard, and the effect on the
 *  `video` prop clears the board exactly where pickVideo used to.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { API, ApiError, apiFetch, apiUrl, errMsg } from '@/lib/api';
import { cn } from '@/lib/cn';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { EmptyState } from '@/components/ui/EmptyState';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { STAGE_HINT } from '@/components/video/PipelineChips';
import { toast } from '@/components/feedback/toast';
import { confirm } from '@/components/feedback/confirm';
import { GroupBoard, type BoardHandle } from '@/components/reid/GroupBoard';
import { EventVideoPlayer, type PlayerHandle } from '@/components/labeling/EventVideoPlayer';
import { useGroupBoard } from '@/components/reid/useGroupBoard';
import { useVideoLabelingData } from '@/components/labeling/useVideoLabelingData';
import { LiveJob } from '@/components/job/LiveJob';
import type { Job, ReidClusters, ReidOptions, ReidPlayers, ReidRecord, ReidVideo } from '@/types/api';
import { reidStatus } from '@/lib/labelStatus';
import { STATUS_OPTIONS, type ModeDescriptor, type RegisterGuard } from './mode';

// Embedders and their threshold-slider calibration both come from
// /reid/options (types/api.ts ReidOptions) — cosine-distance scales differ
// wildly per model and the backend registry is the single source of truth.
// Fallback covers only the pre-fetch instant.
const FALLBACK_THRESHOLD = { min: 0.05, max: 0.95, default: 0.3, step: 0.01 };
// Show enough decimals to tell adjacent slider stops apart.
const fmtThreshold = (v: number, step: number) => v.toFixed(step < 0.01 ? 3 : 2);
const retryEmbeddingRefresh = (failureCount: number, error: Error) =>
  error instanceof ApiError && error.status === 409
    ? failureCount < 20
    : failureCount < 1;
const embeddingRetryDelay = (attempt: number, error: Error) =>
  error instanceof ApiError && error.status === 409
    ? 500
    : Math.min(1000 * 2 ** attempt, 5000);

const selectCls =
  'w-auto cursor-pointer appearance-none rounded-lg border border-border-light bg-surface-50 px-3 py-1 text-xs text-text-primary focus:border-primary/50 focus:outline-none';

export const REID_MODE: ModeDescriptor = {
  key: 'reid',
  label: 'ReID',
  statusOptions: STATUS_OPTIONS,
  status: reidStatus,
  matches: (row, status) => status === 'all' || reidStatus(row) === status,
  available: (row) => Boolean(row.reid?.pipeline.has_records),
  hint: (row) => {
    const blocked = row.reid?.pipeline.blocked_on;
    return blocked ? STAGE_HINT[blocked] : 'No extraction records for this video yet';
  },
  rowExtras: (row) => {
    const v = row.reid;
    if (!v) return null;
    return (
      <>
        <span className="shrink-0 font-mono text-[10px] tabular-nums text-text-muted">{v.event_count}ev</span>
        {(v.player_count ?? 0) > 0 && <Badge tone="brand">{v.player_count}P</Badge>}
      </>
    );
  },
  // No doneApi: the panel's own Done button saves the board first and can
  // record auto actors as confirmed — semantics the page button can't offer.
  listKey: 'reid-videos',
};

export function ReidPanel({ video, registerGuard }: { video: string; registerGuard?: RegisterGuard }) {
  const qc = useQueryClient();
  const [selectedRally, setSelectedRally] = useState<number | 'all'>('all');
  // Where locked groups live on the groups board: pinned on top as full rows,
  // or docked in a sticky right rail showing just 3 crops per group.
  const [lockedDock, setLockedDock] = useState<'top' | 'right'>('top');
  // Embedder + threshold snap to the server's default the moment
  // /reid/options lands (see effect below); queries are gated on `video`,
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
  const extracted = (videosQuery.data ?? []).filter((v) => v.pipeline.has_records);

  const { resultsQuery, records, meta, tracksQuery, actionEvents } = useVideoLabelingData(video);
  const recordById = useMemo(() => new Map(records.map((r) => [r.id, r])), [records]);
  const trackLinks = useMemo(() => tracksQuery.data?.links ?? {}, [tracksQuery.data]);

  const clustersQuery = useQuery({
    queryKey: ['reid-clusters', video, threshold, embedder],
    queryFn: () => apiFetch<ReidClusters>(API.reid.clusters(video, threshold, embedder)),
    enabled: Boolean(video),
    retry: retryEmbeddingRefresh,
    retryDelay: embeddingRetryDelay,
  });
  const clusters = useMemo(() => clustersQuery.data?.clusters ?? [], [clustersQuery.data]);
  const units = useMemo(() => clustersQuery.data?.units ?? {}, [clustersQuery.data]);

  const playersQuery = useQuery({
    queryKey: ['reid-players', video, embedder],
    queryFn: () => apiFetch<ReidPlayers>(API.reid.players(video, embedder)),
    enabled: Boolean(video),
    retry: retryEmbeddingRefresh,
    retryDelay: embeddingRetryDelay,
  });
  const unitNames = useMemo(() => playersQuery.data?.unit_names ?? {}, [playersQuery.data]);
  const matches = playersQuery.data?.matches ?? {};

  // A clusters 404 for a model the video list confirms is missing means "the
  // matrix was never computed" — recoverable right here with a backfill job,
  // no trip to the ReID Predict page. Any other error renders as-is.
  const pickedVideo = extracted.find((v) => v.name === video);
  const matrixMissing =
    clustersQuery.error instanceof ApiError &&
    clustersQuery.error.status === 404 &&
    !!pickedVideo &&
    !pickedVideo.embedded_models.includes(embedder);
  const [backfillJob, setBackfillJob] = useState<Job | null>(null);
  useEffect(() => setBackfillJob(null), [video, embedder]);
  const startBackfill = async () => {
    try {
      const job = await apiFetch<Job>(API.reid.embed, {
        method: 'POST',
        body: { videos: [video], models: [embedder] },
      });
      setBackfillJob(job);
    } catch (e) {
      toast.error(`Backfill start failed: ${errMsg(e)}`);
    }
  };

  const board = useGroupBoard({
    picked: video,
    embedder,
    threshold,
    clusters,
    units,
    unitNames,
  });

  const seekToEvent = (r?: ReidRecord) => {
    // The player owns the whole jump: rally selection, panel expansion,
    // sidebar pinning and the actual seek.
    if (r) playerRef.current?.jumpToEvent({ id: r.id, frame: r.frame, time: r.time ?? null });
  };

  // The parent owns picking; the guard below already asked about unsaved
  // work, so a changed prop empties the board exactly like pickVideo did.
  useEffect(() => {
    board.clearBoard();
    setSelectedRally('all');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [video]);

  const boardDirty = board.dirty;
  useEffect(() => {
    if (!registerGuard) return;
    registerGuard(async () => {
      if (!boardDirty) return true;
      return confirm({
        title: 'Discard unsaved changes?',
        body: 'The current group edits have not been saved.',
        confirmText: 'Discard',
        variant: 'danger',
      });
    });
    return () => registerGuard(null);
  }, [registerGuard, boardDirty]);

  // The denominator is deliberately simple: every action except score (the
  // ball-landing marker — nobody performs it). Off-frame and occluded events
  // stay in the count, so the ratio understates rather than surprises; the
  // Done confirm is a confirm, not a gate.
  const actionableCount = actionEvents.length
    ? actionEvents.filter((a) => a.label !== 'score').length
    : records.length;
  // Events that already carry an identity: member of a named group on the
  // live board (unsaved edits count — the board is the source of truth).
  const assignedCount = new Set(
    board.groups.filter((g) => g.name.trim()).flatMap((g) => g.unitKeys.flatMap(board.eventsOf)),
  ).size;
  // Occluded verdicts count as handled — the user looked and decided. They
  // are crop-less, so they never overlap the assigned (crop-bearing) set.
  const occludedCount = records.filter((r) => r.resolution === 'occluded').length;
  const resolvedCount = assignedCount + occludedCount;

  const isDone = Boolean(extracted.find((v) => v.name === video)?.done);

  // Save, then persist the human "this video is finished" verdict (toggles
  // off when pressed on an already-done video). Warns when actions are still
  // unassigned — done should mean done, but partial is the user's call.
  const markDone = async () => {
    if (!video) return;
    if (!isDone && resolvedCount < actionableCount) {
      const ok = await confirm({
        title: 'Mark as done?',
        body: `${actionableCount - resolvedCount} of ${actionableCount} actions have no player assigned (or occluded verdict) yet. Assigned automatic actor selections will be recorded as human-confirmed association labels.`,
        confirmText: 'Mark done',
      });
      if (!ok) return;
    } else if (!isDone) {
      const ok = await confirm({
        title: 'Mark as done?',
        body: 'Assigned automatic actor selections will be recorded as human-confirmed association labels.',
        confirmText: 'Mark done',
      });
      if (!ok) return;
    }
    if (board.dirty && !(await board.save())) return;
    try {
      await apiFetch(API.reid.done(video), {
        method: 'PUT',
        body: {
          done: !isDone,
          confirm_auto_actors: !isDone,
        },
      });
      toast.success(isDone ? 'Done mark removed' : 'Marked done');
      void qc.invalidateQueries({ queryKey: ['reid-videos'] });
      void qc.invalidateQueries({ queryKey: ['label-stats'] });
      if (!isDone) {
        void qc.invalidateQueries({
          queryKey: ['extraction-records', video],
        });
      }
    } catch (e) {
      toast.error(`Done failed: ${errMsg(e)}`);
    }
  };

  return (
    <div className="space-y-5">
      {video && showVideo && meta.fps && meta.frame_size && (
        <EventVideoPlayer
          ref={playerRef}
          src={apiUrl(API.actionAnnotate.video(video))}
          fps={meta.fps}
          frameSize={meta.frame_size}
          records={records}
          actionEvents={actionEvents}
          matches={matches}
          rallies={meta.rallies ?? []}
          selectedRally={selectedRally}
          onSelectRally={setSelectedRally}
          videoName={video}
          tracklets={tracksQuery.data?.tracklets ?? []}
          onJumpToCrop={(id) => boardRef.current?.jumpToCrop(id)}
          trackLinks={trackLinks}
        />
      )}

      <Card>
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-3">
            <SectionLabel className="mb-0 leading-none">Identities</SectionLabel>
            {video && (
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
              disabled={!video || !board.groups.some((g) => (g.locked || g.name.trim()) && g.unitKeys.length > 0)}
              title="Use every locked/named group as a player anchor: all other events join the nearest anchor (within the threshold); the rest re-cluster into leftover pools"
            >
              Seed regroup
            </Button>
            <Button size="sm" onClick={board.reset} disabled={!board.dirty}>
              Reset
            </Button>
            <Button size="sm" intent="primary" onClick={() => void board.save()} disabled={!video}>
              {board.dirty ? 'Save •' : 'Save'}
            </Button>
            <Button
              size="sm"
              intent={isDone ? 'default' : 'primary'}
              onClick={() => void markDone()}
              disabled={!video}
              title={isDone ? 'Labeling marked finished — click to unmark' : 'Save, then mark this video’s labeling as finished'}
            >
              {isDone ? 'Done ✓' : 'Done'}
            </Button>
          </div>
        </div>

        {!video ? (
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
            picked={video}
            records={records}
            recordById={recordById}
            board={board}
            lockedDock={lockedDock}
            statusFilter={statusFilter}
            showMasked={showMasked}
            trackLinks={trackLinks}
            onSeekToEvent={seekToEvent}
          />
          </>
        )}
        {/* Association stats */}
        {video && records.length > 0 && (
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
