/** Association Label: say which visible person performed each action.
 *
 *  The question this page answers is "who did it", not "who are they" — the
 *  latter is ReID Label, and the two write different annotation files. So
 *  nothing here fetches clusters, players or embedders: an actor verdict is
 *  true regardless of which embedding model happens to be loaded, and asking
 *  the user to pick one would imply otherwise. The server chooses which
 *  matrix to refresh after a fix (see routers/actor_association.py).
 *
 *  Orchestration only — the video player and its actor picker live in
 *  components/labeling/EventVideoPlayer, the work list in
 *  components/association/EventReviewList.
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
import { PipelineChips, STAGE_HINT } from '@/components/video/PipelineChips';
import { VideoCombobox } from '@/components/video/VideoCombobox';
import { confirm } from '@/components/feedback/confirm';
import { toast } from '@/components/feedback/toast';
import { EventReviewList } from '@/components/association/EventReviewList';
import { EventVideoPlayer, type PlayerHandle } from '@/components/labeling/EventVideoPlayer';
import { canConfirm, errMsg, rallyOf, type ActorFix, type Rally, type SidebarAction, type TrackData } from '@/components/labeling/shared';
import type {
  ActionAnnotationData,
  AssociationVideo,
  ReidActorFixResponse,
  ReidRecord,
} from '@/types/api';

const fieldCls =
  'rounded-lg border border-border-light bg-surface-50 px-3 py-2 text-sm text-text-primary focus:border-primary/50 focus:outline-none';

function FieldLabel({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="block min-w-0 space-y-1.5">
      <span className="block text-[10px] font-semibold uppercase tracking-widest text-text-muted">{label}</span>
      {children}
    </label>
  );
}

export function AssociationLabelPage() {
  const qc = useQueryClient();
  const [picked, setPicked] = useState('');
  const [kindFilter, setKindFilter] = useState<'all' | 'broadcast' | 'sideline'>('all');
  // Same four states as the ReID Label picker, read off review progress.
  // Unlike ReID Label's Done — a human "I'm finished" flag that counts can't
  // derive — an actor review IS finished exactly when no event is left
  // unreviewed, so this one is computed rather than stored.
  const [pickStatus, setPickStatus] = useState<'all' | 'unlabeled' | 'labeled' | 'done'>('all');
  const [selectedRally, setSelectedRally] = useState<number | 'all'>('all');
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const playerRef = useRef<PlayerHandle>(null);

  const videosQuery = useQuery({
    queryKey: ['association-videos'],
    queryFn: () => apiFetch<AssociationVideo[]>(API.association.videos),
  });
  const videos = useMemo(() => videosQuery.data ?? [], [videosQuery.data]);
  const pickable = videos.filter((v) => {
    if (kindFilter !== 'all' && v.kind !== kindFilter) return false;
    const done = v.event_count > 0 && v.unreviewed === 0;
    if (pickStatus === 'unlabeled' && (v.reviewed > 0 || done)) return false;
    if (pickStatus === 'labeled' && (v.reviewed === 0 || done)) return false;
    if (pickStatus === 'done' && !done) return false;
    return true;
  });

  // Records are extraction output, shared with ReID Label — same endpoint,
  // same cache key, so switching pages does not re-download them.
  const resultsQuery = useQuery({
    queryKey: ['reid-results', picked],
    queryFn: () =>
      apiFetch<{ meta: Record<string, unknown>; records: ReidRecord[] }>(API.reid.results(picked)),
    enabled: Boolean(picked),
  });
  const records = useMemo(() => resultsQuery.data?.records ?? [], [resultsQuery.data]);
  const meta = (resultsQuery.data?.meta ?? {}) as {
    fps?: number;
    frame_size?: [number, number];
    rallies?: Rally[];
  };
  const rallies = useMemo(() => meta.rallies ?? [], [meta.rallies]);
  const fps = meta.fps ?? 0;

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

  // The full action annotation — the sidebar lists every action's time,
  // including the score / off-frame events extraction skipped.
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

  useEffect(() => setSelectedId(null), [picked]);

  const jumpTo = (r: ReidRecord) => {
    setSelectedId(r.id);
    playerRef.current?.jumpToEvent({ id: r.id, frame: r.frame, time: r.time ?? null });
  };

  // A fix re-crops and re-embeds the event server-side; fixingEvent gates the
  // picker so a double click cannot fire two overlapping writes.
  const [fixingEvent, setFixingEvent] = useState<string | null>(null);
  const fixActor = async (eventId: string, fix: ActorFix) => {
    if (fixingEvent) return;
    setFixingEvent(eventId);
    try {
      const result = await apiFetch<ReidActorFixResponse>(API.association.fix(picked), {
        method: 'POST',
        body: { event_id: eventId, ...fix },
      });
      // The POST returns the changed record and its one track link — patch
      // the two large payloads locally instead of downloading them again.
      qc.setQueryData<{ meta: Record<string, unknown>; records: ReidRecord[] }>(
        ['reid-results', picked],
        (current) =>
          current
            ? {
                ...current,
                records: current.records.map((r) => (r.id === eventId ? result.record : r)),
              }
            : current,
      );
      qc.setQueryData<TrackData | null>(['reid-tracks', picked], (current) => {
        if (!current) return current;
        const links = { ...current.links };
        if (result.track_link) links[eventId] = result.track_link;
        else delete links[eventId];
        return { ...current, links };
      });
      toast.success(
        fix.mode === 'occluded'
          ? 'Marked as occluded'
          : fix.mode === 'pick'
            ? 'Actor updated'
            : 'Reverted to the auto pick',
      );
      // The work list's counts moved, and the identities this event fed into
      // are stale — ReID Label refetches them when it next mounts.
      void qc.invalidateQueries({ queryKey: ['association-videos'] });
      void qc.invalidateQueries({ queryKey: ['reid-clusters', picked], refetchType: 'none' });
      void qc.invalidateQueries({ queryKey: ['reid-players', picked], refetchType: 'none' });
    } catch (e) {
      toast.error(`Actor fix failed: ${errMsg(e)}`);
    } finally {
      setFixingEvent(null);
    }
  };

  // Confirming says "the policy already got this right" — so it only applies
  // where the policy actually picked somebody. A miss has nobody to agree
  // with and needs a real verdict, which is why it is excluded here rather
  // than silently skipped server-side.
  const confirmable = useMemo(() => records.filter(canConfirm), [records]);

  // Identity-stable so the memoized rally sidebar keeps holding. This is the
  // full set — every rally row, the outside row included, offers its own.
  const confirmableIds = useMemo(
    () => new Set(confirmable.map((r) => r.id)),
    [confirmable],
  );

  // The whole-video button deliberately stops at the rally boundaries. An
  // action outside every rally is usually a warm-up hit or a mis-timed
  // annotation, and sweeping those into training truth is exactly the kind
  // of thing nobody notices until the model has learned it. They stay
  // confirmable one rally row (the "outside" one) at a time.
  // With no rally annotation at all there is no boundary to respect.
  const bulkConfirmable = useMemo(
    () =>
      rallies.length
        ? confirmable.filter((r) => rallyOf(rallies, r, fps) !== null)
        : confirmable,
    [confirmable, rallies, fps],
  );
  const outsideCount = confirmable.length - bulkConfirmable.length;

  const confirmAuto = async (ids: string[], { ask }: { ask: boolean }) => {
    if (!ids.length) return;
    if (ask) {
      const ok = await confirm({
        title: `Confirm ${ids.length} automatic picks?`,
        body:
          `They become human-confirmed association labels — training truth, ` +
          `not machine output. Events you already fixed are untouched.` +
          (outsideCount
            ? ` ${outsideCount} action${outsideCount === 1 ? '' : 's'} outside every rally ` +
              `${outsideCount === 1 ? 'is' : 'are'} left out — confirm those from the sidebar's outside row.`
            : ''),
        confirmText: 'Confirm',
      });
      if (!ok) return;
    }
    try {
      const { confirmed } = await apiFetch<{ confirmed: string[] }>(
        API.association.confirm(picked),
        { method: 'POST', body: { event_ids: ids } },
      );
      const done = new Set(confirmed);
      qc.setQueryData<{ meta: Record<string, unknown>; records: ReidRecord[] }>(
        ['reid-results', picked],
        (current) =>
          current
            ? {
                ...current,
                records: current.records.map((r) =>
                  done.has(r.id) ? { ...r, actor_review: 'confirmed_auto' as const } : r,
                ),
              }
            : current,
      );
      void qc.invalidateQueries({ queryKey: ['association-videos'] });
      toast.success(`Confirmed ${confirmed.length} automatic ${confirmed.length === 1 ? 'pick' : 'picks'}`);
    } catch (e) {
      toast.error(`Confirm failed: ${errMsg(e)}`);
    }
  };

  const pickedVideo = videos.find((v) => v.name === picked);
  const reviewed = records.filter((r) => (r.actor_review ?? 'unreviewed') !== 'unreviewed').length;

  return (
    <div className="mx-auto max-w-screen-2xl space-y-5">
      <Card>
        <div className="grid grid-cols-1 items-end gap-3 lg:grid-cols-[8.5rem_8.5rem_minmax(18rem,1fr)]">
          <FieldLabel label="Kind">
            <select
              value={kindFilter}
              onChange={(e) => setKindFilter(e.target.value as typeof kindFilter)}
              className={cn(fieldCls, 'h-9 w-full py-0')}
            >
              <option value="all">All kinds</option>
              <option value="broadcast">Broadcast</option>
              <option value="sideline">Sideline</option>
            </select>
          </FieldLabel>
          <FieldLabel label="Status">
            <select
              value={pickStatus}
              onChange={(e) => setPickStatus(e.target.value as typeof pickStatus)}
              className={cn(fieldCls, 'h-9 w-full py-0')}
            >
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
              onChange={setPicked}
              placeholder={`Search ${pickable.length} extracted videos…`}
              renderItem={(v) => (
                <>
                  <KindBadge kind={v.kind} />
                  <span className="min-w-0 flex-1 break-all font-mono">{v.name}</span>
                  <span className="shrink-0 font-mono text-[10px] tabular-nums text-text-muted">
                    {v.event_count}ev
                  </span>
                  {/* Untouched videos stay bare, so the badges read as the
                      same three states the Status filter selects. */}
                  {v.event_count > 0 && v.unreviewed === 0 ? (
                    <Badge tone="success">✓</Badge>
                  ) : v.reviewed > 0 ? (
                    <Badge tone="warning">{v.unreviewed} left</Badge>
                  ) : null}
                  <PipelineChips pipeline={v.pipeline} />
                </>
              )}
            />
          </FieldLabel>
        </div>
      </Card>

      {pickedVideo?.pipeline.blocked_on && (
        <Card>
          <p className="text-xs text-amber-400">
            This video is not ready for actor review — {STAGE_HINT[pickedVideo.pipeline.blocked_on]}.
          </p>
        </Card>
      )}
      {pickedVideo && !pickedVideo.pipeline.has_masks && pickedVideo.pipeline.has_tracks && (
        <Card>
          <p className="text-xs text-text-muted">
            Tracking for this video predates instance masks — the picker falls back to
            box overlap when resolving who you clicked. Re-run Rally Tracking to restore it.
          </p>
        </Card>
      )}

      {picked && meta.fps && meta.frame_size && (
        <EventVideoPlayer
          ref={playerRef}
          src={apiUrl(API.actionAnnotate.video(picked))}
          fps={meta.fps}
          frameSize={meta.frame_size}
          records={records}
          actionEvents={actionEvents}
          // Player identity is the other page's answer; boxes here name the
          // action so nothing on screen claims to know who somebody is.
          matches={{}}
          rallies={rallies}
          selectedRally={selectedRally}
          onSelectRally={setSelectedRally}
          videoName={picked}
          tracklets={tracksQuery.data?.tracklets ?? []}
          onFixActor={fixActor}
          onConfirmActor={(id) => void confirmAuto([id], { ask: false })}
          confirmableIds={confirmableIds}
          onConfirmRally={(ids) => void confirmAuto(ids, { ask: false })}
          fixing={Boolean(fixingEvent)}
          trackLinks={tracksQuery.data?.links ?? {}}
        />
      )}

      <Card>
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-3">
            <SectionLabel className="mb-0 leading-none">Actor review</SectionLabel>
            {picked && (
              <span
                className="font-mono text-[11px] leading-none tabular-nums text-text-muted"
                title="Events carrying a human verdict / events extraction produced"
              >
                <span className={reviewed >= records.length ? 'text-primary-light' : undefined}>
                  {reviewed}
                </span>
                /{records.length} actions
              </span>
            )}
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {picked && pickedVideo && (
              <div className="flex flex-wrap gap-2 text-[11px]">
                <Badge tone="success">ok {pickedVideo.auto_counts.ok}</Badge>
                <Badge tone="warning">multi {pickedVideo.auto_counts.multi}</Badge>
                <Badge tone="danger">miss {pickedVideo.auto_counts.miss}</Badge>
              </div>
            )}
            {picked && (
              <Button
                size="sm"
                intent="primary"
                disabled={!bulkConfirmable.length}
                onClick={() => void confirmAuto(bulkConfirmable.map((r) => r.id), { ask: true })}
                title={
                  bulkConfirmable.length
                    ? 'Endorse every automatic pick inside a rally that has no verdict yet — the crop and its embedding do not change' +
                      (outsideCount ? `. ${outsideCount} outside every rally stay out; confirm those per rally.` : '')
                    : 'Nothing left to confirm inside a rally; whatever remains needs a real verdict (pick or occluded)'
                }
              >
                Confirm {bulkConfirmable.length} in rallies
              </Button>
            )}
          </div>
        </div>

        {!picked ? (
          <EmptyState
            icon={
              <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M15.042 21.672L13.684 16.6m0 0l-2.51 2.225.569-9.47 5.227 7.917-3.286-.672zm-7.518-.267A8.25 8.25 0 1120.25 10.5M8.288 14.212A5.25 5.25 0 1117.25 10.5"
                />
              </svg>
            }
            title="Pick an extracted video"
            subtitle="Turn on Pick Player, park on an action, then click who performed it"
          />
        ) : resultsQuery.isPending ? (
          <div className="py-8 text-center text-xs text-text-muted">Loading records…</div>
        ) : resultsQuery.isError ? (
          <p className="rounded-lg border border-red-500/20 bg-red-500/10 px-3 py-1.5 text-[11px] text-red-400">
            {errMsg(resultsQuery.error)}
          </p>
        ) : (
          <EventReviewList
            records={records}
            rallies={rallies}
            fps={fps}
            selectedId={selectedId}
            onJump={jumpTo}
          />
        )}
      </Card>
    </div>
  );
}
