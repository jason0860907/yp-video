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
 *  components/labeling/EventVideoPlayer, which is also where the work is
 *  done: the rally sidebar shows what each event still needs and confirming
 *  happens next to the video you are watching.
 */

import { useMemo, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { API, apiFetch, apiUrl, errMsg } from '@/lib/api';
import { Field, fieldCls } from '@/components/train/Field';
import { cn } from '@/lib/cn';
import { Badge } from '@/components/ui/Badge';
import { Card } from '@/components/ui/Card';
import { KindBadge } from '@/components/video/KindBadge';
import { PipelineChips, STAGE_HINT } from '@/components/video/PipelineChips';
import { VideoCombobox } from '@/components/video/VideoCombobox';
import { toast } from '@/components/feedback/toast';
import { EventVideoPlayer, type PlayerHandle } from '@/components/labeling/EventVideoPlayer';
import { canConfirm, type ActorFix, type ActorVerdict, type TrackData } from '@/components/labeling/shared';
import { useVideoLabelingData } from '@/components/labeling/useVideoLabelingData';
import type {
  AssociationVideo,
  ReidActorFixResponse,
  ReidRecord,
} from '@/types/api';



export function AssociationLabelPage() {
  const qc = useQueryClient();
  const [picked, setPicked] = useState('');
  const [kindFilter, setKindFilter] = useState<'all' | 'broadcast' | 'sideline'>('all');
  // Same four states as the ReID Label picker, read off review progress.
  // Unlike ReID Label's Done — a human "I'm finished" flag that counts can't
  // derive — an actor review IS finished exactly when no event is left
  // unreviewed, so this one is computed rather than stored.
  const [pickStatus, setPickStatus] = useState<'all' | 'unlabeled' | 'labeled' | 'done' | 'box_only'>('all');
  const [selectedRally, setSelectedRally] = useState<number | 'all'>('all');
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
    if (pickStatus === 'box_only' && v.box_only === 0) return false;
    return true;
  });

  const { records, meta, tracksQuery, actionEvents } = useVideoLabelingData(picked);
  const rallies = useMemo(() => meta.rallies ?? [], [meta.rallies]);

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
        ['extraction-records', picked],
        (current) =>
          current
            ? {
                ...current,
                records: current.records.map((r) => (r.id === eventId ? result.record : r)),
              }
            : current,
      );
      qc.setQueryData<TrackData | null>(['tracklets', picked], (current) => {
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

  // Confirming is per event or per rally, from the sidebar. There is no
  // whole-video sweep: an action outside every rally is usually a warm-up hit
  // or a mis-timed annotation, and sweeping those into training truth is
  // exactly the kind of thing nobody notices until the model has learned it.
  const confirmAuto = async (ids: string[]) => {
    if (!ids.length) return;
    try {
      // The response says which VERDICT each event got — endorsing a pick
      // lands as `confirmed_auto`, endorsing "nobody is visible" lands as
      // `occluded`. Assuming the first showed the wrong badge for the second.
      const { confirmed } = await apiFetch<{ confirmed: Record<string, ActorVerdict> }>(
        API.association.confirm(picked),
        { method: 'POST', body: { event_ids: ids } },
      );
      qc.setQueryData<{ meta: Record<string, unknown>; records: ReidRecord[] }>(
        ['extraction-records', picked],
        (current) =>
          current
            ? {
                ...current,
                records: current.records.map((r) =>
                  confirmed[r.id] ? { ...r, actor_review: confirmed[r.id] } : r,
                ),
              }
            : current,
      );
      void qc.invalidateQueries({ queryKey: ['association-videos'] });
      const n = Object.keys(confirmed).length;
      const occluded = Object.values(confirmed).filter((v) => v === 'occluded').length;
      toast.success(
        occluded
          ? `Confirmed ${n} — ${n - occluded} pick(s), ${occluded} occluded`
          : `Confirmed ${n} automatic ${n === 1 ? 'pick' : 'picks'}`,
      );
    } catch (e) {
      toast.error(`Confirm failed: ${errMsg(e)}`);
    }
  };

  const pickedVideo = videos.find((v) => v.name === picked);

  return (
    <div className="mx-auto max-w-screen-2xl space-y-5">
      <Card>
        <div className="grid grid-cols-1 items-end gap-3 lg:grid-cols-[8.5rem_8.5rem_minmax(18rem,1fr)]">
          <Field label="Kind">
            <select
              value={kindFilter}
              onChange={(e) => setKindFilter(e.target.value as typeof kindFilter)}
              className={cn(fieldCls, 'h-9 w-full py-0')}
            >
              <option value="all">All kinds</option>
              <option value="broadcast">Broadcast</option>
              <option value="sideline">Sideline</option>
            </select>
          </Field>
          <Field label="Status">
            <select
              value={pickStatus}
              onChange={(e) => setPickStatus(e.target.value as typeof pickStatus)}
              className={cn(fieldCls, 'h-9 w-full py-0')}
            >
              <option value="all">All</option>
              <option value="unlabeled">Unlabeled</option>
              <option value="labeled">In progress</option>
              <option value="done">Done</option>
              <option value="box_only">Needs re-pick</option>
            </select>
          </Field>
          <Field label="Video">
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
                  {v.box_only > 0 && (
                    <span title="Verdicts naming no tracklet — re-pick these players so tracklet training can use them">
                      <Badge tone="warning">{v.box_only} box</Badge>
                    </span>
                  )}
                  <PipelineChips pipeline={v.pipeline} />
                </>
              )}
            />
          </Field>
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
          onConfirmActor={(id) => void confirmAuto([id])}
          confirmableIds={confirmableIds}
          onConfirmRally={(ids) => void confirmAuto(ids)}
          fixing={Boolean(fixingEvent)}
          trackLinks={tracksQuery.data?.links ?? {}}
        />
      )}

    </div>
  );
}
