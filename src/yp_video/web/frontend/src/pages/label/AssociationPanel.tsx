/** Association Label panel: say which visible person performed each action.
 *
 *  The question this panel answers is "who did it", not "who are they" — the
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
 *
 *  No dirty guard: every verdict is written the moment it is made.
 */

import { useMemo, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { API, apiFetch, apiUrl, errMsg } from '@/lib/api';
import { Badge } from '@/components/ui/Badge';
import { Card } from '@/components/ui/Card';
import { STAGE_HINT } from '@/components/video/PipelineChips';
import { toast } from '@/components/feedback/toast';
import { EventVideoPlayer, type PlayerHandle } from '@/components/labeling/EventVideoPlayer';
import { canConfirm, type ActorFix, type ActorVerdict, type TrackData } from '@/components/labeling/shared';
import { useVideoLabelingData } from '@/components/labeling/useVideoLabelingData';
import type {
  AssociationVideo,
  ReidActorFixResponse,
  ReidRecord,
} from '@/types/api';
import { assocStatus } from '@/lib/labelStatus';
import { STATUS_OPTIONS, type ModeDescriptor, type PlaybackClock } from './mode';

export const ASSOCIATION_MODE: ModeDescriptor = {
  key: 'association',
  label: 'Association',
  statusOptions: [...STATUS_OPTIONS, { value: 'unresolved', label: 'Needs re-pick' }],
  status: assocStatus,
  matches: (row, status) => {
    if (status === 'all') return true;
    if (status === 'unresolved') return (row.assoc?.unresolved ?? 0) > 0;
    return assocStatus(row) === status;
  },
  available: (row) => Boolean(row.assoc),
  hint: (row) => {
    const blocked = row.reid?.pipeline.blocked_on;
    return blocked ? STAGE_HINT[blocked] : 'Not ready for actor review — needs rallies, action labels and extraction records';
  },
  rowExtras: (row) => {
    const v = row.assoc;
    if (!v) return null;
    return (
      <>
        <span className="shrink-0 font-mono text-[10px] tabular-nums text-text-muted">{v.event_count}ev</span>
        {v.reviewed > 0 && v.unreviewed > 0 && <Badge tone="warning">{v.unreviewed} left</Badge>}
        {v.unresolved > 0 && (
          <span title="Verdicts resolving to no tracklet — re-pick these players so tracklet training can use them">
            <Badge tone="warning">{v.unresolved} re-pick</Badge>
          </span>
        )}
      </>
    );
  },
  doneApi: (video) => API.association.done(video),
  listKey: 'association-videos',
};

export function AssociationPanel({ video, clock }: { video: string; clock?: PlaybackClock }) {
  const qc = useQueryClient();
  const [selectedRally, setSelectedRally] = useState<number | 'all'>('all');
  const playerRef = useRef<PlayerHandle>(null);

  const videosQuery = useQuery({
    queryKey: ['association-videos'],
    queryFn: () => apiFetch<AssociationVideo[]>(API.association.videos),
  });
  const videos = useMemo(() => videosQuery.data ?? [], [videosQuery.data]);

  const { records, meta, tracksQuery, actionEvents } = useVideoLabelingData(video);
  const rallies = useMemo(() => meta.rallies ?? [], [meta.rallies]);

  // A fix re-crops and re-embeds the event server-side; fixingEvent gates the
  // picker so a double click cannot fire two overlapping writes.
  const [fixingEvent, setFixingEvent] = useState<string | null>(null);
  const fixActor = async (eventId: string, fix: ActorFix) => {
    if (fixingEvent) return;
    setFixingEvent(eventId);
    try {
      const result = await apiFetch<ReidActorFixResponse>(API.association.fix(video), {
        method: 'POST',
        body: { event_id: eventId, ...fix },
      });
      // The POST returns the changed record and its one track link — patch
      // the two large payloads locally instead of downloading them again.
      qc.setQueryData<{ meta: Record<string, unknown>; records: ReidRecord[] }>(
        ['extraction-records', video],
        (current) =>
          current
            ? {
                ...current,
                records: current.records.map((r) => (r.id === eventId ? result.record : r)),
              }
            : current,
      );
      qc.setQueryData<TrackData | null>(['tracklets', video], (current) => {
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
      void qc.invalidateQueries({ queryKey: ['label-stats'] });
      void qc.invalidateQueries({ queryKey: ['reid-clusters', video], refetchType: 'none' });
      void qc.invalidateQueries({ queryKey: ['reid-players', video], refetchType: 'none' });
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
        API.association.confirm(video),
        { method: 'POST', body: { event_ids: ids } },
      );
      qc.setQueryData<{ meta: Record<string, unknown>; records: ReidRecord[] }>(
        ['extraction-records', video],
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
      void qc.invalidateQueries({ queryKey: ['label-stats'] });
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

  const pickedVideo = videos.find((v) => v.name === video);

  return (
    <div className="space-y-5">
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

      {video && meta.fps && meta.frame_size && (
        <EventVideoPlayer
          ref={playerRef}
          clock={clock}
          src={apiUrl(API.actionAnnotate.video(video))}
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
          videoName={video}
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
