/** Association Predict — re-decide who performed each action, in place.
 *
 *  The pick used to be a side effect of extraction, so changing it meant
 *  paying for person detection again. The detections are already stored; only
 *  the choice among them is revisited here, which is why this is its own
 *  stage and not a checkbox on ReID Predict.
 *
 *  Human verdicts are never re-decided — the count is shown per video so it
 *  is visible rather than promised.
 */
import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { API, ApiError, apiFetch } from '@/lib/api';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { PageHeader } from '@/components/ui/PageHeader';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { StatTile } from '@/components/ui/StatTile';
import { PipelineChips, STAGE_HINT } from '@/components/video/PipelineChips';
import { VideoMultiSelectList } from '@/components/video/VideoMultiSelectList';
import { LiveJob } from '@/components/job/LiveJob';
import { TRACKING_JOB_TYPE, useRallyTracking } from '@/lib/useRallyTracking';
import { toast } from '@/components/feedback/toast';
import type {
  AssociationVideo,
  Job,
  ReidAssociationStatus,
} from '@/types/api';

const errMsg = (e: unknown) =>
  e instanceof ApiError ? e.body : e instanceof Error ? e.message : String(e);

const PREDICT_JOB_TYPE = 'actor_association_predict';
// Tracking is offered here too, so its job card belongs here too.
const PAGE_JOB_TYPES = new Set([PREDICT_JOB_TYPE, TRACKING_JOB_TYPE]);

/** The rule is a policy like any other, and the only one that is always
 *  available — it needs no checkpoint and no tracking. */
const RULE = 'rule-based';

export function AssociationPredictPage() {
  const navigate = useNavigate();
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [policy, setPolicy] = useState(RULE);
  const [stopVllm, setStopVllm] = useState(false);
  const [jobOverrides, setJobOverrides] = useState<Record<string, Job>>({});

  const jobsQuery = useQuery({
    queryKey: ['jobs-list'],
    queryFn: () => apiFetch<Job[]>(API.jobs.list),
  });
  const videosQuery = useQuery({
    queryKey: ['association-videos'],
    queryFn: () => apiFetch<AssociationVideo[]>(API.association.videos),
  });
  const statusQuery = useQuery({
    queryKey: ['association-status'],
    queryFn: () => apiFetch<ReidAssociationStatus>(API.association.status),
  });

  const videos = videosQuery.data ?? [];
  const checkpoints = statusQuery.data?.checkpoints ?? [];
  const chosenCheckpoint = checkpoints.find((c) => c.name === policy);
  // A tracklet ranker chooses among tracklets, so a video that was never
  // tracked has nothing for it to choose from.
  const needsTracks = chosenCheckpoint?.feature_set === 'track-v1';

  const upsertJob = (job: Job) =>
    setJobOverrides((prev) => ({ ...prev, [job.id]: job }));
  const jobs = useMemo(() => {
    const merged = new Map<string, Job>();
    for (const job of jobsQuery.data ?? []) {
      if (PAGE_JOB_TYPES.has(job.type ?? '')) merged.set(job.id, job);
    }
    for (const job of Object.values(jobOverrides)) merged.set(job.id, job);
    return [...merged.values()].sort(
      (a, b) => (b.created_at ?? 0) - (a.created_at ?? 0),
    );
  }, [jobsQuery.data, jobOverrides]);

  const tracking = useRallyTracking({ videos, selected, onJob: upsertJob });
  const chosen = videos.filter((v) => selected.has(v.name));
  const blocked = useMemo(() => {
    const stages = new Set<string>();
    for (const video of chosen) {
      if (!video.pipeline.has_records) stages.add('records');
      else if (needsTracks && !video.pipeline.has_tracks) stages.add('tracks');
    }
    if (!stages.size) return null;
    return [...stages]
      .map((k) => STAGE_HINT[k as keyof typeof STAGE_HINT])
      .join('; ');
  }, [chosen, needsTracks]);

  // What the run will and will not touch, before it is started.
  const scope = useMemo(
    () =>
      chosen.reduce(
        (acc, v) => ({
          events: acc.events + v.event_count,
          kept: acc.kept + v.reviewed,
        }),
        { events: 0, kept: 0 },
      ),
    [chosen],
  );

  const run = async () => {
    const names = [...selected];
    if (!names.length) {
      toast.warning('Select at least one video');
      return;
    }
    try {
      const job = await apiFetch<Job>(API.association.predict, {
        method: 'POST',
        body: {
          videos: names,
          checkpoint: policy === RULE ? null : policy,
          stop_vllm: stopVllm,
        },
      });
      upsertJob(job);
      toast.success(`Started Association Predict for ${names.length} video(s)`);
    } catch (e) {
      toast.error(`Association Predict start failed: ${errMsg(e)}`);
    }
  };

  return (
    <div className="mx-auto max-w-screen-2xl space-y-5">
      <PageHeader
        actions={
          <>
            <Button size="sm" onClick={() => navigate('/association-label')}>
              Open Association Label
            </Button>
            <Button intent="primary" onClick={run} disabled={Boolean(blocked)}>
              Run Association Predict
            </Button>
          </>
        }
      />

      <div className="grid grid-cols-2 gap-3.5 lg:grid-cols-4">
        <StatTile label="Videos" value={videos.length} tintClass="text-primary-light" />
        <StatTile label="Selected" value={selected.size} tintClass="text-primary-light" />
        <StatTile label="Events in scope" value={scope.events} tintClass="text-text-muted" />
        <StatTile
          label="Labeled — kept"
          value={scope.kept}
          tintClass="text-emerald-400"
        />
      </div>

      <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.6fr)]">
        <Card>
          <SectionLabel>Policy</SectionLabel>
          <p className="mb-3 text-xs leading-relaxed text-text-muted">
            Re-decides the actor for every event <em>nobody has ruled on</em>,
            from the detections already stored in the records — no person
            detection, no GPU. Only picks that actually move are re-cropped, so
            re-running with the same policy is close to free. Embeddings of
            changed events go stale; back them up on ReID Predict afterwards.
          </p>
          <label className="block text-xs text-text-secondary">
            <span className="mb-1 block">Decide with</span>
            <select
              value={policy}
              onChange={(e) => setPolicy(e.target.value)}
              className="w-full cursor-pointer appearance-none rounded-lg border border-border-light bg-surface-50 px-3 py-1.5 text-xs text-text-primary focus:border-primary/50 focus:outline-none"
            >
              <option value={RULE}>rule-based (geometry, never abstains)</option>
              {checkpoints.map((c) => (
                <option key={c.name} value={c.name}>
                  {c.name} — {c.feature_set}
                </option>
              ))}
            </select>
          </label>

          {chosenCheckpoint && (
            <dl className="mt-3 space-y-1 rounded-lg border border-border bg-surface-50 px-3 py-2 text-[11px]">
              {(
                [
                  ['Answers', 'auto_coverage'],
                  ['Right when it answers', 'selective_accuracy'],
                  ['Spots occlusion', 'occluded_rejection_rate'],
                ] as const
              ).map(([label, key]) => {
                const value = chosenCheckpoint.metrics.grouped_oof?.[key];
                return (
                  <div key={key} className="flex justify-between">
                    <dt className="text-text-muted">{label}</dt>
                    <dd className="font-mono tabular-nums text-text-primary">
                      {value == null ? '—' : `${(value * 100).toFixed(1)}%`}
                    </dd>
                  </div>
                );
              })}
              <p className="pt-1 text-text-muted">
                Grouped out-of-fold over {chosenCheckpoint.training.stems.length}{' '}
                videos — measured on videos the model had not seen.
              </p>
            </dl>
          )}

          <label className="mt-3 flex cursor-pointer items-center gap-2 text-xs text-text-secondary">
            <input
              type="checkbox"
              checked={stopVllm}
              onChange={(e) => setStopVllm(e.target.checked)}
              className="h-3.5 w-3.5 accent-primary"
            />
            Stop vLLM first
          </label>

          <Button
            intent="primary"
            onClick={run}
            disabled={Boolean(blocked)}
            className="mt-4 w-full"
            title={
              blocked
                ? `Cannot start: ${blocked}`
                : 'Re-decide the automatic picks and re-crop only what moved'
            }
          >
            Run Association Predict
          </Button>
          {blocked && (
            <p className="mt-2 rounded-lg border border-amber-500/20 bg-amber-500/10 px-3 py-1.5 text-[11px] text-amber-400">
              {blocked}
            </p>
          )}
          <Button
            onClick={tracking.run}
            disabled={Boolean(tracking.blocked)}
            className="mt-2 w-full"
            title={
              tracking.blocked
                ? `Cannot start: ${tracking.blocked}`
                : 'Dense RF-DETR + ByteTrack over every rally span. Needs rally spans only — it does not wait for action labeling.'
            }
          >
            Run Rally Tracking
            {tracking.missing > 0 ? ` (${tracking.missing} untracked)` : ''}
          </Button>
          {scope.kept > 0 && (
            <p className="mt-2 text-[11px] text-text-muted">
              {scope.kept} reviewed event(s) in the selection will be left
              exactly as they are.
            </p>
          )}
        </Card>

        <Card>
          <VideoMultiSelectList
            videos={videos}
            selected={selected}
            onSelectedChange={setSelected}
            statusOptions={[
              { value: 'all', label: 'All', predicate: () => true },
              {
                value: 'unreviewed',
                label: 'Has unreviewed',
                predicate: (v) => v.unreviewed > 0,
              },
              {
                value: 'reviewed',
                label: 'Reviewed',
                predicate: (v) => v.reviewed > 0,
              },
            ]}
            renderMeta={(v) => (
              <>
                <span className="font-mono text-[11px] tabular-nums text-text-muted">
                  {v.event_count}
                </span>
                {v.reviewed > 0 && (
                  <span title="Human verdicts — Association Predict leaves these untouched">
                    <Badge tone="success">{v.reviewed} kept</Badge>
                  </span>
                )}
                {v.auto_counts.miss > 0 && (
                  <Badge tone="warning">{v.auto_counts.miss} miss</Badge>
                )}
                <PipelineChips pipeline={v.pipeline} />
              </>
            )}
            emptyTitle="No extracted videos"
            emptySubtitle="Run ReID Predict first — association re-decides existing records"
          />
        </Card>
      </div>

      {jobs.length > 0 && (
        <Card>
          <SectionLabel>Association Predict jobs</SectionLabel>
          <div className="space-y-3">
            {jobs.map((job) => (
              <LiveJob key={job.id} job={job} onUpdate={upsertJob} />
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
