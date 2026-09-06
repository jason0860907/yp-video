import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { API, apiFetch, errMsg } from '@/lib/api';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { PageHeader } from '@/components/ui/PageHeader';
import { StatTile } from '@/components/ui/StatTile';
import { Prereqs } from '@/components/video/PipelineChips';
import { VideoMultiSelectList } from '@/components/video/VideoMultiSelectList';
import { JobsCard } from '@/components/job/JobsCard';
import {
  PredictConfigCard,
  SpotProblemBanner,
  type NumField,
} from '@/components/spot/PredictConfigCard';
import { useSpotStatus } from '@/components/spot/useSpotStatus';
import { toast } from '@/components/feedback/toast';
import { confirm } from '@/components/feedback/confirm';
import { useTypedJobs } from '@/lib/useTypedJobs';
import type { InferenceVideo, Job } from '@/types/api';

interface PredSettings {
  checkpoint: string;
  rally_min_score: number;
  max_gap_s: number;
  min_duration_s: number;
  action_min_score: number;
  batch_size: number;
  clip_len: number;
  num_workers: number;
  overwrite: boolean;
  stop_vllm: boolean;
}
const DEFAULTS: PredSettings = {
  checkpoint: '',
  rally_min_score: 0.5,
  max_gap_s: 2.0,
  min_duration_s: 4,
  action_min_score: 0.15,
  batch_size: 16,
  clip_len: 64,
  num_workers: 4,
  overwrite: false,
  stop_vllm: false,
};

const NUM_FIELDS: Array<NumField<PredSettings>> = [
  { key: 'rally_min_score', label: 'Rally min score', min: 0, max: 1, step: 0.05 },
  { key: 'max_gap_s', label: 'Merge gap (s)', min: 0, max: 30, step: 0.5 },
  { key: 'min_duration_s', label: 'Min rally (s)', min: 0, max: 60, step: 0.5 },
  { key: 'action_min_score', label: 'Action min score', min: 0, max: 1, step: 0.05 },
  { key: 'batch_size', label: 'Batch', min: 1, max: 128, step: 1 },
  { key: 'clip_len', label: 'Clip len', min: 8, max: 256, step: 8 },
  { key: 'num_workers', label: 'Workers', min: 1, max: 32, step: 1 },
];

const associationReady = (v: InferenceVideo) => v.association_blocker === null;
const complete = (v: InferenceVideo) => v.has_rally_spot && v.has_action_pre && associationReady(v);

/** Every answer the fusion model gives, in one run per video: rally spans
 *  and winners, action events inside them, and who acted. Association also
 *  needs tracking and player detection, which keep their own pages — a
 *  video missing either still gets its rallies and actions here. */
export function InferencePage() {
  const navigate = useNavigate();
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [settings, setSettings] = useState<PredSettings>(DEFAULTS);
  const { jobs, upsertJob } = useTypedJobs(['fusion_inference']);

  const videosQuery = useQuery({
    queryKey: ['inference-videos'],
    queryFn: () => apiFetch<InferenceVideo[]>(API.inference.videos),
  });
  const { spot, checkpoints, ready: spotReady, problem: spotProblem } = useSpotStatus(
    ['inference-spot'],
    API.inference.spot,
  );

  const videos = videosQuery.data ?? [];
  const readyCount = videos.filter(associationReady).length;
  const runningCount = jobs.filter((j) => j.status === 'running').length;

  const run = async () => {
    const names = [...selected];
    if (!names.length) {
      toast.warning('Select at least one video');
      return;
    }
    const existing = names
      .map((n) => videos.find((v) => v.name === n))
      .filter((v): v is InferenceVideo => Boolean(v))
      .filter((v) => v.has_rally_spot || v.has_action_pre);
    if (existing.length && settings.overwrite) {
      const ok = await confirm({
        title: 'Redo existing stages?',
        body: `This regenerates the machine rally and action output for ${existing.length} video(s). Human labels are never touched.`,
        confirmText: 'Redo',
        variant: 'danger',
      });
      if (!ok) return;
    }
    try {
      const job = await apiFetch<Job>(API.inference.start, {
        method: 'POST',
        body: {
          videos: names,
          checkpoint: settings.checkpoint,
          rally_min_score: settings.rally_min_score,
          max_gap_s: settings.max_gap_s,
          min_duration_s: settings.min_duration_s,
          action_min_score: settings.action_min_score,
          batch_size: settings.batch_size,
          clip_len: settings.clip_len,
          num_workers: settings.num_workers,
          overwrite: settings.overwrite,
          stop_vllm: settings.stop_vllm,
        },
      });
      upsertJob(job);
      toast.success(`Started Inference for ${names.length} video(s)`);
    } catch (e) {
      toast.error(`Inference start failed: ${errMsg(e)}`);
    }
  };

  return (
    <div className="mx-auto max-w-screen-2xl space-y-5">
      <PageHeader
        subtitle={
          <Prereqs
            extras={[
              { label: 'Fusion Checkpoint', hint: 'Train an Action + Rally + Winner recipe on the Train page' },
              { label: 'Track + Extract for association', hint: 'Run Rally Tracking and Player Detection; rallies and actions run without them' },
            ]}
          />
        }
        actions={
          <>
            <Button size="sm" onClick={() => navigate('/label?mode=rally')}>
              Open Label
            </Button>
            <Button intent="primary" onClick={run} disabled={!spotReady}>
              Run Inference
            </Button>
          </>
        }
      />

      <div className="grid grid-cols-2 gap-3.5 lg:grid-cols-4">
        <StatTile label="Videos" value={videos.length} tintClass="text-primary-light" />
        <StatTile label="Selected" value={selected.size} tintClass="text-primary-light" />
        <StatTile label="Association ready" value={readyCount} tintClass="text-primary-light" />
        <StatTile label="Running" value={runningCount} tintClass={runningCount ? 'text-primary-light' : 'text-text-muted'} />
      </div>

      <SpotProblemBanner problem={spotProblem} />

      <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.6fr)]">
        <PredictConfigCard
          settings={settings}
          onChange={(patch) => setSettings((s) => ({ ...s, ...patch }))}
          checkpoints={checkpoints}
          defaultCheckpoint={spot?.default_checkpoint}
          numFields={NUM_FIELDS}
          overwriteLabel="Redo stages that already have output"
          runDisabled={!spotReady}
          onRun={run}
          runLabel="Run Inference"
        />

        <Card>
          <VideoMultiSelectList
            videos={videos}
            query={videosQuery}
            selected={selected}
            onSelectedChange={setSelected}
            statusOptions={[
              { value: 'all', label: 'All', predicate: () => true },
              { value: 'no-rally', label: 'No rally output', predicate: (v) => !v.has_rally_spot },
              { value: 'no-action', label: 'No action output', predicate: (v) => !v.has_action_pre },
              { value: 'assoc-ready', label: 'Association ready', predicate: associationReady },
              { value: 'assoc-blocked', label: 'Association blocked', predicate: (v) => !associationReady(v) },
              { value: 'complete', label: 'Complete', predicate: complete },
            ]}
            quickSelects={[
              { label: 'Missing output', predicate: (v) => !v.has_rally_spot || !v.has_action_pre },
            ]}
            renderMeta={(v) => (
              <>
                {v.has_rally_spot && <Badge tone="accent">rally</Badge>}
                {v.has_action_pre && <Badge tone="accent">action</Badge>}
                {associationReady(v) ? (
                  <Badge tone="success">assoc ready</Badge>
                ) : (
                  <Badge tone="neutral">{v.pipeline.has_tracks ? 'no detections' : 'no tracks'}</Badge>
                )}
              </>
            )}
          />
        </Card>
      </div>

      <JobsCard title="Inference jobs" jobs={jobs} onUpdate={upsertJob} />
    </div>
  );
}
