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
import { STATUS_FILTER_OPTIONS } from '@/lib/labelStatus';
import { useTypedJobs } from '@/lib/useTypedJobs';
import type { Job, RallyPredictVideo } from '@/types/api';

interface PredSettings {
  checkpoint: string;
  min_score: number;
  max_gap_s: number;
  min_duration_s: number;
  batch_size: number;
  clip_len: number;
  num_workers: number;
  overwrite: boolean;
  stop_vllm: boolean;
}
const DEFAULTS: PredSettings = {
  checkpoint: '',
  min_score: 0.5,
  max_gap_s: 2.0,
  min_duration_s: 4,
  batch_size: 8,
  clip_len: 64,
  num_workers: 4,
  overwrite: false,
  stop_vllm: false,
};

const NUM_FIELDS: Array<NumField<PredSettings>> = [
  { key: 'min_score', label: 'Min score', min: 0, max: 1, step: 0.05 },
  { key: 'max_gap_s', label: 'Merge gap (s)', min: 0, max: 30, step: 0.5 },
  { key: 'min_duration_s', label: 'Min rally (s)', min: 0, max: 60, step: 0.5 },
  { key: 'batch_size', label: 'Batch', min: 1, max: 64, step: 1 },
  { key: 'clip_len', label: 'Clip len', min: 8, max: 256, step: 8 },
  { key: 'num_workers', label: 'Workers', min: 1, max: 32, step: 1 },
];

export function SpotPredictPage() {
  const navigate = useNavigate();
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [settings, setSettings] = useState<PredSettings>(DEFAULTS);
  const { jobs, upsertJob } = useTypedJobs(['rally_spot_predict']);

  const videosQuery = useQuery({
    queryKey: ['spot-predict-videos'],
    queryFn: () => apiFetch<RallyPredictVideo[]>(API.spotPredict.videos),
  });
  const { spot, checkpoints, ready: spotReady, problem: spotProblem } = useSpotStatus(
    ['spot-predict-info'],
    API.spotPredict.spot,
  );

  const videos = videosQuery.data ?? [];
  const predictedCount = videos.filter((v) => v.has_pre_annotation).length;
  const runningCount = jobs.filter((j) => j.status === 'running').length;

  const run = async () => {
    const names = [...selected];
    if (!names.length) {
      toast.warning('Select at least one video');
      return;
    }
    const existing = names
      .map((n) => videos.find((v) => v.name === n))
      .filter((v): v is RallyPredictVideo => Boolean(v))
      .filter((v) => v.has_pre_annotation);
    if (existing.length && settings.overwrite) {
      const ok = await confirm({
        title: 'Overwrite rally pre-annotations?',
        body: `This replaces the existing rally pre-annotations for ${existing.length} video(s).`,
        confirmText: 'Overwrite',
        variant: 'danger',
      });
      if (!ok) return;
    }
    try {
      const job = await apiFetch<Job>(API.spotPredict.start, {
        method: 'POST',
        body: {
          videos: names,
          checkpoint: settings.checkpoint,
          min_score: settings.min_score,
          max_gap_s: settings.max_gap_s,
          min_duration_s: settings.min_duration_s,
          batch_size: settings.batch_size,
          clip_len: settings.clip_len,
          num_workers: settings.num_workers,
          overwrite: settings.overwrite,
          stop_vllm: settings.stop_vllm,
        },
      });
      upsertJob(job);
      toast.success(`Started Rally SPOT Predict for ${names.length} video(s)`);
    } catch (e) {
      toast.error(`SPOT start failed: ${errMsg(e)}`);
    }
  };

  return (
    <div className="mx-auto max-w-screen-2xl space-y-5">
      <PageHeader
        subtitle={<Prereqs extras={[{ label: 'Rally Checkpoint', hint: 'Train one on Rally SPOT Train' }]} />}
        actions={
          <>
            <Button size="sm" onClick={() => navigate('/label?mode=rally')}>
              Open Rally Label
            </Button>
            <Button intent="primary" onClick={run} disabled={!spotReady}>
              Run SPOT
            </Button>
          </>
        }
      />

      <div className="grid grid-cols-2 gap-3.5 lg:grid-cols-4">
        <StatTile label="Videos" value={videos.length} tintClass="text-primary-light" />
        <StatTile label="Selected" value={selected.size} tintClass="text-primary-light" />
        <StatTile label="Pre-annotated" value={predictedCount} tintClass="text-primary-light" />
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
          overwriteLabel="Overwrite existing pre-annotations"
          runDisabled={!spotReady}
          onRun={run}
        />

        <Card>
          <VideoMultiSelectList
            videos={videos}
            query={videosQuery}
            selected={selected}
            onSelectedChange={setSelected}
            statusOptions={[
              ...STATUS_FILTER_OPTIONS,
              { value: 'unpredicted', label: 'No SPOT output', predicate: (v) => !v.has_pre_annotation },
              { value: 'predicted', label: 'SPOT output', predicate: (v) => Boolean(v.has_pre_annotation) },
            ]}
            renderMeta={(v) => (
              <>
                {v.has_annotation && <Badge tone="brand">labeled</Badge>}
                {v.has_pre_annotation && <Badge tone="accent">spot</Badge>}
                {v.has_vlm_pre_annotation && <Badge tone="neutral">vlm</Badge>}
              </>
            )}
          />
        </Card>
      </div>

      <JobsCard title="Rally SPOT Predict jobs" jobs={jobs} onUpdate={upsertJob} />
    </div>
  );
}
