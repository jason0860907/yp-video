import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { API, apiFetch, errMsg } from '@/lib/api';
import { cn } from '@/lib/cn';
import { fieldCls } from '@/components/form/Field';
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
import type { ActionVideo, Job } from '@/types/api';

interface PredSettings {
  checkpoint: string;
  min_score: number;
  batch_size: number;
  clip_len: number;
  decoder: 'opencv' | 'nvdec';
  decode_producers: number;
  prefetch_factor: number;
  decode_chunk_frames: number;
  overwrite: boolean;
  stop_vllm: boolean;
}
const DEFAULTS: PredSettings = {
  checkpoint: '',
  min_score: 0.15,
  batch_size: 16,
  clip_len: 64,
  decoder: 'opencv',
  decode_producers: 2,
  prefetch_factor: 2,
  decode_chunk_frames: 256,
  overwrite: false,
  stop_vllm: false,
};

const NUM_FIELDS: Array<NumField<PredSettings>> = [
  { key: 'min_score', label: 'Min score', min: 0, max: 1, step: 0.05 },
  { key: 'batch_size', label: 'Batch', min: 1, max: 128, step: 1 },
  { key: 'clip_len', label: 'Clip len', min: 8, max: 256, step: 8 },
  { key: 'decode_producers', label: 'Producers', min: 1, max: 8, step: 1 },
  { key: 'prefetch_factor', label: 'Prefetch', min: 1, max: 8, step: 1 },
  { key: 'decode_chunk_frames', label: 'Chunk', min: 1, max: 512, step: 16 },
];

const hasLabels = (v: ActionVideo) => Boolean(v.has_action_annotation);

export function ActionPredictPage() {
  const navigate = useNavigate();
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [settings, setSettings] = useState<PredSettings>(DEFAULTS);
  const { jobs, upsertJob } = useTypedJobs(['spot_prelabel_batch']);

  const videosQuery = useQuery({
    queryKey: ['action-videos'],
    queryFn: () => apiFetch<ActionVideo[]>(API.actionAnnotate.videos),
  });
  const { spot, checkpoints, ready: spotReady, problem: spotProblem } = useSpotStatus(
    ['spot-info'],
    API.actionAnnotate.spot,
  );

  const videos = videosQuery.data ?? [];
  const labeledCount = videos.filter(hasLabels).length;
  const runningCount = jobs.filter((j) => j.status === 'running').length;

  const run = async () => {
    const names = [...selected];
    if (!names.length) {
      toast.warning('Select at least one video');
      return;
    }
    const existing = names
      .map((n) => videos.find((v) => v.name === n))
      .filter((v): v is ActionVideo => Boolean(v))
      .filter((v) => Boolean(v.has_action_pre_annotation));
    if (existing.length && !settings.overwrite) {
      toast.warning(`${existing.length} selected video(s) already have pre-labels`);
      return;
    }
    if (existing.length && settings.overwrite) {
      const ok = await confirm({
        title: 'Rebuild pre-labels?',
        body: `This regenerates machine pre-labels for ${existing.length} video(s). Hand-made annotations are never touched — the editor keeps preferring them.`,
        confirmText: 'Rebuild',
      });
      if (!ok) return;
    }
    try {
      const job = await apiFetch<Job>(API.actionAnnotate.prelabelBatch, {
        method: 'POST',
        body: {
          videos: names,
          checkpoint: settings.checkpoint,
          min_score: settings.min_score,
          batch_size: settings.batch_size,
          clip_len: settings.clip_len,
          num_workers: settings.decode_producers,
          decoder: settings.decoder,
          decode_producers: settings.decode_producers,
          prefetch_factor: settings.prefetch_factor,
          decode_chunk_frames: settings.decode_chunk_frames,
          use_amp: true,
          overwrite: settings.overwrite,
          stop_vllm: settings.stop_vllm,
        },
      });
      upsertJob(job);
      toast.success(`Started Action Predict for ${names.length} video(s)`);
    } catch (e) {
      toast.error(`SPOT start failed: ${errMsg(e)}`);
    }
  };

  return (
    <div className="mx-auto max-w-screen-2xl space-y-5">
      <PageHeader
        subtitle={
          <Prereqs
            stages={['rallies']}
            extras={[{ label: 'Action Checkpoint', hint: 'Train an Action recipe on the Train page' }]}
          />
        }
        actions={
          <>
            <Button size="sm" onClick={() => navigate('/label?mode=action')}>
              Open Label
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
        <StatTile label="Labeled" value={labeledCount} tintClass="text-primary-light" />
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
          overwriteLabel="Overwrite existing action pre-annotations"
          runDisabled={!spotReady}
          onRun={run}
        >
          <div>
            <label className="mb-1 block text-[10px] uppercase tracking-wide text-text-muted">Decoder</label>
            <select
              value={settings.decoder}
              onChange={(e) => setSettings((s) => ({ ...s, decoder: e.target.value as PredSettings['decoder'] }))}
              className={cn(fieldCls, 'cursor-pointer appearance-none')}
            >
              <option value="opencv">OpenCV (CPU)</option>
              <option value="nvdec">NVDEC (GPU)</option>
            </select>
          </div>
        </PredictConfigCard>

        <Card>
          <VideoMultiSelectList
            videos={videos}
            query={videosQuery}
            selected={selected}
            onSelectedChange={setSelected}
            statusOptions={STATUS_FILTER_OPTIONS}
            renderMeta={(v) => (
              <span className={cn('font-mono text-[11px] tabular-nums', hasLabels(v) ? 'text-primary-light' : 'text-text-muted')}>
                {v.event_count || 0}
              </span>
            )}
          />
        </Card>
      </div>

      <JobsCard title="Action Predict jobs" jobs={jobs} onUpdate={upsertJob} />
    </div>
  );
}
