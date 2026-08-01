import { useEffect, useMemo, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';

import { toast } from '@/components/feedback/toast';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { PageHeader } from '@/components/ui/PageHeader';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { VideoMultiSelectList } from '@/components/video/VideoMultiSelectList';
import { API, apiFetch, errMsg } from '@/lib/api';
import {
  Field,
  InitCheckpointSelect,
  NumberField,
  SelectArch,
  fieldCls,
} from '@/components/train/Field';
import { CheckpointsCard } from '@/components/train/CheckpointsCard';
import { TrainJobCard } from '@/components/train/TrainJobCard';
import { TrainPerfCard } from '@/components/train/TrainPerfCard';
import { useTrainPerformance } from '@/components/train/useTrainPerformance';
import { useSingleJob } from '@/lib/useSingleJob';
import type {
  AssociationVideo,
  Job,
  ReidAssociationStatus,
} from '@/types/api';

type Form = {
  run_name: string;
  init_checkpoint: string;
  num_epochs: number;
  batch_size: number;
  learning_rate: number;
  backbone_learning_rate: number;
  warm_up_epochs: number;
  gpu: number;
  backbone: 'rny002' | 'rny002_gsm' | 'rny008' | 'rny008_gsm' | 'rn18' | 'rn50';
  crop_dim: number;
  num_workers: number;
  stop_vllm: boolean;
};

const INITIAL_FORM: Form = {
  run_name: '',
  init_checkpoint: '',
  num_epochs: 40,
  batch_size: 8,
  learning_rate: 0.0003,
  backbone_learning_rate: 0.00003,
  warm_up_epochs: 3,
  gpu: 0,
  backbone: 'rny002',
  crop_dim: 224,
  num_workers: 4,
  stop_vllm: false,
};

const BACKBONES = ['rny002', 'rny002_gsm', 'rny008', 'rny008_gsm', 'rn18', 'rn50'] as const;

const isDone = (video: AssociationVideo) =>
  video.event_count > 0 && video.unreviewed === 0;

export function AssociationTrainPage() {
  const valSeeded = useRef(false);
  const startingRef = useRef(false);
  const [valVideos, setValVideos] = useState<Set<string>>(new Set());
  const [form, setForm] = useState<Form>(INITIAL_FORM);
  const [starting, setStarting] = useState(false);

  const videosQuery = useQuery({
    queryKey: ['association-videos'],
    queryFn: () => apiFetch<AssociationVideo[]>(API.association.videos),
  });
  const statusQuery = useQuery({
    queryKey: ['actor-association-status'],
    queryFn: () => apiFetch<ReidAssociationStatus>(API.association.status),
  });

  const videos = useMemo(() => videosQuery.data ?? [], [videosQuery.data]);
  const status = statusQuery.data;

  const { job, setJob, running: training, cancel } = useSingleJob({
    activeJob: status?.active_job,
    label: 'yp-association training',
  });
  const { perf, setPerfRun } = useTrainPerformance(
    'association-train-performance',
    API.association.trainPerformance,
    training,
  );

  // Seed the validation split once: roughly 20% of the Done videos. The
  // training set is implicit — every other Done video.
  useEffect(() => {
    if (valSeeded.current || videosQuery.data === undefined) return;
    const ready = videosQuery.data.filter(isDone);
    const valCount = ready.length >= 2
      ? Math.max(1, Math.round(ready.length * 0.2))
      : 0;
    setValVideos(new Set(ready.slice(ready.length - valCount).map((video) => video.name)));
    valSeeded.current = true;
  }, [videosQuery.data]);

  const set = <K extends keyof Form>(key: K, value: Form[K]) =>
    setForm((previous) => ({ ...previous, [key]: value }));

  const split = useMemo(() => {
    const done = videos.filter(isDone);
    const train = done.filter((video) => !valVideos.has(video.name));
    const reviewed = (rows: AssociationVideo[]) =>
      rows.reduce((total, video) => total + video.reviewed, 0);
    return {
      done,
      train,
      trainEvents: reviewed(train),
      valEvents: reviewed(videos.filter((video) => valVideos.has(video.name))),
    };
  }, [videos, valVideos]);

  const busy = starting || training;
  const canTrain =
    !busy
    && status?.spot_available === true
    && split.train.length > 0
    && valVideos.size > 0
    && split.trainEvents > 0
    && split.valEvents > 0;

  const start = async () => {
    if (startingRef.current) return;
    if (!canTrain) {
      toast.warning('Keep at least one reviewed video in each split');
      return;
    }
    startingRef.current = true;
    setStarting(true);
    toast.info('Validating the selected videos and creating the training job…');
    try {
      const started = await apiFetch<Job>(API.association.train, {
        method: 'POST',
        body: {
          train_videos: split.train.map((video) => video.name),
          val_videos: [...valVideos],
          run_name: form.run_name.trim() || null,
          init_checkpoint: form.init_checkpoint || null,
          num_epochs: form.num_epochs,
          batch_size: form.batch_size,
          learning_rate: form.learning_rate,
          backbone_learning_rate: form.backbone_learning_rate,
          warm_up_epochs: form.warm_up_epochs,
          gpu: form.gpu,
          backbone: form.backbone,
          crop_dim: form.crop_dim,
          num_workers: form.num_workers,
          stop_vllm: form.stop_vllm,
        },
      });
      setJob(started);
      toast.success('yp-association training started');
    } catch (error) {
      toast.error(`Training failed to start: ${errMsg(error)}`);
    } finally {
      startingRef.current = false;
      setStarting(false);
    }
  };

  return (
    <div className="mx-auto max-w-screen-2xl space-y-5">
      <PageHeader />

      <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
        {/* Training config */}
        <Card>
          <SectionLabel>Training config</SectionLabel>
          <p className="mb-4 text-xs leading-relaxed text-text-muted">
            One reviewed action event is one training example. The model reads
            nine synchronized frames, the contact point and every candidate
            track, then chooses a player, occluded or untracked. It has its own
            dataset, optimizer, validation metrics and checkpoint; action
            classification and location loss are not part of this run.
          </p>

          {status?.spot_available === false ? (
            <p className="mb-4 rounded-lg border border-red-500/20 bg-red-500/10 px-3 py-2 text-xs text-red-400">
              yp-spot or its Python environment is unavailable on this machine.
            </p>
          ) : null}

          <div className="grid grid-cols-2 gap-2.5 md:grid-cols-3">
            <Field label="Run name">
              <input
                value={form.run_name}
                onChange={(event) => set('run_name', event.target.value)}
                placeholder="yp_actor_YYYYMMDD-HHMMSS"
                className={fieldCls}
              />
            </Field>
            <InitCheckpointSelect
              value={form.init_checkpoint}
              onChange={(value) => set('init_checkpoint', value)}
              options={status?.init_checkpoints ?? []}
              emptyLabel="— New model (ImageNet-initialized backbone) —"
            />
            <NumberField label="Epochs" value={form.num_epochs} min={1} onChange={(value) => set('num_epochs', value)} />
            <NumberField label="Batch" value={form.batch_size} min={1} onChange={(value) => set('batch_size', value)} />
            <NumberField label="Head LR" value={form.learning_rate} min={0} step={0.00001} onChange={(value) => set('learning_rate', value)} />
            <NumberField label="Backbone LR" value={form.backbone_learning_rate} min={0} step={0.00001} onChange={(value) => set('backbone_learning_rate', value)} />
            <NumberField label="Warm-up" value={form.warm_up_epochs} min={0} onChange={(value) => set('warm_up_epochs', value)} />
            <NumberField label="GPU" value={form.gpu} min={0} max={7} onChange={(value) => set('gpu', value)} />
            <Field label="Visual backbone">
              <SelectArch
                value={form.backbone}
                options={BACKBONES}
                onChange={(value) => set('backbone', value as Form['backbone'])}
              />
            </Field>
            <NumberField label="Image size" value={form.crop_dim} min={64} max={512} onChange={(value) => set('crop_dim', value)} />
            <NumberField label="Workers" value={form.num_workers} min={0} onChange={(value) => set('num_workers', value)} />
          </div>

          <div className="mt-5 border-t border-border pt-4">
            <SectionLabel>Validation split</SectionLabel>
            <p className="mb-2 text-[11px] text-text-muted">
              trains on {split.train.length} videos · {split.trainEvents} reviewed events
            </p>
            <VideoMultiSelectList
              videos={videos}
              query={videosQuery}
              selected={valVideos}
              onSelectedChange={setValVideos}
              title="Validation videos"
              statusOptions={[
                { value: 'all', label: 'All', predicate: () => true },
                { value: 'done', label: 'Done', predicate: isDone },
                {
                  value: 'partial',
                  label: 'In progress',
                  predicate: (video) => video.reviewed > 0 && video.unreviewed > 0,
                },
              ]}
              quickSelects={[{ label: 'Clear', predicate: () => false }]}
              renderMeta={(video) => (
                <>
                  <span className="font-mono text-[11px] tabular-nums text-text-muted">
                    {video.reviewed}/{video.event_count}
                  </span>
                  {isDone(video) ? (
                    <Badge tone="success">Done</Badge>
                  ) : video.reviewed > 0 ? (
                    <Badge tone="warning">{video.unreviewed} left</Badge>
                  ) : (
                    <Badge>Unlabeled</Badge>
                  )}
                </>
              )}
              maxHeightClass="max-h-[42vh]"
              emptyTitle="No association videos"
              emptySubtitle="Finish tracking, actions and Association Label first"
            />
            <p className="mt-2 text-[10px] leading-relaxed text-text-muted">
              Selected videos are validation only; every other Done video trains.
            </p>
          </div>

          <label className="mt-4 inline-flex cursor-pointer items-center gap-2 text-xs text-text-secondary">
            <input
              type="checkbox"
              checked={form.stop_vllm}
              onChange={(event) => set('stop_vllm', event.target.checked)}
              className="h-3.5 w-3.5 accent-primary"
            />
            Stop vLLM before taking the GPU
          </label>

          <div className="mt-4 flex items-center gap-2">
            <Button intent="primary" disabled={!canTrain} onClick={() => void start()} className="flex-1">
              {starting ? 'Starting…' : training ? 'Training…' : 'Start Training'}
            </Button>
            {training && <Button onClick={() => void cancel()}>Cancel</Button>}
          </div>
          {!canTrain && !busy ? (
            <p className="mt-2 text-[11px] text-amber-400">
              Keep at least one reviewed video in each split.
            </p>
          ) : null}
        </Card>

        {/* Dataset summary */}
        <Card>
          <SectionLabel>Association dataset</SectionLabel>
          <div className="space-y-1.5 text-[11.5px]">
            {[
              ['Done', `${split.done.length} video(s)`],
              ['Reviewed', `${split.trainEvents + split.valEvents} events`],
              ['Training', `${split.train.length} vid / ${split.trainEvents} ev`],
              ['Validation', `${valVideos.size} vid / ${split.valEvents} ev`],
              ['Frame dir', status?.frame_dir || '—'],
            ].map(([label, value]) => (
              <div key={label} className="flex items-center gap-3">
                <span className="w-16 flex-shrink-0 text-text-muted">{label}</span>
                <span className="min-w-0 flex-1 truncate font-mono tabular-nums text-text-secondary" title={String(value)}>
                  {value}
                </span>
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* Training job */}
      <TrainJobCard
        job={job}
        progressKey="association_train_progress"
        epochsFallback={form.num_epochs}
        onCancel={() => void cancel()}
      />

      {/* Per-epoch task metrics for the selected (or latest) run */}
      {perf && perf.entries.length > 0 && (
        <TrainPerfCard data={perf} onSelectRun={setPerfRun} />
      )}

      <CheckpointsCard
        title="Available association checkpoints"
        checkpoints={status?.association_checkpoints ?? []}
      />
    </div>
  );
}
