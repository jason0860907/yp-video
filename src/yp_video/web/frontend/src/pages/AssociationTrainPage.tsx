import { useEffect, useMemo, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';

import associationTrainSchema from '@contracts/association_train_request.schema.json';
import { toast } from '@/components/feedback/toast';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { Collapsible } from '@/components/ui/Collapsible';
import { PageHeader } from '@/components/ui/PageHeader';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { VideoMultiSelectList } from '@/components/video/VideoMultiSelectList';
import { API, apiFetch, errMsg } from '@/lib/api';
import { useSchemaForm } from '@/lib/schemaForm';
import { SchemaForm } from '@/components/form/SchemaForm';
import {
  SchemaCheckboxField,
  SchemaNumberField,
  SchemaSearchSelectField,
  SchemaSelectField,
  SchemaTextField,
} from '@/components/form/SchemaFields';
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
import type { AssociationTrainRequest } from '@/types/contracts/association_train_request.schema';

// min_length=1 generates non-empty tuple types; the form legitimately starts
// empty and the backend enforces the minimum at submit.
type AssociationForm = Omit<Required<AssociationTrainRequest>, 'train_videos' | 'val_videos'> & {
  train_videos: string[];
  val_videos: string[];
};

const isDone = (video: AssociationVideo) =>
  video.event_count > 0 && video.unreviewed === 0;

export function AssociationTrainPage() {
  const valSeeded = useRef(false);
  const startingRef = useRef(false);
  const [valVideos, setValVideos] = useState<Set<string>>(new Set());
  const form = useSchemaForm<AssociationForm>(associationTrainSchema, {
    train_videos: [],
    val_videos: [],
  });
  const { values } = form;
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
      const body: AssociationForm = {
        ...values,
        train_videos: split.train.map((video) => video.name),
        val_videos: [...valVideos],
      };
      const started = await apiFetch<Job>(API.association.train, { method: 'POST', body });
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

          <SchemaForm form={form}>
            <div className="grid grid-cols-2 gap-2.5 md:grid-cols-3">
              <SchemaTextField
                name="run_name"
                label="Run name"
                placeholder="yp_actor_YYYYMMDD-HHMMSS"
              />
              <SchemaSearchSelectField
                name="init_checkpoint"
                label="Init checkpoint"
                options={status?.init_checkpoints ?? []}
                placeholder="— New model (ImageNet-initialized backbone) —"
                className="col-span-2"
              />
              <SchemaSelectField name="backbone" label="Visual backbone" />
              <SchemaNumberField name="num_epochs" label="Epochs" />
              <SchemaNumberField name="batch_size" label="Batch" />
              <SchemaNumberField name="learning_rate" label="Head LR" />
              <SchemaNumberField name="backbone_learning_rate" label="Backbone LR" />
            </div>

            <Collapsible label="Advanced" className="mt-4">
              <div className="grid grid-cols-2 gap-2.5 md:grid-cols-3">
                <SchemaNumberField name="warm_up_epochs" label="Warm-up" />
                <SchemaNumberField name="crop_dim" label="Image size" />
                <SchemaNumberField name="num_workers" label="Workers" />
                <SchemaNumberField name="gpu" label="GPU" />
              </div>
            </Collapsible>

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

            <div className="mt-4">
              <SchemaCheckboxField name="stop_vllm" label="Stop vLLM before taking the GPU" />
            </div>
          </SchemaForm>

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
        epochsFallback={values.num_epochs}
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
