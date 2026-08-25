import { useQuery } from '@tanstack/react-query';
import spotTrainSchema from '@contracts/spot_train_request.schema.json';
import { API, apiFetch, errMsg } from '@/lib/api';
import { useSchemaForm } from '@/lib/schemaForm';
import { useSingleJob } from '@/lib/useSingleJob';
import { SchemaForm } from '@/components/form/SchemaForm';
import {
  SchemaCheckboxField,
  SchemaNumberField,
  SchemaSearchSelectField,
  SchemaSelectField,
} from '@/components/form/SchemaFields';
import { useTrainPerformance } from '@/components/train/useTrainPerformance';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { Collapsible } from '@/components/ui/Collapsible';
import { PageHeader } from '@/components/ui/PageHeader';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { TrainJobCard } from '@/components/train/TrainJobCard';
import { TrainPerfCard } from '@/components/train/TrainPerfCard';
import { toast } from '@/components/feedback/toast';
import type { Job, RallyTrainStatus } from '@/types/api';
import type { RallyTrainRequest } from '@/types/contracts/spot_train_request.schema';

type SpotForm = Required<RallyTrainRequest>;

export function SpotTrainPage() {
  const form = useSchemaForm<SpotForm>(spotTrainSchema);
  const { values } = form;

  const statusQuery = useQuery({
    queryKey: ['spot-train-status'],
    queryFn: () => apiFetch<RallyTrainStatus>(API.spotTrain.status),
  });
  const status = statusQuery.data;

  const { job, setJob, running, cancel } = useSingleJob({
    activeJob: status?.active_job,
    label: 'SPOT rally training',
  });
  const { perf, setPerfRun } = useTrainPerformance(
    'spot-train-performance',
    API.spotTrain.performance,
    running,
  );

  const ann = status?.rally_annotations;
  const usable = Math.max(0, Number(ann?.with_local_video) || 0);
  const trainingVideos = values.video_limit > 0 ? Math.min(values.video_limit, usable) : usable;
  const ready = usable > 0;
  const canStart = !running && ready && Boolean(status?.spot_available);
  const initCheckpoints = status?.init_checkpoints ?? [];
  // Rough JPEG footprint of the frame cache this run would need (~15 KB/frame).
  const estCacheGb = ((Number(ann?.total_hours) || 0) * (trainingVideos / Math.max(1, usable)) * 3600 * values.extract_fps * 15) / 1e6;

  const start = async () => {
    try {
      const started = await apiFetch<Job>(API.spotTrain.start, { method: 'POST', body: values });
      setJob(started);
      toast.success('SPOT rally training started');
    } catch (e) {
      toast.error(`SPOT rally training failed to start: ${errMsg(e)}`);
    }
  };

  return (
    <div className="mx-auto max-w-screen-2xl space-y-5">
      <PageHeader />

      <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
        {/* Training config */}
        <Card>
          <SectionLabel>Training config</SectionLabel>
          <SchemaForm form={form}>
            <div className="grid grid-cols-2 gap-2.5 md:grid-cols-3">
              <SchemaSelectField
                name="extract_fps"
                label="Extract fps"
                options={[1, 2, 5]}
                optionLabels={{ 1: '1 fps', 2: '2 fps', 5: '5 fps' }}
              />
              <SchemaSearchSelectField
                name="init_checkpoint"
                label="Init checkpoint"
                options={initCheckpoints}
                placeholder="— From scratch —"
                className="col-span-2"
              />
              <SchemaSelectField name="feature_arch" label="Feature" />
              <SchemaSelectField name="temporal_arch" label="Temporal" />
              <SchemaSelectField
                name="camera_view"
                label="Camera view"
                optionLabels={{ all: 'All Views', broadcast: 'Broadcast', sideline: 'Sideline' }}
              />
              <SchemaNumberField name="video_limit" label="Video limit" />
              <SchemaNumberField name="num_epochs" label="Epochs" />
              <SchemaNumberField name="batch_size" label="Batch" />
              <SchemaNumberField name="learning_rate" label="LR" />
            </div>

            <Collapsible label="Advanced" className="mt-4">
              <div className="grid grid-cols-2 gap-2.5 md:grid-cols-3">
                <SchemaNumberField name="clip_len" label="Clip len" />
                <SchemaNumberField name="warm_up_epochs" label="Warmup" />
                <SchemaNumberField name="num_workers" label="Workers" />
                <SchemaNumberField name="gpu" label="GPU" />
                <SchemaSelectField name="criterion" />
                <SchemaNumberField name="start_val_epoch" label="Start val" />
                <SchemaNumberField name="epoch_num_frames" label="Epoch frames" />
              </div>
            </Collapsible>

            <div className="mt-5 border-t border-border pt-4">
              <SectionLabel>Validation split</SectionLabel>
              <div className="grid grid-cols-2 gap-2.5 md:grid-cols-3">
                <SchemaNumberField name="val_ratio" label="Val ratio" step={0.01} />
                <SchemaNumberField name="split_seed" label="Split seed" />
              </div>
              <p className="mt-2 text-[11px] text-text-muted">
                Rally training splits by ratio — per-video holdout is not supported by this trainer yet.
              </p>
            </div>

            <p className="mt-2 text-xs text-text-secondary">
              Trains on {trainingVideos} video(s); frames are extracted once at {values.extract_fps} fps (~{estCacheGb.toFixed(0)} GB cache) and reused. Video limit 0 = all annotated videos.
            </p>

            <div className="mt-3 flex flex-wrap items-center gap-3 text-xs text-text-secondary">
              <SchemaCheckboxField name="predict_winner" label="Winner head（得分方）" />
              <SchemaCheckboxField name="stop_vllm" label="Stop vLLM" />
            </div>
          </SchemaForm>

          <div className="mt-4 flex items-center gap-2">
            <Button intent="primary" onClick={start} disabled={!canStart} className="flex-1">
              {running ? 'Training…' : 'Start Training'}
            </Button>
            {running && <Button onClick={cancel}>Cancel</Button>}
          </div>
        </Card>

        {/* Dataset summary */}
        <Card>
          <SectionLabel>Rally Labels</SectionLabel>
          <div className="space-y-1.5 text-[11.5px]">
            {[
              ['Labels', `${usable} vid / ${(ann?.rallies ?? 0).toLocaleString()} rallies`],
              ['Coverage', `${(Number(ann?.rally_hours) || 0).toFixed(1)}h rally / ${(Number(ann?.total_hours) || 0).toFixed(1)}h video`],
              ['Missing', `${ann?.missing_videos ?? 0} annotation(s) without local video`],
              ['View', values.camera_view === 'all' ? 'all views' : values.camera_view],
              ['Label dir', ann?.label_dir || '—'],
              ['Ckpt dir', status?.rally_checkpoints?.dir ? `${status.rally_checkpoints.dir}/<auto run>` : '—'],
            ].map(([label, value]) => (
              <div key={label} className="flex items-center gap-3">
                <span className="w-16 flex-shrink-0 text-text-muted">{label}</span>
                <span className="min-w-0 flex-1 truncate font-mono tabular-nums text-text-secondary" title={String(value)}>
                  {value}
                </span>
              </div>
            ))}
          </div>

          {(status?.frame_caches?.length ?? 0) > 0 && (
            <div className="mt-3 space-y-1 text-[11.5px]">
              <div className="text-[10px] font-semibold uppercase tracking-widest text-text-muted">Frame caches</div>
              {status!.frame_caches!.map((c) => (
                <div key={c.fps} className="flex items-center gap-3">
                  <span className="w-16 flex-shrink-0 text-text-muted">{c.fps} fps</span>
                  <span className="font-mono tabular-nums text-text-secondary">{c.videos} video(s) cached</span>
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>

      {/* Training job */}
      <TrainJobCard
        job={job}
        progressKey="rally_train_progress"
        epochsFallback={values.num_epochs}
        onCancel={() => void cancel()}
        mapLabel="Seg mAP"
        eventNoun="rallies"
      />

      {/* Per-epoch curve + per-video mAP for the selected (or latest) run */}
      {perf && perf.entries.length > 0 && <TrainPerfCard data={perf} onSelectRun={setPerfRun} />}
    </div>
  );
}
