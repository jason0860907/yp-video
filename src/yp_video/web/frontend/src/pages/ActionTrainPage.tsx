import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import actionTrainSchema from '@contracts/action_train_request.schema.json';
import { API, apiFetch, errMsg } from '@/lib/api';
import { useSchemaForm } from '@/lib/schemaForm';
import { useSingleJob } from '@/lib/useSingleJob';
import { SchemaForm } from '@/components/form/SchemaForm';
import {
  SchemaCheckboxField,
  SchemaNumberField,
  SchemaSearchSelectField,
  SchemaSelectField,
  SchemaTextField,
} from '@/components/form/SchemaFields';
import { useTrainPerformance } from '@/components/train/useTrainPerformance';
import { VideoMultiSelectList } from '@/components/video/VideoMultiSelectList';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { Collapsible } from '@/components/ui/Collapsible';
import { PageHeader } from '@/components/ui/PageHeader';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { TrainJobCard } from '@/components/train/TrainJobCard';
import { TrainPerfCard } from '@/components/train/TrainPerfCard';
import { toast } from '@/components/feedback/toast';
import type { ActionTrainStatus, CutKind, Job } from '@/types/api';
import type { AnnotationActionTrainRequest } from '@/types/contracts/action_train_request.schema';

type ActionForm = Required<AnnotationActionTrainRequest>;

export function ActionTrainPage() {
  const form = useSchemaForm<ActionForm>(actionTrainSchema, { training_mode: 'holdout' });
  const { values } = form;
  const [holdoutVideos, setHoldoutVideos] = useState<Set<string>>(new Set());

  const statusQuery = useQuery({
    queryKey: ['action-train-status'],
    queryFn: () => apiFetch<ActionTrainStatus>(API.actionTrain.status),
  });
  const status = statusQuery.data;

  const { job, setJob, running, cancel } = useSingleJob({
    activeJob: status?.active_job,
    label: 'Action training',
  });
  // Per-epoch validation curve + per-video breakdown; refresh while training.
  const { perf, setPerfRun } = useTrainPerformance(
    'action-train-performance',
    API.actionTrain.performance,
    running,
  );

  // Counts reflect the selected camera view (all = totals; otherwise the
  // per-view breakdown from the status endpoint).
  const ann = status?.action_annotations;
  const viewStats = values.camera_view === 'all' ? undefined : ann?.by_view?.[values.camera_view];
  const stats = {
    videos: Math.max(0, Number((viewStats ?? ann)?.videos) || 0),
    actions: Math.max(0, Number((viewStats ?? ann)?.events) || 0),
    frames: Math.max(0, Number((viewStats ?? ann)?.frames) || 0),
  };
  // Per-video action counts for the selected view, most-labelled first.
  const perVideo = (ann?.per_video ?? [])
    .filter((v) => values.camera_view === 'all' || v.view === values.camera_view)
    .slice()
    .sort((a, b) => b.events - a.events);
  // Validation is always a manual per-video holdout; the list is the same
  // view-filtered corpus the run trains on.
  const holdoutChoices = useMemo(
    () => perVideo.map((v) => ({ ...v, name: v.video, kind: v.view as CutKind })),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [ann?.per_video, values.camera_view],
  );
  const ready = stats.actions > 0;
  const canStart = !running && ready && Boolean(status?.spot_available) && holdoutVideos.size > 0;

  const start = async () => {
    try {
      const body: ActionForm = {
        ...values,
        holdout_videos: holdoutChoices
          .filter((v) => holdoutVideos.has(v.name))
          .map((v) => `${v.name}_actions.jsonl`),
      };
      const started = await apiFetch<Job>(API.actionTrain.start, { method: 'POST', body });
      setJob(started);
      toast.success('Action training started');
    } catch (e) {
      toast.error(`Action training failed to start: ${errMsg(e)}`);
    }
  };

  const initCheckpoints = status?.init_checkpoints ?? [];

  return (
    <div className="mx-auto max-w-screen-2xl space-y-5">
      <PageHeader />

      <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
        {/* Training config */}
        <Card>
          <SectionLabel>Training config</SectionLabel>
          <SchemaForm form={form}>
            <div className="grid grid-cols-2 gap-2.5 md:grid-cols-3">
              <SchemaTextField name="dataset" />
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
                name="audio_backend"
                label="Audio"
                optionLabels={{ logmel: 'logmel (late fusion)', none: 'none (visual only)' }}
              />
              <SchemaSelectField
                name="camera_view"
                label="Camera view"
                optionLabels={{ all: 'All Views', broadcast: 'Broadcast', sideline: 'Sideline' }}
              />
              <SchemaNumberField name="num_epochs" label="Epochs" />
              <SchemaNumberField name="batch_size" label="Batch" />
              <SchemaNumberField name="learning_rate" label="LR" />
            </div>

            <Collapsible label="Advanced" className="mt-4">
              <div className="grid grid-cols-2 gap-2.5 md:grid-cols-3">
                <SchemaNumberField name="clip_len" label="Clip len" />
                <SchemaNumberField name="sample_fps" label="Sample fps" />
                <SchemaNumberField name="acc_grad_iter" label="Grad accum" />
                <SchemaNumberField name="warm_up_epochs" label="Warmup" />
                <SchemaNumberField name="num_workers" label="Workers" />
                <SchemaNumberField name="gpu" label="GPU" />
                <SchemaSelectField name="criterion" />
                <SchemaNumberField name="start_val_epoch" label="Start val" />
                <SchemaNumberField name="epoch_num_frames" label="Epoch frames" />
              </div>
            </Collapsible>

            <div className="mt-5 border-t border-border pt-4">
              <VideoMultiSelectList
                videos={holdoutChoices}
                selected={holdoutVideos}
                onSelectedChange={setHoldoutVideos}
                title="Validation split"
                quickSelects={[
                  { label: 'Load saved val set', predicate: (v) => Boolean(v.is_val) },
                  { label: 'Clear', predicate: () => false },
                ]}
                renderMeta={(v) => (
                  <>
                    {v.is_val && <Badge tone="neutral">saved val</Badge>}
                    <span className="font-mono text-[10px] tabular-nums text-text-muted">{v.events}</span>
                  </>
                )}
                maxHeightClass="max-h-64"
                emptyTitle="No labelled videos in this camera view"
                query={statusQuery}
              />
              <p className="mt-2 text-[10px] leading-relaxed text-text-muted">
                Selected videos are validation only; every other matching action annotation trains.
              </p>
            </div>

            <div className="mt-3 flex flex-wrap items-center gap-3 text-xs text-text-secondary">
              <SchemaCheckboxField name="predict_location" label="Predict location" />
              <SchemaCheckboxField name="stop_vllm" label="Stop vLLM" />
            </div>
          </SchemaForm>

          <div className="mt-4 flex items-center gap-2">
            <Button intent="primary" onClick={start} disabled={!canStart} className="flex-1">
              {running ? 'Training…' : 'Start Training'}
            </Button>
            {running && (
              <Button onClick={cancel}>Cancel</Button>
            )}
          </div>
        </Card>

        {/* Dataset summary */}
        <Card>
          <SectionLabel>YP Action Labels</SectionLabel>
          <div className="space-y-1.5 text-[11.5px]">
            {[
              ['Labels', `${stats.videos} vid / ${stats.actions} ev`],
              ['Frames', stats.frames.toLocaleString()],
              ['Mode', 'manual holdout'],
              ['View', values.camera_view === 'all' ? 'all views' : values.camera_view],
              ['Label dir', status?.action_annotations?.label_dir || '—'],
              ['Frame dir', status?.action_annotations?.frame_dir || '—'],
              ['Ckpt dir', status?.action_annotations?.checkpoint_dir ? `${status.action_annotations.checkpoint_dir}/<auto run>` : '—'],
            ].map(([label, value]) => (
              <div key={label} className="flex items-center gap-3">
                <span className="w-16 flex-shrink-0 text-text-muted">{label}</span>
                <span className="min-w-0 flex-1 truncate font-mono tabular-nums text-text-secondary" title={String(value)}>
                  {value}
                </span>
              </div>
            ))}
          </div>

          {perVideo.length > 0 && (
            <details className="mt-3">
              <summary className="cursor-pointer text-[11px] text-text-muted hover:text-text-primary">
                Per-video actions ({perVideo.length})
              </summary>
              <div className="mt-1.5 max-h-64 overflow-y-auto rounded-lg border border-border">
                <table className="w-full text-[10.5px] tabular-nums">
                  <thead className="sticky top-0 bg-surface-100 text-text-muted">
                    <tr>
                      <th className="px-2 py-1 text-left font-normal">Video</th>
                      <th className="px-2 py-1 text-right font-normal">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {perVideo.map((v) => (
                      <tr key={v.video} className="border-t border-border/50">
                        <td className="max-w-0 truncate px-2 py-1 text-text-secondary" title={v.video}>
                          {v.is_val && (
                            <span className="mr-1 rounded bg-accent/20 px-1 text-[9px] font-semibold uppercase text-accent">val</span>
                          )}
                          {v.video}
                        </td>
                        <td className="px-2 py-1 text-right font-mono text-text-secondary">{v.events}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          )}
        </Card>
      </div>

      {/* Training job */}
      <TrainJobCard
        job={job}
        progressKey="action_train_progress"
        epochsFallback={values.num_epochs}
        onCancel={() => void cancel()}
      />

      {/* Per-epoch curve + per-video mAP for the selected (or latest) run */}
      {perf && perf.entries.length > 0 && <TrainPerfCard data={perf} onSelectRun={setPerfRun} />}
    </div>
  );
}
